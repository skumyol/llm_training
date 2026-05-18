import json
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from src.metrics_report import log_metrics_to_mlflow, write_metrics_bundle


SLOW_PATH_TRIGGERS = {
    "value_conflict": {"strong"},
    "response_policy": {"threaten", "negotiate"},
}


def should_route_slow(D_t: dict, N_t: dict) -> bool:
    if N_t.get("value_conflict") in SLOW_PATH_TRIGGERS["value_conflict"]:
        return True
    if N_t.get("secrecy_pressure") == "high" and D_t.get("reveal_decision") in {"hint", "partial", "full"}:
        return True
    if D_t.get("response_policy") in SLOW_PATH_TRIGGERS["response_policy"]:
        return True
    return False


def _load_predicted_zt(predicted_zt_file: str) -> dict[tuple[str, int], dict]:
    """Load predicted Z_t JSONL into a lookup keyed by (episode_id, turn_idx)."""
    lookup: dict[tuple[str, int], dict] = {}
    with open(predicted_zt_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ep = str(rec.get("episode_id", ""))
            turn = rec.get("turn_idx", rec.get("turn", None))
            if ep and turn is not None:
                lookup[(ep, int(turn))] = rec
    return lookup


def eval_routing(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    import mlflow
    from src.mlflow_utils import setup_mlflow

    setup_mlflow(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    test_trace = cfg["data"].get("test_trace_file")
    if not test_trace or not Path(test_trace).exists():
        print(f"Test trace file not found: {test_trace}")
        return {}

    mode = cfg.get("routing_mode", "gold")
    missing_prediction_policy = cfg.get("routing_missing_predictions", "error")
    if mode not in {"gold", "predicted"}:
        raise ValueError(f"routing_mode must be 'gold' or 'predicted', got {mode!r}")
    if missing_prediction_policy not in {"error", "skip"}:
        raise ValueError(
            "routing_missing_predictions must be 'error' or 'skip', "
            f"got {missing_prediction_policy!r}"
        )
    predicted_zt_file = cfg.get("predicted_zt_file")
    pred_lookup: dict[tuple[str, int], dict] = {}
    if mode == "predicted":
        if not predicted_zt_file or not Path(predicted_zt_file).exists():
            print(f"Predicted Z_t file required for mode='predicted' but not found: {predicted_zt_file}")
            print("Run eval_latent first; it writes eval_results/predicted_zt.jsonl")
            return {}
        pred_lookup = _load_predicted_zt(predicted_zt_file)
        print(f"Routing on predicted Z_t: loaded {len(pred_lookup)} predictions")

    results_dir = Path(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    total = 0
    missing_predictions = 0
    trace_records = 0

    with open(test_trace) as f:
        for line in tqdm(f, desc="Evaluating routing"):
            line = line.strip()
            if not line:
                continue
            trace_records += 1
            record = json.loads(line)
            D_t = record.get("D_t", {})
            N_t = record.get("N_t", {})

            gold_slow = _gold_slow_path(D_t, N_t, record)

            if mode == "predicted":
                ep = str(record.get("episode_id", ""))
                turn = record.get("turn_idx", record.get("turn", None))
                pred_rec = pred_lookup.get((ep, int(turn))) if turn is not None else None
                if pred_rec is None:
                    missing_predictions += 1
                    if missing_prediction_policy == "skip":
                        continue
                    raise ValueError(
                        "Missing predicted Z_t for "
                        f"(episode_id={ep!r}, turn_idx={turn!r}). "
                        "Run eval_latent on the same split/order as test_trace, "
                        "or set routing_missing_predictions: skip to report coverage."
                    )
                else:
                    # Overlay predicted routing fields onto the gold structure
                    pred_D_t = {**D_t}
                    pred_N_t = {**N_t}
                    for key in ["response_policy", "reveal_decision"]:
                        if key in pred_rec:
                            pred_D_t[key] = pred_rec[key]
                    for key in ["value_conflict", "secrecy_pressure"]:
                        if key in pred_rec:
                            pred_N_t[key] = pred_rec[key]
                pred_slow = should_route_slow(pred_D_t, pred_N_t)
            else:
                pred_slow = should_route_slow(D_t, N_t)

            if gold_slow and pred_slow:
                true_positives += 1
            elif not gold_slow and pred_slow:
                false_positives += 1
            elif gold_slow and not pred_slow:
                false_negatives += 1
            else:
                true_negatives += 1
            total += 1

    precision = true_positives / max(1, true_positives + false_positives)
    recall    = true_positives / max(1, true_positives + false_negatives)
    fp_rate   = false_positives / max(1, false_positives + true_negatives)
    f1        = 2 * precision * recall / max(1e-9, precision + recall)

    metrics = {
        "routing_mode":      mode,
        "routing_precision": precision,
        "routing_recall":    recall,
        "routing_f1":        f1,
        "false_positive_rate": fp_rate,
        "slow_path_rate":    (true_positives + false_positives) / max(1, total),
        "n_evaluated":       total,
        "n_trace_records":    trace_records,
        "missing_predictions": missing_predictions,
        "prediction_coverage": total / max(1, trace_records),
    }

    with open(results_dir / "routing_eval_metrics.json", "w") as f:
        json.dump({"summary": metrics}, f, indent=2)
    write_metrics_bundle(results_dir, "routing_eval_report", {"summary": metrics}, title="Routing Evaluation Report")

    with mlflow.start_run(run_name="routing_eval"):
        log_metrics_to_mlflow(metrics, prefix="eval")
        mlflow.log_artifact(str(results_dir / "routing_eval_report.md"))

    _print_summary(metrics, cfg["thresholds"])
    return metrics


def _gold_slow_path(D_t: dict, N_t: dict, record: dict) -> bool:
    if N_t.get("value_conflict") == "strong":
        return True
    if N_t.get("secrecy_pressure") == "high" and D_t.get("reveal_decision") in {"hint", "partial", "full"}:
        return True
    if D_t.get("response_policy") in {"threaten", "negotiate"}:
        return True
    return False


def _print_summary(metrics: dict, thresholds: dict) -> None:
    mode = metrics.get("routing_mode", "gold")
    print(f"\n=== Routing Evaluation Summary (mode={mode}) ===")
    fp_thresh = thresholds.get("router_false_positive_rate", 0.15)
    fp_rate   = metrics.get("false_positive_rate", 0.0)
    status = "PASS" if fp_rate <= fp_thresh else "FAIL"
    print(f"  [{status}] False Positive Rate: {fp_rate:.4f} (threshold ≤ {fp_thresh})")
    print(f"  Precision:    {metrics.get('routing_precision', 0):.4f}")
    print(f"  Recall:       {metrics.get('routing_recall', 0):.4f}")
    print(f"  F1:           {metrics.get('routing_f1', 0):.4f}")
    print(f"  Slow path %:  {metrics.get('slow_path_rate', 0)*100:.1f}%")
    if mode == "predicted":
        print(f"  Coverage:     {metrics.get('prediction_coverage', 0):.4f}")
        print(f"  Missing preds:{metrics.get('missing_predictions', 0)}")
    print()
