import json
from pathlib import Path

import torch
import yaml
from tqdm import tqdm


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

    results_dir = Path(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    total = 0

    with open(test_trace) as f:
        for line in tqdm(f, desc="Evaluating routing"):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            D_t = record.get("D_t", {})
            N_t = record.get("N_t", {})

            gold_slow = _gold_slow_path(D_t, N_t, record)
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
        "routing_precision": precision,
        "routing_recall":    recall,
        "routing_f1":        f1,
        "false_positive_rate": fp_rate,
        "slow_path_rate":    (true_positives + false_positives) / max(1, total),
        "n_evaluated":       total,
    }

    with open(results_dir / "routing_eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with mlflow.start_run(run_name="routing_eval"):
        for k, v in metrics.items():
            mlflow.log_metric(f"eval/{k}", v)

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
    print("\n=== Routing Evaluation Summary ===")
    fp_thresh = thresholds.get("router_false_positive_rate", 0.15)
    fp_rate   = metrics.get("false_positive_rate", 0.0)
    status = "PASS" if fp_rate <= fp_thresh else "FAIL"
    print(f"  [{status}] False Positive Rate: {fp_rate:.4f} (threshold ≤ {fp_thresh})")
    print(f"  Precision:    {metrics.get('routing_precision', 0):.4f}")
    print(f"  Recall:       {metrics.get('routing_recall', 0):.4f}")
    print(f"  F1:           {metrics.get('routing_f1', 0):.4f}")
    print(f"  Slow path %:  {metrics.get('slow_path_rate', 0)*100:.1f}%")
    print()
