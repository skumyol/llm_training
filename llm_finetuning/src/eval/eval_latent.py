import json
from pathlib import Path

import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.dataset import HeadSupervisionDataset, collate_head_batch, LABEL_MAPS
from src.training.model import load_predictor
from src.metrics_report import compute_latent_metrics, log_metrics_to_mlflow, write_metrics_bundle


STANCE_DIMS = ["affection", "respect", "dominance", "familiarity", "trust", "obligation"]


def eval_latent(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    import mlflow
    from src.mlflow_utils import setup_mlflow

    setup_mlflow(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    checkpoint = cfg["latent_predictor_checkpoint"]
    model_name = cfg.get("base_model", "Qwen/Qwen3-4B")

    quantization = cfg.get("quantization", "4bit")
    torch_dtype = cfg.get("torch_dtype", "bfloat16")
    print(f"Loading predictor from {checkpoint} (quantization={quantization}, dtype={torch_dtype})")
    predictor, tokenizer = load_predictor(checkpoint, model_name, quantization=quantization, torch_dtype=torch_dtype)

    test_ds = HeadSupervisionDataset(
        cfg["data"]["test_heads_file"],
        tokenizer,
        max_seq_len=cfg.get("generation", {}).get("max_seq_len", 1024),
        exclude_counterfactual=cfg["data"].get("exclude_counterfactual", False),
    )
    test_loader = DataLoader(
        test_ds, batch_size=8, shuffle=False,
        collate_fn=collate_head_batch, num_workers=0,
    )

    all_preds: dict[str, list] = {}
    all_golds: dict[str, list] = {}
    secret_leakage = 0
    total_turns = 0

    # Per-record predictions for routing export
    per_record_preds: list[dict] = []
    idx_to_label = {field: {i: name for i, name in enumerate(LABEL_MAPS[field])} for field in LABEL_MAPS}

    predictor.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating latent heads"):
            input_ids      = batch["input_ids"].to(predictor.backbone.device)
            attention_mask = batch["attention_mask"].to(predictor.backbone.device)
            out = predictor(input_ids=input_ids, attention_mask=attention_mask)

            bsz = input_ids.size(0)
            episode_ids = batch.get("episode_ids", [""] * bsz)

            for field in out["logits"]:
                label_key = f"label_{field}"
                if label_key not in batch:
                    continue
                gold = batch[label_key]
                if not isinstance(gold, torch.Tensor):
                    continue
                gold = gold.to(predictor.backbone.device)
                if field == "dialogue_act":
                    valid = gold.sum(dim=1) > 0
                    if not valid.any():
                        continue
                    pred = (out["logits"][field].sigmoid() >= 0.5).to(torch.long)
                    if field not in all_preds:
                        all_preds[field] = []
                        all_golds[field] = []
                    all_preds[field].extend(pred[valid].cpu().tolist())
                    all_golds[field].extend(gold[valid].cpu().long().tolist())
                    continue

                valid = gold != -1
                if not valid.any():
                    continue
                pred = out["logits"][field].argmax(dim=-1)
                if field not in all_preds:
                    all_preds[field] = []
                    all_golds[field] = []
                all_preds[field].extend(pred[valid].cpu().tolist())
                all_golds[field].extend(gold[valid].cpu().tolist())

            # Collect per-record predictions for routing fields. Keep the base
            # index stable for the whole batch; len(per_record_preds) changes
            # as we append inside this loop.
            routing_fields = ["value_conflict", "response_policy", "reveal_decision", "secrecy_pressure"]
            base_idx = len(per_record_preds)
            for i in range(bsz):
                record = test_ds.records[base_idx + i]
                rec_pred = {
                    "record_idx": base_idx + i,
                    "episode_id": str(record.get("episode_id", episode_ids[i])),
                    "turn_idx": record.get("turn_idx", record.get("turn", -1)),
                }
                for field in routing_fields:
                    if field in out["logits"]:
                        idx = int(out["logits"][field][i].argmax(dim=-1).cpu().item())
                        rec_pred[field] = idx_to_label[field].get(idx, "")
                per_record_preds.append(rec_pred)

            reveal_logits = out["logits"].get("reveal_decision")
            secrecy_label_key = "label_secrecy_pressure"
            if reveal_logits is not None and secrecy_label_key in batch:
                reveal_pred = reveal_logits.argmax(dim=-1)
                secrecy_gold = batch[secrecy_label_key]
                if isinstance(secrecy_gold, torch.Tensor):
                    secrecy_gold = secrecy_gold.to(predictor.backbone.device)
                    high_secrecy = secrecy_gold == 2
                    full_reveal  = reveal_pred == 3
                    secret_leakage += (high_secrecy & full_reveal).sum().item()
                    total_turns    += len(reveal_pred)

    results_dir = Path(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict = {}
    for field, golds in all_golds.items():
        label_names = LABEL_MAPS.get(field, None)
        if not label_names:
            continue
        if field == "dialogue_act":
            continue
        if cfg["output"].get("save_confusion_matrices", True):
            try:
                cm = confusion_matrix(golds, all_preds[field], labels=list(range(len(label_names))))
                _save_confusion_matrix(cm, field, results_dir, label_names)
            except Exception as e:
                print(f"  [WARN] confusion_matrix failed for {field}: {e}")

    metrics = compute_latent_metrics(
        all_preds,
        all_golds,
        secret_leakage_rate=secret_leakage / max(1, total_turns),
    )

    pred_path = results_dir / "predicted_zt.jsonl"
    with open(pred_path, "w") as f:
        for rec in per_record_preds:
            f.write(json.dumps(rec) + "\n")
    print(f"Saved predicted Z_t for {len(per_record_preds)} records to {pred_path}")

    with open(results_dir / "latent_eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    write_metrics_bundle(results_dir, "latent_eval_report", metrics, title="Latent Evaluation Report")

    with mlflow.start_run(run_name="latent_eval"):
        if pred_path.exists():
            mlflow.log_artifact(str(pred_path))
        log_metrics_to_mlflow(metrics.get("summary", {}), prefix="eval")
        if metrics.get("groups"):
            log_metrics_to_mlflow(metrics["groups"], prefix="eval/groups")
        if metrics.get("fields"):
            log_metrics_to_mlflow(metrics["fields"], prefix="eval/fields")
        mlflow.log_artifact(str(results_dir / "latent_eval_metrics.json"))
        mlflow.log_artifact(str(results_dir / "latent_eval_report.md"))

    _print_summary(metrics, cfg["thresholds"])
    return metrics


def _save_confusion_matrix(cm, field: str, results_dir: Path, label_names: list) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)
        n = cm.shape[0]
        names = [str(l) for l in label_names[:n]]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_title(f"Confusion Matrix: {field}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.tight_layout()
        fig.savefig(results_dir / f"cm_{field}.png", dpi=100)
        plt.close(fig)
    except Exception:
        pass


def _print_summary(metrics: dict, thresholds: dict) -> None:
    print("\n=== Latent State Evaluation Summary ===")
    checks = [
        ("response_policy_f1",   thresholds.get("response_policy_f1", 0.75),    "≥"),
        ("trust_delta_accuracy", thresholds.get("stance_delta_accuracy", 0.70), "≥"),
        ("secret_leakage_rate",  thresholds.get("secret_leakage_rate", 0.05),   "≤"),
    ]
    for key, threshold, op in checks:
        val = metrics.get("summary", {}).get(key, 0.0)
        passed = (val >= threshold) if op == "≥" else (val <= threshold)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {key}: {val:.4f} (threshold {op} {threshold})")
    print()
