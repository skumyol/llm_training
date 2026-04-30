import json
from pathlib import Path

import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.dataset import HeadSupervisionDataset, collate_head_batch, LABEL_MAPS
from src.training.model import load_predictor


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
    )
    test_loader = DataLoader(
        test_ds, batch_size=8, shuffle=False,
        collate_fn=collate_head_batch, num_workers=0,
    )

    all_preds: dict[str, list] = {}
    all_golds: dict[str, list] = {}
    secret_leakage = 0
    total_turns = 0

    predictor.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating latent heads"):
            input_ids      = batch["input_ids"].to(predictor.backbone.device)
            attention_mask = batch["attention_mask"].to(predictor.backbone.device)
            out = predictor(input_ids=input_ids, attention_mask=attention_mask)

            for field in out["logits"]:
                label_key = f"label_{field}"
                if label_key not in batch or field == "dialogue_act":
                    continue
                gold = batch[label_key]
                if not isinstance(gold, torch.Tensor):
                    continue
                
                # Move gold to device for valid mask and comparison
                gold = gold.to(predictor.backbone.device)
                
                valid = gold != -1
                if not valid.any():
                    continue
                pred = out["logits"][field].argmax(dim=-1)
                if field not in all_preds:
                    all_preds[field] = []
                    all_golds[field] = []
                all_preds[field].extend(pred[valid].cpu().tolist())
                all_golds[field].extend(gold[valid].cpu().tolist())

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
    per_field_reports: dict = {}

    for field, golds in all_golds.items():
        preds = all_preds[field]
        label_names = LABEL_MAPS.get(field, None)
        correct = sum(p == g for p, g in zip(preds, golds))
        acc = correct / len(golds) if golds else 0.0
        metrics[f"{field}_accuracy"] = acc
        if label_names:
            n_classes = len(label_names)
            all_labels = list(range(n_classes))
            target_names = [str(l) for l in label_names]
            try:
                report = classification_report(
                    golds, preds,
                    labels=all_labels,
                    target_names=target_names,
                    output_dict=True, zero_division=0,
                )
                per_field_reports[field] = report
            except Exception as e:
                print(f"  [WARN] classification_report failed for {field}: {e}")
            if cfg["output"].get("save_confusion_matrices", True):
                try:
                    cm = confusion_matrix(golds, preds, labels=all_labels)
                    _save_confusion_matrix(cm, field, results_dir, label_names)
                except Exception as e:
                    print(f"  [WARN] confusion_matrix failed for {field}: {e}")

    metrics["secret_leakage_rate"] = secret_leakage / max(1, total_turns)
    metrics["response_policy_f1"] = per_field_reports.get("response_policy", {}).get("macro avg", {}).get("f1-score", 0.0)
    metrics["trust_delta_accuracy"] = metrics.get("trust_delta_accuracy", 0.0)

    with open(results_dir / "latent_eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with mlflow.start_run(run_name="latent_eval"):
        for k, v in metrics.items():
            mlflow.log_metric(f"eval/{k}", v)
        mlflow.log_artifact(str(results_dir / "latent_eval_metrics.json"))

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
        val = metrics.get(key, 0.0)
        passed = (val >= threshold) if op == "≥" else (val <= threshold)
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {key}: {val:.4f} (threshold {op} {threshold})")
    print()
