"""
Compute Expected Calibration Error (ECE) and selective-routing curves
for the latent predictor heads.

Usage:
    python -m src.eval.eval_calibration --config configs/eval.yaml

Outputs:
    - calibration_metrics.json   (per-head ECE, accuracy, confidence)
    - selective_routing.json     (coverage vs accuracy trade-off)
    - reliability_scorecard.md   (human-readable head reliability summary)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

from src.training.model import load_predictor
from src.training.dataset import HeadSupervisionDataset, collate_head_batch, LABEL_MAPS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--n-bins", type=int, default=10)
    p.add_argument("--confidence-thresholds", type=int, default=20,
                   help="Number of threshold steps for selective classification curve")
    return p.parse_args()


def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """ECE with equal-width bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lower, upper = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correct[mask].mean()
        weight = mask.mean()
        ece += weight * abs(bin_acc - bin_conf)
    return float(ece)


def selective_classification_curve(confidences: np.ndarray, correct: np.ndarray, n_steps: int = 20) -> list[dict]:
    """Return coverage vs accuracy for varying confidence thresholds."""
    thresholds = np.linspace(0.0, 1.0, n_steps + 1)
    curve = []
    for tau in thresholds:
        mask = confidences >= tau
        if mask.sum() == 0:
            curve.append({"threshold": float(tau), "coverage": 0.0, "accuracy": 0.0})
            continue
        acc = correct[mask].mean()
        cov = mask.mean()
        curve.append({"threshold": float(tau), "coverage": float(cov), "accuracy": float(acc)})
    return curve


def eval_calibration(config_path: str, n_bins: int = 10, n_threshold_steps: int = 20) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    checkpoint = cfg["latent_predictor_checkpoint"]
    model_name = cfg.get("base_model", "Qwen/Qwen3-4B")
    quantization = cfg.get("quantization", "4bit")
    torch_dtype = cfg.get("torch_dtype", "bfloat16")

    print(f"Loading predictor from {checkpoint}")
    predictor, tokenizer = load_predictor(checkpoint, model_name, quantization=quantization, torch_dtype=torch_dtype)
    predictor.eval()

    # Use the same test file as latent eval
    test_ds = HeadSupervisionDataset(
        cfg["data"]["test_heads_file"],
        tokenizer,
        max_seq_len=cfg.get("generation", {}).get("max_seq_len", 1024),
    )
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_ds, batch_size=8, shuffle=False,
        collate_fn=collate_head_batch, num_workers=0,
    )

    # Gather confidences and correctness per head
    per_head_conf: dict[str, list[float]] = {}
    per_head_correct: dict[str, list[int]] = {}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Calibration eval"):
            input_ids = batch["input_ids"].to(predictor.backbone.device)
            attention_mask = batch["attention_mask"].to(predictor.backbone.device)
            out = predictor(input_ids=input_ids, attention_mask=attention_mask)

            for field, logits in out["logits"].items():
                label_key = f"label_{field}"
                if label_key not in batch:
                    continue
                gold = batch[label_key]
                if field == "dialogue_act":
                    continue  # skip multi-label for calibration simplicity
                if not isinstance(gold, torch.Tensor):
                    continue
                gold = gold.to(predictor.backbone.device)
                valid = gold != -1
                if not valid.any():
                    continue

                probs = torch.softmax(logits, dim=-1)
                confidences, preds = probs.max(dim=-1)
                confidences = confidences[valid].cpu().numpy()
                preds = preds[valid].cpu().numpy()
                gold_valid = gold[valid].cpu().numpy()
                correct = (preds == gold_valid).astype(int)

                per_head_conf.setdefault(field, []).extend(confidences.tolist())
                per_head_correct.setdefault(field, []).extend(correct.tolist())

    results_dir = Path(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Per-head metrics
    head_metrics = {}
    selective_curves = {}
    for field in sorted(per_head_conf.keys()):
        conf = np.array(per_head_conf[field])
        corr = np.array(per_head_correct[field])
        if len(conf) == 0:
            continue
        ece = expected_calibration_error(conf, corr, n_bins=n_bins)
        acc = corr.mean()
        mean_conf = conf.mean()
        selective_curves[field] = selective_classification_curve(conf, corr, n_steps=n_threshold_steps)

        # Find threshold that gives ~90% accuracy if achievable
        best_tau_90 = None
        for pt in selective_curves[field]:
            if pt["accuracy"] >= 0.90:
                best_tau_90 = pt["threshold"]
                break

        head_metrics[field] = {
            "ece": ece,
            "accuracy": float(acc),
            "mean_confidence": float(mean_conf),
            "n_samples": len(conf),
            "threshold_for_90_acc": best_tau_90,
        }

    # Reliability scorecard
    scorecard_lines = [
        "# Head Reliability Scorecard\n",
        "| Head | Accuracy | ECE | Mean Conf | 90%-Acc Threshold | Recommended Use |\n",
        "|------|----------|-----|-----------|-------------------|-----------------|\n",
    ]
    for field, m in sorted(head_metrics.items(), key=lambda x: x[1]["ece"]):
        tau = m["threshold_for_90_acc"]
        tau_str = f"{tau:.2f}" if tau is not None else "N/A"
        if m["ece"] < 0.10 and m["accuracy"] > 0.70:
            use = "Hard routing"
        elif m["ece"] < 0.15 and m["accuracy"] > 0.60:
            use = "Hard routing with threshold"
        elif m["ece"] < 0.20:
            use = "Advisory only"
        else:
            use = "Not for routing"
        scorecard_lines.append(
            f"| {field} | {m['accuracy']:.3f} | {m['ece']:.3f} | {m['mean_confidence']:.3f} | {tau_str} | {use} |\n"
        )

    # Save
    with open(results_dir / "calibration_metrics.json", "w") as f:
        json.dump({"heads": head_metrics, "n_bins": n_bins}, f, indent=2)

    with open(results_dir / "selective_routing.json", "w") as f:
        json.dump({"curves": selective_curves}, f, indent=2)

    scorecard_path = results_dir / "reliability_scorecard.md"
    with open(scorecard_path, "w") as f:
        f.writelines(scorecard_lines)

    print(f"\n=== Calibration Summary ===")
    print(f"  Evaluated {len(head_metrics)} heads")
    for field, m in sorted(head_metrics.items(), key=lambda x: x[1]["ece"])[:5]:
        print(f"  {field:25s} acc={m['accuracy']:.3f}  ECE={m['ece']:.3f}")
    print(f"  Scorecard written to {scorecard_path}\n")

    return {"heads": head_metrics, "scorecard_path": str(scorecard_path)}


def main():
    args = parse_args()
    eval_calibration(args.config, args.n_bins, args.confidence_thresholds)


if __name__ == "__main__":
    main()
