#!/usr/bin/env python3
"""
Rigorous latent predictor evaluation with:
  - True Cohen's κ (from confusion matrices, not estimated from accuracy)
  - Bootstrap 95% confidence intervals on per-head metrics
  - Paired bootstrap for joint-vs-separate comparisons
  - Per-head CSV output with CIs

Usage:
  python eval_results/evaluate_latent_rigorous.py --checkpoint checkpoints/latent_predictor_best
  python eval_results/evaluate_latent_rigorous.py --checkpoint checkpoints/latent_predictor_jepa_best --jepa
  python eval_results/evaluate_latent_rigorous.py --checkpoint checkpoints/slm_latent_gpt_best --slm --arch gpt
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from llm_finetuning.src.training.dataset import HeadSupervisionDataset, collate_head_batch, LABEL_MAPS, LABEL_TO_IDX
from llm_finetuning.src.training.model import load_predictor
from llm_finetuning.src.training.loss import GROUP_FIELDS


# ═══════════════════════════════════════════════════════════════════════════════
# Cohen's κ (true, from confusion matrix)
# ═══════════════════════════════════════════════════════════════════════════════

def true_cohen_kappa(y_true: list[int], y_pred: list[int], n_classes: int) -> float:
    """Compute Cohen's kappa from the actual confusion matrix."""
    if len(y_true) == 0 or n_classes < 2:
        return 0.0
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t][p] += 1
    n = cm.sum()
    if n == 0:
        return 0.0
    p_o = np.trace(cm) / n
    row_marginals = cm.sum(axis=1) / n
    col_marginals = cm.sum(axis=0) / n
    p_e = (row_marginals * col_marginals).sum()
    if p_e == 1.0:
        return 1.0
    return float((p_o - p_e) / (1.0 - p_e))


def per_class_f1(y_true: list[int], y_pred: list[int], n_classes: int) -> dict[str, float]:
    """Macro-F1: average of per-class F1 scores."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t][p] += 1
    f1s = []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        if tp + fp + fn == 0:
            continue  # class never appears
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    # Weighted F1
    class_support = cm.sum(axis=1)
    total = class_support.sum()
    weighted_f1 = sum(f * s for f, s in zip(f1s, class_support)) / total if total > 0 else 0.0

    return {"macro_f1": macro_f1, "weighted_f1": weighted_f1}


# ═══════════════════════════════════════════════════════════════════════════════
# Bootstrap
# ═══════════════════════════════════════════════════════════════════════════════

def bootstrap_ci(
    y_true: list[int],
    y_pred: list[int],
    n_classes: int,
    metric_fn,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap 95% CI for a metric. Returns (mean, lower, upper)."""
    rng = np.random.RandomState(seed)
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    n = len(y_true_arr)
    if n < 10:
        val = metric_fn(y_true, y_pred, n_classes)
        return val, val, val

    estimates = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        est = metric_fn(
            y_true_arr[idx].tolist(),
            y_pred_arr[idx].tolist(),
            n_classes,
        )
        estimates.append(est)

    estimates = np.array(estimates)
    mean = float(np.mean(estimates))
    lower = float(np.percentile(estimates, 100 * alpha / 2))
    upper = float(np.percentile(estimates, 100 * (1 - alpha / 2)))
    return mean, lower, upper


def bootstrap_accuracy(y_true: list[int], y_pred: list[int], _n_classes: int) -> float:
    return sum(1 for g, p in zip(y_true, y_pred) if g == p) / max(len(y_true), 1)


def bootstrap_kappa(y_true: list[int], y_pred: list[int], n_classes: int) -> float:
    return true_cohen_kappa(y_true, y_pred, n_classes)


def bootstrap_macro_f1(y_true: list[int], y_pred: list[int], n_classes: int) -> float:
    return per_class_f1(y_true, y_pred, n_classes)["macro_f1"]


def paired_bootstrap_diff(
    y_true: list[int],
    y_pred_a: list[int],
    y_pred_b: list[int],
    n_classes: int,
    metric_fn,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Paired bootstrap: CI on (metric_A - metric_B) using same resample indices.

    Returns {mean_diff, ci_lower, ci_upper, p_superiority} where p_superiority
    is the fraction of bootstrap samples where A > B.
    """
    rng = np.random.RandomState(seed)
    y_t = np.array(y_true)
    y_a = np.array(y_pred_a)
    y_b = np.array(y_pred_b)
    n = len(y_t)
    if n < 10:
        val_a = metric_fn(y_true, y_pred_a, n_classes)
        val_b = metric_fn(y_true, y_pred_b, n_classes)
        d = val_a - val_b
        return {"mean_diff": d, "ci_lower": d, "ci_upper": d, "p_superiority": 0.5}

    diffs = []
    a_wins = 0
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        est_a = metric_fn(y_t[idx].tolist(), y_a[idx].tolist(), n_classes)
        est_b = metric_fn(y_t[idx].tolist(), y_b[idx].tolist(), n_classes)
        d = est_a - est_b
        diffs.append(d)
        if d > 0:
            a_wins += 1

    diffs_arr = np.array(diffs)
    return {
        "mean_diff": round(float(np.mean(diffs_arr)), 6),
        "ci_lower": round(float(np.percentile(diffs_arr, 100 * alpha / 2)), 6),
        "ci_upper": round(float(np.percentile(diffs_arr, 100 * (1 - alpha / 2))), 6),
        "p_superiority": round(a_wins / n_bootstrap, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_checkpoint(
    checkpoint_path: str,
    test_file: str,
    base_model: str = "Qwen/Qwen3-1.7B",
    quantization: str | None = None,
    torch_dtype: str = "float32",
    is_slm: bool = False,
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    print(f"Evaluating: {checkpoint_path}")
    print(f"  Model: {base_model}  |  Quant: {quantization}  |  SLM: {is_slm}")

    if is_slm:
        # Load SLM checkpoint
        from slm_training.src.train.small_lm_architectures import (
            GPTConfig, TinyGPTLM, MambaLikeConfig, MambaLikeLM,
        )
        from slm_training.src.train.train_latent_slm import SLMLatentPredictor
        from transformers import AutoTokenizer

        ckpt = torch.load(Path(checkpoint_path) / "checkpoint.pt", map_location="cpu")
        arch = ckpt["config"].get("architecture", "gpt")
        profile = ckpt["config"].get("hardware_profile", "rtx4070_small")
        from slm_training.src.train.small_lm_architectures import RECOMMENDED_CONFIGS
        arch_cfg_dict = RECOMMENDED_CONFIGS[profile][arch]

        if arch == "gpt":
            slm = TinyGPTLM(GPTConfig(**arch_cfg_dict))
            hidden_dim = arch_cfg_dict["n_embd"]
        elif arch == "mamba_like":
            slm = MambaLikeLM(MambaLikeConfig(**arch_cfg_dict))
            hidden_dim = arch_cfg_dict["n_embd"]
        else:
            raise ValueError(f"Unknown arch: {arch}")

        slm.load_state_dict(ckpt["slm_state_dict"])
        pooling = ckpt["config"].get("pooling", "last")
        predictor = SLMLatentPredictor(slm, hidden_dim, pooling=pooling)

        for field, state in ckpt["heads_state_dict"].items():
            if field in predictor.heads:
                predictor.heads[field].load_state_dict(state)

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        predictor = predictor.to(device)
        predictor.eval()
    else:
        # Load Qwen/Gemma checkpoint
        predictor, tokenizer = load_predictor(
            checkpoint_path, base_model,
            quantization=quantization, torch_dtype=torch_dtype,
        )
        predictor.eval()

    ds = HeadSupervisionDataset(test_file, tokenizer, max_seq_len=512 if not is_slm else 256)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_head_batch)

    all_golds: dict[str, list[int]] = defaultdict(list)
    all_preds: dict[str, list[int]] = defaultdict(list)

    device = predictor.backbone.device if hasattr(predictor, 'backbone') else predictor.device
    with torch.no_grad():
        for batch in tqdm(loader, desc="  eval"):
            input_ids = batch["input_ids"].to(device)
            attn = batch.get("attention_mask")
            if attn is not None:
                attn = attn.to(device)

            out = predictor(input_ids=input_ids, attention_mask=attn)

            for field, logits in out["logits"].items():
                if field == "dialogue_act":
                    continue
                label_key = f"label_{field}"
                if label_key not in batch:
                    continue
                gold = batch[label_key]
                if not isinstance(gold, torch.Tensor):
                    continue
                valid = gold != -1
                if not valid.any():
                    continue
                preds = logits.argmax(dim=-1)
                all_preds[field].extend(preds[valid].cpu().tolist())
                all_golds[field].extend(gold[valid].cpu().tolist())

    # ── Compute metrics with CIs ─────────────────────────────────────────────
    per_head = []
    group_metrics = defaultdict(lambda: {"accs": [], "kappas": [], "f1s": [], "n": []})

    print(f"\n  {'Field':<24s} {'Acc':>6s} {'CI':>16s} {'κ':>6s} {'CI':>16s} {'F1':>6s} {'N':>6s}")
    print(f"  {'─'*24} {'─'*6} {'─'*16} {'─'*6} {'─'*16} {'─'*6} {'─'*6}")

    for field in sorted(all_golds.keys()):
        golds = all_golds[field]
        preds = all_preds[field]
        if len(golds) < 5:
            continue
        n_classes = len(LABEL_MAPS.get(field, [])) or max(max(golds), max(preds)) + 1

        acc_mean, acc_lo, acc_hi = bootstrap_ci(golds, preds, n_classes, bootstrap_accuracy, n_bootstrap)
        kappa_mean, kappa_lo, kappa_hi = bootstrap_ci(golds, preds, n_classes, bootstrap_kappa, n_bootstrap)
        f1_mean, f1_lo, f1_hi = bootstrap_ci(golds, preds, n_classes, bootstrap_macro_f1, n_bootstrap)
        # Point estimate
        kappa_pt = true_cohen_kappa(golds, preds, n_classes)
        acc_pt = sum(1 for g, p in zip(golds, preds) if g == p) / len(golds)

        group = GROUP_FIELDS.get(field, "other")
        group_metrics[group]["accs"].append(acc_pt)
        group_metrics[group]["kappas"].append(kappa_pt)
        group_metrics[group]["f1s"].append(f1_mean)
        group_metrics[group]["n"].append(len(golds))

        per_head.append({
            "field": field,
            "group": group,
            "n_samples": len(golds),
            "n_classes": n_classes,
            "accuracy": round(acc_pt, 4),
            "accuracy_95ci_lo": round(acc_lo, 4),
            "accuracy_95ci_hi": round(acc_hi, 4),
            "cohen_kappa": round(kappa_pt, 4),
            "kappa_95ci_lo": round(kappa_lo, 4),
            "kappa_95ci_hi": round(kappa_hi, 4),
            "macro_f1": round(f1_mean, 4),
            "f1_95ci_lo": round(f1_lo, 4),
            "f1_95ci_hi": round(f1_hi, 4),
        })

        print(f"  {field:<24s} {acc_pt:6.4f} [{acc_lo:6.4f},{acc_hi:6.4f}] "
              f"{kappa_pt:6.4f} [{kappa_lo:6.4f},{kappa_hi:6.4f}] "
              f"{f1_mean:6.4f} {len(golds):6d}")

    # ── Groups ───────────────────────────────────────────────────────────────
    print(f"\n  {'Group':<16s} {'Acc':>8s} {'κ':>8s} {'F1':>8s}")
    print(f"  {'─'*16} {'─'*8} {'─'*8} {'─'*8}")
    groups_summary = {}
    for g, gm in sorted(group_metrics.items()):
        mean_acc = np.mean(gm["accs"]) if gm["accs"] else 0
        mean_kappa = np.mean(gm["kappas"]) if gm["kappas"] else 0
        mean_f1 = np.mean(gm["f1s"]) if gm["f1s"] else 0
        groups_summary[g] = {
            "mean_accuracy": round(float(mean_acc), 4),
            "mean_kappa": round(float(mean_kappa), 4),
            "mean_macro_f1": round(float(mean_f1), 4),
            "n_heads": len(gm["accs"]),
        }
        print(f"  {g:<16s} {mean_acc:8.4f} {mean_kappa:8.4f} {mean_f1:8.4f}")

    # ── Overall ──────────────────────────────────────────────────────────────
    all_accs = [h["accuracy"] for h in per_head]
    all_kappas = [h["cohen_kappa"] for h in per_head]
    all_f1s = [h["macro_f1"] for h in per_head]
    overall = {
        "n_heads": len(per_head),
        "macro_mean_accuracy": round(float(np.mean(all_accs)), 4),
        "macro_mean_kappa": round(float(np.mean(all_kappas)), 4),
        "macro_mean_f1": round(float(np.mean(all_f1s)), 4),
        "weighted_mean_accuracy": round(
            sum(h["accuracy"] * h["n_samples"] for h in per_head) /
            max(sum(h["n_samples"] for h in per_head), 1), 4),
    }
    print(f"\n  Overall: {overall['n_heads']} heads, "
          f"macro-acc={overall['macro_mean_accuracy']:.4f}, "
          f"macro-κ={overall['macro_mean_kappa']:.4f}, "
          f"macro-F1={overall['macro_mean_f1']:.4f}")

    return {
        "checkpoint": checkpoint_path,
        "overall": overall,
        "groups": groups_summary,
        "per_head": per_head,
        "_raw": {"golds": dict(all_golds), "preds": dict(all_preds)},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Rigorous latent predictor evaluation")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory")
    parser.add_argument("--compare", default=None, help="Second checkpoint for paired-bootstrap comparison")
    parser.add_argument("--test-file", default="data/splits/val_heads.jsonl")
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--torch-dtype", default="float32")
    parser.add_argument("--slm", action="store_true", help="SLM checkpoint")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    results = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        test_file=args.test_file,
        base_model=args.base_model,
        quantization=args.quantization,
        torch_dtype=args.torch_dtype,
        is_slm=args.slm,
        n_bootstrap=args.n_bootstrap,
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    ckpt_name = Path(args.checkpoint).name
    results_dir = PROJECT_ROOT / "eval_results"
    results_dir.mkdir(exist_ok=True)

    json_path = args.output_json or str(results_dir / f"eval_{ckpt_name}.json")
    csv_path = args.output_csv or str(results_dir / f"eval_{ckpt_name}.csv")

    if args.compare:
        # ── Paired comparison ───────────────────────────────────────────────
        comp_results = evaluate_checkpoint(
            checkpoint_path=args.compare,
            test_file=args.test_file,
            base_model=args.base_model,
            quantization=args.quantization,
            torch_dtype=args.torch_dtype,
            is_slm=args.slm,
            n_bootstrap=args.n_bootstrap,
        )
        comp_name = Path(args.compare).name

        # Per-head paired diffs using stored raw predictions
        paired_diffs = []
        raw_a = results.get("_raw", {})
        raw_b = comp_results.get("_raw", {})
        golds_a = raw_a.get("golds", {})
        golds_b = raw_b.get("golds", {})
        preds_a = raw_a.get("preds", {})
        preds_b = raw_b.get("preds", {})

        print(f"\n  Paired comparison: {ckpt_name} vs {comp_name}")
        print(f"  {'Field':<24s} {'Δ Acc':>10s} {'95% CI':>18s} {'p(A>B)':>8s}")
        print(f"  {'─'*24} {'─'*10} {'─'*18} {'─'*8}")

        for field in sorted(set(list(golds_a.keys()) + list(golds_b.keys()))):
            if field not in golds_a or field not in golds_b:
                continue
            n_cls = len(LABEL_MAPS.get(field, [])) or max(
                max(golds_a[field]), max(golds_b[field]),
                max(preds_a.get(field, [0])), max(preds_b.get(field, [0]))
            ) + 1
            diff = paired_bootstrap_diff(
                golds_a[field], preds_a.get(field, []), preds_b.get(field, []),
                n_cls, bootstrap_accuracy, args.n_bootstrap,
            )
            paired_diffs.append({"field": field, **diff})
            marker = "⚠" if abs(diff["mean_diff"]) > 0.02 else " "
            print(f"  {field:<24s} {diff['mean_diff']:+10.4f} "
                  f"[{diff['ci_lower']:+7.4f},{diff['ci_upper']:+7.4f}] "
                  f"{diff['p_superiority']:8.4f} {marker}")

        # Aggregate paired diffs
        mean_diffs = []
        for h in paired_diffs:
            mean_diffs.append(h.get("mean_diff", 0))
        if mean_diffs:
            macro_diff = sum(mean_diffs) / len(mean_diffs)
            print(f"\n  Macro mean Δ: {macro_diff:+.4f}")

        results["paired_comparison"] = {
            "checkpoint_a": ckpt_name,
            "checkpoint_b": comp_name,
            "per_head_diffs": paired_diffs,
            "overall": {
                "a": results["overall"],
                "b": comp_results["overall"],
            },
        }

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  ✓ JSON → {json_path}")

    if results["per_head"]:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results["per_head"][0].keys())
            writer.writeheader()
            writer.writerows(results["per_head"])
        print(f"  ✓ CSV  → {csv_path}")


if __name__ == "__main__":
    main()
