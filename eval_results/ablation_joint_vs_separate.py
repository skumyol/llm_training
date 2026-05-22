#!/usr/bin/env python3
"""
Ablation 4: Joint vs Separate Evaluation
========================================
Compares joint (shared-backbone) model against separate (latent + response)
models on:
  - Latent head accuracy & Cohen's κ  
  - Response PPL on SFT eval set
  - Consistency violation count
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "llm_finetuning"))

from src.training.dataset import (
    HeadSupervisionDataset, collate_head_batch,
    SFTDataset, collate_sft_batch, LABEL_MAPS,
)
from src.training.model import load_predictor
from src.training.loss import GROUP_FIELDS


# ═══════════════════════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════════════════════

def cohen_kappa(y_true: list[int], y_pred: list[int], n_classes: int) -> float:
    import numpy as np
    if len(y_true) == 0:
        return 0.0
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t][p] += 1
    n = cm.sum()
    if n == 0:
        return 0.0
    p_o = np.trace(cm) / n
    row = cm.sum(axis=1) / n
    col = cm.sum(axis=0) / n
    p_e = (row * col).sum()
    if p_e == 1.0:
        return 1.0
    return float((p_o - p_e) / (1.0 - p_e))


def compute_consistency_violations(
    predictions: dict[str, list[int]],
) -> dict[str, int]:
    """Count impossible state combinations."""
    # Use the shortest list length as n_samples (different heads may have different lengths)
    n_samples = min((len(v) for v in predictions.values()), default=0)
    violations = {
        "high_secrecy_full_reveal": 0,
        "hostile_affection_high": 0,
        "total_samples": n_samples,
    }
    secrecy_vals = predictions.get("secrecy_pressure", [])
    reveal_vals = predictions.get("reveal_decision", [])
    dom_vals = predictions.get("dominance_level", [])
    aff_vals = predictions.get("affection_level", [])
    for i in range(n_samples):
        secrecy = secrecy_vals[i] if i < len(secrecy_vals) else 0
        reveal = reveal_vals[i] if i < len(reveal_vals) else 0
        dom = dom_vals[i] if i < len(dom_vals) else 0
        aff = aff_vals[i] if i < len(aff_vals) else 0
        # secrecy_pressure: 0=low, 1=medium, 2=high | reveal: 0=none, 1=hint, 2=partial, 3=full
        if secrecy == 2 and reveal == 3:
            violations["high_secrecy_full_reveal"] += 1
        # dominance: 0=submissive, 1=neutral, 2=dominant
        # affection: 0=hostile, 1=neutral, 2=friendly
        if aff == 2 and dom == 0:
            violations["hostile_affection_high"] += 1
    return violations


# ═══════════════════════════════════════════════════════════════════════════════
# §1  Latent head evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def eval_latent_heads(
    checkpoint_path: str,
    base_model: str,
    test_heads_file: str,
    quantization: str | None = None,
    torch_dtype: str = "float32",
) -> dict[str, Any]:
    print(f"\n  Loading predictor from: {checkpoint_path}")
    predictor, tokenizer = load_predictor(
        checkpoint_path, base_model,
        quantization=quantization,
        torch_dtype=torch_dtype,
    )
    predictor.eval()

    ds = HeadSupervisionDataset(test_heads_file, tokenizer, max_seq_len=1024)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collate_head_batch)

    all_golds: dict[str, list[int]] = defaultdict(list)
    all_preds: dict[str, list[int]] = defaultdict(list)

    with torch.no_grad():
        for batch in tqdm(loader, desc="  latent eval", leave=False):
            input_ids = batch["input_ids"].to(predictor.backbone.device)
            attn_mask = batch["attention_mask"].to(predictor.backbone.device)
            out = predictor(input_ids=input_ids, attention_mask=attn_mask)
            for field, logits in out["logits"].items():
                # Skip dialogue_act — uses multi-hot labels
                if field == "dialogue_act":
                    continue
                label_key = f"label_{field}"
                if label_key not in batch:
                    continue
                gold = batch[label_key]
                if not isinstance(gold, torch.Tensor):
                    continue
                # Filter out padding (-1)
                valid = gold != -1
                if not valid.any():
                    continue
                preds = out["logits"][field].argmax(dim=-1)
                all_preds[field].extend(preds[valid].cpu().tolist())
                all_golds[field].extend(gold[valid].cpu().tolist())

    head_metrics = {}
    for field in sorted(all_golds.keys()):
        golds = all_golds[field]
        preds = all_preds[field]
        if not golds:
            continue
        n_classes = len(LABEL_MAPS.get(field, [])) or max(max(golds), max(preds)) + 1
        correct = sum(1 for g, p in zip(golds, preds) if g == p)
        acc = correct / len(golds)
        kappa = cohen_kappa(golds, preds, n_classes)
        head_metrics[field] = {
            "accuracy": round(acc, 4),
            "cohen_kappa": round(kappa, 4),
            "n_classes": n_classes,
            "n_samples": len(golds),
            "group": GROUP_FIELDS.get(field, "other"),
        }

    # Group aggregates
    group_metrics = defaultdict(lambda: {"correct": 0, "total": 0, "kappas": []})
    for field, m in head_metrics.items():
        g = m["group"]
        group_metrics[g]["correct"] += int(m["accuracy"] * m["n_samples"])
        group_metrics[g]["total"] += m["n_samples"]
        group_metrics[g]["kappas"].append(m["cohen_kappa"])

    groups = {}
    for g, gm in group_metrics.items():
        groups[g] = {
            "accuracy": round(gm["correct"] / gm["total"], 4) if gm["total"] else 0,
            "mean_kappa": round(sum(gm["kappas"]) / len(gm["kappas"]), 4) if gm["kappas"] else 0,
        }

    mean_acc = sum(m["accuracy"] for m in head_metrics.values()) / max(1, len(head_metrics))
    mean_kappa = sum(m["cohen_kappa"] for m in head_metrics.values()) / max(1, len(head_metrics))

    violations = compute_consistency_violations(all_preds)

    return {
        "checkpoint": checkpoint_path,
        "n_heads": len(head_metrics),
        "mean_accuracy": round(mean_acc, 4),
        "mean_kappa": round(mean_kappa, 4),
        "per_head": head_metrics,
        "groups": groups,
        "consistency_violations": violations,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# §2  Response PPL evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def eval_response_ppl(
    checkpoint_path: str,
    base_model: str,
    test_sft_file: str,
    tokenizer_path: str | None = None,
    quantization: str | None = None,
    torch_dtype: str = "float32",
) -> dict[str, Any]:
    """Compute PPL on SFT eval set using a LoRA-tuned model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"\n  Loading response model from: {checkpoint_path}")

    tok_path = tokenizer_path or checkpoint_path
    # If tokenizer_path is a directory without tokenizer files, use base_model for tokenizer
    tokenizer = None
    if Path(tok_path).is_dir() and not (Path(tok_path) / "tokenizer_config.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        print(f"  (using base model tokenizer: {base_model})")
    else:
        tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=getattr(torch, torch_dtype),
        trust_remote_code=True,
    )

    # Load LoRA adapter
    adapter_dir = Path(checkpoint_path)
    if (adapter_dir / "adapter_config.json").exists():
        # Direct adapter dir (e.g., response_generator_best)
        model = PeftModel.from_pretrained(model, str(adapter_dir))
    elif (adapter_dir / "backbone" / "adapter_config.json").exists():
        # Joint model structure
        model = PeftModel.from_pretrained(model, str(adapter_dir / "backbone"))
    else:
        print(f"  WARNING: no adapter found at {checkpoint_path}")

    model.eval()

    ds = SFTDataset(test_sft_file, tokenizer, max_seq_len=2048)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_sft_batch)

    total_loss = 0.0
    n_tokens = 0
    n_samples = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="  response PPL", leave=False):
            input_ids = batch["input_ids"].to(model.device)
            attn_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            loss = out.loss.item()
            n_valid_tokens = (labels != -100).sum().item()

            total_loss += loss * n_valid_tokens
            n_tokens += n_valid_tokens
            n_samples += 1

    avg_loss = total_loss / max(n_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))

    return {
        "checkpoint": checkpoint_path,
        "n_samples": n_samples,
        "n_tokens": n_tokens,
        "avg_loss": round(avg_loss, 4),
        "ppl": round(ppl, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    p = argparse.ArgumentParser(description="Ablation 4: Joint vs Separate eval")
    p.add_argument("--config", default="llm_finetuning/configs/eval.yaml")
    p.add_argument("--skip-response", action="store_true", help="Skip PPL eval (faster)")
    args = p.parse_args()

    with open(PROJECT_ROOT / args.config) as f:
        cfg = yaml.safe_load(f)

    base_model = cfg.get("base_model", "Qwen/Qwen3-1.7B")
    quantization = cfg.get("quantization")  # None means full precision
    torch_dtype = cfg.get("torch_dtype", "float32")
    test_heads = cfg["data"]["test_heads_file"]
    test_sft = cfg["data"]["test_sft_file"]

    separate_latent_ckpt = cfg["latent_predictor_checkpoint"]
    separate_response_ckpt = cfg["response_checkpoint"]
    joint_ckpt = cfg["joint_checkpoint"]

    print("=" * 70)
    print("  ABLATION 4: JOINT vs SEPARATE EVALUATION")
    print("=" * 70)

    # ── Latent Head Evaluation ────────────────────────────────────────────────
    print("\n── §1 Latent Head Evaluation ──")
    print(f"  Separate latent: {separate_latent_ckpt}")
    print(f"  Joint model:     {joint_ckpt}")

    separate_latent = eval_latent_heads(
        separate_latent_ckpt, base_model, test_heads,
        quantization=quantization, torch_dtype=torch_dtype,
    )
    joint_latent = eval_latent_heads(
        joint_ckpt, base_model, test_heads,
        quantization=quantization, torch_dtype=torch_dtype,
    )

    print(f"\n  Separate → mean acc={separate_latent['mean_accuracy']:.4f}  κ={separate_latent['mean_kappa']:.4f}")
    print(f"  Joint    → mean acc={joint_latent['mean_accuracy']:.4f}  κ={joint_latent['mean_kappa']:.4f}")
    print(f"  Δ accuracy: {joint_latent['mean_accuracy'] - separate_latent['mean_accuracy']:+.4f}")
    print(f"  Δ kappa:    {joint_latent['mean_kappa'] - separate_latent['mean_kappa']:+.4f}")

    # ── Consistency Violations ────────────────────────────────────────────────
    print("\n── Consistency Violations ──")
    for vtype in ["high_secrecy_full_reveal", "hostile_affection_high"]:
        s = separate_latent["consistency_violations"].get(vtype, 0)
        j = joint_latent["consistency_violations"].get(vtype, 0)
        n = separate_latent["consistency_violations"]["total_samples"]
        print(f"  {vtype}: separate={s}/{n}  joint={j}/{n}")

    # ── Per-head comparison ───────────────────────────────────────────────────
    print("\n── Per-Head Accuracy Comparison ──")
    print(f"  {'Head':<25s} {'Separate':>8s} {'Joint':>8s} {'Δ':>8s}  {'Group'}")
    print(f"  {'─'*25} {'─'*8} {'─'*8} {'─'*8}  {'─'*12}")
    for field in sorted(separate_latent["per_head"]):
        s = separate_latent["per_head"][field]
        j = joint_latent["per_head"].get(field, {})
        if not j:
            continue
        delta = j["accuracy"] - s["accuracy"]
        marker = "⚠" if abs(delta) > 0.02 else " "
        print(f"  {field:<25s} {s['accuracy']:8.4f} {j['accuracy']:8.4f} {delta:+8.4f} {marker} {s['group']}")

    # ── Response PPL ──────────────────────────────────────────────────────────
    if not args.skip_response:
        print("\n── §2 Response PPL ──")
        print(f"  Separate response: {separate_response_ckpt}")
        print(f"  Joint model:       {joint_ckpt}")

        separate_ppl = eval_response_ppl(
            separate_response_ckpt, base_model, test_sft,
            quantization=quantization, torch_dtype=torch_dtype,
        )
        joint_ppl = eval_response_ppl(
            joint_ckpt, base_model, test_sft,
            quantization=quantization, torch_dtype=torch_dtype,
        )

        print(f"\n  Separate → PPL={separate_ppl['ppl']:.2f}  (loss={separate_ppl['avg_loss']:.4f}, tokens={separate_ppl['n_tokens']})")
        print(f"  Joint    → PPL={joint_ppl['ppl']:.2f}  (loss={joint_ppl['avg_loss']:.4f}, tokens={joint_ppl['n_tokens']})")
        print(f"  Δ PPL:    {joint_ppl['ppl'] - separate_ppl['ppl']:+.2f}")
    else:
        separate_ppl = {}
        joint_ppl = {}

    # ── Summary ───────────────────────────────────────────────────────────────
    results = {
        "ablation": "joint_vs_separate",
        "separate": {
            "latent": {k: v for k, v in separate_latent.items() if k != "per_head"},
            "response_ppl": separate_ppl,
        },
        "joint": {
            "latent": {k: v for k, v in joint_latent.items() if k != "per_head"},
            "response_ppl": joint_ppl,
        },
        "per_head_comparison": {},
    }

    for field in sorted(separate_latent["per_head"]):
        s = separate_latent["per_head"][field]
        j = joint_latent["per_head"].get(field, {})
        results["per_head_comparison"][field] = {
            "separate_acc": s["accuracy"],
            "joint_acc": j.get("accuracy", 0),
            "separate_kappa": s["cohen_kappa"],
            "joint_kappa": j.get("cohen_kappa", 0),
            "delta_acc": j.get("accuracy", 0) - s["accuracy"],
            "delta_kappa": j.get("cohen_kappa", 0) - s["cohen_kappa"],
        }

    out_path = PROJECT_ROOT / "eval_results" / "ablation_joint_vs_separate.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✓ Results saved to {out_path}")

    # ── Verdict ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)
    if separate_latent["mean_accuracy"] and joint_latent["mean_accuracy"]:
        delta_acc = joint_latent["mean_accuracy"] - separate_latent["mean_accuracy"]
        if abs(delta_acc) < 0.01:
            print(f"  Latent: Joint ≈ Separate (Δ={delta_acc:+.4f}) — shared backbone preserves latent quality")
        elif delta_acc > 0:
            print(f"  Latent: Joint > Separate (Δ={delta_acc:+.4f}) — multi-task training improves latent heads")
        else:
            print(f"  Latent: Joint < Separate (Δ={delta_acc:+.4f}) — joint training degrades latent heads")

    if separate_ppl and joint_ppl:
        delta_ppl = joint_ppl["ppl"] - separate_ppl["ppl"]
        if abs(delta_ppl) < 0.1:
            print(f"  Response: Joint ≈ Separate (ΔPPL={delta_ppl:+.2f}) — shared backbone preserves generation quality")
        elif delta_ppl < 0:
            print(f"  Response: Joint > Separate (ΔPPL={delta_ppl:+.2f}) — joint training improves generation")
        else:
            print(f"  Response: Joint < Separate (ΔPPL={delta_ppl:+.2f}) — joint training degrades generation")

    print("=" * 70)


if __name__ == "__main__":
    main()
