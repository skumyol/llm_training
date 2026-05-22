#!/usr/bin/env python3
"""
Gold Z_t Conditioning — The Bridge Experiment
=============================================
Tests whether structured social-state (Z_t) information correlates with 
response generation difficulty. If yes, predicting Z_t is useful. If no,
the latent predictor solves a task the generator doesn't need.

Method:
  1. Load existing response generator (OCEAN/VAD conditioned)
  2. Compute per-example PPL on dialogue val set
  3. Join with head supervision labels by (episode_id, turn_idx)
  4. Group PPL by Z_t field values — check if social state predicts difficulty

This requires NO retraining. ~10 min on GPU.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "llm_finetuning"))

from src.training.dataset import LABEL_TO_IDX, LABEL_MAPS

# Z_t fields to analyze — the ones most relevant to generation difficulty
ZT_FIELDS = [
    "response_policy",
    "reveal_decision", 
    "secrecy_pressure",
    "repair_strategy",
    "player_knowledge",
    "trust_delta",
    "respect_delta",
    "dominance_delta",
]

# Human-readable labels for reporting
FIELD_LABELS = {
    "response_policy": LABEL_MAPS.get("response_policy", []),
    "reveal_decision": LABEL_MAPS.get("reveal_decision", []),
    "secrecy_pressure": LABEL_MAPS.get("secrecy_pressure", []),
    "repair_strategy": LABEL_MAPS.get("repair_strategy", []),
    "player_knowledge": LABEL_MAPS.get("player_knowledge", []),
    "trust_delta": LABEL_MAPS.get("trust_delta", []),
    "respect_delta": LABEL_MAPS.get("respect_delta", []),
    "dominance_delta": LABEL_MAPS.get("dominance_delta", []),
}


def build_head_index(heads_jsonl: str) -> dict[tuple[str, int], dict[str, int]]:
    """Index head supervision labels by (episode_id, turn_idx)."""
    index = {}
    with open(heads_jsonl) as f:
        for line in f:
            d = json.loads(line)
            eid = str(d.get("episode_id", ""))
            tidx = d.get("turn_idx", d.get("turn", None))
            if not eid or tidx is None:
                continue
            labels = d.get("labels", {})
            encoded = {}
            for field in ZT_FIELDS:
                val = labels.get(field)
                idx_map = LABEL_TO_IDX.get(field, {})
                if val is None:
                    encoded[field] = -1
                elif isinstance(val, list):
                    encoded[field] = idx_map.get(val[0] if val else "", -1)
                else:
                    encoded[field] = idx_map.get(str(val), -1)
            index[(eid, int(tidx))] = encoded
    return index


def main():
    parser = argparse.ArgumentParser(description="Gold Z_t conditioning bridge experiment")
    parser.add_argument("--checkpoint", default="checkpoints/response_generator_best")
    parser.add_argument("--dialogue", default="slm_training/data/dialogue/val.jsonl")
    parser.add_argument("--heads", default="data/splits/val_heads.jsonl")
    parser.add_argument("--max-samples", type=int, default=0, help="0=all")
    args = parser.parse_args()

    print("=" * 70)
    print("  GOLD Z_t BRIDGE EXPERIMENT")
    print("  Does social state predict generation difficulty?")
    print("=" * 70)

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"\nLoading: {args.checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, device_map="auto", torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()
    device = model.device

    # ── Build Z_t index ────────────────────────────────────────────────────
    print(f"Building Z_t index: {args.heads}")
    head_index = build_head_index(args.heads)
    print(f"  {len(head_index)} turns indexed")

    # ── Load dialogue & match ──────────────────────────────────────────────
    print(f"Loading dialogue: {args.dialogue}")
    matched = []
    unmatched = 0
    with open(args.dialogue) as f:
        for line in f:
            d = json.loads(line)
            meta = d.get("metadata", {})
            eid = str(meta.get("episode_id", ""))
            tidx = meta.get("turn_idx", None)
            if not eid or tidx is None:
                continue
            zt = head_index.get((eid, int(tidx)))
            if zt is None:
                unmatched += 1
                continue
            matched.append({**d, "_zt": zt})

    print(f"  Matched: {len(matched)}/{len(matched) + unmatched} ({100*len(matched)/(len(matched)+unmatched):.1f}%)")
    if args.max_samples > 0:
        matched = matched[:args.max_samples]
        print(f"  Sampled: {len(matched)}")

    # ── Compute per-example PPL ────────────────────────────────────────────
    print(f"\nComputing PPL on {len(matched)} examples...")
    per_example = []
    
    with torch.no_grad():
        for rec in tqdm(matched, desc="PPL"):
            ctx = "\n".join(
                f"{m['speaker']}: {m['text']}" for m in rec.get("dialogue_context", [])
            )
            tgt = rec.get("target_response", "")
            if not ctx or not tgt:
                continue

            full = f"{ctx}\nnpc: {tgt}"
            enc = tokenizer(full, return_tensors="pt", truncation=True, max_length=2048)
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)

            prompt = f"{ctx}\nnpc:"
            prompt_enc = tokenizer(prompt, return_tensors="pt")
            prompt_len = prompt_enc["input_ids"].shape[1]

            labels = input_ids.clone()
            labels[:, :prompt_len] = -100

            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss.item()
            n_tok = (labels != -100).sum().item()
            ppl = math.exp(min(loss, 20))

            per_example.append({
                "loss": loss,
                "ppl": ppl,
                "n_target_tokens": n_tok,
                **{f"zt_{f}": rec["_zt"].get(f, -1) for f in ZT_FIELDS},
            })

    if not per_example:
        print("ERROR: No valid examples. Check data paths.")
        return

    # ── Analysis: group PPL by Z_t field values ────────────────────────────
    print(f"\n{'='*70}")
    print(f"  RESULTS: PPL by Z_t field value")
    print(f"  (Higher PPL = harder to generate = more benefit from knowing Z_t)")
    print(f"{'='*70}")

    summary = {"n_examples": len(per_example), "fields": {}}
    
    for field in ZT_FIELDS:
        groups = defaultdict(list)
        for ex in per_example:
            val = ex.get(f"zt_{field}", -1)
            groups[val].append(ex["ppl"])

        if len(groups) <= 1:
            continue

        label_names = FIELD_LABELS.get(field, [])
        field_summary = {}
        
        print(f"\n  ── {field} ──")
        print(f"  {'Value':<20s} {'N':>6s} {'Mean PPL':>10s} {'Std':>8s} {'Δ from min':>10s}")
        print(f"  {'─'*20} {'─'*6} {'─'*10} {'─'*8} {'─'*10}")

        min_ppl = min(np.mean(ppls) for ppls in groups.values())
        
        for val in sorted(groups.keys()):
            ppls = groups[val]
            mean_ppl = np.mean(ppls)
            std_ppl = np.std(ppls)
            delta = mean_ppl - min_ppl
            label = label_names[val] if val >= 0 and val < len(label_names) else f"idx_{val}"
            marker = " ← hardest" if delta == max(np.mean(groups[v]) - min_ppl for v in groups) else ""
            if delta > 0.05:
                marker += " ⚠"
            print(f"  {label:<20s} {len(ppls):>6d} {mean_ppl:>10.4f} {std_ppl:>8.4f} {delta:>+10.4f}{marker}")

            field_summary[label] = {
                "n": len(ppls), "mean_ppl": round(float(mean_ppl), 4),
                "std_ppl": round(float(std_ppl), 4),
            }

        # Effect size: (max PPL - min PPL) / overall std
        all_ppls = [ex["ppl"] for ex in per_example]
        overall_std = np.std(all_ppls)
        max_mean = max(np.mean(groups[v]) for v in groups)
        min_mean = min(np.mean(groups[v]) for v in groups)
        effect = (max_mean - min_mean) / overall_std if overall_std > 0 else 0
        
        print(f"  Effect size: {effect:.3f}σ  (max-min spread / overall std)")
        field_summary["_effect_size_sigma"] = round(effect, 3)
        summary["fields"][field] = field_summary

    # ── Overall verdict ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  VERDICT")
    print(f"{'='*70}")

    significant_fields = []
    for field, fs in summary["fields"].items():
        es = fs.get("_effect_size_sigma", 0)
        if es > 0.5:
            significant_fields.append((field, es))

    if significant_fields:
        print(f"\n  Z_t fields that predict generation difficulty (effect > 0.5σ):")
        for field, es in sorted(significant_fields, key=lambda x: -x[1]):
            print(f"    {field}: {es:.2f}σ")
        print(f"\n  → Knowing these Z_t values in advance WOULD help the generator.")
        print(f"  → The latent predictor has a job to do.")
    else:
        print(f"\n  No Z_t field significantly predicts generation difficulty (all < 0.5σ).")
        print(f"  → The generator doesn't benefit from knowing Z_t in advance.")
        print(f"  → The latent predictor may be solving a task the generator doesn't need.")

    # ── Consistency violation analysis ─────────────────────────────────────
    print(f"\n  Consistency check:")
    high_sec_full_rev = sum(
        1 for ex in per_example
        if ex.get("zt_secrecy_pressure") == 2 and ex.get("zt_reveal_decision") == 3
    )
    n = len(per_example)
    print(f"    High secrecy + Full reveal: {high_sec_full_rev}/{n} = {high_sec_full_rev/n:.4f}")
    if high_sec_full_rev > 0:
        high_sec_ppl = np.mean([ex["ppl"] for ex in per_example if ex.get("zt_secrecy_pressure") == 2 and ex.get("zt_reveal_decision") == 3])
        other_ppl = np.mean([ex["ppl"] for ex in per_example if not (ex.get("zt_secrecy_pressure") == 2 and ex.get("zt_reveal_decision") == 3)])
        print(f"    PPL on violations: {high_sec_ppl:.4f}  vs  other: {other_ppl:.4f}  Δ={high_sec_ppl - other_ppl:+.4f}")

    # ── Save ───────────────────────────────────────────────────────────────
    out_path = PROJECT_ROOT / "eval_results" / "gold_zt_bridge.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  ✓ Saved → {out_path}")


if __name__ == "__main__":
    main()
