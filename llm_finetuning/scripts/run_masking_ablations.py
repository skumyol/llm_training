#!/usr/bin/env python3
"""
Masking ablation: systematically corrupt/remove heads at inference to measure
which heads the router actually depends on.

Runs three replacement modes for each masked head:
  - majority: replace with empirical majority class
  - random:  replace with random draw from empirical distribution
  - gold:    replace with gold label from test trace

Usage:
    PYTHONPATH=llm_finetuning:. python llm_finetuning/scripts/run_masking_ablations.py \
        --predicted-zt eval_results/predicted_zt.jsonl \
        --test-trace data/splits/val_trace.jsonl \
        --output eval_results/masking_ablations.json
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List

random.seed(42)

ROUTING_HEADS = ["response_policy", "reveal_decision", "value_conflict", "secrecy_pressure"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predicted-zt", required=True)
    p.add_argument("--test-trace", required=True)
    p.add_argument("--output", default="eval_results/masking_ablations.json")
    return p.parse_args()


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_lookup(records: list[dict]) -> dict:
    return {(str(r["episode_id"]), int(r["turn_idx"])): r for r in records}


def gold_slow_path(D_t: dict, N_t: dict) -> bool:
    if N_t.get("value_conflict") == "strong":
        return True
    if N_t.get("secrecy_pressure") == "high" and D_t.get("reveal_decision") in {"hint", "partial", "full"}:
        return True
    if D_t.get("response_policy") in {"threaten", "negotiate"}:
        return True
    return False


def router_decision(pred: dict) -> bool:
    """Deterministic router using predicted fields."""
    D_t = {
        "response_policy": pred.get("response_policy", ""),
        "reveal_decision": pred.get("reveal_decision", ""),
    }
    N_t = {
        "value_conflict": pred.get("value_conflict", ""),
        "secrecy_pressure": pred.get("secrecy_pressure", ""),
    }
    return gold_slow_path(D_t, N_t)


def compute_routing_metrics(pred_lookup: dict, trace: list[dict]) -> dict:
    tp = fp = tn = fn = 0
    total = 0
    for rec in trace:
        ep = str(rec.get("episode_id", ""))
        turn = rec.get("turn_idx", rec.get("turn", 0))
        pred = pred_lookup.get((ep, int(turn)))
        if pred is None:
            continue
        gold_D = {
            "response_policy": rec.get("D_t", {}).get("response_policy", ""),
            "reveal_decision": rec.get("D_t", {}).get("reveal_decision", ""),
        }
        gold_N = {
            "value_conflict": rec.get("N_t", {}).get("value_conflict", ""),
            "secrecy_pressure": rec.get("N_t", {}).get("secrecy_pressure", ""),
        }
        gold_slow = gold_slow_path(gold_D, gold_N)
        pred_slow = router_decision(pred)
        if gold_slow and pred_slow:
            tp += 1
        elif not gold_slow and pred_slow:
            fp += 1
        elif gold_slow and not pred_slow:
            fn += 1
        else:
            tn += 1
        total += 1

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1, precision + recall)
    fpr = fp / max(1, fp + tn)
    fnr = fn / max(1, fn + tp)
    unsafe_fp = fn / max(1, total)
    slow_rate = (tp + fp) / max(1, total)
    cost_fn5 = (5 * fn + fp) / max(1, total)
    cost_fn10 = (10 * fn + fp) / max(1, total)

    return {
        "routing_f1": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "unsafe_fast_path_rate": round(unsafe_fp, 4),
        "slow_path_rate": round(slow_rate, 4),
        "routing_cost_fn5": round(cost_fn5, 4),
        "routing_cost_fn10": round(cost_fn10, 4),
        "n_evaluated": total,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def empirical_distributions(predictions: list[dict]) -> dict:
    dists = {}
    for head in ROUTING_HEADS:
        vals = [p[head] for p in predictions if head in p and p[head]]
        dists[head] = Counter(vals)
    return dists


def majority_class(dist: Counter) -> str:
    return dist.most_common(1)[0][0] if dist else ""


def random_class(dist: Counter) -> str:
    vals, weights = zip(*dist.items()) if dist else ([""], [1])
    return random.choices(vals, weights=weights)[0]


def apply_mask(pred_lookup: dict, trace: list[dict], mask_heads: list[str], mode: str, dists: dict) -> dict:
    """Create a modified pred_lookup with specified heads replaced."""
    trace_lookup = build_lookup(trace)
    modified = {}
    for key, pred in pred_lookup.items():
        new_pred = dict(pred)
        for head in mask_heads:
            if mode == "majority":
                new_pred[head] = majority_class(dists[head])
            elif mode == "random":
                new_pred[head] = random_class(dists[head])
            elif mode == "gold":
                gold_rec = trace_lookup.get(key, {})
                if head in ["response_policy", "reveal_decision"]:
                    new_pred[head] = gold_rec.get("D_t", {}).get(head, pred.get(head, ""))
                else:
                    new_pred[head] = gold_rec.get("N_t", {}).get(head, pred.get(head, ""))
        modified[key] = new_pred
    return modified


def main():
    args = parse_args()
    predictions = load_jsonl(args.predicted_zt)
    trace = load_jsonl(args.test_trace)
    pred_lookup = build_lookup(predictions)
    dists = empirical_distributions(predictions)

    print(f"Loaded {len(predictions)} predictions, {len(trace)} trace records")
    for head, dist in dists.items():
        print(f"  {head}: {dict(dist)}")

    # Baseline: no masking
    baseline_metrics = compute_routing_metrics(pred_lookup, trace)
    print(f"\nBaseline (no mask): F1={baseline_metrics['routing_f1']}, unsafe_FP={baseline_metrics['unsafe_fast_path_rate']}")

    ablation_conditions = [
        (["response_policy"], "response_policy"),
        (["reveal_decision"], "reveal_decision"),
        (["secrecy_pressure"], "secrecy_pressure"),
        (["value_conflict"], "value_conflict"),
        (["response_policy", "reveal_decision", "value_conflict", "secrecy_pressure"], "all_routing"),
    ]

    results = {
        "baseline": baseline_metrics,
        "ablations": [],
    }

    for mask_heads, label in ablation_conditions:
        for mode in ["majority", "random", "gold"]:
            modified = apply_mask(pred_lookup, trace, mask_heads, mode, dists)
            metrics = compute_routing_metrics(modified, trace)
            entry = {
                "ablation": label,
                "mask_heads": mask_heads,
                "mode": mode,
                **metrics,
            }
            results["ablations"].append(entry)
            print(f"  {label:20s} {mode:8s}  F1={metrics['routing_f1']:.3f}  unsafe_FP={metrics['unsafe_fast_path_rate']:.3f}  slow={metrics['slow_path_rate']:.3f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
