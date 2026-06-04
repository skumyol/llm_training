#!/usr/bin/env python3
"""
Threshold sweep for selective router: F1 / leakage vs slow-path rate.

Runs the selective router across a grid of confidence thresholds and records
the trade-off between routing accuracy, leakage safety, and slow-path rate.

Usage:
    PYTHONPATH=. python scripts/sweep_selective_router.py \
        --config configs/eval.yaml \
        --predicted-zt eval_results/predicted_zt.jsonl \
        --test-trace data/splits/test_trace.jsonl \
        --leakage-file eval_results/response_eval_metrics.json \
        --output eval_results/threshold_sweep.json

Outputs:
    - threshold_sweep.json  (per-threshold metrics)
    - threshold_sweep.md    (human-readable trade-off table)
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

from src.eval.selective_router import SelectiveRouter


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--predicted-zt", required=True, help="predicted_zt.jsonl with confidences")
    p.add_argument("--test-trace", required=True, help="Gold test trace JSONL")
    p.add_argument("--leakage-file", default=None, help="Optional: response_eval_metrics.json for leakage rate")
    p.add_argument("--output", default="eval_results/threshold_sweep.json")
    p.add_argument("--n-steps", type=int, default=21, help="Threshold grid steps (0 to 1)")
    p.add_argument("--heads", nargs="+", default=["response_policy", "reveal_decision", "value_conflict", "secrecy_pressure"])
    return p.parse_args()


def load_predicted_zt(path: str) -> dict[tuple[str, int], dict]:
    lookup: dict[tuple[str, int], dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ep = str(rec.get("episode_id", ""))
            turn = rec.get("turn_idx", rec.get("turn", 0))
            if ep:
                lookup[(ep, int(turn))] = rec
    return lookup


def load_test_trace(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def gold_slow_path(D_t: dict, N_t: dict) -> bool:
    if N_t.get("value_conflict") == "strong":
        return True
    if N_t.get("secrecy_pressure") == "high" and D_t.get("reveal_decision") in {"hint", "partial", "full"}:
        return True
    if D_t.get("response_policy") in {"threaten", "negotiate"}:
        return True
    return False


def evaluate_threshold(
    trace: list[dict],
    pred_lookup: dict,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    router = SelectiveRouter(thresholds)

    tp = fp = tn = fn = 0
    slow_count = 0
    missing = 0

    for rec in trace:
        ep = str(rec.get("episode_id", ""))
        turn = rec.get("turn_idx", rec.get("turn", 0))
        pred = pred_lookup.get((ep, int(turn)))
        if pred is None:
            missing += 1
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

        pred_D = {
            "response_policy": pred.get("response_policy", ""),
            "reveal_decision": pred.get("reveal_decision", ""),
        }
        pred_N = {
            "value_conflict": pred.get("value_conflict", ""),
            "secrecy_pressure": pred.get("secrecy_pressure", ""),
        }
        confidences = {
            k: pred.get(f"{k}_conf", 1.0)
            for k in ["response_policy", "reveal_decision", "value_conflict", "secrecy_pressure"]
            if f"{k}_conf" in pred
        }

        pred_slow, _ = router.should_route_slow(pred_D, pred_N, confidences=confidences)

        if pred_slow:
            slow_count += 1
        if gold_slow and pred_slow:
            tp += 1
        elif gold_slow and not pred_slow:
            fn += 1
        elif not gold_slow and pred_slow:
            fp += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1, precision + recall)
    fpr = fp / max(1, fp + tn)
    fnr = fn / max(1, fn + tp)
    unsafe_fast_path = fn / max(1, total)
    slow_rate = slow_count / max(1, total)
    cost_fn5_fp1 = 5 * fn + fp
    cost_fn10_fp1 = 10 * fn + fp

    return {
        "thresholds": thresholds,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "unsafe_fast_path_rate": round(unsafe_fast_path, 4),
        "slow_path_precision": round(precision, 4),
        "slow_path_recall": round(recall, 4),
        "slow_path_rate": round(slow_rate, 4),
        "routing_cost_fn5_fp1": cost_fn5_fp1,
        "routing_cost_fn10_fp1": cost_fn10_fp1,
        "n_evaluated": total,
        "missing_predictions": missing,
    }


def main():
    args = parse_args()

    pred_lookup = load_predicted_zt(args.predicted_zt)
    trace = load_test_trace(args.test_trace)
    print(f"Loaded {len(trace)} trace records, {len(pred_lookup)} predictions")

    # Load optional leakage rate per episode/turn if available
    leakage_lookup: dict[tuple[str, int], bool] = {}
    if args.leakage_file and Path(args.leakage_file).exists():
        with open(args.leakage_file) as f:
            leak_data = json.load(f)
        # Try to extract per-sample leakage flags
        if isinstance(leak_data, list):
            samples = leak_data
        else:
            samples = leak_data.get("records", leak_data.get("samples", []))
        for s in samples:
            ep = str(s.get("episode_id", ""))
            turn = s.get("turn_idx", 0)
            leaked = not s.get("validation", {}).get("accepted", True)
            leakage_lookup[(ep, int(turn))] = leaked
        print(f"Loaded leakage info for {len(leakage_lookup)} samples")

    # Sweep: uniform threshold across all heads, then per-head individual sweeps
    results: list[dict] = []
    thresholds_grid = np.linspace(0.0, 1.0, args.n_steps)

    # Uniform sweep (same threshold for all heads)
    for tau in tqdm(thresholds_grid, desc="Uniform threshold sweep"):
        thresh = {h: float(tau) for h in args.heads}
        metrics = evaluate_threshold(trace, pred_lookup, thresh)
        metrics["sweep_type"] = "uniform"
        metrics["active_head"] = "all"
        results.append(metrics)

    # Per-head sweep (vary one head, keep others at default)
    defaults = {"response_policy": 0.65, "reveal_decision": 0.70, "value_conflict": 0.75, "secrecy_pressure": 0.75}
    for head in tqdm(args.heads, desc="Per-head threshold sweep"):
        for tau in thresholds_grid:
            thresh = dict(defaults)
            thresh[head] = float(tau)
            metrics = evaluate_threshold(trace, pred_lookup, thresh)
            metrics["sweep_type"] = "per_head"
            metrics["active_head"] = head
            results.append(metrics)

    # Build trade-off report
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"results": results}, f, indent=2)

    # Markdown summary: safety-first operating points
    md_lines = [
        "# Selective Router Threshold Sweep\n\n",
        "## Uniform Threshold (all heads equal)\n\n",
        "| Threshold | F1 | Precision | Recall | Unsafe FP | Slow-Path % | Cost(5:1) | Cost(10:1) |\n",
        "|-----------|----|-----------|--------|-----------|-------------|-----------|------------|\n",
    ]
    uniform = [r for r in results if r["sweep_type"] == "uniform"]
    uniform.sort(key=lambda x: x["thresholds"][args.heads[0]])
    for r in uniform:
        tau = r["thresholds"][args.heads[0]]
        md_lines.append(
            f"| {tau:.2f} | {r['f1']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['unsafe_fast_path_rate']:.3f} | {r['slow_path_rate']:.3f} | {r['routing_cost_fn5_fp1']} | {r['routing_cost_fn10_fp1']} |\n"
        )

    md_lines.append("\n## Notes\n")
    md_lines.append("- Lower threshold → more fast-path (higher recall, more unsafe fast-paths).\n")
    md_lines.append("- Higher threshold → more conservative (lower unsafe fast-path, more over-routing).\n")
    md_lines.append("- Choose operating point by minimizing unsafe fast-path rate first, then slow-path rate.\n")

    md_path = out_path.with_suffix(".md")
    with open(md_path, "w") as f:
        f.writelines(md_lines)

    # Safety-first selection: minimize unsafe fast-path, then slow-path rate, then maximize F1
    uniform_sorted = sorted(uniform, key=lambda x: (x["unsafe_fast_path_rate"], x["slow_path_rate"], -x["f1"]))
    safest_point = uniform_sorted[0]
    best_f1 = max(uniform, key=lambda x: x["f1"])

    print("\n=== Threshold Sweep Summary ===")
    print(f"  Evaluated {len(results)} threshold configurations")
    print(f"  Safety-first: unsafe_FP={safest_point['unsafe_fast_path_rate']:.3f}, slow={safest_point['slow_path_rate']:.3f}, F1={safest_point['f1']:.3f} @ τ={safest_point['thresholds'][args.heads[0]]:.2f}")
    print(f"  Best F1:      {best_f1['f1']:.3f} @ τ={best_f1['thresholds'][args.heads[0]]:.2f}")
    print(f"  JSON: {out_path}")
    print(f"  Markdown: {md_path}")


if __name__ == "__main__":
    main()
