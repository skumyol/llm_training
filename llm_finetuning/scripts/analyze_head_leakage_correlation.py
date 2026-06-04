#!/usr/bin/env python3
"""
Correlate predicted head values with response leakage.

Computes per-head association (point-biserial correlation, MI) between
predicted head values and whether the generated response leaked information.

Usage:
    PYTHONPATH=. python scripts/analyze_head_leakage_correlation.py \
        --predicted-zt eval_results/predicted_zt.jsonl \
        --leakage-file eval_results/validated_generations.json \
        --output-dir eval_results/head_leakage_corr

Or with sample_generations.json (must contain 'secret_leak' bool):
    PYTHONPATH=. python scripts/analyze_head_leakage_correlation.py \
        --predicted-zt eval_results/predicted_zt.jsonl \
        --leakage-file eval_results/sample_generations.json \
        --output-dir eval_results/head_leakage_corr
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import numpy as np
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predicted-zt", required=True)
    p.add_argument("--leakage-file", required=True)
    p.add_argument("--output-dir", default="eval_results/head_leakage_corr")
    return p.parse_args()


def load_predicted_zt(path: str) -> tuple[dict[tuple[str, int], dict], list[dict]]:
    """Load predicted Z_t keyed by (episode_id, turn_idx) and as ordered list."""
    lookup: dict[tuple[str, int], dict] = {}
    ordered: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ordered.append(rec)
            ep = str(rec.get("episode_id", ""))
            turn = rec.get("turn_idx", rec.get("turn", 0))
            if ep:
                lookup[(ep, int(turn))] = rec
    return lookup, ordered


def load_leakage_labels(path: str) -> tuple[dict[tuple[str, int], bool], list[bool]]:
    """Load leakage labels keyed by (episode_id, turn_idx) and as ordered list."""
    lookup: dict[tuple[str, int], bool] = {}
    ordered: list[bool] = []
    data = json.load(open(path))

    # Try validated_generations.json format first
    records = data.get("records", data)
    if not isinstance(records, list):
        records = [data]

    for rec in records:
        ep = str(rec.get("episode_id", ""))
        turn = rec.get("turn_idx", 0)

        # Determine leakage from various possible fields
        leaked = False
        if "validation" in rec:
            leaked = not rec["validation"].get("accepted", True)
        elif "secret_leak" in rec:
            leaked = bool(rec["secret_leak"])
        elif "leaked" in rec:
            leaked = bool(rec["leaked"])
        else:
            # Fallback: check if any leakage keywords exist
            generated = rec.get("generated", "")
            leaked = "secret" in generated.lower() or "vault" in generated.lower()

        ordered.append(leaked)
        if ep:
            lookup[(ep, int(turn))] = leaked

    return lookup, ordered


def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            ent -= p * math.log2(p)
    return ent


def _mutual_information(x_vals: list, y_vals: list) -> float:
    joint = Counter(zip(x_vals, y_vals))
    marginal_x = Counter(x_vals)
    marginal_y = Counter(y_vals)
    n = len(x_vals)
    if n == 0:
        return 0.0
    mi = 0.0
    for (xv, yv), count in joint.items():
        if count == 0:
            continue
        p_xy = count / n
        p_x = marginal_x[xv] / n
        p_y = marginal_y[yv] / n
        if p_x > 0 and p_y > 0:
            mi += p_xy * math.log2(p_xy / (p_x * p_y))
    return mi


def point_biserial(x_binary: np.ndarray, y_continuous: np.ndarray) -> float:
    """Point-biserial correlation between binary x and continuous y."""
    if len(x_binary) == 0 or len(set(x_binary)) < 2:
        return 0.0
    mask_1 = x_binary == 1
    mask_0 = x_binary == 0
    n1 = mask_1.sum()
    n0 = mask_0.sum()
    if n1 == 0 or n0 == 0:
        return 0.0
    m1 = y_continuous[mask_1].mean()
    m0 = y_continuous[mask_0].mean()
    s = y_continuous.std(ddof=1)
    if s == 0:
        return 0.0
    r = (m1 - m0) / s * math.sqrt(n1 * n0 / (len(x_binary) ** 2))
    return float(np.clip(r, -1.0, 1.0))


def analyze(pred_lookup: dict, leak_lookup: dict) -> dict[str, dict]:
    # Align samples
    aligned: list[tuple[dict, bool]] = []
    for key, pred in pred_lookup.items():
        if key in leak_lookup:
            aligned.append((pred, leak_lookup[key]))

    print(f"Aligned {len(aligned)} samples with leakage labels")
    if len(aligned) < 10:
        print("WARNING: Too few aligned samples for reliable correlation")

    # Collect all head fields
    head_fields = set()
    for pred, _ in aligned:
        head_fields.update(k for k in pred.keys() if not k.endswith("_conf") and k not in {"episode_id", "turn_idx"})

    results: dict[str, dict] = {}
    leakage_binary = [int(leaked) for _, leaked in aligned]

    for field in sorted(head_fields):
        vals = []
        for pred, leaked in aligned:
            v = pred.get(field)
            if v is None:
                continue
            vals.append(str(v))

        if len(vals) != len(leakage_binary) or len(set(vals)) < 2:
            continue

        mi = _mutual_information(vals, [str(l) for l in leakage_binary])
        h_field = _entropy(Counter(vals))
        h_leak = _entropy(Counter(leakage_binary))
        nmi = mi / min(h_field, h_leak) if min(h_field, h_leak) > 0 else 0.0

        # Leak rate per head value
        value_to_leak_rate: dict[str, float] = {}
        value_counts: dict[str, int] = {}
        for v, leaked in zip(vals, leakage_binary):
            value_counts[v] = value_counts.get(v, 0) + 1
            if leaked:
                value_to_leak_rate[v] = value_to_leak_rate.get(v, 0) + 1
        for v in value_to_leak_rate:
            value_to_leak_rate[v] /= max(1, value_counts[v])

        # Find highest-leak value
        max_leak_value = max(value_to_leak_rate, key=value_to_leak_rate.get) if value_to_leak_rate else None
        max_leak_rate = value_to_leak_rate.get(max_leak_value, 0.0) if max_leak_value else 0.0

        results[field] = {
            "mi": round(mi, 4),
            "nmi": round(nmi, 4),
            "entropy": round(h_field, 4),
            "n_samples": len(vals),
            "value_leak_rates": {k: round(v, 4) for k, v in value_to_leak_rate.items()},
            "max_leak_value": max_leak_value,
            "max_leak_rate": round(max_leak_rate, 4),
        }

    return results


def main():
    args = parse_args()

    pred_lookup, pred_ordered = load_predicted_zt(args.predicted_zt)
    leak_lookup, leak_ordered = load_leakage_labels(args.leakage_file)

    results = analyze(pred_lookup, leak_lookup)
    if len(results) == 0:
        print("Key-based alignment failed; falling back to positional alignment...")
        n = min(len(pred_ordered), len(leak_ordered))
        pred_lookup_pos = {i: pred_ordered[i] for i in range(n)}
        leak_lookup_pos = {i: leak_ordered[i] for i in range(n)}
        results = analyze(pred_lookup_pos, leak_lookup_pos)
        print(f"Positional alignment: {n} samples")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "head_leakage_correlation.json", "w") as f:
        json.dump(results, f, indent=2)

    # Markdown report
    ranked = sorted(results.items(), key=lambda x: x[1]["nmi"], reverse=True)
    lines = [
        "# Head–Leakage Correlation Report\n\n",
        "Mutual information between predicted head values and response leakage.\n",
        "Higher NMI = head is more predictive of whether a leak occurs.\n\n",
        "| Rank | Head | MI | NMI | Max Leak Value | Max Leak Rate | Interpretation |\n",
        "|------|------|----|-----|----------------|---------------|----------------|\n",
    ]
    for rank, (field, m) in enumerate(ranked, 1):
        max_val = m.get("max_leak_value", "N/A") or "N/A"
        max_rate = m.get("max_leak_rate", 0.0)
        if max_rate > 0.3:
            interp = "HIGH RISK: this head value strongly predicts leakage"
        elif max_rate > 0.1:
            interp = "MODERATE RISK"
        elif m["nmi"] > 0.05:
            interp = "Weak but non-random association"
        else:
            interp = "No significant association"
        lines.append(
            f"| {rank} | {field} | {m['mi']:.3f} | {m['nmi']:.3f} | {max_val} | {max_rate:.2%} | {interp} |\n"
        )

    lines.append("\n## Usage for Router Improvement\n\n")
    lines.append("- Heads with high NMI should be monitored closely in the router.\n")
    lines.append("- If a specific head value (e.g., `reveal_decision=partial`) has high leak rate,\n")
    lines.append("  the router should always slow-path when that value is predicted.\n")

    with open(out_dir / "head_leakage_report.md", "w") as f:
        f.writelines(lines)

    print(f"\n=== Head–Leakage Correlation Summary ===")
    print(f"  Analyzed {len(results)} heads")
    for field, m in ranked[:5]:
        print(f"  {field:30s} NMI={m['nmi']:.3f}  max_leak_rate={m.get('max_leak_rate', 0):.2%}")
    print(f"  JSON: {out_dir / 'head_leakage_correlation.json'}")
    print(f"  Report: {out_dir / 'head_leakage_report.md'}")


if __name__ == "__main__":
    main()
