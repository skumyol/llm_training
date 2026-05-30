#!/usr/bin/env python3
"""
Analyze head utility for routing decisions.

Computes per-head mutual information with routing labels and redundancy
scores to rank heads by operational necessity.

Usage:
    PYTHONPATH=. python scripts/analyze_head_utility.py \
        --heads-file data/splits/val_heads.jsonl \
        --output-dir eval_results/head_utility

Outputs:
    - head_utility_report.md   (human-readable ranking)
    - head_utility.json        (raw MI and redundancy scores)
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--heads-file", required=True, help="Head supervision JSONL")
    p.add_argument("--output-dir", default="eval_results/head_utility")
    p.add_argument("--min-support", type=int, default=10,
                   help="Minimum samples for a head value to be considered")
    return p.parse_args()


ROUTING_FIELDS = {
    "response_policy", "reveal_decision", "value_conflict", "secrecy_pressure"
}
AFFECT_FIELDS = {"valence", "threat", "control", "arousal"}
RELATIONAL_FIELDS = {
    "trust_level", "trust_delta", "respect_level", "respect_delta",
    "affection_level", "affection_delta", "familiarity_level", "familiarity_delta",
    "dominance_level", "dominance_delta", "obligation_level", "obligation_delta",
}
DESCRIPTIVE_FIELDS = {"tone", "dialogue_act", "risk_type", "repair_strategy"}
META_FIELDS = {"player_intent", "player_knowledge", "player_credibility", "duty_pressure", "face_pressure"}


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
    """Compute MI(X; Y) from paired discrete samples."""
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


def _conditional_entropy(x_vals: list, y_vals: list) -> float:
    """H(X | Y) = H(X, Y) - H(Y)."""
    joint_counts = Counter(zip(x_vals, y_vals))
    y_counts = Counter(y_vals)
    n = len(x_vals)
    if n == 0:
        return 0.0
    h = 0.0
    for (xv, yv), c in joint_counts.items():
        p_xy = c / n
        p_y = y_counts[yv] / n
        if p_xy > 0 and p_y > 0:
            h -= p_xy * math.log2(p_xy / p_y)
    return h


def _gold_slow_path(D_t: dict, N_t: dict) -> bool:
    if N_t.get("value_conflict") == "strong":
        return True
    if N_t.get("secrecy_pressure") == "high" and D_t.get("reveal_decision") in {"hint", "partial", "full"}:
        return True
    if D_t.get("response_policy") in {"threaten", "negotiate"}:
        return True
    return False


def analyze_head_utility(heads_path: str, output_dir: str, min_support: int = 10) -> dict:
    records = []
    with open(heads_path) as f:
        for line in tqdm(f, desc="Loading heads"):
            rec = json.loads(line.strip())
            records.append(rec)

    # Extract routing decisions and episode IDs
    routing_labels = []
    episode_ids = []
    head_values: dict[str, list] = {}
    for rec in records:
        labels = rec.get("labels", {})
        D_t = {
            "response_policy": labels.get("response_policy", ""),
            "reveal_decision": labels.get("reveal_decision", ""),
        }
        N_t = {
            "value_conflict": labels.get("value_conflict", ""),
            "secrecy_pressure": labels.get("secrecy_pressure", ""),
        }
        routing_labels.append(int(_gold_slow_path(D_t, N_t)))
        episode_ids.append(str(rec.get("episode_id", "")))

        for field, val in labels.items():
            if field not in head_values:
                head_values[field] = []
            if isinstance(val, list):
                val = tuple(sorted(val)) if val else ()
            head_values[field].append(str(val) if val is not None else "__none__")

    # Ensure all head lists align with routing_labels
    n = len(routing_labels)
    for field, vals in head_values.items():
        if len(vals) != n:
            print(f"[WARN] {field} has {len(vals)} values vs {n} routing labels; skipping")
            continue

    # Compute MI and redundancy
    routing_entropy = _entropy(Counter(routing_labels))
    results: dict[str, dict] = {}

    for field, vals in head_values.items():
        if len(vals) != n:
            continue
        # Filter out under-supported values
        value_counts = Counter(vals)
        supported_vals = [v for v, c in value_counts.items() if c >= min_support]
        filtered = [v if v in supported_vals else "__other__" for v in vals]

        mi = _mutual_information(filtered, routing_labels)
        # Normalized MI: MI / min(H(X), H(Y))
        h_field = _entropy(Counter(filtered))
        nmi = mi / min(h_field, routing_entropy) if min(h_field, routing_entropy) > 0 else 0.0

        results[field] = {
            "mi": mi,
            "nmi": nmi,
            "entropy": h_field,
            "support": len(set(filtered)),
            "n_samples": n,
        }

    # Redundancy: I(field; routing | other_routing_field)
    # For each routing head, compute how much MI is *unique* vs shared
    routing_head_list = sorted(ROUTING_FIELDS)
    for field in routing_head_list:
        if field not in head_values:
            continue
        vals = head_values[field]
        if len(vals) != n:
            continue
        # I(field; routing)
        mi_total = results[field]["mi"]
        # For each other routing field, compute I(field; routing | other)
        cond_mi = []
        for other in routing_head_list:
            if other == field or other not in head_values:
                continue
            other_vals = head_values[other]
            if len(other_vals) != n:
                continue
            # Approximate conditional MI by grouping
            groups = {}
            for i in range(n):
                key = other_vals[i]
                groups.setdefault(key, []).append(i)
            cmi_sum = 0.0
            for group in groups.values():
                if len(group) < min_support:
                    continue
                sub_field = [vals[i] for i in group]
                sub_routing = [routing_labels[i] for i in group]
                cmi_sum += (len(group) / n) * _mutual_information(sub_field, sub_routing)
            cond_mi.append({
                "other": other,
                "conditional_mi": cmi_sum,
                "redundancy": mi_total - cmi_sum,
            })
        results[field]["conditional_mi_with_others"] = cond_mi
        if cond_mi:
            avg_redundancy = sum(c["redundancy"] for c in cond_mi) / len(cond_mi)
            results[field]["avg_redundancy"] = avg_redundancy

    # Rank heads
    ranked = sorted(results.items(), key=lambda x: x[1]["nmi"], reverse=True)

    # Categorize
    def _category(field: str) -> str:
        if field in ROUTING_FIELDS:
            return "routing"
        if field in AFFECT_FIELDS:
            return "affect"
        if field in RELATIONAL_FIELDS:
            return "relational"
        if field in DESCRIPTIVE_FIELDS:
            return "descriptive"
        if field in META_FIELDS:
            return "meta"
        return "other"

    # Report
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "# Head Utility Report\n",
        "Mutual information between each head and the routing decision (slow-path vs fast-path).\n",
        "Higher NMI = more operationally necessary.\n\n",
        "> **Caveat:** MI/NMI is used for exploratory ranking only, not as the sole pruning criterion.\n",
        "> A head may have low individual MI but still be useful in combination with another head.\n",
        "> The real criterion is **ablation impact on routing and leakage**.\n\n",
        "| Rank | Head | Category | MI | NMI | Entropy | Support | Avg Redundancy | Verdict |\n",
        "|------|------|----------|----|-----|---------|---------|----------------|---------|\n",
    ]

    for rank, (field, m) in enumerate(ranked, 1):
        cat = _category(field)
        redundancy = m.get("avg_redundancy", None)
        red_str = f"{redundancy:.3f}" if redundancy is not None else "N/A"
        # Verdict
        if cat == "routing":
            verdict = "Keep (hard gate)"
        elif m["nmi"] > 0.15:
            verdict = "Keep (strong predictor)"
        elif m["nmi"] > 0.05:
            verdict = "Advisory"
        else:
            verdict = "Drop"
        report_lines.append(
            f"| {rank} | {field} | {cat} | {m['mi']:.3f} | {m['nmi']:.3f} | {m['entropy']:.3f} | {m['support']} | {red_str} | {verdict} |\n"
        )

    report_lines.append("\n## Redundancy Analysis (Routing Heads Only)\n")
    report_lines.append("Redundancy = MI(head; routing) − avg conditional MI given other routing heads.\n")
    report_lines.append("Low redundancy = unique contribution. High redundancy = overlapping signal.\n\n")

    for field in routing_head_list:
        if field not in results:
            continue
        cond = results[field].get("conditional_mi_with_others", [])
        if not cond:
            continue
        report_lines.append(f"### {field}\n")
        for c in cond:
            report_lines.append(f"  vs {c['other']}: conditional MI={c['conditional_mi']:.3f}, redundancy={c['redundancy']:.3f}\n")
        report_lines.append("\n")

    report_path = out_dir / "head_utility_report.md"
    with open(report_path, "w") as f:
        f.writelines(report_lines)

    with open(out_dir / "head_utility.json", "w") as f:
        json.dump({"routing_entropy": routing_entropy, "heads": results, "ranking": [f for f, _ in ranked]}, f, indent=2)

    print(f"Wrote report to {report_path}")
    print(f"Top 5 heads by NMI:")
    for field, m in ranked[:5]:
        print(f"  {field:30s} NMI={m['nmi']:.3f}  MI={m['mi']:.3f}")
    print()

    return {"ranking": ranked, "output_dir": str(out_dir)}


def main():
    args = parse_args()
    analyze_head_utility(args.heads_file, args.output_dir, args.min_support)


if __name__ == "__main__":
    main()
