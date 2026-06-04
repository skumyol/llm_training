#!/usr/bin/env python3
"""
Build ablation comparison table from masking_ablations.json.

Produces a LaTeX-ready markdown table for the paper.
"""
import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="eval_results/masking_ablations.json")
    p.add_argument("--output", default="eval_results/ablation_table.md")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.input) as f:
        data = json.load(f)

    baseline = data["baseline"]
    ablations = data["ablations"]

    # Build rows: for each head, show majority/random/gold
    lines = [
        "# Head Masking Ablation Results\n\n",
        "Baseline (no mask): "
        f"F1={baseline['routing_f1']:.3f}, "
        f"unsafe_FP={baseline['unsafe_fast_path_rate']:.3f}, "
        f"slow={baseline['slow_path_rate']:.3f}\n\n",
        "## Full Table\n\n",
        "| Masked head(s) | Mode | Routing F1 | Unsafe fast-path | Slow-path rate | FPR | FNR | Cost(5:1) |\n",
        "|----------------|------|-----------:|-----------------:|---------------:|----:|----:|----------:|\n",
    ]

    for entry in ablations:
        heads = ", ".join(entry["mask_heads"])
        lines.append(
            f"| {heads:20s} | {entry['mode']:8s} |"
            f" {entry['routing_f1']:.3f} |"
            f" {entry['unsafe_fast_path_rate']:.3f} |"
            f" {entry['slow_path_rate']:.3f} |"
            f" {entry['false_positive_rate']:.3f} |"
            f" {entry['false_negative_rate']:.3f} |"
            f" {entry['routing_cost_fn5']:.2f} |\n"
        )

    # Key findings
    lines.append("\n## Key Findings\n\n")
    for entry in ablations:
        if entry["mode"] == "gold" and entry["ablation"] != "all_routing":
            delta_f1 = entry["routing_f1"] - baseline["routing_f1"]
            delta_unsafe = baseline["unsafe_fast_path_rate"] - entry["unsafe_fast_path_rate"]
            heads = ", ".join(entry["mask_heads"])
            lines.append(
                f"- Gold **{heads}**: F1 {'+' if delta_f1>=0 else ''}{delta_f1:.3f}, "
                f"unsafe_FP {'-' if delta_unsafe>=0 else ''}{abs(delta_unsafe):.3f}\n"
            )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.writelines(lines)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
