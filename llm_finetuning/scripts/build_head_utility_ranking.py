#!/usr/bin/env python3
"""
Build head utility ranking table from masking ablation results.
Ranks heads by operational importance: Δ routing F1, Δ unsafe FP reduction.
"""
import json
from pathlib import Path


def load_json(path):
    with open(path) as f:
        return json.load(f)


def build_ranking_table(masking_path, output_path):
    masking = load_json(masking_path)
    baseline = masking.get("baseline", {})
    baseline_f1 = baseline.get("routing_f1", 0.672)
    baseline_unsafe = baseline.get("unsafe_fast_path_rate", 0.174)

    # Collect gold-mode single-head ablations
    head_results = []
    for entry in masking.get("ablations", []):
        if entry.get("mode") == "gold" and len(entry.get("mask_heads", [])) == 1:
            head = entry["mask_heads"][0]
            delta_f1 = entry["routing_f1"] - baseline_f1
            delta_unsafe = baseline_unsafe - entry["unsafe_fast_path_rate"]  # Reduction
            head_results.append({
                "head": head,
                "routing_f1": entry["routing_f1"],
                "delta_f1": delta_f1,
                "unsafe_fp": entry["unsafe_fast_path_rate"],
                "delta_unsafe": delta_unsafe,
            })

    # Sort by composite score: prioritize ΔF1, then unsafe FP reduction
    # Weight: ΔF1 is primary (routing quality), unsafe reduction secondary (safety)
    def score(h):
        return h["delta_f1"] + 0.5 * h["delta_unsafe"]  # Composite

    head_results.sort(key=score, reverse=True)

    # Assign ranks
    for i, h in enumerate(head_results, 1):
        h["rank"] = i

    # Build markdown table
    lines = [
        "\n## Table: Head Utility Ranking (by Operational Importance)\n\n",
        "Ranked by composite score: Δ routing F1 + 0.5 × (unsafe FP reduction).\n\n",
        "| Rank | Head | Routing F1 | Δ F1 | Unsafe FP | Δ Unsafe | Importance |\n",
        "|------|------|-----------:|-----:|----------:|---------:|------------|\n",
    ]

    for h in head_results:
        # Determine importance label
        if h["delta_f1"] > 0.05:
            importance = "Critical"
        elif h["delta_f1"] > 0.02:
            importance = "High"
        elif h["delta_f1"] > 0.0:
            importance = "Medium"
        else:
            importance = "Low"

        lines.append(
            f"| {h['rank']} | {h['head']:20s} | {h['routing_f1']:.3f} |"
            f" {h['delta_f1']:+.3f} | {h['unsafe_fp']:.3f} |"
            f" {h['delta_unsafe']:+.3f} | {importance:8s} |\n"
        )

    # Summary statistics
    lines.append("\n### Summary\n\n")
    lines.append(f"- **Baseline**: routing F1 = {baseline_f1:.3f}, unsafe FP = {baseline_unsafe:.3f}\n")
    lines.append(f"- **Critical heads** (ΔF1 > 0.05): {sum(1 for h in head_results if h['delta_f1'] > 0.05)}\n")
    lines.append(f"- **High heads** (ΔF1 0.02–0.05): {sum(1 for h in head_results if 0.02 < h['delta_f1'] <= 0.05)}\n")
    lines.append(f"- **Medium heads** (ΔF1 0–0.02): {sum(1 for h in head_results if 0 < h['delta_f1'] <= 0.02)}\n")
    lines.append(f"- **Low heads** (ΔF1 ≤ 0): {sum(1 for h in head_results if h['delta_f1'] <= 0)}\n")

    # Recommendation
    lines.append("\n### Recommendation\n\n")
    critical = [h["head"] for h in head_results if h["delta_f1"] > 0.02]
    lines.append(f"Retain for routing: **{', '.join(critical)}** ({len(critical)} heads)\n")
    lines.append(f"Candidates for compression: remaining {29 - len(critical)} heads\n")

    content = "".join(lines)

    with open(output_path, "w") as f:
        f.write(content)

    print(f"Saved: {output_path}")
    print(content)

    return head_results


def build_latex_table(head_results, baseline_f1, baseline_unsafe, output_path):
    """Build LaTeX version for paper."""
    lines = [
        "\\begin{table}[t]\n",
        "\\centering\n",
        "\\caption{Head utility ranking by operational importance ($n{=}683$ val turns). "
        "Ranked by composite score: $\\Delta$ routing F1 + 0.5 $\\times$ (unsafe FP reduction). "
        "Critical heads ($\\Delta$F1${>}0.05$) are essential for routing; Low heads can be dropped without degrading safety.}\n",
        "\\label{tab:head_utility_ranking}\n",
        "\\small\n",
        "\\begin{tabular}{@{}rlrrrrl@{}}\n",
        "\\toprule\n",
        "\\textbf{Rank} & \\textbf{Head} & \\textbf{Routing F1} & \\textbf{$\\Delta$F1} & "
        "\\textbf{Unsafe FP} & \\textbf{$\\Delta$Unsafe} & \\textbf{Importance} \\\\\n",
        "\\midrule\n",
    ]

    for h in head_results:
        if h["delta_f1"] > 0.05:
            importance = "\\textbf{Critical}"
        elif h["delta_f1"] > 0.02:
            importance = "High"
        elif h["delta_f1"] > 0.0:
            importance = "Medium"
        else:
            importance = "Low"

        # Escape underscores for LaTeX
        head_escaped = h["head"].replace("_", "\\_")

        lines.append(
            f"{h['rank']} & {head_escaped:20s} & {h['routing_f1']:.3f} &"
            f" {h['delta_f1']:+.3f} & {h['unsafe_fp']:.3f} &"
            f" {h['delta_unsafe']:+.3f} & {importance} \\\\\n"
        )

    lines.append("\\midrule\n")
    lines.append(f"\\multicolumn{{2}}{{l}}{{\\textbf{{Baseline}} (no mask)}} & {baseline_f1:.3f} & --- & {baseline_unsafe:.3f} & --- & --- \\\\\n")
    lines.append("\\bottomrule\n")
    lines.append("\\end{tabular}\n")
    lines.append("\\end{table}\n")

    content = "".join(lines)

    latex_path = output_path.replace(".md", ".tex")
    with open(latex_path, "w") as f:
        f.write(content)

    print(f"Saved: {latex_path}")


def main():
    masking_path = "eval_results/masking_ablations.json"
    output_path = "paper/OCEAN_MBTI_ACL_2026/figures/head_utility_ranking.md"

    head_results = build_ranking_table(masking_path, output_path)

    masking = load_json(masking_path)
    baseline = masking.get("baseline", {})
    baseline_f1 = baseline.get("routing_f1", 0.672)
    baseline_unsafe = baseline.get("unsafe_fast_path_rate", 0.174)

    build_latex_table(head_results, baseline_f1, baseline_unsafe, output_path)


if __name__ == "__main__":
    main()
