#!/usr/bin/env python3
"""
Generate publication-ready tables and plots from evaluation results.

Usage:
    cd /home/skumyol/llm_training
    python llm_finetuning/scripts/generate_paper_figures.py \
        --output-dir paper/OCEAN_MBTI_ACL_2026/figures
"""
import argparse
import json
import csv
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="paper/OCEAN_MBTI_ACL_2026/figures")
    return p.parse_args()


# ── Group mapping ─────────────────────────────────────────────────────────────
GROUP_LABELS = {
    "A": "Affect",
    "C": "Context",
    "D": "Decision",
    "M": "Mental",
    "N": "Normative",
    "R": "Relational",
}

ROUTING_IMPACT = {
    "response_policy": "High",
    "reveal_decision": "High",
    "value_conflict": "Low",
    "secrecy_pressure": "Medium",
}


# ── Load data ─────────────────────────────────────────────────────────────────
def load_per_head_metrics(path: str):
    heads = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            heads.append(row)
    return heads


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


# ── Table 1: Head Utility ──────────────────────────────────────────────────
def build_head_utility_table(per_head_metrics, masking_ablations):
    """Build head utility table with κ, macro-F1, routing impact."""
    lines = [
        "\n## Table 1: Head Utility\n\n",
        "| Head Group | Head | Cohen's κ | Macro-F1 | Routing Impact | Operational? |\n",
        "|------------|------|----------:|---------:|---------------:|---------------|\n",
    ]

    # Get Δ routing F1 from masking ablations (gold mode)
    delta_f1 = {}
    baseline_f1 = masking_ablations.get("baseline", {}).get("routing_f1", 0.672)
    for entry in masking_ablations.get("ablations", []):
        if entry.get("mode") == "gold" and len(entry.get("mask_heads", [])) == 1:
            head = entry["mask_heads"][0]
            delta = entry["routing_f1"] - baseline_f1
            delta_f1[head] = delta

    for row in per_head_metrics:
        group = row.get("group", "?")
        group_label = GROUP_LABELS.get(group, group)
        head = row["head"]
        kappa = float(row.get("true_cohen_kappa", 0))
        macro_f1 = float(row.get("macro_f1", 0))
        delta = delta_f1.get(head, 0.0)

        # Determine operational status
        if delta > 0.05:
            operational = "Yes"
            impact = "High"
        elif delta > 0.02:
            operational = "Advisory"
            impact = "Medium"
        elif delta > 0.0:
            operational = "Optional"
            impact = "Low"
        else:
            operational = "Drop"
            impact = "None"

        # Override for known routing heads
        if head in ["response_policy", "reveal_decision"]:
            operational = "Yes"
            impact = "High"
        elif head in ["value_conflict", "secrecy_pressure"]:
            operational = "Yes"
            impact = "Medium"

        lines.append(
            f"| {group_label:10s} | {head:20s} | {kappa:.3f} | {macro_f1:.3f} |"
            f" {impact:6s} ({delta:+.3f}) | {operational:8s} |\n"
        )

    return "".join(lines)


# ── Table 2: Routing Performance ──────────────────────────────────────────────
def build_routing_table(ablation_dir):
    """Build routing performance table from retraining ablations."""
    systems = [
        ("M0: Full 29-head", "exp_d_full_29head"),
        ("M1: Routing only (4-head)", "exp_a_routing_only"),
        ("M2: +Affect (7-head)", "exp_b_plus_affect"),
        ("M3: +Relational (6-head)", "exp_c_plus_relational"),
    ]

    lines = [
        "\n## Table 2: Routing Performance (Retraining Ablations)\n\n",
        "| System | Routing F1 | Unsafe fast-path | Slow-path rate | Slow-path recall | Cost (5:1) |\n",
        "|--------|-----------:|-----------------:|---------------:|-----------------:|-----------:|\n",
    ]

    for label, subdir in systems:
        path = Path(ablation_dir) / subdir / "ablation_metrics.json"
        if path.exists():
            m = load_json(str(path))
            lines.append(
                f"| {label:28s} | {m['routing_f1']:.3f} |"
                f" {m['unsafe_fast_path_rate']:.3f} |"
                f" {m['slow_path_rate']:.3f} |"
                f" {m['slow_path_recall']:.3f} |"
                f" {m['routing_cost_fn5']:.3f} |\n"
            )
        else:
            lines.append(f"| {label:28s} | N/A | N/A | N/A | N/A | N/A |\n")

    return "".join(lines)


# ── Table 3: Disclosure Metrics ──────────────────────────────────────────────
def build_disclosure_table(baseline_path, card_path):
    """Build disclosure metrics table."""
    baseline = load_json(baseline_path) if Path(baseline_path).exists() else {}
    card = load_json(card_path) if Path(card_path).exists() else {}

    lines = [
        "\n## Table 3: Disclosure Metrics\n\n",
        "| System | Over-disclosure | Under-disclosure | Exact match | Policy consistency |\n",
        "|--------|---------------:|-----------------:|-----------:|-------------------:|\n",
    ]

    def _get(d, key, default="N/A"):
        v = d.get(key, default)
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    lines.append(
        f"| 29-head baseline | {_get(baseline, 'over_disclosure_rate')} |"
        f" {_get(baseline, 'under_disclosure_rate')} |"
        f" {_get(baseline, 'exact_disclosure_match', 'N/A')} |"
        f" {_get(baseline, 'policy_consistency', 'N/A')} |\n"
    )
    lines.append(
        f"| Decision card    | {_get(card, 'over_disclosure_rate')} |"
        f" {_get(card, 'under_disclosure_rate')} |"
        f" {_get(card, 'exact_disclosure_match', 'N/A')} |"
        f" {_get(card, 'policy_consistency', 'N/A')} |\n"
    )

    return "".join(lines)


# ── Plot 1: Head Utility Bar Chart ───────────────────────────────────────────
def plot_head_utility(per_head_metrics, delta_f1, output_dir):
    """Bar plot of Δ routing F1 per head, colored by group."""
    group_colors = {
        "A": "#e74c3c",  # Affect - red
        "C": "#3498db",  # Context - blue
        "D": "#2ecc71",  # Decision - green
        "M": "#9b59b6",  # Mental - purple
        "N": "#f39c12",  # Normative - orange
        "R": "#1abc9c",  # Relational - teal
    }

    # Sort by delta_f1 descending
    sorted_heads = sorted(
        per_head_metrics,
        key=lambda h: delta_f1.get(h["head"], 0),
        reverse=True,
    )

    names = [h["head"] for h in sorted_heads]
    deltas = [delta_f1.get(h["head"], 0) for h in sorted_heads]
    colors = [group_colors.get(h["group"], "#7f8c8d") for h in sorted_heads]

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(range(len(names)), deltas, color=colors, edgecolor="black", linewidth=0.5)

    # Add value labels on bars
    for bar, delta in zip(bars, deltas):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.003 if height >= 0 else height - 0.008,
            f"{delta:+.3f}",
            ha="center", va="bottom" if height >= 0 else "top",
            fontsize=7, rotation=0,
        )

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Δ Routing F1 (gold replacement)", fontsize=10)
    ax.set_title("Head Utility: Impact on Routing F1 When Replaced with Gold Labels", fontsize=11)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylim(min(deltas) - 0.05, max(deltas) + 0.05)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=GROUP_LABELS.get(g, g)) for g, c in group_colors.items()]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    plt.tight_layout()
    out = Path(output_dir) / "head_utility_delta_f1.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Plot 2: Routing Cost Curve (from threshold sweep) ────────────────────────
def plot_routing_cost_curve(threshold_sweep_path, output_dir):
    """Plot unsafe fast-path vs slow-path rate across thresholds."""
    if not Path(threshold_sweep_path).exists():
        print(f"[WARN] Threshold sweep file not found: {threshold_sweep_path}")
        return

    data = load_json(threshold_sweep_path)
    results = data.get("results", data)

    # Extract metrics
    thresholds = []
    unsafe_fp = []
    slow_rate = []
    f1 = []

    for entry in results:
        if isinstance(entry, dict):
            thresholds.append(entry.get("threshold", 0))
            unsafe_fp.append(entry.get("unsafe_fast_path_rate", 0))
            slow_rate.append(entry.get("slow_path_rate", 0))
            f1.append(entry.get("routing_f1", 0))

    if not thresholds:
        print("[WARN] No threshold data found")
        return

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Primary: unsafe fast-path and slow-path rate
    ax1.plot(thresholds, unsafe_fp, "r-o", label="Unsafe fast-path rate", markersize=4)
    ax1.plot(thresholds, slow_rate, "b-s", label="Slow-path rate", markersize=4)
    ax1.set_xlabel("Confidence threshold", fontsize=10)
    ax1.set_ylabel("Rate", fontsize=10)
    ax1.set_ylim(0, 1.05)

    # Secondary: F1
    ax2 = ax1.twinx()
    ax2.plot(thresholds, f1, "g--^", label="Routing F1", markersize=4)
    ax2.set_ylabel("Routing F1", fontsize=10)
    ax2.set_ylim(0, 1.05)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)

    ax1.set_title("Selective Router: Threshold Sweep", fontsize=11)
    plt.tight_layout()
    out = Path(output_dir) / "routing_threshold_sweep.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Plot 3: Disclosure Stacked Bar ───────────────────────────────────────────
def plot_disclosure_stacked(baseline_path, card_path, output_dir):
    """Stacked bar chart: over/under/exact disclosure."""
    baseline = load_json(baseline_path) if Path(baseline_path).exists() else {}
    card = load_json(card_path) if Path(card_path).exists() else {}

    def _get_rates(d):
        return [
            d.get("over_disclosure_rate", 0),
            d.get("exact_disclosure_match", d.get("exact_disclosure_rate", 0)),
            d.get("under_disclosure_rate", 0),
        ]

    rates_baseline = _get_rates(baseline)
    rates_card = _get_rates(card)

    labels = ["Over-disclosure\n(safety failure)", "Exact match\n(success)", "Under-disclosure\n(interaction failure)"]
    x = [0, 1]
    systems = ["29-head baseline", "Decision card"]
    rates = [rates_baseline, rates_card]

    colors = ["#e74c3c", "#2ecc71", "#f39c12"]  # red, green, orange

    fig, ax = plt.subplots(figsize=(6, 5))
    bottom_baseline = 0
    bottom_card = 0

    for i, (label, color) in enumerate(zip(labels, colors)):
        vals = [rates_baseline[i], rates_card[i]]
        ax.bar(x[0], rates_baseline[i], bottom=bottom_baseline, color=color, edgecolor="white", width=0.5, label=label if i == 0 or i == 1 or i == 2 else "")
        ax.bar(x[1], rates_card[i], bottom=bottom_card, color=color, edgecolor="white", width=0.5)
        bottom_baseline += rates_baseline[i]
        bottom_card += rates_card[i]

    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=10)
    ax.set_ylabel("Proportion", fontsize=10)
    ax.set_title("Disclosure Policy Accuracy", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=9)

    # Add text annotations
    for i, (rb, rc) in enumerate(zip(rates_baseline, rates_card)):
        ax.text(0, sum(rates_baseline[:i]) + rb/2, f"{rb:.1%}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        ax.text(1, sum(rates_card[:i]) + rc/2, f"{rc:.1%}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    plt.tight_layout()
    out = Path(output_dir) / "disclosure_stacked_bar.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Plot 4: Retraining Ablation Comparison ───────────────────────────────────
def plot_retraining_ablation(ablation_dir, output_dir):
    """Bar plot comparing retraining ablations."""
    systems = [
        ("M0: Full 29-head", "exp_d_full_29head"),
        ("M1: Routing only", "exp_a_routing_only"),
        ("M2: +Affect", "exp_b_plus_affect"),
        ("M3: +Relational", "exp_c_plus_relational"),
    ]

    names = []
    f1_vals = []
    unsafe_vals = []
    cost_vals = []

    for label, subdir in systems:
        path = Path(ablation_dir) / subdir / "ablation_metrics.json"
        if path.exists():
            m = load_json(str(path))
            names.append(label)
            f1_vals.append(m["routing_f1"])
            unsafe_vals.append(m["unsafe_fast_path_rate"])
            cost_vals.append(m["routing_cost_fn5"])

    if not names:
        print("[WARN] No ablation data found")
        return

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width, f1_vals, width, label="Routing F1", color="#3498db")
    bars2 = ax.bar(x, unsafe_vals, width, label="Unsafe fast-path", color="#e74c3c")
    bars3 = ax.bar(x + width, cost_vals, width, label="Cost (5:1)", color="#f39c12")

    ax.set_ylabel("Score / Rate", fontsize=10)
    ax.set_title("Retraining Ablations: Compressed vs Full State", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", fontsize=9)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, max(max(f1_vals), max(cost_vals)) + 0.1)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01, f"{height:.3f}",
                    ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    out = Path(output_dir) / "retraining_ablation_comparison.pdf"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    per_head = load_per_head_metrics("eval_results/per_head_metrics.csv")
    masking = load_json("eval_results/masking_ablations.json")

    # Build delta_f1 map
    baseline_f1 = masking.get("baseline", {}).get("routing_f1", 0.672)
    delta_f1 = {}
    for entry in masking.get("ablations", []):
        if entry.get("mode") == "gold" and len(entry.get("mask_heads", [])) == 1:
            head = entry["mask_heads"][0]
            delta_f1[head] = entry["routing_f1"] - baseline_f1

    # Generate tables (print to console + save to markdown)
    tables_md = ""
    tables_md += build_head_utility_table(per_head, masking)
    tables_md += build_routing_table("eval_results/ablation")
    tables_md += build_disclosure_table(
        "eval_results/disclosure_eval_baseline.json",
        "eval_results/disclosure_eval_card.json",
    )

    tables_path = out_dir / "generated_tables.md"
    with open(tables_path, "w") as f:
        f.write(tables_md)
    print(f"\nSaved tables: {tables_path}")

    # Generate plots
    plot_head_utility(per_head, delta_f1, out_dir)
    plot_routing_cost_curve("eval_results/threshold_sweep.json", out_dir)
    plot_disclosure_stacked(
        "eval_results/disclosure_eval_baseline.json",
        "eval_results/disclosure_eval_card.json",
        out_dir,
    )
    plot_retraining_ablation("eval_results/ablation", out_dir)

    print("\n✅ All figures and tables generated successfully.")


if __name__ == "__main__":
    main()
