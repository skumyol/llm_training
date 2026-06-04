#!/usr/bin/env python3
"""
Plot trade-off curves from threshold sweep output.

Produces:
  - leakage_vs_slowpath.png  (leakage rate on y, slow-path rate on x)
  - f1_vs_slowpath.png       (routing F1 on y, slow-path rate on x)
  - f1_vs_threshold.png      (routing F1 on y, threshold on x)

Usage:
    PYTHONPATH=. python scripts/plot_tradeoff_curves.py \
        --input eval_results/threshold_sweep.json \
        --output-dir eval_results/figures

Requires: matplotlib
"""
import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="threshold_sweep.json")
    p.add_argument("--output-dir", default="eval_results/figures")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--fmt", default="png", choices=["png", "pdf", "svg"])
    return p.parse_args()


def try_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("ERROR: matplotlib not installed. Install with: pip install matplotlib")
        import sys
        sys.exit(1)


def plot_f1_vs_slowpath(results: list[dict], output_path: str, plt) -> None:
    uniform = [r for r in results if r.get("sweep_type") == "uniform"]
    if not uniform:
        print("No uniform sweep data found; skipping F1 vs slow-path plot")
        return

    uniform.sort(key=lambda x: x["slow_path_rate"])
    x = [r["slow_path_rate"] for r in uniform]
    y = [r["f1"] for r in uniform]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, marker="o", linewidth=1.5, markersize=4, color="#2E86AB")
    ax.fill_between(x, y, alpha=0.15, color="#2E86AB")
    ax.set_xlabel("Slow-Path Rate (fraction routed to slow path)", fontsize=10)
    ax.set_ylabel("Routing F1", fontsize=10)
    ax.set_title("Routing F1 vs Slow-Path Rate", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Annotate best F1 point
    best = max(uniform, key=lambda r: r["f1"])
    ax.annotate(
        f"Best F1={best['f1']:.3f}\n@ slow={best['slow_path_rate']:.2f}",
        xy=(best["slow_path_rate"], best["f1"]),
        xytext=(best["slow_path_rate"] + 0.05, best["f1"] - 0.1),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="#A23B72"),
        color="#A23B72",
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_f1_vs_threshold(results: list[dict], output_path: str, plt) -> None:
    uniform = [r for r in results if r.get("sweep_type") == "uniform"]
    if not uniform:
        return

    # Extract threshold value (they're all the same in uniform sweep)
    uniform.sort(key=lambda x: list(x["thresholds"].values())[0])
    x = [list(r["thresholds"].values())[0] for r in uniform]
    y_f1 = [r["f1"] for r in uniform]
    y_fp = [r["false_positive_rate"] for r in uniform]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    color1 = "#2E86AB"
    ax1.plot(x, y_f1, marker="o", color=color1, label="Routing F1", linewidth=1.5)
    ax1.set_xlabel("Confidence Threshold (uniform across heads)", fontsize=10)
    ax1.set_ylabel("Routing F1", color=color1, fontsize=10)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(0, 1)

    ax2 = ax1.twinx()
    color2 = "#A23B72"
    ax2.plot(x, y_fp, marker="s", color=color2, label="FP Rate", linewidth=1.5, linestyle="--")
    ax2.set_ylabel("False Positive Rate", color=color2, fontsize=10)
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.set_ylim(0, 1)

    ax1.set_xlim(0, 1)
    ax1.grid(True, alpha=0.3)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=9)

    ax1.set_title("Router Performance vs Confidence Threshold", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_per_head(results: list[dict], output_path: str, plt) -> None:
    per_head = {}
    for r in results:
        if r.get("sweep_type") != "per_head":
            continue
        head = r.get("active_head", "unknown")
        per_head.setdefault(head, []).append(r)

    if not per_head:
        print("No per-head sweep data found; skipping per-head plot")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = plt.cm.tab10(np.linspace(0, 1, len(per_head)))

    for (head, data), color in zip(sorted(per_head.items()), colors):
        data.sort(key=lambda x: list(x["thresholds"].values())[0])
        thresh_vals = [list(r["thresholds"].values())[0] for r in data]
        # Get the actual varied threshold for this head
        x = [r["thresholds"][head] for r in data]
        y = [r["f1"] for r in data]
        ax.plot(x, y, marker="o", label=head, linewidth=1.5, color=color, markersize=3)

    ax.set_xlabel(f"Head-Specific Confidence Threshold", fontsize=10)
    ax.set_ylabel("Routing F1", fontsize=10)
    ax.set_title("Per-Head Threshold Sensitivity", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    args = parse_args()
    plt = try_import_matplotlib()

    with open(args.input) as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        print("No results found in input file")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Plotting trade-off curves from {len(results)} sweep points...")
    plot_f1_vs_slowpath(results, str(out_dir / f"f1_vs_slowpath.{args.fmt}"), plt)
    plot_f1_vs_threshold(results, str(out_dir / f"f1_vs_threshold.{args.fmt}"), plt)
    plot_per_head(results, str(out_dir / f"per_head_threshold.{args.fmt}"), plt)

    print(f"\nAll figures saved to {out_dir}")


if __name__ == "__main__":
    main()
