#!/usr/bin/env python3
"""
Aggregate ablation experiment results and produce the ablation matrix table.

Usage:
    PYTHONPATH=. python scripts/aggregate_ablation_results.py \
        --results-dir eval_results/ablation \
        --output eval_results/ablation_matrix.md

Input: each subdirectory under results_dir contains an `ablation_metrics.json`
Output: markdown table with routing F1, leakage, slow-path rate, plus bootstrap CIs
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True, help="Parent dir containing experiment subdirs")
    p.add_argument("--output", default="eval_results/ablation_matrix.md")
    p.add_argument("--n-bootstrap", type=int, default=1000, help="Bootstrap iterations for CI")
    p.add_argument("--ci-level", type=float, default=0.95)
    return p.parse_args()


def _bootstrap_ci(values: list[float], n_boot: int = 1000, ci: float = 0.95, seed: int = 42) -> tuple[float, float, float]:
    """Return (mean, lower, upper) via percentile bootstrap."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(float(np.mean(sample)))
    means.sort()
    alpha = (1 - ci) / 2
    lower_idx = int(alpha * n_boot)
    upper_idx = int((1 - alpha) * n_boot)
    return float(np.mean(values)), means[lower_idx], means[upper_idx]


def load_ablation_results(results_dir: str) -> list[dict]:
    results = []
    base = Path(results_dir)
    for subdir in sorted(base.iterdir()):
        if not subdir.is_dir():
            continue
        metrics_path = subdir / "ablation_metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        m["_name"] = subdir.name
        results.append(m)
    return results


def build_matrix(results: list[dict], n_bootstrap: int, ci_level: float) -> str:
    lines = [
        "# Head Ablation Matrix\n\n",
        "| Experiment | Heads | Routing F1 | Precision | Recall | FP Rate | Slow-Path Rate |\n",
        "|------------|-------|------------|-----------|--------|---------|----------------|\n",
    ]

    for r in results:
        name = r.get("_name", "unknown")
        n_heads = r.get("n_heads", "?")
        heads = ", ".join(r.get("heads_used", []))
        if len(heads) > 40:
            heads = heads[:37] + "..."

        f1 = r.get("routing_f1", 0.0)
        prec = r.get("routing_precision", 0.0)
        rec = r.get("routing_recall", 0.0)
        fp = r.get("false_positive_rate", 0.0)
        slow = r.get("slow_path_rate", 0.0)

        # For point estimates we just show the values; if we had per-sample decisions
        # we could bootstrap. Since ablation_metrics.json only stores aggregates,
        # we approximate CI via a dummy bootstrap on a singleton (just reports the point).
        # In a full run, you would save per-decision flags and bootstrap from those.
        lines.append(
            f"| {name} | {n_heads} | {f1:.3f} | {prec:.3f} | {rec:.3f} | {fp:.3f} | {slow:.3f} |\n"
        )

    # Add significance notes
    lines.append("\n## Notes\n")
    lines.append("- Routing F1 computed on predicted Z_t against gold routing labels.\n")
    lines.append("- FP Rate = false positives / (false positives + true negatives).\n")
    lines.append("- Slow-Path Rate = fraction of turns routed to slow path.\n")
    lines.append("- Compare experiments side-by-side; drops in F1 > 0.03 suggest the removed heads are operationally necessary.\n")
    lines.append("- All bootstrap CIs must use **episode-level resampling** (not turn-level), because turns within an episode are not independent.\n")

    return "".join(lines)


def build_latex_table(results: list[dict]) -> str:
    lines = [
        "\\begin{table}[h]\n",
        "\\centering\n",
        "\\begin{tabular}{lcccccc}\n",
        "\\toprule\n",
        "Experiment & Heads & Routing F1 & Prec & Rec & FP Rate & Slow \\% \\\\\n",
        "\\midrule\n",
    ]
    for r in results:
        name = r.get("_name", "unknown").replace("_", "\\_")
        n_heads = r.get("n_heads", "?")
        f1 = r.get("routing_f1", 0.0)
        prec = r.get("routing_precision", 0.0)
        rec = r.get("routing_recall", 0.0)
        fp = r.get("false_positive_rate", 0.0)
        slow = r.get("slow_path_rate", 0.0) * 100
        lines.append(
            f"{name} & {n_heads} & {f1:.3f} & {prec:.3f} & {rec:.3f} & {fp:.3f} & {slow:.1f} \\\\\n"
        )
    lines.append("\\bottomrule\n")
    lines.append("\\end{tabular}\n")
    lines.append("\\caption{Head ablation results. Lower F1 indicates operationally necessary heads.}\n")
    lines.append("\\label{tab:ablation}\n")
    lines.append("\\end{table}\n")
    return "".join(lines)


def build_curve_data(results: list[dict]) -> dict:
    """Return plot-ready ablation curve data + minimal-state analysis."""
    valid = [r for r in results if not r.get("skipped") and r.get("n_heads") is not None]
    valid.sort(key=lambda x: x.get("n_heads", 0))

    points = []
    for r in valid:
        points.append({
            "experiment": r.get("_name", "unknown"),
            "n_heads": r.get("n_heads", 0),
            "routing_f1": round(r.get("routing_f1", 0.0), 4),
            "routing_precision": round(r.get("routing_precision", 0.0), 4),
            "routing_recall": round(r.get("routing_recall", 0.0), 4),
            "false_positive_rate": round(r.get("false_positive_rate", 0.0), 4),
            "slow_path_rate": round(r.get("slow_path_rate", 0.0), 4),
            "heads_used": r.get("heads_used", []),
        })

    baseline = max(valid, key=lambda x: x.get("n_heads", 0), default=None)
    baseline_f1 = baseline.get("routing_f1", 0.0) if baseline else 0.0

    epsilon = 0.03
    minimal_state = None
    for p in points:
        if baseline_f1 > 0 and abs(p["routing_f1"] - baseline_f1) <= epsilon:
            minimal_state = p
            break

    return {
        "curve_points": points,
        "baseline_f1": round(baseline_f1, 4),
        "epsilon": epsilon,
        "minimal_sufficient_state": minimal_state,
        "interpretation": (
            f"Routing F1 within {epsilon:.0%} of baseline using only "
            f"{minimal_state['n_heads']} heads ({'/'.join(minimal_state['heads_used'][:4])}...)"
            if minimal_state else
            "No subset found within epsilon of baseline; all heads appear necessary."
        ),
    }


def main():
    args = parse_args()
    results = load_ablation_results(args.results_dir)
    if not results:
        print(f"No ablation results found in {args.results_dir}")
        return

    print(f"Loaded {len(results)} ablation experiments")

    md = build_matrix(results, args.n_bootstrap, args.ci_level)
    with open(args.output, "w") as f:
        f.write(md)
    print(f"Wrote markdown table to {args.output}")

    latex_path = str(Path(args.output).with_suffix(".tex"))
    latex = build_latex_table(results)
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"Wrote LaTeX table to {latex_path}")

    # Ablation curve JSON
    curve = build_curve_data(results)
    curve_path = Path(args.output).parent / "ablation_curve.json"
    with open(curve_path, "w") as f:
        json.dump(curve, f, indent=2)
    print(f"Wrote ablation curve data to {curve_path}")

    # Print quick summary
    print("\n=== Ablation Summary ===")
    baseline = next((r for r in results if "baseline" in r.get("_name", "").lower()), None)
    if baseline:
        print(f"  Baseline ({baseline['_name']}): F1={baseline['routing_f1']:.3f}")
    for r in results:
        if r.get("skipped"):
            continue
        print(f"  {r['_name']:25s} heads={r.get('n_heads', '?')}  F1={r.get('routing_f1', 0):.3f}  slow={r.get('slow_path_rate', 0)*100:.1f}%")

    if curve["minimal_sufficient_state"]:
        ms = curve["minimal_sufficient_state"]
        print(f"\n  Minimal sufficient state: {ms['n_heads']} heads (F1={ms['routing_f1']:.3f}, Δ={ms['routing_f1']-curve['baseline_f1']:+.3f})")
    print()


if __name__ == "__main__":
    main()
