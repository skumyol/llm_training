#!/usr/bin/env python3
"""
Generate v2 paper figures and tables:
1. Summary comparison table (oracle, predicted, decision-card)
2. Per-head contribution to unsafe FP and routing F1
3. Episode-level metrics (first-leak turn, secret persistence)
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_json(path):
    with open(path) as f:
        return json.load(f)


def build_summary_table():
    """Create LaTeX summary table comparing all systems."""
    
    # Data from existing tables
    systems = {
        "Oracle (gold 29D)": {
            "routing_f1": 1.000,
            "unsafe_fp": 0.000,
            "slow_path_rate": 0.548,
            "cost_fn5": 0.000,
            "policy_consistency": "---",
            "over_disclosure": "---",
            "rouge_l": "---",
        },
        "Gold-head oracle (4 heads)": {
            "routing_f1": 0.893,
            "unsafe_fp": 0.061,
            "slow_path_rate": 0.529,
            "cost_fn5": 0.183,
            "policy_consistency": "---",
            "over_disclosure": "---",
            "rouge_l": "---",
        },
        "Predicted 29D (current)": {
            "routing_f1": 0.672,
            "unsafe_fp": 0.174,
            "slow_path_rate": 0.543,
            "cost_fn5": 1.051,
            "policy_consistency": 0.735,
            "over_disclosure": 0.0029,
            "rouge_l": 0.126,
        },
        "4-head retrained (M1)": {
            "routing_f1": 0.697,
            "unsafe_fp": 0.003,
            "slow_path_rate": 0.997,
            "cost_fn5": 0.477,
            "policy_consistency": "---",
            "over_disclosure": "---",
            "rouge_l": "---",
        },
        "Decision card (compressed)": {
            "routing_f1": "---",
            "unsafe_fp": "---",
            "slow_path_rate": "---",
            "cost_fn5": "---",
            "policy_consistency": 0.850,
            "over_disclosure": 0.0015,
            "rouge_l": 0.080,
        },
    }
    
    lines = [
        "\\begin{table*}[t]\n",
        "\\centering\n",
        "\\caption{Summary comparison across all systems ($n{=}683$ val turns). Routing metrics (F1, unsafe FP, cost) apply to predictor+router pipeline; generation metrics (policy consistency, over-disclosure, ROUGE-L) apply to response quality.}\n",
        "\\label{tab:summary_comparison}\n",
        "\\small\n",
        "\\begin{tabular}{@{}lccccccc@{}}\n",
        "\\toprule\n",
        "\\textbf{System} & \\textbf{Routing F1} & \\textbf{Unsafe FP} & \\textbf{Slow-path} & \\textbf{Cost (5:1)} & \\textbf{Policy Cons.} & \\textbf{Over-disl.} & \\textbf{ROUGE-L} \\\\\n",
        "\\midrule\n",
    ]
    
    for name, metrics in systems.items():
        vals = []
        for key in ["routing_f1", "unsafe_fp", "slow_path_rate", "cost_fn5", 
                    "policy_consistency", "over_disclosure", "rouge_l"]:
            v = metrics.get(key, "---")
            if isinstance(v, float):
                if key == "cost_fn5":
                    vals.append(f"{v:.3f}")
                elif key in ["policy_consistency", "rouge_l"]:
                    vals.append(f"{v:.3f}")
                else:
                    vals.append(f"{v:.3f}")
            else:
                vals.append(str(v))
        
        lines.append(f"{name} & {' & '.join(vals)} \\\\\n")
    
    lines.extend([
        "\\bottomrule\n",
        "\\end{tabular}\n",
        "\\end{table*}\n",
    ])
    
    output_path = "paper/OCEAN_MBTI_ACL_2026/figures/summary_comparison.tex"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("".join(lines))
    
    print(f"Saved: {output_path}")


def build_perhead_contribution_figure():
    """Figure showing per-head contribution to unsafe FP and routing F1 delta."""
    
    data = load_json("eval_results/masking_ablations.json")
    baseline = data["baseline"]
    ablations = data["ablations"]
    
    # Filter gold-mode ablations only
    gold_ablations = [a for a in ablations if a["mode"] == "gold"]
    
    # Compute deltas
    heads = []
    delta_f1 = []
    delta_unsafe_fp = []
    
    for a in gold_ablations:
        head = a["ablation"]
        d_f1 = a["routing_f1"] - baseline["routing_f1"]
        d_ufp = baseline["unsafe_fast_path_rate"] - a["unsafe_fast_path_rate"]  # reduction
        
        heads.append(head)
        delta_f1.append(d_f1)
        delta_unsafe_fp.append(d_ufp)
    
    # Sort by delta F1 descending
    sorted_idx = np.argsort(delta_f1)[::-1]
    heads = [heads[i] for i in sorted_idx]
    delta_f1 = [delta_f1[i] for i in sorted_idx]
    delta_unsafe_fp = [delta_unsafe_fp[i] for i in sorted_idx]
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    x = np.arange(len(heads))
    width = 0.35
    
    # Color by head group
    group_colors = {
        "secrecy_pressure": "#2ecc71", "reveal_decision": "#2ecc71",
        "response_policy": "#2ecc71", "value_conflict": "#2ecc71",
        "affection": "#e74c3c", "respect": "#e74c3c",
        "dominance": "#3498db", "familiarity": "#3498db", "trust": "#3498db",
    }
    colors_f1 = [group_colors.get(h, "#95a5a6") for h in heads]
    
    bars1 = ax1.bar(x - width/2, delta_f1, width, label="$\\Delta$ Routing F1", 
                    color=colors_f1, edgecolor="black", linewidth=0.5)
    
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, delta_unsafe_fp, width, label="$\\Delta$ Unsafe FP (reduction)", 
                    color=colors_f1, edgecolor="black", linewidth=0.5, alpha=0.6, hatch="//")
    
    # Add value labels
    for bar, val in zip(bars1, delta_f1):
        if abs(val) > 0.005:
            ax1.text(bar.get_x() + bar.get_width()/2, val + (0.005 if val > 0 else -0.015),
                    f"{val:+.3f}", ha="center", va="bottom" if val > 0 else "top", fontsize=8)
    
    # Reference line at 0
    ax1.axhline(0, color="black", linewidth=0.5)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", edgecolor="black", label="Decision heads"),
        Patch(facecolor="#e74c3c", edgecolor="black", label="Affect heads"),
        Patch(facecolor="#3498db", edgecolor="black", label="Relational heads"),
        Patch(facecolor="#95a5a6", edgecolor="black", label="Other heads"),
    ]
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=8)
    
    ax1.set_xlabel("Head", fontsize=10)
    ax1.set_ylabel("$\\Delta$ Routing F1 (vs baseline)", fontsize=10)
    ax2.set_ylabel("Unsafe FP Reduction (vs baseline)", fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels([h.replace("_", "\\_") for h in heads], rotation=45, ha="right", fontsize=8)
    ax1.set_title("Per-head Contribution to Routing F1 and Safety", fontsize=12, fontweight="bold")
    
    plt.tight_layout()
    output_path = "paper/OCEAN_MBTI_ACL_2026/figures/perhead_contribution.pdf"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    
    print(f"Saved: {output_path}")


def compute_episode_metrics():
    """Compute episode-level metrics from generation files."""
    
    baseline = load_json("eval_results/response_baseline/sample_generations.json")
    card = load_json("eval_results/response_card/sample_generations.json")
    
    def _organize_by_episode(samples):
        episodes = defaultdict(list)
        for s in samples:
            episodes[s["episode_id"]].append(s)
        # Sort by turn_idx within each episode
        for ep_id in episodes:
            episodes[ep_id].sort(key=lambda x: x.get("turn_idx", 0))
        return episodes
    
    baseline_ep = _organize_by_episode(baseline)
    card_ep = _organize_by_episode(card)
    
    # Metrics
    def _compute_episode_stats(episodes):
        first_leak_turns = []
        secret_persistence = []  # num turns with reveal=none before first leak
        episode_lengths = []
        
        for ep_id, turns in episodes.items():
            episode_lengths.append(len(turns))
            
            # Find first leak
            leak_turn = None
            for i, t in enumerate(turns):
                reveal = t.get("reveal_decision", "none")
                if reveal in ("hint", "partial", "full"):
                    leak_turn = i + 1
                    break
            
            if leak_turn:
                first_leak_turns.append(leak_turn)
                secret_persistence.append(leak_turn - 1)
            else:
                # Never leaked - persistence = episode length
                secret_persistence.append(len(turns))
        
        return {
            "first_leak_turns": first_leak_turns,
            "secret_persistence": secret_persistence,
            "episode_lengths": episode_lengths,
            "n_episodes": len(episodes),
        }
    
    baseline_stats = _compute_episode_stats(baseline_ep)
    card_stats = _compute_episode_stats(card_ep)
    
    return baseline_stats, card_stats


def build_episode_level_figures():
    """Generate episode-level plots."""
    
    baseline_stats, card_stats = compute_episode_metrics()
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # 1. First-leak turn distribution
    ax1 = axes[0]
    max_turn = max(
        max(baseline_stats["first_leak_turns"] or [0]),
        max(card_stats["first_leak_turns"] or [0])
    ) + 1
    
    bins = range(1, max_turn + 2)
    ax1.hist(baseline_stats["first_leak_turns"], bins=bins, alpha=0.6, label="Baseline", 
             color="#3498db", edgecolor="black")
    ax1.hist(card_stats["first_leak_turns"], bins=bins, alpha=0.6, label="Decision card", 
             color="#e74c3c", edgecolor="black")
    ax1.set_xlabel("First leak turn", fontsize=10)
    ax1.set_ylabel("Number of episodes", fontsize=10)
    ax1.set_title("Distribution of First Leak Turn", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    
    # Add mean lines
    if baseline_stats["first_leak_turns"]:
        b_mean = np.mean(baseline_stats["first_leak_turns"])
        ax1.axvline(b_mean, color="#3498db", linestyle="--", linewidth=1.5, label=f"Baseline mean: {b_mean:.1f}")
    if card_stats["first_leak_turns"]:
        c_mean = np.mean(card_stats["first_leak_turns"])
        ax1.axvline(c_mean, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"Card mean: {c_mean:.1f}")
    
    # 2. Secret persistence (turns before first leak)
    ax2 = axes[1]
    max_persist = max(
        max(baseline_stats["secret_persistence"] or [0]),
        max(card_stats["secret_persistence"] or [0])
    ) + 1
    
    bins2 = range(0, max_persist + 2)
    ax2.hist(baseline_stats["secret_persistence"], bins=bins2, alpha=0.6, label="Baseline", 
             color="#3498db", edgecolor="black", density=True)
    ax2.hist(card_stats["secret_persistence"], bins=bins2, alpha=0.6, label="Decision card", 
             color="#e74c3c", edgecolor="black", density=True)
    ax2.set_xlabel("Secret persistence (turns)", fontsize=10)
    ax2.set_ylabel("Density", fontsize=10)
    ax2.set_title("Secret Persistence Before First Leak", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    
    # 3. Episode length distribution
    ax3 = axes[2]
    bins3 = range(1, max(max(baseline_stats["episode_lengths"]), max(card_stats["episode_lengths"])) + 2)
    ax3.hist(baseline_stats["episode_lengths"], bins=bins3, alpha=0.6, label="Baseline", 
             color="#3498db", edgecolor="black", density=True)
    ax3.hist(card_stats["episode_lengths"], bins=bins3, alpha=0.6, label="Decision card", 
             color="#e74c3c", edgecolor="black", density=True)
    ax3.set_xlabel("Episode length (turns)", fontsize=10)
    ax3.set_ylabel("Density", fontsize=10)
    ax3.set_title("Episode Length Distribution", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=9)
    
    plt.tight_layout()
    output_path = "paper/OCEAN_MBTI_ACL_2026/figures/episode_level_metrics.pdf"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    
    print(f"Saved: {output_path}")
    
    # Print summary stats
    print("\nEpisode-level summary:")
    for name, stats in [("Baseline", baseline_stats), ("Decision card", card_stats)]:
        print(f"\n{name}:")
        print(f"  Episodes: {stats['n_episodes']}")
        if stats["first_leak_turns"]:
            print(f"  Mean first-leak turn: {np.mean(stats['first_leak_turns']):.2f} (SD: {np.std(stats['first_leak_turns']):.2f})")
            print(f"  Mean secret persistence: {np.mean(stats['secret_persistence']):.2f}")
        print(f"  Mean episode length: {np.mean(stats['episode_lengths']):.2f}")
    
    # Save JSON
    output_json = {
        "baseline": {
            "n_episodes": baseline_stats["n_episodes"],
            "mean_first_leak_turn": float(np.mean(baseline_stats["first_leak_turns"])) if baseline_stats["first_leak_turns"] else None,
            "mean_secret_persistence": float(np.mean(baseline_stats["secret_persistence"])),
            "mean_episode_length": float(np.mean(baseline_stats["episode_lengths"])),
        },
        "decision_card": {
            "n_episodes": card_stats["n_episodes"],
            "mean_first_leak_turn": float(np.mean(card_stats["first_leak_turns"])) if card_stats["first_leak_turns"] else None,
            "mean_secret_persistence": float(np.mean(card_stats["secret_persistence"])),
            "mean_episode_length": float(np.mean(card_stats["episode_lengths"])),
        },
    }
    
    json_path = "eval_results/episode_level_metrics.json"
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"\nSaved: {json_path}")


def main():
    print("Building summary comparison table...")
    build_summary_table()
    
    print("\nBuilding per-head contribution figure...")
    build_perhead_contribution_figure()
    
    print("\nBuilding episode-level figures...")
    build_episode_level_figures()
    
    print("\n✅ All v2 figures and tables generated.")


if __name__ == "__main__":
    main()
