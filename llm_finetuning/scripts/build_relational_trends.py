#!/usr/bin/env python3
"""
Build per-turn relational delta trend figure from predicted Z_t data.
Shows how relational heads evolve across turns within episodes.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from pathlib import Path


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def build_relational_trends():
    # Load predicted Z_t (validation set)
    pred_records = load_jsonl("eval_results/predicted_zt.jsonl")
    
    # Organize by episode
    episodes = defaultdict(list)
    for r in pred_records:
        ep_id = r.get("episode_id")
        turn_idx = r.get("turn_idx", 0)
        if ep_id:
            episodes[ep_id].append((turn_idx, r))
    
    # Sort by turn_idx within each episode
    for ep_id in episodes:
        episodes[ep_id].sort(key=lambda x: x[0])
    
    # Extract relational heads per turn
    relational_heads = ["affection_level", "respect_level", "dominance_level", 
                       "familiarity_level", "trust_level", "obligation_level"]
    
    # Convert categorical levels to numeric for plotting
    level_map = {
        "VL": 1, "L": 2, "N": 3, "H": 4, "VH": 5,
        "VL-": 0.5, "VL+": 1.5, "L+": 2.5, "N+": 3.5, "H+": 4.5, "VH+": 5.5,
        "+": 3.5, "-": 2.5, "++": 4.5, "--": 1.5,
    }
    
    # Collect per-turn averages across episodes
    max_turns = max(len(turns) for turns in episodes.values())
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    for idx, head in enumerate(relational_heads):
        ax = axes[idx]
        
        # For each episode, plot trajectory
        plotted = 0
        for ep_id, turns in episodes.items():
            if len(turns) < 3:  # Skip very short episodes
                continue
            
            x_vals = []
            y_vals = []
            for turn_idx, record in turns:
                val = record.get(head, "N")
                if val in level_map:
                    y = level_map[val]
                else:
                    # Try to extract base level from variants like "H(++)" or "VH(--)"
                    base = val.split("(")[0].split("-")[0].strip()
                    if base in level_map:
                        y = level_map[base]
                    else:
                        continue
                x_vals.append(turn_idx)
                y_vals.append(y)
            
            if len(x_vals) >= 3 and plotted < 20:  # Limit to 20 episodes for clarity
                ax.plot(x_vals, y_vals, alpha=0.3, color="#3498db", linewidth=0.8)
                plotted += 1
        
        # Compute mean trajectory
        turn_values = defaultdict(list)
        for ep_id, turns in episodes.items():
            for turn_idx, record in turns:
                val = record.get(head, "N")
                if val in level_map:
                    y = level_map[val]
                else:
                    base = val.split("(")[0].split("-")[0].strip()
                    if base in level_map:
                        y = level_map[base]
                    else:
                        continue
                turn_values[turn_idx].append(y)
        
        if turn_values:
            x_mean = sorted(turn_values.keys())
            y_mean = [np.mean(turn_values[x]) for x in x_mean]
            y_std = [np.std(turn_values[x]) for x in x_mean]
            
            ax.plot(x_mean, y_mean, color="#e74c3c", linewidth=2, marker="o", 
                   markersize=4, label="Mean", zorder=5)
            ax.fill_between(x_mean, 
                          [max(0, m - s) for m, s in zip(y_mean, y_std)],
                          [min(6, m + s) for m, s in zip(y_mean, y_std)],
                          alpha=0.2, color="#e74c3c")
        
        ax.set_xlabel("Turn index", fontsize=9)
        ax.set_ylabel("Level (VL=1, L=2, N=3, H=4, VH=5)", fontsize=8)
        ax.set_title(head.replace("_", " ").title(), fontsize=10, fontweight="bold")
        ax.set_ylim(0.5, 5.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    plt.suptitle("Per-Turn Relational Head Trajectories ($n{=}69$ episodes)", 
                fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    output_path = "paper/OCEAN_MBTI_ACL_2026/figures/relational_trends.pdf"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    
    print(f"Saved: {output_path}")
    
    # Compute transition statistics
    print("\nRelational head transition statistics:")
    for head in relational_heads:
        transitions = defaultdict(int)
        total = 0
        for ep_id, turns in episodes.items():
            for i in range(1, len(turns)):
                prev_val = turns[i-1][1].get(head, "N")
                curr_val = turns[i][1].get(head, "N")
                # Simplify to base level
                def simplify(v):
                    base = v.split("(")[0].split("-")[0].strip()
                    return base if base in level_map else "N"
                p = simplify(prev_val)
                c = simplify(curr_val)
                if p != c:
                    transitions[f"{p}->{c}"] += 1
                total += 1
        
        change_rate = sum(transitions.values()) / max(1, total)
        print(f"  {head}: {change_rate:.3f} change rate per turn ({sum(transitions.values())}/{total} transitions)")


if __name__ == "__main__":
    build_relational_trends()
