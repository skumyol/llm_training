#!/usr/bin/env python3
"""
Build router vs predictor error decomposition visual.

Creates a waterfall/stacked bar chart showing error decomposition.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def build_error_decomposition_figure(output_dir):
    """
    Create waterfall bar chart showing error decomposition:
    - Oracle (gold 29D): F1 = 1.000
    - Gold-head oracle (4 heads): F1 = 0.893  [predictor error: 0.107]
    - Current (predicted 29D): F1 = 0.672  [router error: 0.221]
    """
    
    # Data from Table tab:oracle
    systems = [
        "Oracle\n(gold 29D)",
        "Gold-head oracle\n(gold 4 routing)",
        "Current\n(predicted 29D)",
    ]
    
    f1_values = [1.000, 0.893, 0.672]
    gaps = [0.000, 0.107, 0.328]  # Gap to oracle
    
    # Error decomposition
    # Gap from predicted to gold-head = predictor error (0.221)
    # Gap from gold-head to oracle = router error (0.107)
    predictor_error = 0.893 - 0.672  # 0.221
    router_error = 1.000 - 0.893  # 0.107
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left panel: F1 values waterfall
    colors_f1 = ["#2ecc71", "#3498db", "#e74c3c"]  # Green, Blue, Red
    x = np.arange(len(systems))
    bars = ax1.bar(x, f1_values, color=colors_f1, edgecolor="black", linewidth=0.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, f1_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.02, 
                f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    # Add gap annotations
    ax1.annotate("", xy=(1, 0.893), xytext=(0, 1.0),
                arrowprops=dict(arrowstyle="<->", color="gray", lw=1.5))
    ax1.text(0.5, 0.95, f"Router error\n{router_error:.3f}", ha="center", fontsize=9, color="gray")
    
    ax1.annotate("", xy=(2, 0.672), xytext=(1, 0.893),
                arrowprops=dict(arrowstyle="<->", color="darkred", lw=1.5))
    ax1.text(1.5, 0.78, f"Predictor error\n{predictor_error:.3f}", ha="center", fontsize=9, color="darkred")
    
    ax1.set_ylabel("Routing F1", fontsize=11)
    ax1.set_ylim(0, 1.15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(systems, fontsize=9)
    ax1.set_title("Error Decomposition: Routing F1", fontsize=12, fontweight="bold")
    ax1.axhline(1.0, color="green", linestyle="--", alpha=0.3, label="Oracle")
    
    # Right panel: Stacked bar showing error composition
    categories = ["Gap to Oracle"]
    predictor = [predictor_error]  # 0.221
    router = [router_error]  # 0.107
    
    width = 0.5
    x2 = [0]
    
    # Stacked bar
    p1 = ax2.bar(x2, predictor, width, label=f"Predictor error ({predictor_error:.3f})", 
                 color="#e74c3c", edgecolor="black")
    p2 = ax2.bar(x2, router, width, bottom=predictor, label=f"Router error ({router_error:.3f})", 
                 color="#f39c12", edgecolor="black")
    
    # Total error bar on top
    total_error = predictor_error + router_error
    ax2.bar(x2, [0.01], width, bottom=[total_error], color="black", label=f"Total ({total_error:.3f})")
    
    # Add percentage annotations
    pred_pct = predictor_error / total_error * 100
    router_pct = router_error / total_error * 100
    
    ax2.text(0, predictor_error/2, f"{pred_pct:.1f}%", ha="center", va="center", 
             fontsize=11, fontweight="bold", color="white")
    ax2.text(0, predictor_error + router_error/2, f"{router_pct:.1f}%", ha="center", va="center", 
             fontsize=11, fontweight="bold", color="white")
    
    ax2.set_ylabel("Gap to Oracle (F1)", fontsize=11)
    ax2.set_ylim(0, 0.4)
    ax2.set_xticks([])
    ax2.set_title("Error Composition (Total Gap: 0.328)", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=9)
    
    plt.tight_layout()
    
    out_path = Path(output_dir) / "error_decomposition.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    
    print(f"Saved: {out_path}")
    
    # Also create unsafe fast-path version
    fig2, ax3 = plt.subplots(figsize=(8, 5))
    
    # Unsafe FP data
    systems_ufp = ["Gold-head oracle", "Current (predicted)"]
    unsafe_fp = [0.061, 0.174]
    
    x3 = np.arange(len(systems_ufp))
    colors_ufp = ["#3498db", "#e74c3c"]
    bars3 = ax3.bar(x3, unsafe_fp, color=colors_ufp, edgecolor="black", width=0.5)
    
    for bar, val in zip(bars3, unsafe_fp):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, height + 0.005, 
                f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    
    # Add reduction arrow
    ax3.annotate("", xy=(1, 0.174), xytext=(0, 0.061),
                arrowprops=dict(arrowstyle="->", color="red", lw=2))
    ax3.text(0.5, 0.12, f"+{0.174-0.061:.3f}\npredictor error", ha="center", fontsize=10, color="red")
    
    ax3.set_ylabel("Unsafe Fast-Path Rate", fontsize=11)
    ax3.set_ylim(0, 0.25)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(systems_ufp, fontsize=10)
    ax3.set_title("Predictor Error: Unsafe Fast-Path Rate", fontsize=12, fontweight="bold")
    
    plt.tight_layout()
    out_path2 = Path(output_dir) / "unsafe_fp_decomposition.pdf"
    plt.savefig(out_path2, bbox_inches="tight")
    plt.close()
    
    print(f"Saved: {out_path2}")


def main():
    output_dir = "paper/OCEAN_MBTI_ACL_2026/figures"
    build_error_decomposition_figure(output_dir)
    print("\n✅ Error decomposition figures generated successfully.")


if __name__ == "__main__":
    main()
