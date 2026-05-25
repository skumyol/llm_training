#!/usr/bin/env python3
"""Regenerate best_model_diagram.pdf from code."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.linewidth": 0.8,
})

fig, ax = plt.subplots(figsize=(9, 5.2))
ax.set_xlim(0, 100)
ax.set_ylim(0, 58)
ax.axis("off")

# Title
ax.text(50, 55, "ConditionalDialogue Soft-Prefix Conditioning",
        fontsize=13, fontweight="bold", ha="center", va="center")

# Colors
c_input = "#E8F4F8"      # light blue
c_proj = "#FFF4E6"       # light orange
c_prefix = "#E8F8E8"     # light green
c_model = "#F0E8F8"      # light purple
c_placebo = "#F8E8E8"      # light red

# Helper: draw box
def box(ax, x, y, w, h, text, color, fontsize=8.5, bold=False):
    style = "round,pad=0.3"
    bb = FancyBboxPatch((x - w/2, y - h/2), w, h,
                        boxstyle=style, facecolor=color,
                        edgecolor="#333333", linewidth=0.8)
    ax.add_patch(bb)
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, fontsize=fontsize, ha="center", va="center",
            fontweight=weight, wrap=True)

# Helper: arrow
def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2))

# --- Row 1: Inputs ---
box(ax, 12, 42, 16, 6, "OCEAN\n(5 values)", c_input)
box(ax, 32, 42, 16, 6, "VAD\n(3 values)", c_input)

# --- Row 2: Conditioning vector ---
box(ax, 22, 32, 20, 6, "8D Conditioning\nVector  $c_t$", c_proj, bold=True)
arrow(ax, 12, 39, 16, 35)
arrow(ax, 32, 39, 28, 35)

# --- Row 3: Projection + Prefix ---
box(ax, 48, 32, 14, 6, "Projection\n(8 × 8 learnable)", c_proj)
arrow(ax, 32, 32, 41, 32)

box(ax, 70, 32, 18, 6, "8 Soft-Prefix\nTokens", c_prefix, bold=True)
arrow(ax, 55, 32, 61, 32)

# --- Row 4: Concatenate + Model ---
box(ax, 70, 20, 18, 6, "Concatenated\nwith Prompt", c_prefix)
arrow(ax, 70, 29, 70, 23)

box(ax, 48, 20, 16, 6, "Conditional\nDialogue", c_model, bold=True)
arrow(ax, 61, 20, 56, 20)

# --- Row 5: LoRA + Tokenizer ---
box(ax, 48, 10, 16, 6, "LoRA\n(r=16, α=32)", c_model)
arrow(ax, 48, 17, 48, 13)

box(ax, 70, 10, 18, 6, "GPT-2\nTokenizer", c_model)
arrow(ax, 56, 10, 61, 10)

# --- Placebo controls (bottom) ---
ax.text(22, 3, "Placebo controls:", fontsize=8, fontweight="bold", ha="center")
box(ax, 12, 3, 14, 4, "Shuffled\nOCEAN", c_placebo, fontsize=7.5)
box(ax, 28, 3, 12, 4, "Random\nVAD", c_placebo, fontsize=7.5)
box(ax, 44, 3, 16, 4, "Shuffled +\nRandom", c_placebo, fontsize=7.5)

# Dashed lines from placebo to projection
for px in [12, 28, 44]:
    ax.plot([px, 48], [5, 29], color="#999999", lw=0.8, ls="--", alpha=0.6)

plt.tight_layout()
plt.savefig("figures/best_model_diagram.pdf", bbox_inches="tight", pad_inches=0.15)
plt.savefig("figures/best_model_diagram.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
print("Saved figures/best_model_diagram.{pdf,png}")
