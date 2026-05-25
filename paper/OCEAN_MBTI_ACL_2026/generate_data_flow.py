#!/usr/bin/env python3
"""Regenerate data_flow.pdf from code."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8.5,
    "axes.linewidth": 0.8,
})

fig, ax = plt.subplots(figsize=(10, 6.5))
ax.set_xlim(0, 100)
ax.set_ylim(0, 65)
ax.axis("off")

# Title
ax.text(50, 62, "Data Flow from Scenario Specifications to Validated Splits and Paper Artifacts",
        fontsize=11, fontweight="bold", ha="center", va="center")

# Colors
c_bank = "#E8F4F8"      # light blue
c_plan = "#FFF4E6"       # light orange
c_gen = "#E8F8E8"        # light green
c_val = "#F0E8F8"        # light purple
c_split = "#F8F0E8"      # light tan
c_audit = "#F8E8E8"        # light red
c_state = "#E8E8F8"        # light indigo

# Helper: draw box
def box(ax, x, y, w, h, text, color, fontsize=8, bold=False):
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

# --- Main pipeline (left to right) ---
box(ax, 12, 50, 18, 7, "ScenarioBank\n(7 YAML scenarios)", c_bank, bold=True)
arrow(ax, 21, 46.5, 21, 41)

box(ax, 12, 37, 18, 7, "Episode\nPlanner", c_plan, bold=True)
arrow(ax, 21, 33.5, 21, 29)

box(ax, 12, 25, 18, 7, "Dialogue\nGenerator", c_gen, bold=True)
arrow(ax, 21, 21.5, 21, 17)

box(ax, 12, 13, 18, 7, "Turn\nValidator", c_val, bold=True)
arrow(ax, 21, 9.5, 21, 5)

box(ax, 12, 1, 18, 7, "Train / Val / Test\nSplit", c_split, bold=True)

# --- Right-side branches ---
# Validator → Audit branch
arrow(ax, 30, 13, 40, 13)
box(ax, 52, 13, 18, 7, "Human\nAudit", c_audit)
arrow(ax, 61, 16.5, 61, 21)
box(ax, 52, 25, 18, 7, "Model-Based\nValidator", c_audit)
arrow(ax, 61, 28.5, 61, 33)
box(ax, 52, 37, 18, 7, "Audit\nReport", c_audit, bold=True)

# Validator → State branch (upward)
arrow(ax, 30, 15, 40, 37)
box(ax, 52, 50, 20, 7, "Inspectable latent\nstate $Z_t$", c_state, bold=True)

# Episode Planner → State
arrow(ax, 21, 40.5, 42, 47)

# Dialogue Generator → State
arrow(ax, 21, 28.5, 42, 47)

# Turn Validator → State
arrow(ax, 21, 16.5, 42, 47)

# State → Audit Report
arrow(ax, 61, 46.5, 61, 40.5)

# State → Train/Val/Test split (downward)
arrow(ax, 42, 46.5, 30, 4)

plt.tight_layout()
plt.savefig("figures/data_flow.pdf", bbox_inches="tight", pad_inches=0.15)
plt.savefig("figures/data_flow.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
print("Saved figures/data_flow.{pdf,png}")
