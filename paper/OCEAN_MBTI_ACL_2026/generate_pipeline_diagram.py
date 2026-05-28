#!/usr/bin/env python3
"""Generate the main bottleneck figure for the OCEAN/MBTI paper."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.linewidth": 0.8,
})

fig, ax = plt.subplots(figsize=(12.2, 3.7))
ax.set_xlim(8, 176)
ax.set_ylim(0, 48)
ax.axis("off")

COLORS = {
    "input": "#E8F4F8",
    "model": "#F0E8F8",
    "state": "#E8F8E8",
    "router": "#FFF4E6",
    "slow": "#FFF8DC",
    "output": "#F8E8E8",
    "claim": "#FFFFFF",
}


def box(x, y, w, h, text, color, fontsize=10, weight="bold", lw=0.9):
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.28",
        facecolor=color,
        edgecolor="#333333",
        linewidth=lw,
    )
    ax.add_patch(rect)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
    )


def arrow(x1, y1, x2, y2, color="#555555", lw=1.3, label=None, label_pos=0.5, dy=2.2):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="->",
            color=color,
            lw=lw,
            shrinkA=0,
            shrinkB=0,
        ),
    )
    if label:
        ax.text(
            x1 + (x2 - x1) * label_pos,
            y1 + (y2 - y1) * label_pos + dy,
            label,
            fontsize=10,
            ha="center",
            va="center",
            color=color,
        )


# Left column: raw input and predictor.
box(18, 31, 25, 9, "Raw dialogue\ncontext\n+\nNPC profile", COLORS["input"], fontsize=10.5)
box(47, 33, 28, 9, "QLoRA-adapted\nQwen3-4B\nlatent predictor", COLORS["model"], fontsize=10.5)
box(47, 17, 28, 8, "29 supervised heads\nfrom final hidden state", COLORS["model"], fontsize=10.2)

arrow(30.5, 31, 33, 33)        # input -> predictor
arrow(47, 28.5, 47, 21, label="classification heads", dy=0)  # predictor -> heads

# Center: bottleneck.
box(
    88,
    26,
    45,
    27,
    "Inspectable $\\hat{Z}_t$\n\n"
    "C: intent / dialogue act\n"
    "A: valence, arousal, threat, control\n"
    "M: player intent, knowledge, credibility\n"
    "R: trust, respect, obligation\n"
    "N: secrecy, face, duty\n"
    "D: response policy, reveal decision",
    COLORS["state"],
    fontsize=10.1,
    lw=1.4,
)
ax.text(
    88,
    41.5,
    "central bottleneck: named variables before generation",
    fontsize=10.8,
    ha="center",
    va="center",
    color="#2E7D32",
    fontweight="bold",
)
arrow(61, 17, 66, 21)        # heads -> bottleneck (lower)
arrow(61, 33, 66, 35)        # predictor -> bottleneck (upper)

# Right column: operational use.
box(128, 36, 28, 7, "Fast/slow router\nreads $\\hat{Z}_t$", COLORS["router"], fontsize=10.4)
box(128, 18, 31, 7, "Disclosure constraints\nfor risky turns", COLORS["slow"], fontsize=10.3)
box(162, 29, 27, 9, "Qwen3-4B\nresponse generator", COLORS["model"], fontsize=10.6)
box(162, 10, 23, 6.5, "NPC response", COLORS["output"], fontsize=10.8)

arrow(111, 32.5, 114, 36)        # bottleneck -> router
arrow(111, 20.5, 112.5, 18, color="#B85C00")  # bottleneck -> disclosure
arrow(142, 36, 149, 31, color="#2E7D32")  # router -> response gen
ax.text(146, 37, "fast", fontsize=10, ha="center", color="#2E7D32")
arrow(143.5, 20, 149, 26, color="#B85C00")  # disclosure -> response gen
ax.text(148.5, 19, "slow", fontsize=10, ha="center", color="#B85C00")
arrow(162, 24.5, 162, 13.25)      # response gen -> output

plt.tight_layout()
plt.savefig("figures/pipeline_diagram.pdf", bbox_inches="tight", pad_inches=0.15)
plt.savefig("figures/pipeline_diagram.png", dpi=240, bbox_inches="tight", pad_inches=0.15)
print("Saved figures/pipeline_diagram.{pdf,png}")
