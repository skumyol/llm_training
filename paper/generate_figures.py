#!/usr/bin/env python3
"""Generate paper figures from local experiment summaries.

The script writes vector PDFs for LaTeX and PNG previews for quick browsing.
It intentionally uses only local JSON/CSV artifacts so figures are reproducible
without loading model checkpoints.
"""
from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


COLORS = {
    "blue": "#3563a9",
    "green": "#2f855a",
    "orange": "#c05621",
    "red": "#b83232",
    "purple": "#6b46c1",
    "gray": "#4a5568",
    "light_blue": "#e9f1fb",
    "light_green": "#e8f5ee",
    "light_orange": "#fff1e8",
    "light_purple": "#f1ebfb",
    "light_gray": "#edf2f7",
}


def load_json(rel: str) -> dict:
    with open(ROOT / rel) as f:
        return json.load(f)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def box(ax, xy, wh, text, fc, ec=None, fontsize=9, weight="normal"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.04",
        fc=fc,
        ec=ec or COLORS["gray"],
        lw=1.2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        wrap=True,
    )
    return patch


def arrow(ax, start, end, color=COLORS["gray"], rad=0.0, lw=1.3):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            lw=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def figure_architecture_stack() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 6.3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis("off")
    ax.set_title("Four-Track Evaluation Stack", fontsize=16, fontweight="bold", pad=12)

    box(ax, (0.35, 5.75), (2.05, 0.7), "Scenario bank\n+ world contexts", COLORS["light_gray"])
    box(ax, (0.35, 4.7), (2.05, 0.7), "LLM-generated\nturn traces", COLORS["light_gray"])
    box(ax, (0.35, 3.65), (2.05, 0.7), "Validated splits\ntrain / val / test", COLORS["light_gray"])
    arrow(ax, (1.38, 5.75), (1.38, 5.4))
    arrow(ax, (1.38, 4.7), (1.38, 4.35))

    tracks = [
        ("Track A\nFrom-scratch SLMs\nGPT / Prefix / MoE / SSM", COLORS["light_blue"], 3.2, 5.55),
        ("Track B\nConditioning encoders\nOCEAN + VAD", COLORS["light_green"], 6.05, 5.55),
        ("Track C\nResponse models\nTinyLlama / Gemma / Conditional", COLORS["light_orange"], 3.2, 3.45),
        ("Track D\nStructured Qwen3\n29 heads + response + joint", COLORS["light_purple"], 6.05, 3.45),
    ]
    for text, fc, x, y in tracks:
        box(ax, (x, y), (2.45, 1.1), text, fc, fontsize=9, weight="bold")
        arrow(ax, (2.4, 4.0), (x, y + 0.55), rad=0.08)

    box(ax, (9.25, 4.5), (2.15, 0.95), "Unified evaluation\nPPL, F1, κ, leakage,\ndiversity, routing", "#fff8db", fontsize=9, weight="bold")
    for x, y in [(5.65, 6.1), (8.5, 6.1), (5.65, 4.0), (8.5, 4.0)]:
        arrow(ax, (x, y), (9.25, 4.98), rad=0.05)
    save(fig, "architecture_stack")


def figure_data_flow() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("Data Generation and Evaluation Flow", fontsize=16, fontweight="bold", pad=12)

    nodes = [
        (0.35, 3.0, "7 scenario types\nNPC profiles\nsecrets/goals"),
        (2.55, 3.0, "Teacher LLM\nmulti-turn dialogue\n+ Z_t labels"),
        (4.85, 3.0, "Schema validation\nconsistency checks\ncounterfactuals"),
        (7.15, 3.0, "Packaged splits\n6,175 train\n683 val\n884 test"),
        (9.45, 3.0, "Model training\nTracks A-D"),
        (9.45, 1.25, "Analysis artifacts\nfigures, tables,\nLaTeX paper"),
    ]
    for x, y, text in nodes:
        box(ax, (x, y), (1.8, 0.95), text, COLORS["light_gray"], fontsize=8.6, weight="bold")
    for i in range(4):
        arrow(ax, (nodes[i][0] + 1.8, nodes[i][1] + 0.48), (nodes[i + 1][0], nodes[i + 1][1] + 0.48))
    arrow(ax, (10.35, 3.0), (10.35, 2.2))
    box(ax, (0.35, 0.75), (7.0, 0.8), "Auditable latent state Z_t = (C_t, A_t, M_t, R_t, N_t, D_t): 29 heads over social intent, affect, relationship, norms, and policy", "#fff8db", fontsize=9)
    arrow(ax, (5.75, 3.0), (4.2, 1.55), rad=-0.18)
    arrow(ax, (7.35, 1.15), (9.45, 1.65), rad=0.08)
    save(fig, "data_flow")


def figure_structured_llm() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.8)
    ax.axis("off")
    ax.set_title("Structured Qwen3 Social-State Pipeline", fontsize=16, fontweight="bold", pad=12)

    box(ax, (0.4, 2.65), (1.75, 0.85), "Dialogue context\nNPC + player\nhistory", COLORS["light_gray"], weight="bold")
    box(ax, (2.75, 3.65), (2.15, 0.9), "Stage 1\nQwen3-1.7B + LoRA\n29 classification heads", COLORS["light_purple"], weight="bold")
    box(ax, (5.5, 3.65), (2.0, 0.9), "Latent state Z_t\nC,A,M,R,N,D", "#fff8db", weight="bold")
    box(ax, (8.05, 3.65), (2.0, 0.9), "Rule router\nfast vs slow path", COLORS["light_blue"], weight="bold")
    box(ax, (5.5, 1.35), (2.2, 0.95), "Stage 2\nQwen3 response SFT\ngold Z_t in prompt", COLORS["light_orange"], weight="bold")
    box(ax, (8.25, 1.35), (2.2, 0.95), "Stage 3\njoint heads + LM\nconsistency loss", COLORS["light_green"], weight="bold")
    box(ax, (10.65, 2.65), (1.0, 0.85), "NPC\nreply", COLORS["light_gray"], weight="bold")

    arrow(ax, (2.15, 3.08), (2.75, 4.1))
    arrow(ax, (4.9, 4.1), (5.5, 4.1))
    arrow(ax, (7.5, 4.1), (8.05, 4.1))
    arrow(ax, (9.05, 3.65), (9.1, 2.3))
    arrow(ax, (2.15, 3.08), (5.5, 1.82), rad=-0.08)
    arrow(ax, (7.7, 1.82), (8.25, 1.82))
    arrow(ax, (10.45, 1.82), (10.9, 2.65))
    arrow(ax, (10.05, 4.1), (11.0, 3.5))

    ax.text(3.8, 0.45, "Key evaluation: latent heads are interpretable; response metrics expose verbosity/repetition; routing is deterministic over Z_t.", ha="center", fontsize=9)
    save(fig, "structured_llm_pipeline")


def figure_best_model() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    ax.set_title("Best Response Model: ConditionalDialogue Soft-Prefix Conditioning", fontsize=15.5, fontweight="bold", pad=12)

    box(ax, (0.45, 3.55), (2.0, 0.8), "NPC dialogue\ncontext", COLORS["light_gray"], weight="bold")
    box(ax, (0.45, 1.45), (2.0, 0.8), "NPC profile\nhistory/persona", COLORS["light_gray"], weight="bold")
    box(ax, (3.0, 3.55), (2.0, 0.8), "Affect encoder\nVAD (3D)", COLORS["light_green"], weight="bold")
    box(ax, (3.0, 1.45), (2.0, 0.8), "Personality cache\nOCEAN (5D)", COLORS["light_green"], weight="bold")
    box(ax, (5.7, 2.5), (1.85, 0.9), "8D social vector\nOCEAN + VAD", "#fff8db", weight="bold")
    box(ax, (8.05, 2.5), (1.75, 0.9), "MLP projection\n4 soft-prefix\ntokens", COLORS["light_blue"], weight="bold")
    box(ax, (10.25, 2.5), (1.45, 0.9), "TinyLlama\n+ LoRA\nresponse", COLORS["light_orange"], weight="bold")

    arrow(ax, (2.45, 3.95), (3.0, 3.95))
    arrow(ax, (2.45, 1.85), (3.0, 1.85))
    arrow(ax, (5.0, 3.95), (5.7, 3.05), rad=-0.08)
    arrow(ax, (5.0, 1.85), (5.7, 2.85), rad=0.08)
    arrow(ax, (7.55, 2.95), (8.05, 2.95))
    arrow(ax, (9.8, 2.95), (10.25, 2.95))
    ax.text(6.0, 0.75, "Validation PPL: 2.90 vs. 3.30 for unconditioned TinyLlama SFT (12.3% relative reduction).", fontsize=10, ha="left")
    save(fig, "best_model_diagram")


def figure_slm_results() -> None:
    runs = {
        "GPT": "slm_training/artifacts/small_lm/slurm_816183_slm_small_lm_20260502_170210/run_summary.json",
        "PrefixGPT": "slm_training/artifacts/small_lm/slurm_816185_slm_small_lm_20260502_170217/run_summary.json",
        "MoE": "slm_training/artifacts/small_lm/slurm_816186_slm_small_lm_20260502_170219/run_summary.json",
        "Mamba-like": "slm_training/artifacts/small_lm/slurm_816184_slm_small_lm_20260502_170213/run_summary.json",
    }
    rows = []
    for name, rel in runs.items():
        d = load_json(rel)
        rows.append((name, d["best"]["val_ppl"], d["model_params"] / 1e6))
    rows.sort(key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [COLORS["green"] if n == "MoE" else COLORS["blue"] for n in names]
    bars = ax.bar(names, vals, color=colors)
    ax.set_ylabel("Validation perplexity (lower is better)")
    ax.set_title("From-Scratch SLM Architecture Benchmark", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    for bar, (_, ppl, params) in zip(bars, rows):
        ax.text(bar.get_x() + bar.get_width() / 2, ppl + 0.8, f"{ppl:.1f}\n{params:.1f}M", ha="center", fontsize=8.5)
    ax.set_ylim(0, max(vals) * 1.22)
    save(fig, "slm_ppl_comparison")


def figure_response_results() -> None:
    rows = [
        ("ConditionalDialogue", 2.8967, COLORS["green"]),
        ("TinyLlama LoRA", 3.3048, COLORS["blue"]),
        ("Gemma-2-2B-it QLoRA", 6.3838, COLORS["orange"]),
        ("Gemma-4-E2B QLoRA\nexploratory", 16.2395, COLORS["gray"]),
    ]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    bars = ax.bar(names, vals, color=[r[2] for r in rows])
    ax.set_ylabel("Validation perplexity (lower is better)")
    ax.set_title("Response Model Perplexity", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.35, f"{val:.2f}", ha="center", fontsize=9)
    ax.tick_params(axis="x", rotation=15)
    ax.set_ylim(0, max(vals) * 1.2)
    save(fig, "response_ppl_comparison")


def figure_latent_group_results() -> None:
    comp = load_json("eval_results/comprehensive_results.json")
    groups = comp["latent_state"]["groups"]
    order = ["C", "A", "M", "R", "N", "D"]
    names = ["Conv.", "Affect", "Mental", "Rel.", "Norm.", "Decision"]
    acc = [groups[g]["mean_accuracy"] for g in order]
    kappa = [groups[g]["mean_kappa"] for g in order]

    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    w = 0.36
    ax.bar(x - w / 2, acc, width=w, label="Accuracy", color=COLORS["blue"])
    ax.bar(x + w / 2, kappa, width=w, label="Estimated κ", color=COLORS["green"])
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Latent Social-State Prediction by Group", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    for xi, a in zip(x, acc):
        ax.text(xi - w / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)
    for xi, k in zip(x, kappa):
        ax.text(xi + w / 2, k + 0.02, f"{k:.2f}", ha="center", fontsize=8)
    save(fig, "latent_group_scores")


def figure_latent_head_heatmap() -> None:
    comp = load_json("eval_results/comprehensive_results.json")
    rows = []
    group_of = {}
    for group, info in comp["latent_state"]["groups"].items():
        for h in info["heads"]:
            group_of[h] = group
    for head, m in comp["latent_state"]["per_head"].items():
        rows.append((group_of.get(head, "?"), head, m["accuracy"], m["cohen_kappa_est"]))
    rows.sort(key=lambda r: (["C", "A", "M", "R", "N", "D", "?"].index(r[0]), -r[2]))

    fig, ax = plt.subplots(figsize=(8.4, 8.8))
    data = np.array([[r[2], r[3]] for r in rows])
    im = ax.imshow(data, aspect="auto", cmap="YlGnBu", vmin=0.3, vmax=0.9)
    labels = [f"{g}: {h.replace('_', ' ')}" for g, h, _, _ in rows]
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Accuracy", "Estimated κ"])
    ax.set_title("Per-Head Latent Metrics", fontweight="bold")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    save(fig, "latent_head_heatmap")


def figure_qwen_training_curves() -> None:
    metric_dir = ROOT / "checkpoints" / "latent_predictor" / "metrics"
    epochs = []
    mean_acc = []
    resp_f1 = []
    trust_f1 = []
    for p in sorted(metric_dir.glob("epoch_*_latent.json")):
        d = json.load(open(p))["summary"]
        epochs.append(int(p.name.split("_")[1]))
        mean_acc.append(d["mean_accuracy"])
        resp_f1.append(d["response_policy_f1"])
        trust_f1.append(d["trust_delta_f1"])

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(epochs, mean_acc, marker="o", label="Mean accuracy", color=COLORS["blue"])
    ax.plot(epochs, resp_f1, marker="o", label="Response-policy macro-F1", color=COLORS["orange"])
    ax.plot(epochs, trust_f1, marker="o", label="Trust-delta macro-F1", color=COLORS["green"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation score")
    ax.set_title("Qwen3 Latent Predictor Training Dynamics", fontweight="bold")
    ax.set_xticks(epochs)
    ax.set_ylim(0.3, 0.75)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    save(fig, "qwen_latent_training")


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure_architecture_stack()
    figure_data_flow()
    figure_structured_llm()
    figure_best_model()
    figure_slm_results()
    figure_response_results()
    figure_latent_group_results()
    figure_latent_head_heatmap()
    figure_qwen_training_curves()
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
