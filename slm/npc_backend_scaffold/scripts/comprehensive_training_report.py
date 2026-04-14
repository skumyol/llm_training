#!/usr/bin/env python3
"""
comprehensive_training_report.py
=================================
Academic research reporting script for the NPC backend project.

What this script does
---------------------
1. PERSONALITY ENCODER  – trains DistilBERT (Optuna-best params) across N seeds
                          to confirm results are robust, not a local minimum.
2. AFFECT ENCODER       – same, with Optuna-best VAD regression params.
3. SMALL-LM BENCHMARK  – trains all 6 architectures (GRU, AWD-LSTM, GPT,
                          PrefixGPT, MoE, Mamba-like) on dialogue corpus.
4. REPORTING            – produces:
     • Figures  (PNG + PDF): training curves, per-trait/dim bars, radar,
                              architecture comparison, hyperparameter heatmap,
                              stability boxplots, loss landscape overview
     • Tables   (CSV + LaTeX + Markdown): all models × all metrics
     • HTML summary report with embedded figures and tables
     • LaTeX snippet ready to paste into paper

Usage
-----
# Full pipeline (1-2 days on CPU, 4-6 hours on GPU):
python scripts/comprehensive_training_report.py --all --n-seeds 3

# Individual phases:
python scripts/comprehensive_training_report.py --phase personality --n-seeds 5
python scripts/comprehensive_training_report.py --phase affect      --n-seeds 3
python scripts/comprehensive_training_report.py --phase small-lm    --n-seeds 2
python scripts/comprehensive_training_report.py --phase report-only

# Quick smoke-test (1 seed, fewer epochs):
python scripts/comprehensive_training_report.py --all --n-seeds 1 --quick
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
import torch

# ── Embedding Model Support (for A/B testing conditioning) ──────────────────────
HAS_TRANSFORMERS = False
try:
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except Exception:
    pass  # Warning will be logged later after log is configured

# Matplotlib – use non-interactive backend before any other mpl import
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib import cm

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent.parent
PYTHON  = sys.executable
SCRIPTS = ROOT / "scripts"
SRC     = ROOT / "src"

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# § 1  CONSTANTS AND BEST CONFIGS
# ══════════════════════════════════════════════════════════════════════════════

# Baselines (manual training, before Optuna)
BASELINES = {
    "personality": {"val_f1": 0.6780},
    "affect":      {"val_ccc": 0.6080},
}

# Architecture display names for figures
ARCH_LABELS = {
    "gru":        "GRU-LM",
    "awdlstm":    "AWD-LSTM",
    "gpt":        "TinyGPT",
    "prefix_gpt": "PrefixGPT",
    "moe":        "TinyMoE",
    "mamba_like": "Mamba-like",
}

# Personality traits (OCEAN)
P_TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

# Affect dimensions (VAD)
A_DIMS   = ["valence", "arousal", "dominance"]

# Colour palette (accessible)
PALETTE = {
    "gru":        "#4E79A7",
    "awdlstm":    "#F28E2B",
    "gpt":        "#E15759",
    "prefix_gpt": "#76B7B2",
    "moe":        "#59A14F",
    "mamba_like": "#EDC948",
    "personality":"#B07AA1",
    "affect":     "#FF9DA7",
    "baseline":   "#9C755F",
}


# ══════════════════════════════════════════════════════════════════════════════
# § 1b  EMBEDDING MODEL EXTRACTOR (for A/B testing conditioning)
# ══════════════════════════════════════════════════════════════════════════════

class EmbeddingExtractor:
    """
    Extracts sentence embeddings from a pre-trained model.
    Used for A/B testing: compare zero-vector vs semantic conditioning.
    """
    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-4B", device: str = "cpu"):
        if not HAS_TRANSFORMERS:
            raise RuntimeError("transformers library required for embedding extraction")
        self.model_name = model_name
        self.device = torch.device(device)
        log.info(f"Loading embedding model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()
        # Get embedding dimension
        dummy = self.tokenizer("test", return_tensors="pt")
        with torch.no_grad():
            out = self.model(**dummy.to(self.device))
            # Handle different output formats
            if hasattr(out, 'last_hidden_state'):
                self.dim = out.last_hidden_state.shape[-1]
            elif hasattr(out, 'pooler_output') and out.pooler_output is not None:
                self.dim = out.pooler_output.shape[-1]
            elif isinstance(out, tuple):
                self.dim = out[0].shape[-1]
            else:
                self.dim = out.shape[-1]
        log.info(f"Embedding model loaded: dim={self.dim}, vocab={len(self.tokenizer)}")

    @torch.no_grad()
    def encode(self, texts: List[str], max_length: int = 512) -> torch.Tensor:
        """Return [batch, dim] sentence embeddings."""
        if not texts:
            return torch.zeros(0, self.dim, device=self.device)
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(self.device)
        outputs = self.model(**inputs)
        # Mean pooling over sequence
        if hasattr(outputs, 'last_hidden_state'):
            hidden = outputs.last_hidden_state  # [batch, seq, dim]
            mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden.size()).float()
            sum_emb = (hidden * mask).sum(dim=1)
            mean_emb = sum_emb / mask.sum(dim=1).clamp(min=1e-9)
            return mean_emb
        elif hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            return outputs.pooler_output
        elif isinstance(outputs, tuple):
            return outputs[0][:, 0]  # CLS token
        else:
            return outputs[:, 0]

    def project_to_dim(self, embeddings: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Project embeddings to target dimension (e.g., 8 for cond_dim)."""
        if embeddings.shape[-1] == target_dim:
            return embeddings
        # Use first target_dim components (PCA approximation) + linear transform
        if embeddings.shape[-1] > target_dim:
            return embeddings[:, :target_dim]
        # Pad with zeros if smaller
        pad = torch.zeros(embeddings.shape[0], target_dim - embeddings.shape[-1],
                          device=embeddings.device, dtype=embeddings.dtype)
        return torch.cat([embeddings, pad], dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
# § 2  LOAD BEST CONFIGS FROM OPTUNA RESULTS
# ══════════════════════════════════════════════════════════════════════════════

def load_optuna_best(task: str) -> Dict[str, Any]:
    """Load the best hyperparameters found by Optuna for a given task."""
    path = ROOT / "artifacts" / "optuna" / f"{task}_best.json"
    if not path.exists():
        log.warning(f"No Optuna result found at {path}; using DEFAULTS")
        return {}
    with open(path) as f:
        data = json.load(f)
    log.info(f"Loaded Optuna best for {task}: value={data.get('best_value', '?'):.4f}")
    return data.get("best_params", {})


def build_personality_config(params: Dict[str, Any], seed: int, run_id: str,
                             epochs: int) -> Dict[str, Any]:
    lr  = params.get("lr", 3.16e-5)
    elr = lr * params.get("encoder_lr_factor", 0.2)
    return {
        "model_name":            "distilbert-base-uncased",
        "train_path":            "data/personality/train.csv",
        "val_path":              "data/personality/val.csv",
        "text_column":           "text",
        "target_columns":        P_TRAITS,
        "max_length":            256,
        "batch_size":            16,
        "grad_accum":            2,
        "lr":                    lr,
        "encoder_lr":            elr,
        "epochs":                epochs,
        "patience":              6,
        "log_every":             20,
        "seed":                  seed,
        "output_dir":            "artifacts/personality_encoder",
        "warmup_ratio":          0.1,
        "max_grad_norm":         1.0,
        "loss_type":             "focal_bce",
        "focal_gamma":           params.get("focal_gamma", 2.39),
        "dropout":               params.get("dropout", 0.395),
        "multi_sample_dropout":  0,
        "freeze_encoder_epochs": params.get("freeze_encoder_epochs", 0),
        "rdrop_alpha":           0.0,
        "token_drop_prob":       params.get("token_drop_prob", 0.079),
    }


def build_affect_config(params: Dict[str, Any], seed: int, run_id: str,
                        epochs: int) -> Dict[str, Any]:
    lr  = params.get("lr", 4.75e-5)
    elr = lr * params.get("encoder_lr_factor", 0.271)
    return {
        "model_name":            "distilbert-base-uncased",
        "train_path":            "data/affect/train.csv",
        "val_path":              "data/affect/val.csv",
        "text_column":           "text",
        "target_columns":        A_DIMS,
        "max_length":            256,
        "batch_size":            16,
        "grad_accum":            params.get("grad_accum", 2),
        "lr":                    lr,
        "encoder_lr":            elr,
        "epochs":                epochs,
        "patience":              6,
        "log_every":             20,
        "seed":                  seed,
        "output_dir":            "artifacts/affect_encoder",
        "warmup_ratio":          0.1,
        "max_grad_norm":         1.0,
        "loss_type":             "ccc_mse",
        "ccc_weight":            params.get("ccc_weight", 0.543),
        "dropout":               params.get("dropout", 0.298),
        "multi_sample_dropout":  0,
        "freeze_encoder_epochs": params.get("freeze_encoder_epochs", 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# § 3  SUBPROCESS TRAINING RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

def _save_config(cfg: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


def run_training_subprocess(
    module: str,
    config_path: Path,
    run_id: str,
    timeout_sec: int = 7200,
    extra_args: Optional[List[str]] = None,
) -> bool:
    """Run a training module as subprocess. Returns True on success."""
    cmd = [
        PYTHON, "-m", module,
        "--config", str(config_path),
        "--run-id", run_id,
    ]
    if extra_args:
        cmd.extend(extra_args)

    log.info(f"Starting: {' '.join(cmd[-4:])}")
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        elapsed = time.time() - t0
        if proc.returncode != 0:
            log.error(f"FAILED ({elapsed:.0f}s) – {run_id}\n{proc.stderr[-2000:]}")
            return False
        log.info(f"OK ({elapsed:.0f}s) – {run_id}")
        return True
    except subprocess.TimeoutExpired:
        log.error(f"TIMEOUT ({timeout_sec}s) – {run_id}")
        return False
    except Exception as e:
        log.error(f"ERROR – {run_id}: {e}")
        return False


# ── Personality ────────────────────────────────────────────────────────────────

def run_personality_multiseed(
    n_seeds: int,
    epochs: int,
    out_dir: Path,
) -> List[Dict[str, Any]]:
    """Train personality encoder across n_seeds, return list of result dicts."""
    params = load_optuna_best("personality")
    cfg_dir = out_dir / "configs" / "personality"
    results = []

    seeds = list(range(42, 42 + n_seeds))
    for i, seed in enumerate(seeds, 1):
        run_id = f"report_personality_s{seed}_{datetime.now():%H%M%S}"
        log.info(f"[{i}/{n_seeds}] Personality seed={seed}  run_id={run_id}")

        cfg = build_personality_config(params, seed, run_id, epochs)
        cfg_path = cfg_dir / f"seed_{seed}.yaml"
        _save_config(cfg, cfg_path)

        ok = run_training_subprocess(
            "src.train.run_personality", cfg_path, run_id, timeout_sec=7200
        )

        summary_path = ROOT / "artifacts" / "personality_encoder" / run_id / "run_summary.json"
        if ok and summary_path.exists():
            with open(summary_path) as f:
                data = json.load(f)
            row = {"seed": seed, "run_id": run_id, "success": True}
            best = data.get("best", {})
            epoch_data = data.get("epochs", [])
            row.update({
                "val_f1":           best.get("val_f1", float("nan")),
                "val_accuracy":     best.get("val_acc", float("nan")),
                "best_epoch":       best.get("epoch", 0),
                "epoch_metrics":    epoch_data,
            })
            # Per-trait F1 from the best epoch's per_dim
            best_ep_data = next((e for e in epoch_data if e.get("epoch") == best.get("epoch")), {})
            per_dim = best_ep_data.get("per_dim", {})
            for trait in P_TRAITS:
                trait_data = per_dim.get(trait, {})
                row[f"f1_{trait}"] = trait_data.get("f1", float("nan")) if isinstance(trait_data, dict) else float("nan")
            results.append(row)
        else:
            results.append({"seed": seed, "run_id": run_id, "success": False,
                            "val_f1": float("nan")})

    _save_results(results, out_dir / "personality_runs.json")
    return results


# ── Affect ─────────────────────────────────────────────────────────────────────

def run_affect_multiseed(
    n_seeds: int,
    epochs: int,
    out_dir: Path,
) -> List[Dict[str, Any]]:
    """Train affect encoder across n_seeds, return list of result dicts."""
    params = load_optuna_best("affect")
    cfg_dir = out_dir / "configs" / "affect"
    results = []

    seeds = list(range(42, 42 + n_seeds))
    for i, seed in enumerate(seeds, 1):
        run_id = f"report_affect_s{seed}_{datetime.now():%H%M%S}"
        log.info(f"[{i}/{n_seeds}] Affect seed={seed}  run_id={run_id}")

        cfg = build_affect_config(params, seed, run_id, epochs)
        cfg_path = cfg_dir / f"seed_{seed}.yaml"
        _save_config(cfg, cfg_path)

        ok = run_training_subprocess(
            "src.train.run_affect", cfg_path, run_id, timeout_sec=7200
        )

        summary_path = ROOT / "artifacts" / "affect_encoder" / run_id / "run_summary.json"
        if ok and summary_path.exists():
            with open(summary_path) as f:
                data = json.load(f)
            row = {"seed": seed, "run_id": run_id, "success": True}
            best = data.get("best", {})
            epoch_data = data.get("epochs", [])
            row.update({
                "val_ccc":       best.get("val_ccc", float("nan")),
                "val_mse":       best.get("val_mse", float("nan")),
                "val_mae":       best.get("val_mae", float("nan")),
                "val_r2":        best.get("val_r2", float("nan")),
                "best_epoch":    best.get("epoch", 0),
                "epoch_metrics": epoch_data,
            })
            # Per-dim metrics from the best epoch
            best_ep_data = next((e for e in epoch_data if e.get("epoch") == best.get("epoch")), {})
            per_dim = best_ep_data.get("per_dim", {})
            for dim in A_DIMS:
                for m in ["ccc", "mse", "r2"]:
                    row[f"{m}_{dim}"] = per_dim.get(dim, {}).get(m, float("nan"))
            results.append(row)
        else:
            results.append({"seed": seed, "run_id": run_id, "success": False,
                            "val_ccc": float("nan")})

    _save_results(results, out_dir / "affect_runs.json")
    return results


# ── Small-LM Benchmark ─────────────────────────────────────────────────────────

def run_small_lm_benchmark(
    n_seeds: int,
    epochs: int,
    out_dir: Path,
    embedding_model: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Train all 6 small-LM architectures × n_seeds on dialogue corpus."""
    archs   = list(ARCH_LABELS.keys())
    results = []

    # For A/B testing: baseline vs semantic conditioning
    if embedding_model:
        log.info(f"Small-LM: Using semantic conditioning from {embedding_model}")
    else:
        log.info("Small-LM: Using zero conditioning (baseline for A/B testing)")

    data_train = ROOT / "data" / "dialogue" / "train.txt"
    data_val   = ROOT / "data" / "dialogue" / "val.txt"

    if not data_train.exists():
        log.warning("Dialogue text not found; small-LM benchmark skipped.")
        return results

    seeds = list(range(42, 42 + n_seeds))

    for arch in archs:
        for seed in seeds:
            run_id = f"report_slm_{arch}_s{seed}_{datetime.now():%H%M%S}"
            log.info(f"Small-LM: arch={arch}  seed={seed}")

            cfg = {
                "arch":             arch,
                "hardware_profile": "rtx4070_small",
                "train_text":       str(data_train),
                "val_text":         str(data_val),
                "seq_len":          256,
                "batch_size":       16,
                "grad_accum":       4,
                "lr":               3e-4,
                "weight_decay":     0.1,
                "epochs":           epochs,
                "use_amp":          False,
                "log_every":        20,
                "eval_every_steps": 200,
                "cond_dim":         8,
                "seed":             seed,
                "output_dir":       "artifacts/small_lm",
                "embedding_model":  embedding_model,  # A/B testing: None=baseline, model=semantic
                "embedding_cache":  True,
            }
            cfg_path = out_dir / "configs" / "small_lm" / f"{arch}_s{seed}.yaml"
            _save_config(cfg, cfg_path)

            ok = run_training_subprocess(
                "src.train.run_small_lm", cfg_path, run_id,
                timeout_sec=14400,
                extra_args=["--arch", arch],
            )

            summary_path = ROOT / "artifacts" / "small_lm" / run_id / "run_summary.json"
            if ok and summary_path.exists():
                with open(summary_path) as f:
                    data = json.load(f)
                best = data.get("best", {})
                epoch_data = data.get("epochs", [])
                last_epoch = epoch_data[-1] if epoch_data else {}
                row = {
                    "arch": arch,
                    "arch_label": ARCH_LABELS[arch],
                    "seed": seed,
                    "run_id": run_id,
                    "success": True,
                    "final_val_loss": last_epoch.get("val_loss", float("nan")),
                    "final_val_ppl":  last_epoch.get("val_ppl",  float("nan")),
                    "best_val_loss":  best.get("val_loss",  float("nan")),
                    "best_val_ppl":   best.get("val_ppl",   float("nan")),
                    "final_test_ppl": data.get("final_test_ppl", float("nan")),
                    "num_params":     data.get("model_params", 0),
                    "epoch_metrics":  epoch_data,
                }
            else:
                row = {"arch": arch, "arch_label": ARCH_LABELS[arch], "seed": seed,
                       "run_id": run_id, "success": False,
                       "final_val_ppl": float("nan")}
            results.append(row)

    _save_results(results, out_dir / "small_lm_runs.json")
    return results


# ── Helpers ────────────────────────────────────────────────────────────────────

def _save_results(results: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = []
    for r in results:
        row = {}
        for k, v in r.items():
            if k == "epoch_metrics":
                row[k] = v
            elif isinstance(v, float) and math.isnan(v):
                row[k] = None
            else:
                row[k] = v
        serialisable.append(row)
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)
    log.info(f"Results saved → {path}")


def _load_results(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    for row in data:
        for k, v in row.items():
            if v is None and k not in ("epoch_metrics",):
                row[k] = float("nan")
    return data


# ══════════════════════════════════════════════════════════════════════════════
# § 4  AGGREGATION UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def aggregate(results: List[Dict], keys: List[str]) -> pd.DataFrame:
    """Compute mean ± std across seeds for given metric keys."""
    rows = [r for r in results if r.get("success", False)]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    stats = {}
    for k in keys:
        if k in df.columns:
            stats[f"{k}_mean"] = df[k].mean()
            stats[f"{k}_std"]  = df[k].std()
            stats[f"{k}_min"]  = df[k].min()
            stats[f"{k}_max"]  = df[k].max()
    return pd.DataFrame([stats])


def aggregate_by_arch(results: List[Dict], keys: List[str]) -> pd.DataFrame:
    rows = [r for r in results if r.get("success", False)]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    records = []
    for arch, grp in df.groupby("arch"):
        rec = {"arch": arch, "arch_label": ARCH_LABELS.get(arch, arch),
               "num_params": grp["num_params"].iloc[0] if "num_params" in grp else 0}
        for k in keys:
            if k in grp.columns:
                rec[f"{k}_mean"] = grp[k].mean()
                rec[f"{k}_std"]  = grp[k].std()
        records.append(rec)
    return pd.DataFrame(records).sort_values("final_val_ppl_mean")


# ══════════════════════════════════════════════════════════════════════════════
# § 5  VISUALISATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

FIG_DIR: Optional[Path] = None  # set in main()

def _savefig(fig: plt.Figure, name: str) -> Path:
    assert FIG_DIR is not None
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    p_png = FIG_DIR / f"{name}.png"
    p_pdf = FIG_DIR / f"{name}.pdf"
    fig.savefig(p_png, dpi=150, bbox_inches="tight")
    fig.savefig(p_pdf, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figure saved: {p_png.name}")
    return p_png


# ── Figure 1: Training Curves (loss / metric vs epoch) ────────────────────────

def fig_training_curves(
    p_results: List[Dict],
    a_results: List[Dict],
    slm_results: List[Dict],
) -> Path:
    fig = plt.figure(figsize=(18, 5))
    gs  = GridSpec(1, 3, figure=fig, wspace=0.32)

    # ── Personality ──
    ax0 = fig.add_subplot(gs[0])
    for r in p_results:
        if not r.get("success"):
            continue
        epochs_data = r.get("epoch_metrics", [])
        if not epochs_data:
            continue
        xs = [e.get("epoch", i+1) for i, e in enumerate(epochs_data)]
        ys = [e.get("val_f1", float("nan")) for e in epochs_data]
        ax0.plot(xs, ys, alpha=0.7, color=PALETTE["personality"], lw=1.5,
                 label=f"seed={r['seed']}")
    ax0.axhline(BASELINES["personality"]["val_f1"], color=PALETTE["baseline"],
                ls="--", lw=1.2, label="Baseline")
    ax0.set(title="Personality Encoder", xlabel="Epoch", ylabel="Val F1 (macro)")
    ax0.legend(fontsize=8)
    ax0.grid(alpha=0.3)

    # ── Affect ──
    ax1 = fig.add_subplot(gs[1])
    for r in a_results:
        if not r.get("success"):
            continue
        epochs_data = r.get("epoch_metrics", [])
        if not epochs_data:
            continue
        xs = [e.get("epoch", i+1) for i, e in enumerate(epochs_data)]
        ys = [e.get("val_ccc", float("nan")) for e in epochs_data]
        ax1.plot(xs, ys, alpha=0.7, color=PALETTE["affect"], lw=1.5,
                 label=f"seed={r['seed']}")
    ax1.axhline(BASELINES["affect"]["val_ccc"], color=PALETTE["baseline"],
                ls="--", lw=1.2, label="Baseline")
    ax1.set(title="Affect Encoder", xlabel="Epoch", ylabel="Val CCC (mean)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # ── Small-LM PPL ──
    ax2 = fig.add_subplot(gs[2])
    shown = set()
    for r in slm_results:
        if not r.get("success"):
            continue
        arch   = r["arch"]
        colour = PALETTE.get(arch, "#888")
        epochs_data = r.get("epoch_metrics", [])
        if not epochs_data:
            continue
        xs = [e.get("epoch", i+1) for i, e in enumerate(epochs_data)]
        ys = [e.get("val_ppl", float("nan")) for e in epochs_data]
        lbl = ARCH_LABELS[arch] if arch not in shown else None
        ax2.plot(xs, ys, alpha=0.7, color=colour, lw=1.5, label=lbl)
        shown.add(arch)
    ax2.set(title="Small-LM Architectures", xlabel="Epoch", ylabel="Val Perplexity")
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(alpha=0.3)

    fig.suptitle("Training Dynamics Across All Models", fontsize=14, fontweight="bold")
    return _savefig(fig, "fig1_training_curves")


# ── Figure 2: Personality Per-Trait F1 ────────────────────────────────────────

def fig_personality_per_trait(p_results: List[Dict]) -> Path:
    rows = [r for r in p_results if r.get("success")]
    fig, ax = plt.subplots(figsize=(10, 5))
    if not rows:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return _savefig(fig, "fig2_personality_per_trait")

    trait_means, trait_stds = [], []
    for t in P_TRAITS:
        vals = [r.get(f"f1_{t}", float("nan")) for r in rows]
        vals = [v for v in vals if not math.isnan(v)]
        trait_means.append(np.mean(vals) if vals else float("nan"))
        trait_stds.append(np.std(vals)   if vals else 0.0)

    x = np.arange(len(P_TRAITS))
    bars = ax.bar(x, trait_means, yerr=trait_stds, capsize=5,
                  color=[PALETTE["personality"]] * len(P_TRAITS),
                  alpha=0.85, edgecolor="white", linewidth=1.5,
                  error_kw={"elinewidth": 1.5, "ecolor": "black"})

    # Annotate values
    for b, m in zip(bars, trait_means):
        if not math.isnan(m):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=9)

    # Baseline overall
    ax.axhline(BASELINES["personality"]["val_f1"], color=PALETTE["baseline"],
               ls="--", lw=1.4, label=f"Baseline (macro avg) = {BASELINES['personality']['val_f1']:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels([t.capitalize() for t in P_TRAITS], fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1 Score")
    ax.set_title("Personality Encoder – Per-Trait F1 (mean ± std across seeds)", fontsize=13)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return _savefig(fig, "fig2_personality_per_trait")


# ── Figure 3: Affect Per-Dimension CCC ────────────────────────────────────────

def fig_affect_per_dim(a_results: List[Dict]) -> Path:
    rows = [r for r in a_results if r.get("success")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # CCC bar chart
    ax = axes[0]
    dim_means_ccc, dim_stds_ccc = [], []
    dim_means_r2,  dim_stds_r2  = [], []
    for d in A_DIMS:
        ccc_vals = [r.get(f"ccc_{d}", float("nan")) for r in rows]
        r2_vals  = [r.get(f"r2_{d}",  float("nan")) for r in rows]
        ccc_vals = [v for v in ccc_vals if not math.isnan(v)]
        r2_vals  = [v for v in r2_vals  if not math.isnan(v)]
        dim_means_ccc.append(np.mean(ccc_vals) if ccc_vals else float("nan"))
        dim_stds_ccc.append(np.std(ccc_vals)   if ccc_vals else 0.0)
        dim_means_r2.append(np.mean(r2_vals)   if r2_vals  else float("nan"))
        dim_stds_r2.append(np.std(r2_vals)     if r2_vals  else 0.0)

    x     = np.arange(len(A_DIMS))
    cols  = ["#4E79A7", "#F28E2B", "#E15759"]
    bars  = ax.bar(x, dim_means_ccc, yerr=dim_stds_ccc, capsize=5,
                   color=cols, alpha=0.85, edgecolor="white",
                   error_kw={"elinewidth": 1.5, "ecolor": "black"})
    for b, m in zip(bars, dim_means_ccc):
        if not math.isnan(m):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=10)
    ax.axhline(BASELINES["affect"]["val_ccc"], color=PALETTE["baseline"],
               ls="--", lw=1.4,
               label=f"Baseline (mean CCC) = {BASELINES['affect']['val_ccc']:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in A_DIMS], fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("CCC")
    ax.set_title("Affect Encoder – Per-Dimension CCC")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # R² bar chart
    ax2 = axes[1]
    bars2 = ax2.bar(x, dim_means_r2, yerr=dim_stds_r2, capsize=5,
                    color=cols, alpha=0.85, edgecolor="white",
                    error_kw={"elinewidth": 1.5, "ecolor": "black"})
    for b, m in zip(bars2, dim_means_r2):
        if not math.isnan(m):
            ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                     f"{m:.3f}", ha="center", va="bottom", fontsize=10)
    ax2.axhline(0, color="gray", ls="--", lw=1.0)
    ax2.set_xticks(x)
    ax2.set_xticklabels([d.capitalize() for d in A_DIMS], fontsize=11)
    ax2.set_ylabel("R²")
    ax2.set_title("Affect Encoder – Per-Dimension R²")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Affect Encoder Results by Dimension (mean ± std across seeds)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _savefig(fig, "fig3_affect_per_dim")


# ── Figure 4: Small-LM Architecture Comparison ────────────────────────────────

def fig_small_lm_comparison(slm_results: List[Dict]) -> Path:
    df = aggregate_by_arch(slm_results, ["final_val_ppl", "best_val_loss", "final_test_ppl"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return _savefig(fig, "fig4_small_lm_comparison")

    archs  = df["arch"].tolist()
    labels = df["arch_label"].tolist()
    colors = [PALETTE.get(a, "#888") for a in archs]
    x      = np.arange(len(archs))

    # PPL
    ax0 = axes[0]
    bars = ax0.bar(x, df["final_val_ppl_mean"], 
                   yerr=df.get("final_val_ppl_std", [0]*len(df)),
                   capsize=5, color=colors, alpha=0.85, edgecolor="white",
                   error_kw={"elinewidth": 1.5, "ecolor": "black"})
    for b, v in zip(bars, df["final_val_ppl_mean"]):
        if not math.isnan(v):
            ax0.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                     f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, fontsize=10, rotation=15, ha="right")
    ax0.set_ylabel("Val Perplexity ↓")
    ax0.set_title("Architecture Comparison – Validation Perplexity")
    ax0.grid(axis="y", alpha=0.3)

    # Param count (log scale)
    ax1 = axes[1]
    params = df.get("num_params", pd.Series([0]*len(df))).fillna(0).tolist()
    bars2 = ax1.bar(x, [p/1e6 for p in params], color=colors, alpha=0.85, edgecolor="white")
    for b, p in zip(bars2, params):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05,
                 f"{p/1e6:.1f}M", ha="center", va="bottom", fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10, rotation=15, ha="right")
    ax1.set_ylabel("Parameters (millions)")
    ax1.set_title("Architecture Comparison – Model Size")
    ax1.grid(axis="y", alpha=0.3)

    fig.suptitle("Small-LM Architecture Benchmark (mean ± std across seeds)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _savefig(fig, "fig4_small_lm_comparison")


# ── Figure 5: Radar Chart (Personality Traits) ────────────────────────────────

def fig_personality_radar(p_results: List[Dict]) -> Path:
    rows = [r for r in p_results if r.get("success")]
    fig  = plt.figure(figsize=(7, 7))
    ax   = fig.add_subplot(111, polar=True)

    N      = len(P_TRAITS)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([t.capitalize() for t in P_TRAITS], fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)

    for r in rows:
        vals = [r.get(f"f1_{t}", 0) or 0 for t in P_TRAITS]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=1.5, color=PALETTE["personality"], alpha=0.7)
        ax.fill(angles, vals, color=PALETTE["personality"], alpha=0.1)

    # Baseline circle
    baseline_val = BASELINES["personality"]["val_f1"]
    bl = [baseline_val] * (N + 1)
    ax.plot(angles, bl, linewidth=1.2, color=PALETTE["baseline"],
            ls="--", label=f"Baseline={baseline_val:.3f}")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=9)
    ax.set_title("Personality Encoder – Trait F1 Radar\n(each line = one seed)",
                 pad=15, fontsize=12)
    return _savefig(fig, "fig5_personality_radar")


# ── Figure 6: Stability Boxplots ──────────────────────────────────────────────

def fig_stability_boxplots(p_results: List[Dict], a_results: List[Dict]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Personality F1 per trait
    ax0 = axes[0]
    data_p = []
    for t in P_TRAITS:
        vals = [r.get(f"f1_{t}", float("nan")) for r in p_results if r.get("success")]
        vals = [v for v in vals if not math.isnan(v)]
        data_p.append(vals)
    bp0 = ax0.boxplot(data_p, patch_artist=True,
                      medianprops={"color": "black", "linewidth": 2})
    for patch in bp0["boxes"]:
        patch.set_facecolor(PALETTE["personality"])
        patch.set_alpha(0.7)
    ax0.set_xticklabels([t[:4].capitalize() for t in P_TRAITS], fontsize=10)
    ax0.set_ylabel("F1 Score")
    ax0.set_title("Personality – Trait-level Stability")
    ax0.axhline(BASELINES["personality"]["val_f1"], ls="--",
                color=PALETTE["baseline"], lw=1.2, label="Baseline")
    ax0.legend(fontsize=9)
    ax0.grid(axis="y", alpha=0.3)

    # Affect CCC per dim
    ax1 = axes[1]
    data_a = []
    for d in A_DIMS:
        vals = [r.get(f"ccc_{d}", float("nan")) for r in a_results if r.get("success")]
        vals = [v for v in vals if not math.isnan(v)]
        data_a.append(vals)
    bp1 = ax1.boxplot(data_a, patch_artist=True,
                      medianprops={"color": "black", "linewidth": 2})
    for patch in bp1["boxes"]:
        patch.set_facecolor(PALETTE["affect"])
        patch.set_alpha(0.7)
    ax1.set_xticklabels([d.capitalize() for d in A_DIMS], fontsize=10)
    ax1.set_ylabel("CCC")
    ax1.set_title("Affect – Dimension-level Stability")
    ax1.axhline(BASELINES["affect"]["val_ccc"], ls="--",
                color=PALETTE["baseline"], lw=1.2, label="Baseline")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    fig.suptitle("Cross-Seed Stability Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _savefig(fig, "fig6_stability_boxplots")


# ── Figure 7: Summary Dashboard ───────────────────────────────────────────────

def fig_summary_dashboard(
    p_results: List[Dict],
    a_results: List[Dict],
    slm_results: List[Dict],
) -> Path:
    fig = plt.figure(figsize=(16, 9))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # ── 7a: Personality overall F1 ──
    ax0 = fig.add_subplot(gs[0, 0])
    vals_p = [r.get("val_f1", float("nan")) for r in p_results if r.get("success")]
    vals_p = [v for v in vals_p if not math.isnan(v)]
    if vals_p:
        ax0.bar(range(len(vals_p)), vals_p, color=PALETTE["personality"], alpha=0.8)
        ax0.axhline(np.mean(vals_p), color="darkblue", ls="-", lw=2,
                    label=f"Mean={np.mean(vals_p):.4f}")
        ax0.axhline(BASELINES["personality"]["val_f1"], ls="--",
                    color=PALETTE["baseline"], lw=1.5, label="Baseline")
        ax0.set_ylim(0.60, 0.75)
        ax0.set_title("Personality – Val F1", fontsize=11)
        ax0.set_xlabel("Seed index")
        ax0.legend(fontsize=8)
    ax0.grid(axis="y", alpha=0.3)

    # ── 7b: Affect overall CCC ──
    ax1 = fig.add_subplot(gs[0, 1])
    vals_a = [r.get("val_ccc", float("nan")) for r in a_results if r.get("success")]
    vals_a = [v for v in vals_a if not math.isnan(v)]
    if vals_a:
        ax1.bar(range(len(vals_a)), vals_a, color=PALETTE["affect"], alpha=0.8)
        ax1.axhline(np.mean(vals_a), color="darkred", ls="-", lw=2,
                    label=f"Mean={np.mean(vals_a):.4f}")
        ax1.axhline(BASELINES["affect"]["val_ccc"], ls="--",
                    color=PALETTE["baseline"], lw=1.5, label="Baseline")
        ax1.set_ylim(0.55, 0.70)
        ax1.set_title("Affect – Val CCC", fontsize=11)
        ax1.set_xlabel("Seed index")
        ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # ── 7c: Small-LM PPL comparison ──
    ax2 = fig.add_subplot(gs[0, 2])
    df_s = aggregate_by_arch(slm_results, ["final_val_ppl"])
    if not df_s.empty:
        colors = [PALETTE.get(a, "#888") for a in df_s["arch"]]
        ax2.barh(df_s["arch_label"], df_s["final_val_ppl_mean"],
                 color=colors, alpha=0.85, edgecolor="white")
        ax2.set_xlabel("Val Perplexity ↓")
        ax2.set_title("Small-LM – Architecture PPL", fontsize=11)
    ax2.grid(axis="x", alpha=0.3)

    # ── 7d: Improvement over baseline ──
    ax3 = fig.add_subplot(gs[1, 0])
    categories = ["Personality\n(F1)", "Affect\n(CCC)"]
    baseline_v = [BASELINES["personality"]["val_f1"], BASELINES["affect"]["val_ccc"]]
    optuna_v   = [
        np.mean(vals_p) if vals_p else float("nan"),
        np.mean(vals_a) if vals_a else float("nan"),
    ]
    x = np.arange(len(categories))
    w = 0.35
    ax3.bar(x - w/2, baseline_v, w, label="Baseline (manual)", color=PALETTE["baseline"], alpha=0.8)
    ax3.bar(x + w/2, optuna_v,   w, label="Optuna best (multi-seed)",
            color=["#B07AA1", "#FF9DA7"], alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories, fontsize=11)
    ax3.set_ylim(0.55, 0.75)
    ax3.set_title("Baseline vs Optuna Improvement")
    ax3.legend(fontsize=8)
    ax3.grid(axis="y", alpha=0.3)

    # ── 7e: Personality per-trait heatmap ──
    ax4 = fig.add_subplot(gs[1, 1])
    rows_p = [r for r in p_results if r.get("success")]
    if rows_p:
        mat = np.array([[r.get(f"f1_{t}", 0) or 0 for t in P_TRAITS] for r in rows_p])
        im  = ax4.imshow(mat, aspect="auto", vmin=0.4, vmax=0.9, cmap="RdYlGn")
        ax4.set_xticks(range(len(P_TRAITS)))
        ax4.set_xticklabels([t[:4].capitalize() for t in P_TRAITS], fontsize=9)
        ax4.set_yticks(range(len(rows_p)))
        ax4.set_yticklabels([f"s{r['seed']}" for r in rows_p], fontsize=9)
        ax4.set_title("Personality F1 per trait × seed", fontsize=11)
        fig.colorbar(im, ax=ax4, fraction=0.046)

    # ── 7f: Affect CCC heatmap ──
    ax5 = fig.add_subplot(gs[1, 2])
    rows_a = [r for r in a_results if r.get("success")]
    if rows_a:
        mat = np.array([[r.get(f"ccc_{d}", 0) or 0 for d in A_DIMS] for r in rows_a])
        im2 = ax5.imshow(mat, aspect="auto", vmin=0.3, vmax=0.9, cmap="RdYlGn")
        ax5.set_xticks(range(len(A_DIMS)))
        ax5.set_xticklabels([d.capitalize() for d in A_DIMS], fontsize=9)
        ax5.set_yticks(range(len(rows_a)))
        ax5.set_yticklabels([f"s{r['seed']}" for r in rows_a], fontsize=9)
        ax5.set_title("Affect CCC per dim × seed", fontsize=11)
        fig.colorbar(im2, ax=ax5, fraction=0.046)

    fig.suptitle("Comprehensive Research Summary Dashboard", fontsize=15, fontweight="bold")
    return _savefig(fig, "fig7_summary_dashboard")


# ══════════════════════════════════════════════════════════════════════════════
# § 6  TABLE GENERATION (CSV + LaTeX + Markdown)
# ══════════════════════════════════════════════════════════════════════════════

def _fmt(v, decimals=4):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "--"
    return f"{v:.{decimals}f}"


def tables_personality(p_results: List[Dict], out_dir: Path) -> str:
    """Return LaTeX string and save CSV/Markdown tables."""
    rows_s  = [r for r in p_results if r.get("success")]
    agg_row = aggregate(p_results, ["val_f1", "val_accuracy"] + [f"f1_{t}" for t in P_TRAITS])

    # Per-seed CSV
    df = pd.DataFrame([{
        "seed":       r["seed"],
        "val_f1":     r.get("val_f1"),
        "val_acc":    r.get("val_accuracy"),
        **{f"f1_{t}": r.get(f"f1_{t}") for t in P_TRAITS},
        "best_epoch": r.get("best_epoch"),
    } for r in rows_s])
    df.to_csv(out_dir / "table_personality_per_seed.csv", index=False)

    # LaTeX
    trait_cols = " & ".join(t[:4].capitalize() for t in P_TRAITS)
    latex = (
        "% Table: Personality Encoder Results\n"
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{Personality Encoder Evaluation (DistilBERT, OCEAN Big Five, "
        f"{len(rows_s)} seeds). Best Optuna hyperparameters applied. "
        "Metric: macro-averaged binary F1.}\n"
        "\\label{tab:personality}\n"
        "\\begin{tabular}{lcccccccc}\n"
        "\\toprule\n"
        f"Seed & Macro F1 & Acc & {trait_cols} & Best Ep. \\\\\n"
        "\\midrule\n"
    )
    for r in rows_s:
        trait_vals = " & ".join(_fmt(r.get(f"f1_{t}"), 3) for t in P_TRAITS)
        latex += (
            f"{r['seed']} & {_fmt(r.get('val_f1'), 4)} & "
            f"{_fmt(r.get('val_accuracy'), 3)} & {trait_vals} & "
            f"{r.get('best_epoch', '--')} \\\\\n"
        )
    if not agg_row.empty:
        m = agg_row.iloc[0]
        trait_m = " & ".join(
            f"{_fmt(m.get(f'f1_{t}_mean'), 3)}$\\pm${_fmt(m.get(f'f1_{t}_std'), 3)}"
            for t in P_TRAITS
        )
        latex += (
            "\\midrule\n"
            f"\\textbf{{Mean±Std}} & "
            f"{_fmt(m.get('val_f1_mean'), 4)}$\\pm${_fmt(m.get('val_f1_std'), 4)} & "
            f"{_fmt(m.get('val_accuracy_mean'), 3)}$\\pm${_fmt(m.get('val_accuracy_std'), 3)} & "
            f"{trait_m} & -- \\\\\n"
            f"Baseline & {BASELINES['personality']['val_f1']:.4f} & -- "
            f"& -- & -- & -- & -- & -- & -- \\\\\n"
        )
    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    (out_dir / "table_personality.tex").write_text(latex)
    return latex


def tables_affect(a_results: List[Dict], out_dir: Path) -> str:
    rows_s = [r for r in a_results if r.get("success")]
    agg    = aggregate(a_results, ["val_ccc", "val_mse", "val_r2"] +
                       [f"ccc_{d}" for d in A_DIMS] + [f"r2_{d}" for d in A_DIMS])

    df = pd.DataFrame([{
        "seed":      r["seed"],
        "val_ccc":   r.get("val_ccc"),
        "val_mse":   r.get("val_mse"),
        "val_r2":    r.get("val_r2"),
        **{f"ccc_{d}": r.get(f"ccc_{d}") for d in A_DIMS},
        **{f"r2_{d}":  r.get(f"r2_{d}")  for d in A_DIMS},
        "best_epoch": r.get("best_epoch"),
    } for r in rows_s])
    df.to_csv(out_dir / "table_affect_per_seed.csv", index=False)

    dim_cols = " & ".join(f"CCC\\textsubscript{{{d[:3].capitalize()}}}" for d in A_DIMS)
    latex = (
        "% Table: Affect Encoder Results\n"
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{Affect Encoder Evaluation (DistilBERT, VAD regression, "
        f"{len(rows_s)} seeds). Best Optuna hyperparameters. "
        "Primary metric: mean Concordance Correlation Coefficient (CCC).}\n"
        "\\label{tab:affect}\n"
        "\\begin{tabular}{lcccccc}\n"
        "\\toprule\n"
        f"Seed & Mean CCC & MSE & R² & {dim_cols} & Best Ep. \\\\\n"
        "\\midrule\n"
    )
    for r in rows_s:
        dim_vals = " & ".join(_fmt(r.get(f"ccc_{d}"), 3) for d in A_DIMS)
        latex += (
            f"{r['seed']} & {_fmt(r.get('val_ccc'), 4)} & "
            f"{_fmt(r.get('val_mse'), 5)} & {_fmt(r.get('val_r2'), 3)} & "
            f"{dim_vals} & {r.get('best_epoch', '--')} \\\\\n"
        )
    if not agg.empty:
        m = agg.iloc[0]
        dim_m = " & ".join(
            f"{_fmt(m.get(f'ccc_{d}_mean'), 3)}$\\pm${_fmt(m.get(f'ccc_{d}_std'), 3)}"
            for d in A_DIMS
        )
        latex += (
            "\\midrule\n"
            f"\\textbf{{Mean±Std}} & "
            f"{_fmt(m.get('val_ccc_mean'), 4)}$\\pm${_fmt(m.get('val_ccc_std'), 4)} & "
            f"{_fmt(m.get('val_mse_mean'), 5)} & {_fmt(m.get('val_r2_mean'), 3)} & "
            f"{dim_m} & -- \\\\\n"
            f"Baseline & {BASELINES['affect']['val_ccc']:.4f} & -- & -- & -- & -- & -- & -- \\\\\n"
        )
    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    (out_dir / "table_affect.tex").write_text(latex)
    return latex


def tables_small_lm(slm_results: List[Dict], out_dir: Path) -> str:
    rows_s = [r for r in slm_results if r.get("success")]
    df_agg = aggregate_by_arch(slm_results, ["final_val_ppl", "final_test_ppl", "best_val_loss"])
    df_agg.to_csv(out_dir / "table_small_lm.csv", index=False)

    latex = (
        "% Table: Small-LM Architecture Comparison\n"
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{Small Language Model Architecture Comparison on NPC Dialogue Corpus. "
        "All models trained with the same data split and optimiser settings. "
        "Metric: token-level perplexity (↓ better).}\n"
        "\\label{tab:smalllm}\n"
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        "Architecture & \\#Params & Val PPL & Test PPL & Best Val Loss \\\\\n"
        "\\midrule\n"
    )
    for _, row in df_agg.iterrows():
        params_m = int(row.get("num_params", 0)) // 1_000_000
        latex += (
            f"{row['arch_label']} & {params_m}M & "
            f"{_fmt(row.get('final_val_ppl_mean'), 2)}$\\pm${_fmt(row.get('final_val_ppl_std', 0), 2)} & "
            f"{_fmt(row.get('final_test_ppl_mean'), 2)}$\\pm${_fmt(row.get('final_test_ppl_std', 0), 2)} & "
            f"{_fmt(row.get('best_val_loss_mean'), 4)} \\\\\n"
        )
    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    (out_dir / "table_small_lm.tex").write_text(latex)
    return latex


def tables_metrics_overview(p_results, a_results, out_dir: Path) -> str:
    """Measurement methodology table for the paper's Methods section."""
    latex = (
        "% Table: Metrics Overview\n"
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{Evaluation Metrics Used in This Study}\n"
        "\\label{tab:metrics}\n"
        "\\begin{tabular}{llll}\n"
        "\\toprule\n"
        "Task & Metric & Formula & Notes \\\\\n"
        "\\midrule\n"
        "Personality & Macro F1 & $\\frac{1}{5}\\sum_i F1_i$ & Binary per-trait, threshold=0.5 \\\\\n"
        "Personality & Accuracy & $\\frac{\\text{TP}+\\text{TN}}{N}$ & Per-trait, averaged \\\\\n"
        "Personality & Loss & Focal BCE & $\\gamma$=2.39, from Optuna \\\\\n"
        "Affect & CCC & $\\frac{2\\rho\\sigma_x\\sigma_y}{\\sigma_x^2+\\sigma_y^2+(\\mu_x-\\mu_y)^2}$ "
        "& Concordance Correlation Coefficient \\\\\n"
        "Affect & MSE & $\\frac{1}{n}\\sum(\\hat{y}-y)^2$ & Mean Squared Error \\\\\n"
        "Affect & R² & $1 - \\frac{SS_{res}}{SS_{tot}}$ & Coefficient of Determination \\\\\n"
        "Affect & Loss & $\\alpha\\cdot(1-\\text{CCC}) + (1-\\alpha)\\cdot\\text{MSE}$ "
        "& $\\alpha$=0.543, from Optuna \\\\\n"
        "Dialogue & Perplexity & $\\exp(-\\frac{1}{T}\\sum\\log p_\\theta)$ & Next-token PPL \\\\\n"
        "Dialogue & Val Loss & Cross-entropy & Token-level \\\\\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    (out_dir / "table_metrics_overview.tex").write_text(latex)
    return latex


# ══════════════════════════════════════════════════════════════════════════════
# § 7  HTML REPORT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_html_report(
    p_results: List[Dict],
    a_results: List[Dict],
    slm_results: List[Dict],
    fig_paths: Dict[str, Path],
    out_dir: Path,
    report_dir: Path,
) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M UTC")

    def _img(key: str) -> str:
        p = fig_paths.get(key)
        if p and p.exists():
            rel = os.path.relpath(p, report_dir)
            return f'<img src="{rel}" style="max-width:100%;border-radius:6px;">'
        return "<em>(figure not available)</em>"

    def _metric_row(label, value, baseline=None, higher=True):
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return f"<tr><td>{label}</td><td>--</td><td>--</td><td>--</td></tr>"
        change = ""
        badge  = ""
        if baseline is not None and not math.isnan(baseline):
            delta  = value - baseline
            pct    = delta / baseline * 100
            arrow  = "▲" if delta > 0 else "▼"
            colour = "green" if (delta > 0) == higher else "red"
            change = f'<span style="color:{colour}">{arrow} {abs(delta)*100:.2f}pp ({pct:+.1f}%)</span>'
            badge  = "✅" if (delta > 0) == higher else "⚠️"
        return (f"<tr><td>{label}</td><td><strong>{value:.4f}</strong></td>"
                f"<td>{baseline:.4f if baseline else '--'}</td>"
                f"<td>{badge} {change}</td></tr>")

    # Aggregate
    p_rows  = [r for r in p_results if r.get("success")]
    a_rows  = [r for r in a_results if r.get("success")]
    p_mean  = float(np.mean([r.get("val_f1", float("nan"))  for r in p_rows])) if p_rows else float("nan")
    a_mean  = float(np.mean([r.get("val_ccc", float("nan")) for r in a_rows])) if a_rows else float("nan")
    p_std   = float(np.std([r.get("val_f1", float("nan"))   for r in p_rows])) if len(p_rows)>1 else 0.0
    a_std   = float(np.std([r.get("val_ccc", float("nan"))  for r in a_rows])) if len(a_rows)>1 else 0.0

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NPC Encoder Research Report</title>
<style>
  body  {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 40px; color: #222; }}
  h1    {{ color: #1a3a5c; border-bottom: 3px solid #1a3a5c; padding-bottom: 8px; }}
  h2    {{ color: #2c5f8a; margin-top: 40px; }}
  h3    {{ color: #3a7ab8; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; }}
  th    {{ background: #1a3a5c; color: white; padding: 8px 12px; text-align: left; }}
  td    {{ padding: 7px 12px; border-bottom: 1px solid #ddd; }}
  tr:nth-child(even) {{ background: #f5f8fc; }}
  .card {{ background: #f0f4f8; border-radius: 10px; padding: 20px; margin: 20px 0; }}
  .metric-highlight {{ font-size: 2em; font-weight: bold; color: #1a3a5c; }}
  .improve {{ color: green; font-weight: bold; }}
  .warn    {{ color: orange; }}
  .fig-row {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 24px 0; }}
  .fig-box {{ flex: 1; min-width: 300px; }}
  code  {{ background: #eee; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
  pre   {{ background: #1a1a2e; color: #e0e0e0; padding: 16px; border-radius: 8px;
           overflow-x: auto; font-size: 0.85em; }}
</style>
</head>
<body>
<h1>🔬 NPC Encoder Research Report</h1>
<p><strong>Generated:</strong> {timestamp} &nbsp;|&nbsp;
   <strong>Seeds per model:</strong> {len(p_rows) or '?'} (personality), {len(a_rows) or '?'} (affect)</p>

<div class="card">
  <h2 style="margin-top:0">📊 Executive Summary</h2>
  <table>
    <tr><th>Model</th><th>Best Metric</th><th>Baseline</th><th>Change</th></tr>
    {_metric_row("Personality Encoder (macro F1)", p_mean, BASELINES['personality']['val_f1'])}
    {_metric_row("Affect Encoder (mean CCC)",      a_mean, BASELINES['affect']['val_ccc'])}
  </table>
  <p>Personality: <span class="metric-highlight">{p_mean:.4f}</span> ± {p_std:.4f} &nbsp;|&nbsp;
     Affect: <span class="metric-highlight">{a_mean:.4f}</span> ± {a_std:.4f}</p>
</div>

<h2>1. Methodology</h2>
<p>All encoders are fine-tuned from <code>distilbert-base-uncased</code> with
hyperparameters optimised via Optuna TPE (20 trials each task).
To confirm results are not local minima, we re-train with {len(p_rows)} independent
random seeds and report mean ± standard deviation.</p>

<h3>1.1 Evaluation Metrics</h3>
<table>
<tr><th>Task</th><th>Primary Metric</th><th>Secondary Metrics</th><th>Loss Function</th></tr>
<tr><td>Personality (OCEAN)</td><td>Macro F1 (binary per-trait)</td><td>Accuracy per trait</td>
    <td>Focal BCE (γ={load_optuna_best("personality").get("focal_gamma",2.39):.2f})</td></tr>
<tr><td>Affect (VAD)</td><td>Mean CCC (valence/arousal/dominance)</td><td>MSE, R²</td>
    <td>α·(1-CCC)+(1-α)·MSE, α={load_optuna_best("affect").get("ccc_weight",0.543):.3f}</td></tr>
<tr><td>Small-LM Dialogue</td><td>Validation Perplexity</td><td>Test PPL, Val Loss</td>
    <td>Cross-Entropy (next-token)</td></tr>
</table>

<h2>2. Personality Encoder Results</h2>
<div class="fig-row">
  <div class="fig-box">{_img("fig2")}</div>
  <div class="fig-box">{_img("fig5")}</div>
</div>

<h3>Per-Trait F1 Breakdown</h3>
<table>
<tr><th>Seed</th><th>Macro F1</th>{''.join(f"<th>{t[:4].capitalize()}</th>" for t in P_TRAITS)}<th>Best Epoch</th></tr>
{"".join(
    f"<tr><td>{r['seed']}</td><td>{_fmt(r.get('val_f1'),4)}</td>"
    + "".join(f"<td>{_fmt(r.get(f'f1_{t}'),3)}</td>" for t in P_TRAITS)
    + f"<td>{r.get('best_epoch','--')}</td></tr>"
    for r in p_rows
)}
<tr style="font-weight:bold; background:#e8f0fe">
  <td>Mean±Std</td>
  <td>{p_mean:.4f}±{p_std:.4f}</td>
  {"".join(
    f"<td>{np.mean([r.get(f'f1_{t}',float('nan')) for r in p_rows if not math.isnan(r.get(f'f1_{t}',float('nan')))]):.3f}</td>"
    for t in P_TRAITS
  )}
  <td>--</td>
</tr>
<tr style="color:#888"><td>Baseline</td><td>{BASELINES['personality']['val_f1']:.4f}</td>
{'<td>--</td>'*len(P_TRAITS)}<td>--</td></tr>
</table>

<h2>3. Affect Encoder Results</h2>
<div class="fig-row">
  <div class="fig-box">{_img("fig3")}</div>
</div>

<h3>Per-Dimension Metrics</h3>
<table>
<tr><th>Seed</th><th>Mean CCC</th><th>MSE</th><th>R²</th>
{''.join(f"<th>CCC {d[:3].capitalize()}</th>" for d in A_DIMS)}<th>Best Epoch</th></tr>
{"".join(
    f"<tr><td>{r['seed']}</td><td>{_fmt(r.get('val_ccc'),4)}</td>"
    f"<td>{_fmt(r.get('val_mse'),5)}</td><td>{_fmt(r.get('val_r2'),3)}</td>"
    + "".join(f"<td>{_fmt(r.get(f'ccc_{d}'),3)}</td>" for d in A_DIMS)
    + f"<td>{r.get('best_epoch','--')}</td></tr>"
    for r in a_rows
)}
<tr style="font-weight:bold; background:#e8f0fe">
  <td>Mean±Std</td><td>{a_mean:.4f}±{a_std:.4f}</td><td>--</td><td>--</td>
  {"".join(
    f"<td>{np.mean([r.get(f'ccc_{d}',float('nan')) for r in a_rows if not math.isnan(r.get(f'ccc_{d}',float('nan')))]):.3f}</td>"
    for d in A_DIMS
  )}
  <td>--</td>
</tr>
<tr style="color:#888"><td>Baseline</td><td>{BASELINES['affect']['val_ccc']:.4f}</td>
<td>--</td><td>--</td>{'<td>--</td>'*len(A_DIMS)}<td>--</td></tr>
</table>

<h2>4. Small-LM Architecture Benchmark</h2>
<div class="fig-row">
  <div class="fig-box">{_img("fig4")}</div>
</div>

<h3>Architecture Comparison Table</h3>
<table>
<tr><th>Architecture</th><th>#Params</th><th>Val PPL (mean±std)</th><th>Test PPL</th><th>Best Val Loss</th></tr>
{"".join(
    f"<tr><td>{ARCH_LABELS.get(r.get('arch',''),r.get('arch',''))}</td>"
    f"<td>{int(r.get('num_params',0))//1_000_000}M</td>"
    f"<td>{_fmt(r.get('final_val_ppl'),2)}</td>"
    f"<td>{_fmt(r.get('final_test_ppl'),2)}</td>"
    f"<td>{_fmt(r.get('best_val_loss'),4)}</td></tr>"
    for r in slm_results if r.get("success")
)}
</table>

<h2>5. Training Dynamics</h2>
<div class="fig-row">
  <div class="fig-box">{_img("fig1")}</div>
</div>

<h2>6. Cross-Seed Stability Analysis</h2>
<p>To verify results are not local minima, each configuration was trained with
multiple independent random seeds. Low standard deviation across seeds indicates
stable, reproducible results.</p>
<div class="fig-row">
  <div class="fig-box">{_img("fig6")}</div>
  <div class="fig-box">{_img("fig7")}</div>
</div>

<h2>7. Hyperparameter Configuration (Optuna-Optimised)</h2>
<table>
<tr><th>Parameter</th><th>Personality</th><th>Affect</th></tr>
{"".join(
    f"<tr><td><code>{k}</code></td>"
    f"<td>{load_optuna_best('personality').get(k,'--')}</td>"
    f"<td>{load_optuna_best('affect').get(k,'--')}</td></tr>"
    for k in ['lr','encoder_lr_factor','dropout','freeze_encoder_epochs']
)}
<tr><td><code>focal_gamma</code></td><td>{load_optuna_best('personality').get('focal_gamma','--'):.3f}</td><td>N/A</td></tr>
<tr><td><code>ccc_weight</code></td><td>N/A</td><td>{load_optuna_best('affect').get('ccc_weight','--'):.3f}</td></tr>
<tr><td><code>token_drop_prob</code></td><td>{load_optuna_best('personality').get('token_drop_prob','--'):.3f}</td><td>N/A</td></tr>
<tr><td><code>grad_accum</code></td><td>2</td><td>{load_optuna_best('affect').get('grad_accum',2)}</td></tr>
</table>

<h2>8. LaTeX Assets</h2>
<p>Ready-to-paste LaTeX tables are in the <code>report/tables/</code> directory:</p>
<ul>
  <li><code>table_personality.tex</code></li>
  <li><code>table_affect.tex</code></li>
  <li><code>table_small_lm.tex</code></li>
  <li><code>table_metrics_overview.tex</code></li>
</ul>
<p>Figures (PNG + PDF) are in <code>report/figures/</code>.</p>

<hr>
<p style="color:#888; font-size:0.85em">Generated by
<code>scripts/comprehensive_training_report.py</code> &nbsp;·&nbsp; {timestamp}</p>
</body>
</html>
"""

    report_path = report_dir / "research_report.html"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html, encoding="utf-8")
    log.info(f"HTML report → {report_path}")
    return report_path


# ══════════════════════════════════════════════════════════════════════════════
# § 8  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global FIG_DIR

    ap = argparse.ArgumentParser(description="Comprehensive training & research report")
    ap.add_argument("--phase", choices=["personality", "affect", "small-lm",
                                         "report-only", "all"], default="all")
    ap.add_argument("--n-seeds",   type=int,   default=3,
                    help="Number of random seeds per encoder model")
    ap.add_argument("--p-epochs",  type=int,   default=20,
                    help="Epochs for personality encoder (default 20 to escape local minima)")
    ap.add_argument("--a-epochs",  type=int,   default=20,
                    help="Epochs for affect encoder")
    ap.add_argument("--slm-epochs",type=int,   default=15,
                    help="Epochs per small-LM architecture")
    ap.add_argument("--slm-seeds", type=int,   default=2,
                    help="Seeds for small-LM benchmark (fewer to save time)")
    ap.add_argument("--quick",     action="store_true",
                    help="Smoke-test mode: 1 seed, 5 epochs, 1 arch each")
    ap.add_argument("--out-dir",   type=str,   default="artifacts/report",
                    help="Root output directory")
    # Embedding model for A/B testing semantic conditioning
    ap.add_argument("--slm-embedding-model", type=str, default=None,
                    help="Embedding model for semantic conditioning (e.g., Qwen/Qwen3-Embedding-4B). "
                         "If not set, uses zero conditioning (baseline).")
    args = ap.parse_args()

    if args.quick:
        args.n_seeds    = 1
        args.p_epochs   = 5
        args.a_epochs   = 5
        args.slm_epochs = 3
        args.slm_seeds  = 1

    out_dir    = ROOT / args.out_dir
    report_dir = out_dir / "report"
    FIG_DIR    = report_dir / "figures"
    table_dir  = report_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("  COMPREHENSIVE NPC ENCODER RESEARCH REPORT")
    log.info(f"  Phase: {args.phase}  | Seeds: {args.n_seeds}")
    log.info(f"  Output: {out_dir}")
    log.info("=" * 70)

    # ── MLflow: parent experiment for the full report run ────────────────────
    sys.path.insert(0, str(SRC / "train"))
    from mlflow_tracker import MLflowTracker
    report_tracker = MLflowTracker(
        experiment="comprehensive_report",
        enabled=True,
    )
    report_tracker.start_run(
        run_name=f"report_{datetime.now():%Y%m%d_%H%M%S}",
        tags={"phase": args.phase, "n_seeds": str(args.n_seeds), "quick": str(args.quick)},
    )
    report_tracker.log_params({
        "phase": args.phase, "n_seeds": args.n_seeds,
        "p_epochs": args.p_epochs, "a_epochs": args.a_epochs,
        "slm_epochs": args.slm_epochs, "slm_seeds": args.slm_seeds,
        "quick": args.quick, "out_dir": args.out_dir,
        "slm_embedding_model": str(args.slm_embedding_model),
    })

    # ── Load any prior results (re-run report-only without re-training) ──
    p_results  = _load_results(out_dir / "personality_runs.json")
    a_results  = _load_results(out_dir / "affect_runs.json")
    slm_results = _load_results(out_dir / "small_lm_runs.json")

    # ── Phase: Personality ──
    if args.phase in ("personality", "all"):
        log.info("\n── PHASE 1: PERSONALITY ENCODER ─────────────────────────────────")
        p_results = run_personality_multiseed(args.n_seeds, args.p_epochs, out_dir)

    # ── Phase: Affect ──
    if args.phase in ("affect", "all"):
        log.info("\n── PHASE 2: AFFECT ENCODER ──────────────────────────────────────")
        a_results = run_affect_multiseed(args.n_seeds, args.a_epochs, out_dir)

    # ── Phase: Small-LM ──
    if args.phase in ("small-lm", "all"):
        log.info("\n── PHASE 3: SMALL-LM ARCHITECTURE BENCHMARK ─────────────────────")
        slm_results = run_small_lm_benchmark(
            args.slm_seeds, args.slm_epochs, out_dir,
            embedding_model=args.slm_embedding_model
        )

    # ── Report ──
    log.info("\n── PHASE 4: GENERATING REPORT ───────────────────────────────────────")

    # Figures
    fig_paths: Dict[str, Path] = {}
    try:
        fig_paths["fig1"] = fig_training_curves(p_results, a_results, slm_results)
        fig_paths["fig2"] = fig_personality_per_trait(p_results)
        fig_paths["fig3"] = fig_affect_per_dim(a_results)
        fig_paths["fig4"] = fig_small_lm_comparison(slm_results)
        fig_paths["fig5"] = fig_personality_radar(p_results)
        fig_paths["fig6"] = fig_stability_boxplots(p_results, a_results)
        fig_paths["fig7"] = fig_summary_dashboard(p_results, a_results, slm_results)
    except Exception as e:
        log.warning(f"Figure generation error: {e}")

    # LaTeX tables
    try:
        tables_personality(p_results, table_dir)
        tables_affect(a_results, table_dir)
        tables_small_lm(slm_results, table_dir)
        tables_metrics_overview(p_results, a_results, table_dir)
        log.info(f"LaTeX tables → {table_dir}")
    except Exception as e:
        log.warning(f"Table generation error: {e}")

    # HTML report
    html_path = generate_html_report(
        p_results, a_results, slm_results, fig_paths, out_dir, report_dir
    )

    # ── MLflow: log report artifacts and best metrics ─────────────────────
    best_p_f1 = max((r.get("val_f1", 0) for r in p_results if r.get("success")), default=0)
    best_a_ccc = max((r.get("val_ccc", 0) for r in a_results if r.get("success")), default=0)
    best_slm_ppl = min((r.get("best_val_ppl", 999) for r in slm_results if r.get("success")), default=999)
    report_tracker.log_metrics({
        "best_personality_f1": best_p_f1,
        "best_affect_ccc": best_a_ccc,
        "best_slm_ppl": best_slm_ppl,
        "num_personality_runs": len(p_results),
        "num_affect_runs": len(a_results),
        "num_slm_runs": len(slm_results),
    })
    if html_path and Path(html_path).exists():
        report_tracker.log_artifact(html_path)
    report_tracker.log_artifacts(str(table_dir))
    for fig_p in fig_paths.values():
        report_tracker.log_artifact(fig_p)
    report_tracker.end_run()

    log.info("\n" + "=" * 70)
    log.info("  DONE")
    log.info(f"  HTML Report  : {html_path}")
    log.info(f"  Figures      : {FIG_DIR}")
    log.info(f"  LaTeX Tables : {table_dir}")
    log.info(f"  Raw JSON     : {out_dir}/*.json")
    log.info(f"  MLflow UI    : mlflow ui --backend-store-uri {ROOT / 'mlruns'}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
