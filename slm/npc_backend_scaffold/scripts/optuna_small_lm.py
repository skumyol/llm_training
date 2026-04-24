#!/usr/bin/env python3
"""
optuna_small_lm.py
==================
Per-architecture Optuna hyperparameter search for all 6 Small-LM architectures.

Each architecture gets its own Optuna study with a tailored search space that
reflects the fundamentally different training dynamics of RNNs vs Transformers.

Usage:
  python scripts/optuna_small_lm.py --arch gru       --n-trials 20 --epochs 5
  python scripts/optuna_small_lm.py --arch awdlstm   --n-trials 20 --epochs 5
  python scripts/optuna_small_lm.py --arch all       --n-trials 20 --epochs 5
  python scripts/optuna_small_lm.py --arch all       --n-trials 30 --epochs 7 --jobs 1

Outputs per architecture:
  artifacts/optuna/small_lm_{arch}_best.json   — best params for final training
  artifacts/optuna/small_lm_{arch}_study.pkl   — full Optuna study (for analysis)
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT   = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
log    = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

ARCHS = ["gru", "awdlstm", "gpt", "prefix_gpt", "moe", "mamba_like"]

# ── Shared training defaults (non-arch-specific) ───────────────────────────────
SHARED_DEFAULTS = {
    "hardware_profile": "rtx4070_small",
    "train_text":       str(ROOT / "data" / "dialogue" / "train.txt"),
    "val_text":         str(ROOT / "data" / "dialogue" / "val.txt"),
    "cond_dim":         8,
    "use_amp":          False,
    "embedding_model":  None,
    "embedding_cache":  True,
    "log_every":        50,
    "eval_every_steps": 9999,   # disable mid-epoch eval; use end-of-epoch only
    "mlflow_enabled":   False,  # disable MLflow during search to reduce overhead
    "scheduler":        "none", # scheduler is a separate search dimension
}

# ── Architecture-specific search spaces ───────────────────────────────────────

def suggest_gru(trial) -> Dict[str, Any]:
    """
    GRU failure root cause: lr=3e-4 + seq_len=256 → vanishing BPTT gradients.
    Key axes: lr (lower range), seq_len (shorter), model size.
    """
    import optuna
    lr           = trial.suggest_float("lr",          5e-5,  5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 0.3,  log=True)
    seq_len      = trial.suggest_categorical("seq_len",   [64, 128])
    batch_size   = trial.suggest_categorical("batch_size", [16, 32])
    grad_accum   = trial.suggest_categorical("grad_accum", [1, 2, 4])
    dropout      = trial.suggest_float("dropout", 0.1, 0.5)
    hidden_size  = trial.suggest_categorical("hidden_size", [256, 512])
    num_layers   = trial.suggest_categorical("num_layers",  [2, 3])
    embed_dim    = trial.suggest_categorical("embed_dim",   [128, 256])

    training = {
        "lr":           lr,
        "weight_decay": weight_decay,
        "seq_len":      seq_len,
        "batch_size":   batch_size,
        "grad_accum":   grad_accum,
    }
    arch_params = {
        "dropout":      dropout,
        "hidden_size":  hidden_size,
        "num_layers":   num_layers,
        "embed_dim":    embed_dim,
    }
    return training, arch_params


def suggest_awdlstm(trial) -> Dict[str, Any]:
    """
    AWD-LSTM: original paper uses bptt=70, SGD lr=30. We use Adam but must
    drastically lower lr and shorten seq_len vs the transformer default.
    DropConnect (wdrop), embedding drop, hidden drop are key regularisers.
    """
    lr           = trial.suggest_float("lr",           5e-5,  5e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay",  1e-6, 0.1,  log=True)
    seq_len      = trial.suggest_categorical("seq_len",    [64, 70, 128])
    batch_size   = trial.suggest_categorical("batch_size", [16, 32])
    grad_accum   = trial.suggest_categorical("grad_accum", [1, 2])
    wdrop        = trial.suggest_float("wdrop",    0.2, 0.7)
    dropout      = trial.suggest_float("dropout",  0.2, 0.6)
    dropouth     = trial.suggest_float("dropouth", 0.1, 0.4)
    dropouti     = trial.suggest_float("dropouti", 0.3, 0.7)
    hidden_size  = trial.suggest_categorical("hidden_size", [512, 800])
    embed_dim    = trial.suggest_categorical("embed_dim",   [256, 400])

    training = {
        "lr":           lr,
        "weight_decay": weight_decay,
        "seq_len":      seq_len,
        "batch_size":   batch_size,
        "grad_accum":   grad_accum,
    }
    arch_params = {
        "wdrop":        wdrop,
        "dropout":      dropout,
        "dropouth":     dropouth,
        "dropouti":     dropouti,
        "hidden_size":  hidden_size,
        "embed_dim":    embed_dim,
    }
    return training, arch_params


def suggest_gpt(trial) -> Dict[str, Any]:
    """
    GPT already works (PPL ~40) but is over-parameterised (51M on 545K tokens).
    Shrink model + tune lr/dropout to reduce overfitting.
    """
    lr           = trial.suggest_float("lr",           5e-5,  3e-3,  log=True)
    weight_decay = trial.suggest_float("weight_decay",  1e-4,  0.5,  log=True)
    warmup_ratio = trial.suggest_float("warmup_ratio",  0.02,  0.15)
    seq_len      = trial.suggest_categorical("seq_len",    [128, 256])
    batch_size   = trial.suggest_categorical("batch_size", [16, 32])
    grad_accum   = trial.suggest_categorical("grad_accum", [2, 4])
    dropout      = trial.suggest_float("dropout", 0.0, 0.3)
    n_layer      = trial.suggest_categorical("n_layer",  [4, 6])
    n_embd       = trial.suggest_categorical("n_embd",   [128, 256])
    n_head       = trial.suggest_categorical("n_head",   [4, 8])

    training = {
        "lr":           lr,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "seq_len":      seq_len,
        "batch_size":   batch_size,
        "grad_accum":   grad_accum,
    }
    arch_params = {
        "dropout":  dropout,
        "n_layer":  n_layer,
        "n_embd":   n_embd,
        "n_head":   n_head,
    }
    return training, arch_params


def suggest_prefix_gpt(trial) -> Dict[str, Any]:
    """PrefixGPT same as GPT + prefix_length."""
    training, arch_params = suggest_gpt(trial)
    arch_params["prefix_length"] = trial.suggest_categorical("prefix_length", [4, 8, 16])
    return training, arch_params


def suggest_moe(trial) -> Dict[str, Any]:
    """
    MoE: over-parameterised + potential expert collapse.
    Key axes: aux_loss weight for load balancing, n_experts, top_k, model size.
    """
    lr             = trial.suggest_float("lr",             5e-5, 1e-3,  log=True)
    weight_decay   = trial.suggest_float("weight_decay",   1e-4, 0.3,   log=True)
    warmup_ratio   = trial.suggest_float("warmup_ratio",   0.02, 0.15)
    seq_len        = trial.suggest_categorical("seq_len",    [128, 256])
    batch_size     = trial.suggest_categorical("batch_size", [16, 32])
    grad_accum     = trial.suggest_categorical("grad_accum", [2, 4])
    dropout        = trial.suggest_float("dropout", 0.0, 0.2)
    n_layer        = trial.suggest_categorical("n_layer",    [4, 6])
    n_embd         = trial.suggest_categorical("n_embd",     [128, 256])
    n_head         = trial.suggest_categorical("n_head",     [4, 8])
    n_experts      = trial.suggest_categorical("n_experts",  [4, 8])
    top_k          = trial.suggest_categorical("top_k",      [1, 2])

    training = {
        "lr":           lr,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "seq_len":      seq_len,
        "batch_size":   batch_size,
        "grad_accum":   grad_accum,
    }
    arch_params = {
        "dropout":    dropout,
        "n_layer":    n_layer,
        "n_embd":     n_embd,
        "n_head":     n_head,
        "num_experts": n_experts,
        "top_k":      top_k,
    }
    return training, arch_params


def suggest_mamba_like(trial) -> Dict[str, Any]:
    """
    Mamba-like SSM: sequential scan benefits from longer context but we only
    have short dialogues. Key axes: lr, d_state, expand, n_layer.
    """
    lr           = trial.suggest_float("lr",           5e-5,  3e-3,  log=True)
    weight_decay = trial.suggest_float("weight_decay",  1e-4,  0.3,  log=True)
    warmup_ratio = trial.suggest_float("warmup_ratio",  0.02,  0.15)
    # NOTE: seq_len capped at 64 — the pure-Python SSM scan is O(seq_len)
    # so seq_len=128 already takes ~10× longer than a transformer.
    seq_len      = trial.suggest_categorical("seq_len",    [32, 64])
    batch_size   = trial.suggest_categorical("batch_size", [16, 32])
    grad_accum   = trial.suggest_categorical("grad_accum", [2, 4])
    dropout      = trial.suggest_float("dropout", 0.0, 0.2)
    n_layer      = trial.suggest_categorical("n_layer",  [4, 6, 8])
    n_embd       = trial.suggest_categorical("n_embd",   [128, 256])
    d_state      = trial.suggest_categorical("d_state",  [8, 16])
    expand       = trial.suggest_categorical("expand",   [2, 4])

    training = {
        "lr":           lr,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "seq_len":      seq_len,
        "batch_size":   batch_size,
        "grad_accum":   grad_accum,
    }
    arch_params = {
        "dropout": dropout,
        "n_layer": n_layer,
        "n_embd":  n_embd,
        "d_state": d_state,
        "expand":  expand,
    }
    return training, arch_params


SUGGESTERS = {
    "gru":        suggest_gru,
    "awdlstm":    suggest_awdlstm,
    "gpt":        suggest_gpt,
    "prefix_gpt": suggest_prefix_gpt,
    "moe":        suggest_moe,
    "mamba_like": suggest_mamba_like,
}

# ── Trial runner ───────────────────────────────────────────────────────────────

def run_trial(arch: str, training_params: Dict, arch_params: Dict,
              epochs: int, trial_id: str, timeout: int = 1800,
              train_text: str = None, val_text: str = None) -> float:
    """Run a single training trial, return best val_ppl (lower is better)."""
    run_id     = f"optuna_{arch}_{trial_id}_{datetime.now():%H%M%S}"
    cfg_path   = ROOT / "artifacts" / "optuna" / "trials" / f"{run_id}.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = {
        **SHARED_DEFAULTS,
        **training_params,
        "arch_params": arch_params,
        "epochs":      epochs,
        "seed":        42,
        "output_dir":  "artifacts/small_lm",
        "warmup_ratio": training_params.get("warmup_ratio", 0.05),
    }
    # Override data paths if custom ones provided
    if train_text:
        cfg["train_text"] = train_text
    if val_text:
        cfg["val_text"] = val_text
    
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    cmd = [
        PYTHON, "-m", "src.train.run_small_lm",
        "--config", str(cfg_path),
        "--run-id", run_id,
        "--arch",   arch,
    ]
    log.info(f"  Trial {trial_id}: {arch}  lr={training_params.get('lr', '?'):.2e}  "
             f"seq_len={training_params.get('seq_len', '?')}  "
             f"arch_params={arch_params}")

    t0 = time.time()
    result = subprocess.run(
        cmd, cwd=str(ROOT),
        capture_output=True, text=True, timeout=timeout,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        log.warning(f"  Trial {trial_id} failed (rc={result.returncode}) in {elapsed:.0f}s")
        log.debug(f"  STDERR: {result.stderr[-500:]}")
        return float("inf")

    summary_path = ROOT / "artifacts" / "small_lm" / run_id / "run_summary.json"
    if not summary_path.exists():
        log.warning(f"  Trial {trial_id}: no run_summary.json found")
        return float("inf")

    with open(summary_path) as f:
        data = json.load(f)

    best_ppl = data.get("best", {}).get("val_ppl", float("inf"))
    log.info(f"  Trial {trial_id} done in {elapsed:.0f}s → best_val_ppl={best_ppl:.2f}")
    return best_ppl


# ── Per-arch Optuna study ──────────────────────────────────────────────────────

def run_study(arch: str, n_trials: int, epochs: int, jobs: int,
              train_text: str = None, val_text: str = None) -> Dict[str, Any]:
    try:
        import optuna
    except ImportError:
        log.error("optuna not installed — run: pip install optuna")
        sys.exit(1)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study_name = f"small_lm_{arch}_{datetime.now():%Y%m%d_%H%M%S}"
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )

    suggester = SUGGESTERS[arch]
    trial_counter = [0]

    def objective(trial) -> float:
        n = trial_counter[0]
        trial_counter[0] += 1
        training_params, arch_params = suggester(trial)
        return run_trial(arch, training_params, arch_params,
                         epochs=epochs, trial_id=f"t{n:03d}",
                         train_text=train_text, val_text=val_text)

    log.info(f"\n{'='*60}")
    log.info(f"  Optuna HPO: arch={arch}  trials={n_trials}  epochs/trial={epochs}")
    log.info(f"{'='*60}")

    study.optimize(objective, n_trials=n_trials, n_jobs=jobs, show_progress_bar=False)

    completed = [t for t in study.trials if t.value is not None and t.value < float("inf")]
    if not completed:
        log.error(f"  {arch}: ALL trials failed — no best config saved. "
                  f"Check seq_len / timeout / GPU memory.")
        return {"arch": arch, "best_val_ppl": float("inf"), "n_trials": len(study.trials),
                "error": "all_trials_failed"}

    best = study.best_trial
    log.info(f"\n  ✓ {arch} best trial: val_ppl={best.value:.2f}")
    log.info(f"    params: {json.dumps(best.params, indent=4)}")

    # Reconstruct training + arch params from flat Optuna params
    training_params, arch_params = suggester(best)

    result = {
        "arch":          arch,
        "best_val_ppl":  best.value,
        "best_params":   best.params,
        "training":      training_params,
        "arch_params":   arch_params,
        "study_name":    study_name,
        "n_trials":      len(study.trials),
        "timestamp":     datetime.now().isoformat(),
    }

    # Save best config
    out_dir = ROOT / "artifacts" / "optuna"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"small_lm_{arch}_best.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"  Saved → {out_path}")

    # Save full study
    try:
        import pickle
        study_path = out_dir / f"small_lm_{arch}_study.pkl"
        with open(study_path, "wb") as f:
            pickle.dump(study, f)
        log.info(f"  Study  → {study_path}")
    except Exception:
        pass

    return result


# ── Build final training config from Optuna best ───────────────────────────────

def build_final_config(arch: str, seed: int, epochs: int) -> Optional[Dict]:
    """Load Optuna best and build a full training YAML config."""
    best_path = ROOT / "artifacts" / "optuna" / f"small_lm_{arch}_best.json"
    if not best_path.exists():
        log.warning(f"No Optuna best found for {arch} at {best_path}")
        return None

    with open(best_path) as f:
        best = json.load(f)

    cfg = {
        **SHARED_DEFAULTS,
        **best["training"],
        "arch_params":  best["arch_params"],
        "epochs":       epochs,
        "seed":         seed,
        "output_dir":   "artifacts/small_lm",
        "scheduler":    "cosine_warm_restarts",
        "T_0":          5,
        "T_mult":       2,
        "eta_min":      1e-6,
        "mlflow_enabled": True,
        "mlflow_experiment": "small_lm",
    }
    return cfg


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch",     default="all",
                   help=f"Architecture: {', '.join(ARCHS)}, or 'all'")
    p.add_argument("--n-trials", type=int, default=20,
                   help="Number of Optuna trials per architecture")
    p.add_argument("--epochs",   type=int, default=5,
                   help="Epochs per trial (short; keep ≤10 for speed)")
    p.add_argument("--jobs",     type=int, default=1,
                   help="Parallel Optuna jobs (set 1 for single GPU)")
    p.add_argument("--timeout",  type=int, default=1800,
                   help="Seconds per trial before timeout")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip arch if best.json already exists")
    p.add_argument("--train-text", type=str, default=None,
                   help="Path to training text file (default: data/dialogue/train.txt)")
    p.add_argument("--val-text",   type=str, default=None,
                   help="Path to validation text file (default: data/dialogue/val.txt)")
    args = p.parse_args()

    archs_to_run = ARCHS if args.arch == "all" else [args.arch]
    if args.arch not in ARCHS and args.arch != "all":
        p.error(f"Unknown arch '{args.arch}'. Choose from: {', '.join(ARCHS)} or 'all'")

    # Install optuna if needed
    try:
        import optuna  # noqa
    except ImportError:
        log.info("Installing optuna...")
        subprocess.run([PYTHON, "-m", "pip", "install", "optuna", "-q"], check=True)

    all_results = {}
    for arch in archs_to_run:
        best_path = ROOT / "artifacts" / "optuna" / f"small_lm_{arch}_best.json"
        if args.skip_existing and best_path.exists():
            log.info(f"Skipping {arch} — best.json already exists")
            with open(best_path) as f:
                all_results[arch] = json.load(f)
            continue

        result = run_study(arch, n_trials=args.n_trials,
                           epochs=args.epochs, jobs=args.jobs,
                           train_text=args.train_text, val_text=args.val_text)
        all_results[arch] = result

    # Print summary table
    print(f"\n{'='*60}")
    print("  OPTUNA SEARCH SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Arch':<14} {'Best PPL':<12} {'Trials':<8}")
    print(f"  {'-'*40}")
    for arch, r in all_results.items():
        ppl = r.get("best_val_ppl", float("inf"))
        n   = r.get("n_trials", "?")
        print(f"  {arch:<14} {ppl:<12.2f} {n}")

    print(f"\n  Best configs saved to: artifacts/optuna/small_lm_*_best.json")
    print(f"  Next step: retrain with best params × 3 seeds")
    print(f"  Run: python scripts/train_final_small_lms.py --seeds 42 43 44 --epochs 30")


if __name__ == "__main__":
    main()
