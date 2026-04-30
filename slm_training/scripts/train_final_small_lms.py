#!/usr/bin/env python3
"""
train_final_small_lms.py
========================
Retrain all 6 Small-LM architectures using the Optuna-found best hyperparameters,
across multiple seeds for statistical robustness (mean ± std in the paper).

Prerequisite: run optuna_small_lm.py --arch all first.

Usage:
  python scripts/train_final_small_lms.py
  python scripts/train_final_small_lms.py --seeds 42 43 44 --epochs 30
  python scripts/train_final_small_lms.py --arch gpt --seeds 42 43 44 --epochs 30
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

ROOT   = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

ARCHS = ["gru", "awdlstm", "gpt", "prefix_gpt", "moe", "mamba_like"]

SHARED_DEFAULTS = {
    "hardware_profile": "rtx4070_small",
    "train_text":       str(ROOT / "data" / "dialogue" / "train.txt"),
    "val_text":         str(ROOT / "data" / "dialogue" / "val.txt"),
    "cond_dim":         8,
    "use_amp":          False,
    "embedding_model":  None,
    "embedding_cache":  True,
    "log_every":        20,
    "eval_every_steps": 9999,
    "scheduler":        "cosine_warm_restarts",
    "T_0":              5,
    "T_mult":           2,
    "eta_min":          1e-6,
    "mlflow_enabled":   True,
    "mlflow_experiment": "small_lm_final",
}


def load_optuna_best(arch: str) -> Optional[Dict]:
    path = ROOT / "artifacts" / "optuna" / f"small_lm_{arch}_best.json"
    if not path.exists():
        log.warning(f"  [{arch}] No Optuna best found at {path} — skipping")
        return None
    with open(path) as f:
        return json.load(f)


# GPU-optimized batch sizes: increase batch, reduce/eliminate grad_accum
# Uses VRAM headroom on RTX 2080 Ti (22GB, currently ~3GB used)
# NOTE: 
#   - prefix_gpt: n_embd=256, n_head=8, prefix_len=8 → reduced bs to avoid OOM
#   - moe/mamba: Keep original bs=16, ga=2 (OOM risk with ga=1)
BATCH_OVERRIDE = {
    "gpt":        {"batch_size": 32, "grad_accum": 2},   # Reduced from 64,1 - OOM risk
    "prefix_gpt": {"batch_size": 16, "grad_accum": 2},  # Reduced from 32,1 - OOM risk
    # moe and mamba_like: use original configs (bs=16, ga=2) - OOM if ga=1
}


def run_seed(arch: str, seed: int, epochs: int,
             training_params: Dict, arch_params: Dict,
             timeout: int = 7200) -> Dict[str, Any]:
    """Train one arch × seed with the given config."""
    run_id   = f"final_{arch}_s{seed}_{datetime.now():%H%M%S}"
    cfg_path = ROOT / "artifacts" / "optuna" / "final_configs" / f"{run_id}.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)

    # Apply GPU-optimized batch override if available
    opt_params = BATCH_OVERRIDE.get(arch, {})
    effective_params = {**training_params, **opt_params}
    if opt_params:
        log.info(f"  [GPU OPT] {arch}: bs={opt_params.get('batch_size')} ga={opt_params.get('grad_accum')}")

    cfg = {
        **SHARED_DEFAULTS,
        **effective_params,
        "arch_params": arch_params,
        "epochs":      epochs,
        "seed":        seed,
        "output_dir":  "artifacts/small_lm",
    }
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    cmd = [
        PYTHON, "-m", "src.train.run_small_lm",
        "--config", str(cfg_path),
        "--run-id", run_id,
        "--arch",   arch,
    ]
    log.info(f"  Training {arch} seed={seed}  run_id={run_id}")

    result = subprocess.run(
        cmd, cwd=str(ROOT),
        capture_output=True, text=True, timeout=timeout,
    )

    if result.returncode != 0:
        log.error(f"  FAILED: {arch} seed={seed}")
        log.error(f"  STDERR: {result.stderr[-800:]}")
        return {"arch": arch, "seed": seed, "success": False, "run_id": run_id}

    summary_path = ROOT / "artifacts" / "small_lm" / run_id / "run_summary.json"
    if not summary_path.exists():
        log.warning(f"  No summary for {arch} seed={seed}")
        return {"arch": arch, "seed": seed, "success": False, "run_id": run_id}

    with open(summary_path) as f:
        data = json.load(f)

    best = data.get("best", {})
    log.info(f"  ✓ {arch} seed={seed} → best_ppl={best.get('val_ppl', 0):.2f}"
             f"  (epoch {best.get('epoch', '?')})")
    return {
        "arch":     arch,
        "seed":     seed,
        "success":  True,
        "run_id":   run_id,
        "best_ppl": best.get("val_ppl", float("inf")),
        "best_loss": best.get("val_loss", float("inf")),
        "best_epoch": best.get("epoch"),
        "n_epochs": len(data.get("epochs", [])),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch",    default="all",
                   help=f"Architecture or 'all'. Default: all")
    p.add_argument("--seeds",   type=int, nargs="+", default=[42, 43, 44],
                   help="Random seeds (default: 42 43 44)")
    p.add_argument("--epochs",  type=int, default=30,
                   help="Training epochs per run (default: 30)")
    p.add_argument("--timeout", type=int, default=7200,
                   help="Seconds per run before timeout (default: 7200)")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip runs whose run_summary.json already exists")
    p.add_argument("--train-text", type=str, default=None,
                   help="Override training text file (default: data/dialogue/train.txt)")
    p.add_argument("--val-text", type=str, default=None,
                   help="Override validation text file (default: data/dialogue/val.txt)")
    args = p.parse_args()

    if args.train_text:
        SHARED_DEFAULTS["train_text"] = args.train_text
    if args.val_text:
        SHARED_DEFAULTS["val_text"] = args.val_text

    archs = ARCHS if args.arch == "all" else [args.arch]
    if args.arch not in ARCHS and args.arch != "all":
        p.error(f"Unknown arch '{args.arch}'. Choose from: {', '.join(ARCHS)} or 'all'")

    all_results: List[Dict] = []

    for arch in archs:
        best = load_optuna_best(arch)
        if best is None:
            continue

        training_params = best.get("training", {})
        arch_params     = best.get("arch_params", {})

        log.info(f"\n{'='*60}")
        log.info(f"  {arch.upper()}  ({len(args.seeds)} seeds × {args.epochs} epochs)")
        log.info(f"  Training params: lr={training_params.get('lr', '?'):.2e}  "
                 f"seq_len={training_params.get('seq_len', '?')}")
        log.info(f"  Arch params: {arch_params}")
        log.info(f"{'='*60}")

        for seed in args.seeds:
            if args.skip_existing:
                existing = sorted((ROOT / "artifacts" / "small_lm").glob(
                    f"final_{arch}_s{seed}_*/run_summary.json"))
                if existing:
                    with open(existing[-1]) as f:
                        summary = json.load(f)
                    log.info(f"  [skip] {arch} seed={seed} — reusing {existing[-1].parent.name}")
                    all_results.append({
                        "arch": arch, "seed": seed, "success": True,
                        "best_ppl": summary.get("best_ppl"),
                        "best_loss": summary.get("best_loss"),
                        "best_epoch": summary.get("best_epoch"),
                        "epochs": summary.get("epochs"),
                        "run_id": existing[-1].parent.name,
                    })
                    continue
            row = run_seed(arch, seed, args.epochs,
                           training_params, arch_params,
                           timeout=args.timeout)
            all_results.append(row)

    # Save combined results
    out_path = ROOT / "artifacts" / "small_lm_final_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"\n  All results → {out_path}")

    # Print summary
    from statistics import mean, stdev
    print(f"\n{'='*70}")
    print("  FINAL TRAINING RESULTS  (mean ± std across seeds)")
    print(f"{'='*70}")
    print(f"  {'Arch':<14} {'Seeds OK':<10} {'Mean PPL':<12} {'Std PPL':<12} {'Optuna PPL':<12}")
    print(f"  {'-'*60}")

    for arch in archs:
        arch_rows = [r for r in all_results if r["arch"] == arch and r.get("success")]
        best_path = ROOT / "artifacts" / "optuna" / f"small_lm_{arch}_best.json"
        optuna_ppl = "N/A"
        if best_path.exists():
            with open(best_path) as f:
                optuna_ppl = f"{json.load(f).get('best_val_ppl', 0):.2f}"

        if not arch_rows:
            print(f"  {arch:<14} {'0':<10} {'FAILED'}")
            continue

        ppls = [r["best_ppl"] for r in arch_rows if r.get("best_ppl") is not None]
        if not ppls:
            print(f"  {arch:<14} {'0':<10} {'NO VALID PPL'}")
            continue
        avg  = mean(ppls)
        std  = stdev(ppls) if len(ppls) > 1 else 0.0
        print(f"  {arch:<14} {len(arch_rows):<10} {avg:<12.2f} {std:<12.2f} {optuna_ppl:<12}")

    print(f"\n  Next step: evaluate all models")
    print(f"  Run: python scripts/eval_small_lms.py --out-csv artifacts/slm_final_eval.csv")


if __name__ == "__main__":
    main()
