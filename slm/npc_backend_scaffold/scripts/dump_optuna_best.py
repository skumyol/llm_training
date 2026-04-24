#!/usr/bin/env python3
"""Extract best params from each Optuna study into artifacts/optuna/small_lm_<arch>_best.json."""
from __future__ import annotations
import json
import pickle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Keys that belong to training (rest → arch_params)
TRAINING_KEYS = {
    "lr", "weight_decay", "warmup_ratio",
    "seq_len", "batch_size", "grad_accum",
    "scheduler", "T_0", "T_mult", "eta_min",
}

MANUAL_DEFAULTS = {
    "mamba_like": {
        "training": {
            "lr": 0.0008, "weight_decay": 0.01, "warmup_ratio": 0.05,
            "seq_len": 256, "batch_size": 16, "grad_accum": 2,
        },
        "arch_params": {
            "dropout": 0.1, "n_layer": 4, "n_embd": 256,
            "d_state": 16, "expand": 2,
        },
        "note": "manual (all Optuna trials failed)",
    }
}


def split_params(params: dict) -> tuple[dict, dict]:
    training = {k: v for k, v in params.items() if k in TRAINING_KEYS}
    arch_params = {k: v for k, v in params.items() if k not in TRAINING_KEYS}
    return training, arch_params


def main() -> None:
    optuna_dir = ROOT / "artifacts" / "optuna"
    archs = ["gru", "awdlstm", "gpt", "prefix_gpt", "moe", "mamba_like"]
    for arch in archs:
        study_path = optuna_dir / f"small_lm_{arch}_study.pkl"
        out_path = optuna_dir / f"small_lm_{arch}_best.json"
        if not study_path.exists():
            print(f"[skip] {arch}: no study at {study_path}")
            continue
        study = pickle.load(open(study_path, "rb"))
        completed = [t for t in study.trials
                     if t.value is not None and t.value != float("inf")]
        if not completed:
            if arch in MANUAL_DEFAULTS:
                payload = {
                    "arch": arch, "value": None,
                    **MANUAL_DEFAULTS[arch],
                }
                out_path.write_text(json.dumps(payload, indent=2))
                print(f"[manual] {arch}: {MANUAL_DEFAULTS[arch]['note']}")
            else:
                print(f"[skip] {arch}: no completed trials, no manual fallback")
            continue
        best = study.best_trial
        training, arch_params = split_params(best.params)
        payload = {
            "arch": arch,
            "value": best.value,
            "training": training,
            "arch_params": arch_params,
            "trial_number": best.number,
            "n_completed_trials": len(completed),
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"[ok]     {arch}: val={best.value:.3f}  trial={best.number}  "
              f"completed={len(completed)}/{len(study.trials)}")


if __name__ == "__main__":
    main()
