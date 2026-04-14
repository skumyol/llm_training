#!/usr/bin/env python3
"""Hyperparameter search for personality and affect encoders using Optuna."""

import optuna
import yaml
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Base config templates
PERSONALITY_BASE = {
    "model_name": "distilbert-base-uncased",
    "train_path": "data/personality/train.csv",
    "val_path": "data/personality/val.csv",
    "text_column": "text",
    "target_columns": ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"],
    "max_length": 256,
    "batch_size": 16,
    "grad_accum": 2,
    "epochs": 12,
    "patience": 4,
    "log_every": 20,
    "seed": 42,
    "output_dir": "artifacts/personality_encoder",
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
    "loss_type": "focal_bce",
    "rdrop_alpha": 0.0,
    "multi_sample_dropout": 0,
}

AFFECT_BASE = {
    "model_name": "distilbert-base-uncased",
    "train_path": "data/affect/train.csv",
    "val_path": "data/affect/val.csv",
    "text_column": "text",
    "target_columns": ["valence", "arousal", "dominance"],
    "max_length": 256,
    "batch_size": 16,
    "grad_accum": 1,
    "epochs": 15,
    "patience": 5,
    "log_every": 20,
    "seed": 42,
    "output_dir": "artifacts/affect_encoder",
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
    "loss_type": "ccc_mse",
    "multi_sample_dropout": 0,
    "freeze_encoder_epochs": 0,
}


def run_personality_trial(trial: optuna.Trial) -> float:
    """Run a single hyperparameter trial for personality encoder."""
    
    # Define search space
    lr = trial.suggest_float("lr", 1e-5, 5e-5, log=True)
    encoder_lr_factor = trial.suggest_float("encoder_lr_factor", 0.05, 0.3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    focal_gamma = trial.suggest_float("focal_gamma", 0.5, 3.0)
    token_drop_prob = trial.suggest_float("token_drop_prob", 0.0, 0.2)
    freeze_encoder_epochs = trial.suggest_int("freeze_encoder_epochs", 0, 2)
    
    # Build config
    config = PERSONALITY_BASE.copy()
    config.update({
        "lr": lr,
        "encoder_lr": lr * encoder_lr_factor,
        "dropout": dropout,
        "focal_gamma": focal_gamma,
        "token_drop_prob": token_drop_prob,
        "freeze_encoder_epochs": freeze_encoder_epochs,
    })
    
    # Save config
    trial_dir = Path(f"artifacts/optuna/personality/trial_{trial.number}")
    trial_dir.mkdir(parents=True, exist_ok=True)
    config_path = trial_dir / "config.yaml"
    
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Run training
    run_id = f"optuna_p_{trial.number}_{datetime.now():%H%M%S}"
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "src.train.run_personality",
             "--config", str(config_path),
             "--run-id", run_id],
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min timeout
            cwd="/home/serkan/llm_training/slm/npc_backend_scaffold"
        )
        
        # Parse result from summary
        summary_path = Path("artifacts/personality_encoder") / run_id / "run_summary.json"
        
        if summary_path.exists():
            with open(summary_path) as f:
                data = json.load(f)
                best_f1 = data.get("best", {}).get("val_f1", 0.0)
                trial.set_user_attr("run_id", run_id)
                trial.set_user_attr("config_path", str(config_path))
                return best_f1
        
        return 0.0
        
    except subprocess.TimeoutExpired:
        print(f"Trial {trial.number} timed out")
        return 0.0
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return 0.0


def run_affect_trial(trial: optuna.Trial) -> float:
    """Run a single hyperparameter trial for affect encoder."""
    
    # Define search space
    lr = trial.suggest_float("lr", 5e-6, 5e-5, log=True)
    encoder_lr_factor = trial.suggest_float("encoder_lr_factor", 0.05, 0.3, log=True)
    dropout = trial.suggest_float("dropout", 0.05, 0.40)
    ccc_weight = trial.suggest_float("ccc_weight", 0.1, 0.6)
    grad_accum = trial.suggest_categorical("grad_accum", [1, 2])
    freeze_encoder_epochs = trial.suggest_int("freeze_encoder_epochs", 0, 2)
    
    # Build config
    config = AFFECT_BASE.copy()
    config.update({
        "lr": lr,
        "encoder_lr": lr * encoder_lr_factor,
        "dropout": dropout,
        "ccc_weight": ccc_weight,
        "grad_accum": grad_accum,
        "freeze_encoder_epochs": freeze_encoder_epochs,
    })
    
    # Save config
    trial_dir = Path(f"artifacts/optuna/affect/trial_{trial.number}")
    trial_dir.mkdir(parents=True, exist_ok=True)
    config_path = trial_dir / "config.yaml"
    
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Run training
    run_id = f"optuna_a_{trial.number}_{datetime.now():%H%M%S}"
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "src.train.run_affect",
             "--config", str(config_path),
             "--run-id", run_id],
            capture_output=True,
            text=True,
            timeout=5400,  # 90 min — affect trials take ~35 min
            cwd="/home/serkan/llm_training/slm/npc_backend_scaffold"
        )
        
        # Parse result
        summary_path = Path("artifacts/affect_encoder") / run_id / "run_summary.json"
        
        if summary_path.exists():
            with open(summary_path) as f:
                data = json.load(f)
                best_ccc = data.get("best", {}).get("val_ccc", 0.0)
                trial.set_user_attr("run_id", run_id)
                trial.set_user_attr("config_path", str(config_path))
                return best_ccc
        
        return 0.0
        
    except subprocess.TimeoutExpired:
        print(f"Trial {trial.number} timed out")
        return 0.0
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return 0.0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["personality", "affect", "both"], default="both")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--study-name", type=str, default=None)
    args = parser.parse_args()
    
    if args.task in ["personality", "both"]:
        print("=" * 60)
        print("HYPERPARAMETER SEARCH: PERSONALITY ENCODER")
        print("=" * 60)
        
        study_name = args.study_name or f"personality_hpo_{datetime.now():%Y%m%d_%H%M%S}"
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        )
        
        study.optimize(run_personality_trial, n_trials=args.n_trials, show_progress_bar=True)
        
        print("\n" + "=" * 60)
        print("PERSONALITY SEARCH COMPLETE")
        print("=" * 60)
        print(f"Best F1: {study.best_value:.4f}")
        print(f"Best params: {json.dumps(study.best_params, indent=2)}")
        
        # Save results
        results_path = Path("artifacts/optuna/personality_best.json")
        with open(results_path, "w") as f:
            json.dump({
                "best_value": study.best_value,
                "best_params": study.best_params,
                "study_name": study_name,
                "n_trials": len(study.trials),
            }, f, indent=2)
        print(f"Results saved to {results_path}")
    
    if args.task in ["affect", "both"]:
        print("\n" + "=" * 60)
        print("HYPERPARAMETER SEARCH: AFFECT ENCODER")
        print("=" * 60)
        
        study_name = args.study_name or f"affect_hpo_{datetime.now():%Y%m%d_%H%M%S}"
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        )
        
        study.optimize(run_affect_trial, n_trials=args.n_trials, show_progress_bar=True)
        
        print("\n" + "=" * 60)
        print("AFFECT SEARCH COMPLETE")
        print("=" * 60)
        print(f"Best CCC: {study.best_value:.4f}")
        print(f"Best params: {json.dumps(study.best_params, indent=2)}")
        
        # Save results
        results_path = Path("artifacts/optuna/affect_best.json")
        with open(results_path, "w") as f:
            json.dump({
                "best_value": study.best_value,
                "best_params": study.best_params,
                "study_name": study_name,
                "n_trials": len(study.trials),
            }, f, indent=2)
        print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
