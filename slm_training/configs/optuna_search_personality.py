"""Optuna hyperparameter search for personality encoder."""

import optuna
from pathlib import Path
import subprocess
import json

def objective(trial):
    # Search space
    lr = trial.suggest_float("lr", 1e-5, 5e-5, log=True)
    encoder_lr_factor = trial.suggest_float("encoder_lr_factor", 0.05, 0.25)
    dropout = trial.suggest_float("dropout", 0.1, 0.35)
    focal_gamma = trial.suggest_float("focal_gamma", 1.0, 3.0)
    token_drop = trial.suggest_float("token_drop", 0.0, 0.15)
    freeze_epochs = trial.suggest_int("freeze_epochs", 0, 2)
    
    # Run training
    config = {
        "lr": lr,
        "encoder_lr": lr * encoder_lr_factor,
        "dropout": dropout,
        "focal_gamma": focal_gamma,
        "token_drop_prob": token_drop,
        "freeze_encoder_epochs": freeze_epochs,
        "batch_size": 16,
        "grad_accum": 2,
        "epochs": 10,
        "patience": 3,
    }
    
    # Save temp config
    config_path = f"configs/optuna/trial_{trial.number}.yaml"
    Path(config_path).parent.mkdir(exist_ok=True)
    import yaml
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    # Run training
    result = subprocess.run(
        [".venv/bin/python", "-m", "src.train.run_personality", 
         "--config", config_path, "--run-id", f"optuna_p_{trial.number}"],
        capture_output=True, text=True
    )
    
    # Parse best F1 from summary
    summary_path = f"artifacts/personality_encoder/optuna_p_{trial.number}/run_summary.json"
    try:
        with open(summary_path) as f:
            data = json.load(f)
            return data["best"]["val_f1"]
    except:
        return 0.0

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize", study_name="personality_v3")
    study.optimize(objective, n_trials=50, n_jobs=1)
    
    print("Best trial:")
    trial = study.best_trial
    print(f"  F1: {trial.value:.4f}")
    print(f"  Params: {trial.params}")
