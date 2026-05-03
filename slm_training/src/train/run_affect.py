#!/usr/bin/env python3
"""
Affect Encoder Training Script
================================
Trains a DistilBERT regressor on Valence-Arousal-Dominance (VAD) affect data.

Usage:
  python -m src.train.run_affect
  python -m src.train.run_affect --config configs/affect.yaml
  python -m src.train.run_affect --config configs/affect.yaml --lr 1e-5 --epochs 5
  python -m src.train.run_affect --run-id ablation_vad_lr1e5

Artifacts produced under output_dir/run_id/:
  best_model/              saved HuggingFace encoder + full state dict
  run.log                  full structured log
  step_metrics.csv         per-step train loss
  epoch_metrics.csv        per-epoch val MSE / MAE / R² (total + per-dimension)
  predictions_epoch{N}.csv sample predictions vs ground truth for error analysis
  run_summary.json         all hyperparams + results (ablation table row)
  run_summary.md           human-readable summary report
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "train"))

from src.data.dialogue_data import RegressionTextDataset
from src.models.affect import DistilBertRegressor
from metrics_report import write_metrics_bundle


# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULTS: Dict[str, Any] = {
    "model_name":     "distilbert-base-uncased",
    "train_path":     "data/affect/train.csv",
    "val_path":       "data/affect/val.csv",
    "text_column":    "text",
    "target_columns": ["valence", "arousal", "dominance"],
    "max_length":     256,
    "batch_size":     16,
    "grad_accum":     2,
    "lr":             3e-5,
    "encoder_lr":     3e-6,
    "epochs":         15,
    "log_every":      20,
    "seed":           42,
    "output_dir":     "artifacts/affect_encoder",
    "warmup_ratio":   0.1,
    "max_grad_norm":  1.0,
    "patience":       5,
    "dropout":        0.3,
    "multi_sample_dropout": 5,
    "freeze_encoder_epochs": 1,
    "loss_type":      "ccc_mse",
    "ccc_weight":     0.5,
    # MLflow tracking
    "mlflow_experiment": "affect_encoder",
    "mlflow_enabled":    True,
}


# ── CCC Loss ────────────────────────────────────────────────────────────────

def ccc_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Differentiable 1 - CCC loss. Directly optimizes CCC metric."""
    mean_p = preds.mean(dim=0)
    mean_t = targets.mean(dim=0)
    var_p = preds.var(dim=0)
    var_t = targets.var(dim=0)
    cov = ((preds - mean_p) * (targets - mean_t)).mean(dim=0)
    ccc = 2 * cov / (var_p + var_t + (mean_p - mean_t) ** 2 + 1e-8)
    return 1.0 - ccc.mean()


def load_config(config_path: Optional[str], overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(DEFAULTS)
    if config_path:
        with open(config_path) as f:
            cfg.update(yaml.safe_load(f))
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    # Coerce numeric types that YAML may load as strings
    for k in ("lr", "encoder_lr", "warmup_ratio", "max_grad_norm", "dropout", "ccc_weight"):
        if k in cfg and isinstance(cfg[k], str):
            cfg[k] = float(cfg[k])
    for k in ("batch_size", "grad_accum", "epochs", "patience", "multi_sample_dropout",
              "freeze_encoder_epochs", "max_length", "log_every", "seed"):
        if k in cfg and isinstance(cfg[k], str):
            cfg[k] = int(cfg[k])
    return cfg


def setup_logger(log_dir: Path, name: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_dir / "run.log", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


class MetricsWriter:
    def __init__(self, path: Path, fieldnames: List[str]) -> None:
        self.path = path
        self.fieldnames = fieldnames
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def write(self, row: Dict[str, Any]) -> None:
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(
                {k: row.get(k, "") for k in self.fieldnames}
            )


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Concordance Correlation Coefficient."""
    mean_t, mean_p = y_true.mean(), y_pred.mean()
    var_t, var_p = y_true.var(), y_pred.var()
    cov = np.mean((y_true - mean_t) * (y_pred - mean_p))
    denom = var_t + var_p + (mean_t - mean_p) ** 2
    return float(2.0 * cov / denom) if denom > 1e-8 else 0.0


def compute_metrics(preds: np.ndarray, labels: np.ndarray, dims: List[str]) -> Dict[str, Any]:
    mse_per_dim = np.mean((preds - labels) ** 2, axis=0)
    mae_per_dim = np.mean(np.abs(preds - labels), axis=0)
    ss_res      = np.sum((preds - labels) ** 2, axis=0)
    ss_tot      = np.sum((labels - labels.mean(axis=0)) ** 2, axis=0)
    r2_per_dim  = 1.0 - ss_res / (ss_tot + 1e-8)
    ccc_per_dim = np.array([_ccc(labels[:, i], preds[:, i]) for i in range(len(dims))])
    return {
        "val_mse": float(np.mean(mse_per_dim)),
        "val_mae": float(np.mean(mae_per_dim)),
        "val_r2":  float(np.mean(r2_per_dim)),
        "val_ccc": float(np.mean(ccc_per_dim)),
        "per_dim": {
            d: {
                "mse": float(mse_per_dim[i]), "mae": float(mae_per_dim[i]),
                "r2": float(r2_per_dim[i]), "ccc": float(ccc_per_dim[i]),
            }
            for i, d in enumerate(dims)
        },
    }


def train(cfg: Dict[str, Any]) -> Dict[str, Any]:
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    run_id  = cfg.get("run_id") or f"affect_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(cfg["output_dir"]) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    log = setup_logger(out_dir, run_id)
    log.info("=" * 60)
    log.info(f"RUN  : {run_id}")
    log.info(f"MODEL: {cfg['model_name']}")
    log.info(f"DIMS : {cfg['target_columns']}")
    log.info("=" * 60)
    log.debug(f"Full config:\n{json.dumps(cfg, indent=2)}")

    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # ── Data validation ───────────────────────────────────────────────────────
    for key in ("train_path", "val_path"):
        p = Path(cfg[key])
        if not p.exists():
            log.error(f"Required data file missing: {p}")
            log.error("Run data download first:  python -m src.data.datasets --datasets emobank goemotions empathetic_dialogues")
            raise FileNotFoundError(f"Missing: {p}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = RegressionTextDataset(
        path=cfg["train_path"], tokenizer_name=cfg["model_name"],
        text_column=cfg["text_column"], target_columns=cfg["target_columns"],
        max_length=cfg["max_length"],
    )
    val_ds = RegressionTextDataset(
        path=cfg["val_path"], tokenizer_name=cfg["model_name"],
        text_column=cfg["text_column"], target_columns=cfg["target_columns"],
        max_length=cfg["max_length"],
    )
    log.info(f"Train: {len(train_ds):,} samples | Val: {len(val_ds):,} samples")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    device = select_device()
    log.info(f"Device: {device}")
    model = DistilBertRegressor(
        cfg["model_name"], out_dim=len(cfg["target_columns"]),
        dropout=cfg.get("dropout", 0.3),
        multi_sample_dropout=cfg.get("multi_sample_dropout", 0),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters: {n_params:,}  ({n_params/1e6:.1f} M)")
    log.info(f"Loss type: {cfg.get('loss_type', 'mse')}")

    # Differential LR: encoder gets lower LR than head
    encoder_lr = cfg.get("encoder_lr", cfg["lr"] / 10)
    param_groups = [
        {"params": model.encoder.parameters(), "lr": encoder_lr},
        {"params": list(model.layer_norm.parameters()) + list(model.head.parameters()), "lr": cfg["lr"]},
    ]
    if hasattr(model, '_ms_dropouts'):
        param_groups[1]["params"] += list(model._ms_dropouts.parameters())
    optimizer = AdamW(param_groups, weight_decay=0.01)
    grad_accum = cfg.get("grad_accum", 1)
    effective_steps = (len(train_loader) // grad_accum) * cfg["epochs"]
    warmup_steps = int(effective_steps * cfg.get("warmup_ratio", 0.1))
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=effective_steps)
    log.info(f"Differential LR: encoder={encoder_lr:.1e}, head={cfg['lr']:.1e}")
    log.info(f"Scheduler: cosine with {warmup_steps} warmup / {effective_steps} effective steps")
    log.info(f"Gradient accumulation: {grad_accum} (effective batch={cfg['batch_size'] * grad_accum})")

    # ── MLflow tracking ────────────────────────────────────────────────────────
    sys.path.insert(0, str(ROOT / "src" / "train"))
    from mlflow_tracker import MLflowTracker
    tracker = MLflowTracker(
        experiment=cfg.get("mlflow_experiment", "affect_encoder"),
        enabled=cfg.get("mlflow_enabled", True),
    )
    tracker.start_run(run_name=run_id, tags={
        "task": "affect",
        "seed": str(cfg["seed"]),
        "loss_type": cfg.get("loss_type", "ccc_mse"),
    })
    tracker.log_params(cfg)

    # Gradual unfreezing
    freeze_epochs = cfg.get("freeze_encoder_epochs", 0)
    if freeze_epochs > 0:
        for p in model.encoder.parameters():
            p.requires_grad = False
        log.info(f"Encoder frozen for first {freeze_epochs} epoch(s)")

    # ── Metric writers ────────────────────────────────────────────────────────
    dims        = cfg["target_columns"]
    step_writer = MetricsWriter(out_dir / "step_metrics.csv", ["epoch", "global_step", "train_loss"])
    epoch_cols  = (
        ["epoch", "val_mse", "val_mae", "val_r2", "val_ccc"]
        + [f"{d}_mse" for d in dims]
        + [f"{d}_mae" for d in dims]
        + [f"{d}_r2"  for d in dims]
        + [f"{d}_ccc" for d in dims]
    )
    epoch_writer = MetricsWriter(out_dir / "epoch_metrics.csv", epoch_cols)

    best_val_ccc = -1.0
    best_val_mse = math.inf
    patience_ctr = 0
    patience     = cfg.get("patience", 3)
    best_dir     = out_dir / "best_model"
    summary: Dict[str, Any] = {
        "run_id":     run_id,
        "model":      cfg["model_name"],
        "task":       "affect_regression",
        "dimensions": dims,
        "hyperparams": {k: v for k, v in cfg.items() if k not in ("run_id",)},
        "data":        {"train_size": len(train_ds), "val_size": len(val_ds)},
        "epochs":      [],
        "best":        {},
    }

    # ── Training loop ─────────────────────────────────────────────────────────
    global_step  = 0
    running_loss = 0.0
    running_n    = 0

    loss_type = cfg.get("loss_type", "mse")
    ccc_weight = cfg.get("ccc_weight", 0.5)

    for epoch in range(1, cfg["epochs"] + 1):
        log.info(f"── Epoch {epoch}/{cfg['epochs']} ──────────────────────────────────")
        model.train()

        # Gradual unfreezing
        if epoch == freeze_epochs + 1 and freeze_epochs > 0:
            for p in model.encoder.parameters():
                p.requires_grad = True
            log.info("  Encoder unfrozen")

        optimizer.zero_grad()
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"  train ep{epoch}", leave=False, file=sys.stdout)):
            global_step += 1
            batch = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask", "labels")}
            out   = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

            if loss_type == "ccc_mse":
                loss = ccc_weight * ccc_loss(out["preds"], batch["labels"]) + (1 - ccc_weight) * F.mse_loss(out["preds"], batch["labels"])
            elif loss_type == "ccc":
                loss = ccc_loss(out["preds"], batch["labels"])
            else:
                loss = F.mse_loss(out["preds"], batch["labels"])

            loss = loss / grad_accum
            loss.backward()

            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), cfg.get("max_grad_norm", 1.0))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.item()
            running_n    += 1

            if global_step % cfg["log_every"] == 0:
                avg = running_loss / running_n
                log.info(f"  step {global_step:5d} | train_loss={avg:.6f}")
                step_writer.write({"epoch": epoch, "global_step": global_step, "train_loss": avg})
                tracker.log_metrics({"train_loss": avg, "lr": optimizer.param_groups[0]["lr"]}, step=global_step)
                running_loss = 0.0
                running_n    = 0

        # ── Validation ────────────────────────────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"  val   ep{epoch}", leave=False, file=sys.stdout):
                batch = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask", "labels")}
                out   = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                all_preds.append(out["preds"].cpu().numpy())
                all_labels.append(batch["labels"].cpu().numpy())

        preds  = np.concatenate(all_preds,  axis=0)
        labels = np.concatenate(all_labels, axis=0)
        m      = compute_metrics(preds, labels, dims)

        log.info(f"  val_mse={m['val_mse']:.6f}  val_mae={m['val_mae']:.6f}  val_r2={m['val_r2']:.4f}  val_ccc={m['val_ccc']:.4f}")
        for d, dm in m["per_dim"].items():
            log.info(f"    {d:12s}  mse={dm['mse']:.6f}  mae={dm['mae']:.6f}  r2={dm['r2']:.4f}  ccc={dm['ccc']:.4f}")

        epoch_writer.write({
            "epoch":   epoch,
            "val_mse": m["val_mse"],
            "val_mae": m["val_mae"],
            "val_r2":  m["val_r2"],
            "val_ccc": m["val_ccc"],
            **{f"{d}_mse": m["per_dim"][d]["mse"] for d in dims},
            **{f"{d}_mae": m["per_dim"][d]["mae"] for d in dims},
            **{f"{d}_r2":  m["per_dim"][d]["r2"]  for d in dims},
            **{f"{d}_ccc": m["per_dim"][d]["ccc"] for d in dims},
        })

        # ── Error-analysis predictions CSV ────────────────────────────────────
        val_texts = pd.read_csv(cfg["val_path"])[cfg["text_column"]].tolist()
        pred_rows = []
        for i, text in enumerate(val_texts[:len(preds)]):
            row = {"text": text}
            for j, d in enumerate(dims):
                row[f"pred_{d}"]  = round(float(preds[i, j]),  6)
                row[f"true_{d}"]  = round(float(labels[i, j]), 6)
                row[f"error_{d}"] = round(float(preds[i, j] - labels[i, j]), 6)
            pred_rows.append(row)
        pd.DataFrame(pred_rows).to_csv(out_dir / f"predictions_epoch{epoch}.csv", index=False)

        summary["epochs"].append({
            "epoch": epoch,
            **{k: m[k] for k in ("val_mse", "val_mae", "val_r2", "val_ccc")},
            "per_dim": m["per_dim"],
        })
        tracker.log_metrics({"val_ccc": m["val_ccc"], "val_mse": m["val_mse"],
                             "val_r2": m["val_r2"], "val_mae": m["val_mae"]}, step=epoch)
        for d, dm in m["per_dim"].items():
            tracker.log_metrics({f"{d}_ccc": dm["ccc"], f"{d}_r2": dm["r2"]}, step=epoch)

        # ── Checkpoint (best by CCC) ─────────────────────────────────────────
        if m["val_ccc"] > best_val_ccc:
            best_val_ccc = m["val_ccc"]
            best_val_mse = m["val_mse"]
            patience_ctr = 0
            best_dir.mkdir(parents=True, exist_ok=True)
            model.encoder.save_pretrained(best_dir)
            torch.save(model.state_dict(), best_dir / "pytorch_model.bin")
            log.info(f"  ✓ Best model saved → {best_dir}  (val_ccc={best_val_ccc:.4f}  val_mse={best_val_mse:.6f})")
        else:
            patience_ctr += 1
            log.info(f"  No improvement ({patience_ctr}/{patience})")
            if patience_ctr >= patience:
                log.info(f"  Early stopping at epoch {epoch}")
                break

    best_epoch = max(summary["epochs"], key=lambda e: e["val_ccc"])
    summary["best"] = {"epoch": best_epoch["epoch"], "val_mse": best_val_mse,
                       "val_ccc": best_val_ccc, "val_r2": best_epoch.get("val_r2", 0)}

    write_metrics_bundle(out_dir, "run_summary", summary, title="Affect Encoder Run Summary")

    # ── MLflow: log final metrics and artifacts ──────────────────────────────────
    tracker.log_metrics({
        "best_val_ccc": best_val_ccc,
        "best_val_mse": best_val_mse,
        "best_val_r2":  best_epoch.get("val_r2", 0),
        "best_epoch":   best_epoch["epoch"],
    })
    tracker.log_artifact(out_dir / "run_summary.json")
    tracker.log_artifact(out_dir / "run_summary.md")
    tracker.log_artifact(out_dir / "epoch_metrics.csv")
    tracker.end_run()

    log.info("=" * 60)
    log.info(f"DONE  best val_ccc={best_val_ccc:.4f}  val_r2={best_epoch.get('val_r2', 0):.4f}  val_mse={best_val_mse:.6f}  (epoch {best_epoch['epoch']})")
    log.info(f"Artifacts → {out_dir}")
    log.info("=" * 60)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train affect encoder")
    p.add_argument("--config",     type=str)
    p.add_argument("--run-id",     type=str, dest="run_id")
    p.add_argument("--model-name", type=str, dest="model_name")
    p.add_argument("--train-path", type=str, dest="train_path")
    p.add_argument("--val-path",   type=str, dest="val_path")
    p.add_argument("--output-dir", type=str, dest="output_dir")
    p.add_argument("--batch-size", type=int, dest="batch_size")
    p.add_argument("--lr",         type=float)
    p.add_argument("--epochs",     type=int)
    p.add_argument("--max-length", type=int, dest="max_length")
    p.add_argument("--log-every",  type=int, dest="log_every")
    p.add_argument("--seed",       type=int)
    return p.parse_args()


if __name__ == "__main__":
    args      = parse_args()
    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    cfg       = load_config(args.config, overrides)
    train(cfg)
