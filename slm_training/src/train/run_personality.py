#!/usr/bin/env python3
"""
Personality Encoder Training Script
=====================================
Trains a DistilBERT regressor on Big Five (OCEAN) personality data.

Usage:
  # From scaffold root:
  python -m src.train.run_personality
  python -m src.train.run_personality --config configs/personality.yaml
  python -m src.train.run_personality --config configs/personality.yaml --lr 1e-5 --epochs 5
  python -m src.train.run_personality --run-id ablation_lr_1e5

Artifacts produced under output_dir/run_id/:
  best_model/              saved HuggingFace encoder + full state dict
  run.log                  full structured log (console + file)
  step_metrics.csv         per-step train loss
  epoch_metrics.csv        per-epoch val MSE / MAE / R² (total + per-dimension)
  predictions_epoch{N}.csv sample predictions vs ground truth for error analysis
  run_summary.json         all hyperparams + results (one row in your ablation table)
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
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "train"))

from src.data.dialogue_data import RegressionTextDataset
from src.models.personality import DistilBertRegressor
from metrics_report import write_metrics_bundle


# ── Defaults (override via YAML or CLI) ──────────────────────────────────────
DEFAULTS: Dict[str, Any] = {
    "model_name":     "distilbert-base-uncased",
    "train_path":     "data/personality/train.csv",
    "val_path":       "data/personality/val.csv",
    "text_column":    "text",
    "target_columns": ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"],
    "max_length":     512,
    "batch_size":     8,
    "grad_accum":     4,
    "lr":             3e-5,
    "encoder_lr":     3e-6,
    "epochs":         15,
    "log_every":      20,
    "seed":           42,
    "output_dir":     "artifacts/personality_encoder",
    "warmup_ratio":   0.1,
    "max_grad_norm":  1.0,
    "patience":       5,
    "loss_type":      "focal_bce",
    "focal_gamma":    2.0,
    "dropout":        0.3,
    "multi_sample_dropout": 5,
    "freeze_encoder_epochs": 1,
    "rdrop_alpha":    0.5,
    "token_drop_prob": 0.1,
    # MLflow tracking
    "mlflow_experiment": "personality_encoder",
    "mlflow_enabled":    True,
}


# ── Focal BCE Loss ───────────────────────────────────────────────────────────

def focal_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor,
                          gamma: float = 2.0) -> torch.Tensor:
    """Focal loss for binary classification — focuses on hard examples."""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1 - probs) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma
    return (focal_weight * bce).mean()


# ── Token-level dropout augmentation for small datasets ──────────────────────

class TokenDropDataset(torch.utils.data.Dataset):
    """Wraps a dataset and randomly drops tokens during training."""
    def __init__(self, base_dataset, drop_prob: float = 0.1, pad_token_id: int = 0):
        self.base = base_dataset
        self.drop_prob = drop_prob
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        if self.drop_prob > 0:
            ids = item["input_ids"].clone()
            mask = item["attention_mask"].clone()
            # Don't drop [CLS], [SEP], or padding
            active = mask.bool() & (ids != 101) & (ids != 102)
            drop = torch.rand_like(ids.float()) < self.drop_prob
            drop = drop & active
            ids[drop] = self.pad_token_id
            mask[drop] = 0
            item = {**item, "input_ids": ids, "attention_mask": mask}
        return item


# ── Config loading ────────────────────────────────────────────────────────────

def load_config(config_path: Optional[str], overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(DEFAULTS)
    if config_path:
        with open(config_path) as f:
            cfg.update(yaml.safe_load(f))
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    # Coerce numeric types that YAML may load as strings
    for k in ("lr", "encoder_lr", "warmup_ratio", "max_grad_norm", "dropout",
              "focal_gamma", "rdrop_alpha", "token_drop_prob"):
        if k in cfg and isinstance(cfg[k], str):
            cfg[k] = float(cfg[k])
    for k in ("batch_size", "grad_accum", "epochs", "patience", "multi_sample_dropout",
              "freeze_encoder_epochs", "max_length", "log_every", "seed"):
        if k in cfg and isinstance(cfg[k], str):
            cfg[k] = int(cfg[k])
    return cfg


# ── Logging ───────────────────────────────────────────────────────────────────

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


# ── CSV metrics writer ────────────────────────────────────────────────────────

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


# ── Device selection ──────────────────────────────────────────────────────────

def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Validation metrics ────────────────────────────────────────────────────────

def compute_metrics(preds: np.ndarray, labels: np.ndarray, dims: List[str]) -> Dict[str, Any]:
    mse_per_dim  = np.mean((preds - labels) ** 2, axis=0)
    mae_per_dim  = np.mean(np.abs(preds - labels), axis=0)
    ss_res = np.sum((preds - labels) ** 2, axis=0)
    ss_tot = np.sum((labels - labels.mean(axis=0)) ** 2, axis=0)
    r2_per_dim   = 1.0 - ss_res / (ss_tot + 1e-8)

    # Binary classification metrics (threshold at 0.5)
    pred_binary  = (preds >= 0.5).astype(int)
    label_binary = (labels >= 0.5).astype(int)
    acc_per_dim  = np.array([accuracy_score(label_binary[:, i], pred_binary[:, i]) for i in range(len(dims))])
    f1_per_dim   = np.array([f1_score(label_binary[:, i], pred_binary[:, i], zero_division=0) for i in range(len(dims))])

    return {
        "val_mse": float(np.mean(mse_per_dim)),
        "val_mae": float(np.mean(mae_per_dim)),
        "val_r2":  float(np.mean(r2_per_dim)),
        "val_acc": float(np.mean(acc_per_dim)),
        "val_f1":  float(np.mean(f1_per_dim)),
        "per_dim": {
            d: {
                "mse": float(mse_per_dim[i]), "mae": float(mae_per_dim[i]),
                "r2": float(r2_per_dim[i]),
                "acc": float(acc_per_dim[i]), "f1": float(f1_per_dim[i]),
            }
            for i, d in enumerate(dims)
        },
    }


# ── Main training ─────────────────────────────────────────────────────────────

def train(cfg: Dict[str, Any]) -> Dict[str, Any]:
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    run_id  = cfg.get("run_id") or f"personality_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
            log.error("Run data download first:  python -m src.data.datasets --datasets essays_big5 pan15")
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

    # Token-level dropout augmentation for small personality dataset
    token_drop = cfg.get("token_drop_prob", 0.0)
    if token_drop > 0:
        train_ds = TokenDropDataset(train_ds, drop_prob=token_drop, pad_token_id=0)
        log.info(f"Token dropout augmentation: {token_drop:.0%}")

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    device = select_device()
    log.info(f"Device: {device}")
    loss_type = cfg.get("loss_type", "bce")
    use_bce = loss_type in ("bce", "focal_bce")
    model = DistilBertRegressor(
        cfg["model_name"], out_dim=len(cfg["target_columns"]),
        dropout=cfg.get("dropout", 0.3), use_sigmoid=use_bce,
        multi_sample_dropout=cfg.get("multi_sample_dropout", 0),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters: {n_params:,}  ({n_params/1e6:.1f} M)")
    log.info(f"Loss type: {loss_type}")

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
        experiment=cfg.get("mlflow_experiment", "personality_encoder"),
        enabled=cfg.get("mlflow_enabled", True),
    )
    tracker.start_run(run_name=run_id, tags={
        "task": "personality",
        "seed": str(cfg["seed"]),
        "loss_type": cfg.get("loss_type", "focal_bce"),
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
        ["epoch", "val_mse", "val_mae", "val_r2", "val_acc", "val_f1"]
        + [f"{d}_mse" for d in dims]
        + [f"{d}_mae" for d in dims]
        + [f"{d}_r2"  for d in dims]
        + [f"{d}_acc" for d in dims]
        + [f"{d}_f1"  for d in dims]
    )
    epoch_writer = MetricsWriter(out_dir / "epoch_metrics.csv", epoch_cols)

    best_val_f1  = -1.0
    best_val_mse = math.inf
    patience_ctr = 0
    patience     = cfg.get("patience", 3)
    best_dir     = out_dir / "best_model"
    summary: Dict[str, Any] = {
        "run_id":     run_id,
        "model":      cfg["model_name"],
        "task":       "personality_classification" if use_bce else "personality_regression",
        "dimensions": dims,
        "hyperparams": {k: v for k, v in cfg.items() if k not in ("run_id",)},
        "data":       {"train_size": len(train_ds), "val_size": len(val_ds)},
        "epochs":     [],
        "best":       {},
    }

    # ── Training loop ─────────────────────────────────────────────────────────
    global_step  = 0
    running_loss = 0.0
    running_n    = 0

    for epoch in range(1, cfg["epochs"] + 1):
        log.info(f"── Epoch {epoch}/{cfg['epochs']} ──────────────────────────────────")
        model.train()

        # Gradual unfreezing: unfreeze encoder after N epochs
        if epoch == freeze_epochs + 1 and freeze_epochs > 0:
            for p in model.encoder.parameters():
                p.requires_grad = True
            log.info("  Encoder unfrozen")

        optimizer.zero_grad()
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"  train ep{epoch}", leave=False, file=sys.stdout)):
            global_step += 1
            batch = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask", "labels")}
            out   = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

            if loss_type == "focal_bce":
                loss = focal_bce_with_logits(out["logits"], batch["labels"], gamma=cfg.get("focal_gamma", 2.0))
            elif loss_type == "bce":
                loss = F.binary_cross_entropy_with_logits(out["logits"], batch["labels"])
            else:
                loss = F.mse_loss(out["preds"], batch["labels"])

            # R-Drop: run a second forward pass, penalize KL divergence between two stochastic outputs
            rdrop_alpha = cfg.get("rdrop_alpha", 0.0)
            if rdrop_alpha > 0 and model.training:
                out2 = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                p1 = torch.sigmoid(out["logits"])
                p2 = torch.sigmoid(out2["logits"])
                kl1 = F.kl_div(torch.log(p1 + 1e-8), p2, reduction='batchmean')
                kl2 = F.kl_div(torch.log(p2 + 1e-8), p1, reduction='batchmean')
                loss = loss + rdrop_alpha * (kl1 + kl2) / 2

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

        log.info(f"  val_mse={m['val_mse']:.6f}  val_mae={m['val_mae']:.6f}  val_r2={m['val_r2']:.4f}  val_acc={m['val_acc']:.4f}  val_f1={m['val_f1']:.4f}")
        for d, dm in m["per_dim"].items():
            log.info(f"    {d:20s} mse={dm['mse']:.6f}  mae={dm['mae']:.6f}  r2={dm['r2']:.4f}  acc={dm['acc']:.4f}  f1={dm['f1']:.4f}")

        epoch_writer.write({
            "epoch":   epoch,
            "val_mse": m["val_mse"],
            "val_mae": m["val_mae"],
            "val_r2":  m["val_r2"],
            "val_acc": m["val_acc"],
            "val_f1":  m["val_f1"],
            **{f"{d}_mse": m["per_dim"][d]["mse"] for d in dims},
            **{f"{d}_mae": m["per_dim"][d]["mae"] for d in dims},
            **{f"{d}_r2":  m["per_dim"][d]["r2"]  for d in dims},
            **{f"{d}_acc": m["per_dim"][d]["acc"] for d in dims},
            **{f"{d}_f1":  m["per_dim"][d]["f1"]  for d in dims},
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

        summary["epochs"].append({"epoch": epoch, **{k: m[k] for k in ("val_mse", "val_mae", "val_r2", "val_acc", "val_f1")}, "per_dim": m["per_dim"]})
        tracker.log_metrics({"val_f1": m["val_f1"], "val_mse": m["val_mse"],
                             "val_acc": m["val_acc"], "val_r2": m["val_r2"]}, step=epoch)
        for d, dm in m["per_dim"].items():
            tracker.log_metrics({f"{d}_f1": dm["f1"], f"{d}_acc": dm["acc"]}, step=epoch)

        # ── Checkpoint (best by F1 for binary, MSE for regression) ───────────
        improved = False
        if use_bce:
            if m["val_f1"] > best_val_f1:
                best_val_f1 = m["val_f1"]
                best_val_mse = m["val_mse"]
                improved = True
        else:
            if m["val_mse"] < best_val_mse:
                best_val_mse = m["val_mse"]
                improved = True

        if improved:
            patience_ctr = 0
            best_dir.mkdir(parents=True, exist_ok=True)
            model.encoder.save_pretrained(best_dir)
            torch.save(model.state_dict(), best_dir / "pytorch_model.bin")
            log.info(f"  ✓ Best model saved → {best_dir}  (val_f1={m['val_f1']:.4f}  val_acc={m['val_acc']:.4f}  val_mse={m['val_mse']:.6f})")
        else:
            patience_ctr += 1
            log.info(f"  No improvement ({patience_ctr}/{patience})")
            if patience_ctr >= patience:
                log.info(f"  Early stopping at epoch {epoch}")
                break

    best_epoch = max(summary["epochs"], key=lambda e: e["val_f1"]) if use_bce else min(summary["epochs"], key=lambda e: e["val_mse"])
    summary["best"] = {"epoch": best_epoch["epoch"], "val_mse": best_val_mse, "val_f1": best_val_f1,
                       "val_acc": best_epoch.get("val_acc", 0)}

    write_metrics_bundle(out_dir, "run_summary", summary, title="Personality Encoder Run Summary")

    # ── MLflow: log final metrics and artifacts ──────────────────────────────────
    tracker.log_metrics({
        "best_val_f1":  best_val_f1,
        "best_val_mse": best_val_mse,
        "best_val_acc": best_epoch.get("val_acc", 0),
        "best_epoch":   best_epoch["epoch"],
    })
    tracker.log_artifact(out_dir / "run_summary.json")
    tracker.log_artifact(out_dir / "run_summary.md")
    tracker.log_artifact(out_dir / "epoch_metrics.csv")
    tracker.end_run()

    log.info("=" * 60)
    log.info(f"DONE  best val_f1={best_val_f1:.4f}  val_acc={best_epoch.get('val_acc', 0):.4f}  val_mse={best_val_mse:.6f}  (epoch {best_epoch['epoch']})")
    log.info(f"Artifacts → {out_dir}")
    log.info("=" * 60)
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train personality encoder")
    p.add_argument("--config",     type=str, help="YAML config file")
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
    args  = parse_args()
    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    cfg   = load_config(args.config, overrides)
    train(cfg)
