#!/usr/bin/env python3
"""
Dialogue Model Training Script
================================
Trains a LoRA + soft-prefix conditioned dialogue model on NPC conversation data.
Conditioning is driven by personality (OCEAN) and affect (VAD) vectors.

Prerequisites (must exist before running):
  artifacts/affect_encoder/best_model/    ← from run_affect.py
  artifacts/personality_cache.jsonl       ← from src/data/build_caches.py

Usage:
  python -m src.train.run_dialogue
  python -m src.train.run_dialogue --config configs/dialogue.yaml
  python -m src.train.run_dialogue --config configs/dialogue.yaml --lr 1e-4 --epochs 5
  python -m src.train.run_dialogue --run-id ablation_lora_r32

Artifacts produced under output_dir/run_id/:
  best_model/           LoRA adapter weights + prefix encoder + tokenizer
  run.log               full structured log
  step_metrics.csv      per-step train loss, lr, grad_norm
  epoch_metrics.csv     per-epoch val loss + perplexity
  run_summary.json      hyperparams + results (ablation table row)
  run_summary.md        human-readable summary report
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

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, ConstantLR
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "train"))

from src.common.config import DialogueTrainConfig
from src.data.datasets import DialogueJsonlDataset
from src.infer.memory_store import EpisodicMemoryStore
from src.models.dialogue import ConditionalDialogueModel
from src.train.train_dialogue import DialogueBatch, DialogueCollator
from metrics_report import log_metrics_to_mlflow, write_metrics_bundle


# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULTS: Dict[str, Any] = {
    "base_model_name":         "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "train_path":              "data/dialogue/train.jsonl",
    "val_path":                "data/dialogue/val.jsonl",
    "personality_cache_path":  "artifacts/personality_cache.jsonl",
    "affect_encoder_path":     "artifacts/affect_encoder/best_model",
    "sentence_transformer_name": "sentence-transformers/all-MiniLM-L6-v2",
    "lora_r":                  16,
    "lora_alpha":              32,
    "lora_dropout":            0.05,
    "target_modules":          ["q_proj", "k_proj", "v_proj", "o_proj"],
    "prefix_length":           8,
    "max_source_length":       768,
    "max_target_length":       192,
    "batch_size":              2,
    "grad_accum_steps":        8,
    "lr":                      2e-4,
    "warmup_steps":            100,
    "epochs":                  3,
    "log_every":               10,
    "seed":                    42,
    "output_dir":              "artifacts/dialogue_model",
}


def load_config(config_path: Optional[str], overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(DEFAULTS)
    if config_path:
        with open(config_path) as f:
            cfg.update(yaml.safe_load(f))
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
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


# ── MLflow tracking ──────────────────────────────────────────────────────────

def _init_mlflow(cfg: Dict[str, Any], run_id: str, log) -> object:
    """Initialize MLflowTracker. Returns tracker or None for graceful degradation."""
    sys.path.insert(0, str(ROOT / "src" / "train"))
    from mlflow_tracker import MLflowTracker
    tracker = MLflowTracker(
        experiment=cfg.get("mlflow_experiment", "dialogue_model"),
        enabled=cfg.get("mlflow_enabled", True),
    )
    tracker.start_run(run_name=run_id, tags={
        "task":   "dialogue",
        "seed":   str(cfg["seed"]),
        "model":  cfg.get("base_model_name", "?"),
        "lora_r": str(cfg.get("lora_r", "?")),
    })
    tracker.log_params(cfg)
    return tracker


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _cfg_to_dialogue_config(cfg: Dict[str, Any]) -> DialogueTrainConfig:
    """Convert flat dict to DialogueTrainConfig for DialogueCollator."""
    return DialogueTrainConfig(
        base_model_name=cfg["base_model_name"],
        train_path=cfg["train_path"],
        val_path=cfg["val_path"],
        personality_cache_path=cfg["personality_cache_path"],
        affect_encoder_path=cfg["affect_encoder_path"],
        sentence_transformer_name=cfg["sentence_transformer_name"],
        lora_r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        prefix_length=cfg["prefix_length"],
        max_source_length=cfg["max_source_length"],
        max_target_length=cfg["max_target_length"],
        batch_size=cfg["batch_size"],
        grad_accum_steps=cfg["grad_accum_steps"],
        lr=cfg["lr"],
        epochs=cfg["epochs"],
        output_dir=cfg["output_dir"],
        memory_top_k=cfg.get("memory_top_k", 3),
    )


@torch.no_grad()
def evaluate(model: ConditionalDialogueModel, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    for batch in loader:
        out = model(
            input_ids=batch.input_ids.to(device),
            attention_mask=batch.attention_mask.to(device),
            cond_vec=batch.cond_vec.to(device),
            labels=batch.labels.to(device),
        )
        total_loss += out.loss.item()
        n_batches  += 1
    mean_loss = total_loss / max(n_batches, 1)
    ppl = math.exp(min(mean_loss, 20))
    return {"val_loss": mean_loss, "val_ppl": ppl}


def train(cfg: Dict[str, Any]) -> Dict[str, Any]:
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    run_id  = cfg.get("run_id") or f"dialogue_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(cfg["output_dir"]) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    log = setup_logger(out_dir, run_id)
    log.info("=" * 60)
    log.info(f"RUN      : {run_id}")
    log.info(f"BACKBONE : {cfg['base_model_name']}")
    log.info(f"LoRA r   : {cfg['lora_r']}  alpha={cfg['lora_alpha']}")
    log.info(f"Prefix L : {cfg['prefix_length']}")
    log.info(f"Eff.batch: {cfg['batch_size'] * cfg['grad_accum_steps']}")
    log.info("=" * 60)
    log.debug(f"Full config:\n{json.dumps(cfg, indent=2)}")

    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # ── Prerequisite checks ───────────────────────────────────────────────────
    for key, hint in [
        ("train_path",             "dialogue training data"),
        ("val_path",               "dialogue validation data"),
        ("personality_cache_path", "run src/data/build_caches.py first"),
        ("affect_encoder_path",    "run src/train/run_affect.py first"),
    ]:
        p = Path(cfg[key])
        if not p.exists():
            log.error(f"Missing prerequisite: {p}  [{hint}]")
            raise FileNotFoundError(f"Missing: {p}")

    # ── Datasets + collator ───────────────────────────────────────────────────
    device     = select_device()
    log.info(f"Device: {device}")

    dcfg       = _cfg_to_dialogue_config(cfg)
    train_ds   = DialogueJsonlDataset(cfg["train_path"])
    val_ds     = DialogueJsonlDataset(cfg["val_path"])
    log.info(f"Train: {len(train_ds):,} examples | Val: {len(val_ds):,} examples")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ConditionalDialogueModel(
        base_model_name=cfg["base_model_name"],
        cond_dim=8,                           # OCEAN(5) + VAD(3)
        prefix_length=cfg["prefix_length"],
        lora_r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
    ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    log.info(f"Parameters: {total:,} total | {trainable:,} trainable ({100*trainable/total:.2f}%)")

    collator   = DialogueCollator(dcfg, model.tokenizer, device)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  collate_fn=collator)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, collate_fn=collator)

    # ── Optimizer + warmup scheduler ─────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=cfg["lr"])
    warmup_steps = cfg.get("warmup_steps", 0)
    total_steps  = len(train_loader) * cfg["epochs"] // cfg["grad_accum_steps"]
    if warmup_steps > 0 and total_steps > warmup_steps:
        warmup_sched   = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
        constant_sched = ConstantLR(optimizer, factor=1.0, total_iters=total_steps - warmup_steps)
        scheduler      = SequentialLR(optimizer, schedulers=[warmup_sched, constant_sched], milestones=[warmup_steps])
    else:
        scheduler = None

    # ── MLflow tracking ──────────────────────────────────────────────────────
    tracker = _init_mlflow(cfg, run_id, log)

    # ── Metric writers ────────────────────────────────────────────────────────
    step_writer  = MetricsWriter(
        out_dir / "step_metrics.csv",
        ["epoch", "global_step", "train_loss", "lr", "grad_norm"],
    )
    epoch_writer = MetricsWriter(
        out_dir / "epoch_metrics.csv",
        ["epoch", "val_loss", "val_ppl"],
    )

    best_val_loss = math.inf
    best_dir      = out_dir / "best_model"
    summary: Dict[str, Any] = {
        "run_id":      run_id,
        "backbone":    cfg["base_model_name"],
        "task":        "dialogue_lm",
        "hyperparams": {k: v for k, v in cfg.items() if k not in ("run_id",)},
        "data":        {"train_size": len(train_ds), "val_size": len(val_ds)},
        "model_stats": {"total_params": total, "trainable_params": trainable},
        "epochs":      [],
        "best":        {},
    }

    # ── Training loop ─────────────────────────────────────────────────────────
    global_step  = 0
    accum_loss   = 0.0
    accum_n      = 0

    for epoch in range(1, cfg["epochs"] + 1):
        log.info(f"── Epoch {epoch}/{cfg['epochs']} ──────────────────────────────────")
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(
            tqdm(train_loader, desc=f"  train ep{epoch}", leave=False, file=sys.stdout), start=1
        ):
            out  = model(
                input_ids=batch.input_ids.to(device),
                attention_mask=batch.attention_mask.to(device),
                cond_vec=batch.cond_vec.to(device),
                labels=batch.labels.to(device),
            )
            loss = out.loss / cfg["grad_accum_steps"]
            loss.backward()

            accum_loss += out.loss.item()
            accum_n    += 1

            if batch_idx % cfg["grad_accum_steps"] == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
                optimizer.step()
                if scheduler:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % cfg["log_every"] == 0:
                    avg_loss = accum_loss / accum_n
                    cur_lr   = optimizer.param_groups[0]["lr"]
                    ppl      = math.exp(min(avg_loss, 20))
                    log.info(
                        f"  step {global_step:5d} | loss={avg_loss:.4f}  ppl={ppl:.2f}"
                        f"  lr={cur_lr:.2e}  grad_norm={grad_norm:.4f}"
                    )
                    step_writer.write({
                        "epoch": epoch, "global_step": global_step,
                        "train_loss": avg_loss, "lr": cur_lr, "grad_norm": grad_norm,
                    })
                    tracker.log_metrics({
                        "train_loss": avg_loss, "lr": cur_lr, "grad_norm": grad_norm,
                    }, step=global_step)
                    accum_loss = 0.0
                    accum_n    = 0

        # ── Validation ────────────────────────────────────────────────────────
        val_metrics = evaluate(model, val_loader, device)
        log.info(
            f"  val_loss={val_metrics['val_loss']:.4f}  val_ppl={val_metrics['val_ppl']:.2f}"
        )
        epoch_writer.write({"epoch": epoch, **val_metrics})
        summary["epochs"].append({"epoch": epoch, **val_metrics})
        tracker.log_metrics(val_metrics, step=epoch)

        # ── Checkpoint ────────────────────────────────────────────────────────
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_dir.mkdir(parents=True, exist_ok=True)
            model.model.save_pretrained(best_dir)
            model.tokenizer.save_pretrained(best_dir)
            torch.save(model.prefix.state_dict(), best_dir / "prefix_encoder.pt")
            log.info(f"  ✓ Best model saved → {best_dir}  (val_loss={best_val_loss:.4f})")

    best_epoch = min(summary["epochs"], key=lambda e: e["val_loss"])
    summary["best"] = {
        "epoch":      best_epoch["epoch"],
        "val_loss":   best_val_loss,
        "val_ppl":    best_epoch["val_ppl"],
    }

    with open(out_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    write_metrics_bundle(out_dir, "run_summary", summary, title="Dialogue Run Summary")

    log_metrics_to_mlflow(tracker, {"best_val_loss": best_val_loss, "best_val_ppl": best_epoch["val_ppl"]})
    tracker.log_artifact(out_dir / "run_summary.json")
    tracker.log_artifact(out_dir / "run_summary.md")
    tracker.log_artifact(out_dir / "epoch_metrics.csv")
    tracker.end_run()

    log.info("=" * 60)
    log.info(f"DONE  best val_loss={best_val_loss:.4f}  ppl={best_epoch['val_ppl']:.2f}  (epoch {best_epoch['epoch']})")
    log.info(f"Artifacts → {out_dir}")
    log.info("=" * 60)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train NPC dialogue model")
    p.add_argument("--config",              type=str)
    p.add_argument("--run-id",              type=str,   dest="run_id")
    p.add_argument("--base-model-name",     type=str,   dest="base_model_name")
    p.add_argument("--train-path",          type=str,   dest="train_path")
    p.add_argument("--val-path",            type=str,   dest="val_path")
    p.add_argument("--personality-cache",   type=str,   dest="personality_cache_path")
    p.add_argument("--affect-encoder",      type=str,   dest="affect_encoder_path")
    p.add_argument("--output-dir",          type=str,   dest="output_dir")
    p.add_argument("--batch-size",          type=int,   dest="batch_size")
    p.add_argument("--grad-accum",          type=int,   dest="grad_accum_steps")
    p.add_argument("--lr",                  type=float)
    p.add_argument("--warmup-steps",        type=int,   dest="warmup_steps")
    p.add_argument("--epochs",              type=int)
    p.add_argument("--lora-r",              type=int,   dest="lora_r")
    p.add_argument("--lora-alpha",          type=int,   dest="lora_alpha")
    p.add_argument("--prefix-length",       type=int,   dest="prefix_length")
    p.add_argument("--log-every",           type=int,   dest="log_every")
    p.add_argument("--seed",                type=int)
    return p.parse_args()


if __name__ == "__main__":
    args      = parse_args()
    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    cfg       = load_config(args.config, overrides)
    train(cfg)
