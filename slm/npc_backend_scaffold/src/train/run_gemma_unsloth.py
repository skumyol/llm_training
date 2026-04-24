#!/usr/bin/env python3
"""
Gemma 4 Training Runner (E2B / E4B)
=================================
Fine-tunes Gemma 4 MoE models on the scaffold's dialogue JSONL format.

Supported Models:
  - google/gemma-4-E2B: 2 active experts (~2B active params, 16B total)
  - google/gemma-4-E4B: 4 active experts (~4B active params, 16B total)

Gemma 4 uses sparse mixture-of-experts (MoE) with 16B total parameters but
only activates 2B (E2B) or 4B (E4B) per token for efficiency.

Requirements:
  - transformers >= 4.50
  - peft (for LoRA)
  - bitsandbytes (for 4-bit quantization)
  - trl (for SFTTrainer)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import yaml

# Gemma 4 uses native transformers + peft + trl (no Unsloth needed)
# Ensure compatible accelerate version
os.environ['ACCELERATE_FP8'] = '0'

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


DEFAULTS: Dict[str, Any] = {
    # Gemma 4 models: E2B (~2B active), E4B (~4B active)
    "base_model_name": "google/gemma-4-E2B",  # Default: E2B (faster, less VRAM)
    "download_if_missing": True,
    "local_model_root": "models",
    "train_path": "data/dialogue/from_gen_train.jsonl",
    "val_path": "data/dialogue/from_gen_val.jsonl",
    "max_seq_length": 2048,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "weight_decay": 0.01,
    "warmup_steps": 20,
    "epochs": 3,
    "lora_r": 16,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "max_train_samples": None,
    "max_eval_samples": 512,
    "logging_steps": 10,
    "eval_steps": 100,
    "save_steps": 100,
    "seed": 42,
    "output_dir": "artifacts/gemma4",
    "system_prompt_template": (
        "You are roleplaying an NPC in a dialogue simulation. Stay in character, "
        "reply naturally, and use the NPC profile below as a hard constraint.\n"
        "NPC profile: {npc_profile}"
    ),
}


def load_config(config_path: Optional[str], overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(DEFAULTS)
    if config_path:
        with open(config_path, encoding="utf-8") as f:
            cfg.update(yaml.safe_load(f))
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


def setup_logger(log_dir: Path, name: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
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
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def write(self, row: Dict[str, Any]) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(
                {k: row.get(k, "") for k in self.fieldnames}
            )


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_").lower()


def _load_records(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _to_messages(record: Dict[str, Any], system_prompt_template: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{
        "role": "system",
        "content": system_prompt_template.format(npc_profile=record["npc_profile"]),
    }]
    for turn in record.get("dialogue_context", []):
        speaker = turn.get("speaker")
        if speaker not in {"player", "npc"}:
            continue
        messages.append({
            "role": "user" if speaker == "player" else "assistant",
            "content": str(turn.get("text", "")).strip(),
        })
    messages.append({
        "role": "assistant",
        "content": str(record.get("target_response", "")).strip(),
    })
    return messages


def _resolve_model_path(cfg: Dict[str, Any], log: logging.Logger) -> str:
    model_name = cfg["base_model_name"]
    model_path = Path(model_name)
    if model_path.exists():
        log.info(f"Using local base model: {model_path}")
        return str(model_path)

    local_root = ROOT / cfg["local_model_root"]
    local_dir = local_root / _slug(model_name)
    if (local_dir / "config.json").exists():
        log.info(f"Using cached base model: {local_dir}")
        return str(local_dir)

    if not cfg.get("download_if_missing", True):
        raise FileNotFoundError(
            f"Base model '{model_name}' not found locally and download_if_missing=false"
        )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required for automatic Gemma download. "
            "Install it with: pip install huggingface_hub"
        ) from exc

    log.info(f"Downloading base model {model_name} -> {local_dir}")
    local_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_name,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        token=os.environ.get("HF_TOKEN"),
        resume_download=True,
    )
    return str(local_dir)


def _load_gemma4_components(cfg: Dict[str, Any], model_path: str, log: logging.Logger):
    """Load Gemma 4 model with native transformers + PEFT LoRA."""
    base_model = cfg["base_model_name"]
    
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for Gemma 4 training. "
            "Run this stage on a CUDA machine."
        )
    
    log.info(f"Loading Gemma 4 model from {model_path}")
    log.info(f"Model type: {'E2B (2 active experts)' if 'E2B' in base_model else 'E4B (4 active experts)'}")
    
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    
    # 4-bit quantization config for Gemma 4
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    
    # Prepare for LoRA
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config for Gemma 4 (text-only, no vision layers)
    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", 
                      "gate_proj", "up_proj", "down_proj"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f"Gemma 4 loaded: {total_params/1e9:.1f}B total params")
    log.info(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M")
    
    return model, tokenizer


def _build_dataset(records: Iterable[Dict[str, Any]], tokenizer, cfg: Dict[str, Any]):
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required for Gemma 4 training") from exc

    rows: List[Dict[str, str]] = []
    for record in records:
        text = tokenizer.apply_chat_template(
            _to_messages(record, cfg["system_prompt_template"]),
            tokenize=False,
            add_generation_prompt=False,
        ).removeprefix("<bos>")
        rows.append({"text": text})
    return Dataset.from_list(rows)


def _safe_ppl(loss_value: Optional[float]) -> Optional[float]:
    if loss_value is None:
        return None
    return math.exp(min(float(loss_value), 20.0))


def train(cfg: Dict[str, Any]) -> Dict[str, Any]:
    run_id = cfg.get("run_id") or f"gemma4_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(cfg["output_dir"]) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log = setup_logger(out_dir, run_id)

    torch.manual_seed(cfg["seed"])

    log.info("=" * 60)
    log.info(f"RUN      : {run_id}")
    log.info(f"BACKBONE : {cfg['base_model_name']}")
    log.info(f"TRAIN    : {cfg['train_path']}")
    log.info(f"VAL      : {cfg['val_path']}")
    log.info("=" * 60)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    for key in ("train_path", "val_path"):
        if not Path(cfg[key]).exists():
            raise FileNotFoundError(f"Missing dataset file: {cfg[key]}")

    # Validate Gemma 4 only
    if not _is_gemma4(cfg["base_model_name"]):
        raise ValueError(
            f"This script only supports Gemma 4 models (E2B/E4B). "
            f"Got: {cfg['base_model_name']}. "
            f"Use 'google/gemma-4-E2B' or 'google/gemma-4-E4B'"
        )
    
    model_path = _resolve_model_path(cfg, log)
    model, tokenizer = _load_gemma4_components(cfg, model_path, log)

    train_records = _load_records(cfg["train_path"], cfg.get("max_train_samples"))
    val_records = _load_records(cfg["val_path"], cfg.get("max_eval_samples"))

    train_ds = _build_dataset(train_records, tokenizer, cfg)
    val_ds = _build_dataset(val_records, tokenizer, cfg)

    log.info(f"Train: {len(train_ds):,} examples | Val: {len(val_ds):,} examples")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Parameters: {total_params:,} total | {trainable_params:,} trainable")

    try:
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError("trl is required for Gemma 4 training") from exc

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        args=SFTConfig(
            output_dir=str(out_dir / "checkpoints"),
            per_device_train_batch_size=cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            learning_rate=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
            warmup_steps=cfg["warmup_steps"],
            num_train_epochs=cfg["epochs"],
            logging_steps=cfg["logging_steps"],
            eval_steps=cfg["eval_steps"],
            save_steps=cfg["save_steps"],
            eval_strategy="steps",
            save_strategy="steps",
            save_total_limit=2,
            seed=cfg["seed"],
            report_to=[],
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            max_seq_length=cfg["max_seq_length"],
        ),
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    best_dir = out_dir / "best_model"
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(best_dir)

    train_loss = None
    if hasattr(train_result, "training_loss"):
        train_loss = float(train_result.training_loss)

    eval_loss = eval_metrics.get("eval_loss")
    epoch_metrics = {
        "epoch": cfg["epochs"],
        "train_loss": train_loss,
        "val_loss": eval_loss,
        "val_ppl": _safe_ppl(eval_loss),
    }

    MetricsWriter(
        out_dir / "epoch_metrics.csv",
        ["epoch", "train_loss", "val_loss", "val_ppl"],
    ).write(epoch_metrics)

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "backbone": cfg["base_model_name"],
        "task": "dialogue_lm_gemma4",
        "framework": "transformers+peft",
        "hyperparams": {k: v for k, v in cfg.items() if k != "run_id"},
        "data": {
            "train_size": len(train_ds),
            "val_size": len(val_ds),
            "train_path": cfg["train_path"],
            "val_path": cfg["val_path"],
        },
        "model_stats": {
            "total_params": total_params,
            "trainable_params": trainable_params,
        },
        "epochs": [epoch_metrics],
        "best": {
            "epoch": cfg["epochs"],
            "train_loss": train_loss,
            "val_loss": eval_loss,
            "val_ppl": _safe_ppl(eval_loss),
        },
    }

    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log.info("=" * 60)
    log.info(
        f"DONE  val_loss={summary['best']['val_loss']:.4f}  "
        f"val_ppl={summary['best']['val_ppl']:.2f}"
        if summary["best"]["val_loss"] is not None
        else "DONE"
    )
    log.info(f"Artifacts → {out_dir}")
    log.info("=" * 60)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Gemma 4 (E2B/E4B) with native transformers + PEFT")
    p.add_argument("--config", type=str)
    p.add_argument("--run-id", type=str, dest="run_id")
    p.add_argument("--base-model-name", type=str, dest="base_model_name")
    p.add_argument("--train-path", type=str, dest="train_path")
    p.add_argument("--val-path", type=str, dest="val_path")
    p.add_argument("--output-dir", type=str, dest="output_dir")
    p.add_argument("--local-model-root", type=str, dest="local_model_root")
    p.add_argument("--download-if-missing", type=str, dest="download_if_missing")
    p.add_argument("--per-device-train-batch-size", type=int, dest="per_device_train_batch_size")
    p.add_argument("--gradient-accumulation-steps", type=int, dest="gradient_accumulation_steps")
    p.add_argument("--learning-rate", type=float, dest="learning_rate")
    p.add_argument("--weight-decay", type=float, dest="weight_decay")
    p.add_argument("--warmup-steps", type=int, dest="warmup_steps")
    p.add_argument("--epochs", type=int)
    p.add_argument("--max-seq-length", type=int, dest="max_seq_length")
    p.add_argument("--lora-r", type=int, dest="lora_r")
    p.add_argument("--lora-alpha", type=int, dest="lora_alpha")
    p.add_argument("--lora-dropout", type=float, dest="lora_dropout")
    p.add_argument("--max-train-samples", type=int, dest="max_train_samples")
    p.add_argument("--max-eval-samples", type=int, dest="max_eval_samples")
    p.add_argument("--logging-steps", type=int, dest="logging_steps")
    p.add_argument("--eval-steps", type=int, dest="eval_steps")
    p.add_argument("--save-steps", type=int, dest="save_steps")
    p.add_argument("--seed", type=int)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    if isinstance(overrides.get("download_if_missing"), str):
        overrides["download_if_missing"] = overrides["download_if_missing"].lower() in {"1", "true", "yes"}
    cfg = load_config(args.config, overrides)
    train(cfg)
