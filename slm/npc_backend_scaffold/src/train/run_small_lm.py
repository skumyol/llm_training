#!/usr/bin/env python3
"""
Small-LM Training Runner
=========================
Trains one from-scratch small language model for NPC dialogue.
Produces the same logging/artifact format as run_dialogue.py for direct A/B comparison.

Supported architectures:
  gru | awdlstm | gpt | prefix_gpt | moe | mamba_like

Usage:
  python -m src.train.run_small_lm --arch gpt
  python -m src.train.run_small_lm --arch prefix_gpt --config configs/small_lm.yaml
  python -m src.train.run_small_lm --arch awdlstm --run-id ablation_awdlstm_01 \\
      --train-text data/dialogue/train.txt --val-text data/dialogue/val.txt

  # Benchmark all architectures sequentially on the same data split:
  for arch in gru awdlstm gpt prefix_gpt moe mamba_like; do
    python -m src.train.run_small_lm --arch $arch --run-id bench_$(date +%s)_$arch
  done

A/B comparison note:
  val_ppl in run_summary.json is on your NPC dialogue corpus using tiktoken (GPT-2 BPE).
  This is comparable across all six architectures here, and to ConditionalDialogueModel
  (which uses the same tokenizer family). Use val_ppl as the primary ablation metric.

Artifacts under output_dir/run_id/:
  best_model.pt          checkpoint (state_dict + config)
  run.log                structured log
  step_metrics.csv       per-step train_loss, lr, grad_norm
  epoch_metrics.csv      per-epoch val_loss, val_ppl
  run_summary.json       hyperparams + results (ablation row)
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
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "train"))  # for small_lm_architectures
sys.path.insert(0, str(ROOT))

from small_lm_architectures import (
    LMOutput,
    PrefixTinyGPTLM,
    RECOMMENDED_CONFIGS,
    build_model,
    select_device,
)

try:
    import tiktoken
    _TIKTOKEN_OK = True
except ImportError:
    _TIKTOKEN_OK = False


# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULTS: Dict[str, Any] = {
    "arch":              "gpt",
    "hardware_profile":  "m1_small",
    "train_text":        "data/dialogue/train.txt",
    "val_text":          "data/dialogue/val.txt",
    "seq_len":           256,
    "batch_size":        16,
    "grad_accum":        4,
    "lr":                3e-4,
    "weight_decay":      0.1,
    "epochs":            3,
    "log_every":         20,
    "eval_every_steps":  200,
    "seed":              42,
    "output_dir":        "artifacts/small_lm",
    "cond_dim":          8,    # OCEAN(5)+VAD(3); used by prefix_gpt only
    "use_amp":           False,
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


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logger(log_dir: Path, name: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch  = logging.StreamHandler(sys.stdout); ch.setLevel(logging.INFO); ch.setFormatter(fmt)
    fh  = logging.FileHandler(log_dir / "run.log", mode="w"); fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
    logger.addHandler(ch); logger.addHandler(fh)
    return logger


class MetricsWriter:
    def __init__(self, path: Path, fieldnames: List[str]) -> None:
        self.path, self.fieldnames = path, fieldnames
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def write(self, row: Dict) -> None:
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(
                {k: row.get(k, "") for k in self.fieldnames}
            )


# ── Tokenizer ─────────────────────────────────────────────────────────────────

class CharTokenizer:
    def __init__(self, text: str) -> None:
        vocab = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(vocab)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(vocab)
        self.name = "char"

    def encode(self, text: str) -> List[int]:
        return [self.stoi[c] for c in text if c in self.stoi]


def build_tokenizer(text: str):
    if _TIKTOKEN_OK:
        enc = tiktoken.get_encoding("gpt2")
        enc.name = "tiktoken:gpt2"  # type: ignore[attr-defined]
        enc.vocab_size = enc.n_vocab  # type: ignore[attr-defined]
        return enc
    return CharTokenizer(text)


# ── Dataset ───────────────────────────────────────────────────────────────────

class TokenDataset(Dataset):
    def __init__(self, ids: List[int], seq_len: int) -> None:
        self.t   = torch.tensor(ids, dtype=torch.long)
        self.seq = seq_len

    def __len__(self) -> int:
        return max(0, (len(self.t) - 1) // self.seq)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = idx * self.seq
        x = self.t[s:s + self.seq]
        y = self.t[s + 1:s + self.seq + 1]
        if len(x) < self.seq:
            x = torch.cat([x, torch.zeros(self.seq - len(x), dtype=torch.long)])
        if len(y) < self.seq:
            y = torch.cat([y, torch.full((self.seq - len(y),), -100, dtype=torch.long)])
        return x, y


# ── AMP context ───────────────────────────────────────────────────────────────

def amp_ctx(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return torch.autocast(device_type="cpu", enabled=False)


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device,
    cond_dim: int, use_amp: bool, max_batches: int = 200,
) -> Dict[str, float]:
    model.eval()
    losses = []
    for bi, (x, y) in enumerate(loader):
        if bi >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with amp_ctx(device, use_amp):
            if isinstance(model, PrefixTinyGPTLM):
                cond = torch.zeros(x.size(0), cond_dim, device=device)
                out  = model(x, cond, y)
            else:
                out  = model(x, y)
        losses.append(out.loss.item())
    mean = sum(losses) / max(len(losses), 1)
    return {"val_loss": mean, "val_ppl": math.exp(min(mean, 20))}


# ── Main ──────────────────────────────────────────────────────────────────────

def train(cfg: Dict[str, Any]) -> Dict[str, Any]:
    random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])

    arch   = cfg["arch"].lower()
    run_id = cfg.get("run_id") or f"{arch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(cfg["output_dir"]) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    log = setup_logger(out_dir, run_id)
    log.info("=" * 60)
    log.info(f"RUN      : {run_id}")
    log.info(f"ARCH     : {arch}")
    log.info(f"PROFILE  : {cfg['hardware_profile']}")
    log.info("=" * 60)
    log.debug(f"Config:\n{json.dumps(cfg, indent=2)}")

    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # ── Data ──────────────────────────────────────────────────────────────────
    for key in ("train_text", "val_text"):
        p = Path(cfg[key])
        if not p.exists():
            log.error(f"Missing text file: {p}")
            log.error("Run prepare_dialogue_data.py first to produce .txt splits.")
            raise FileNotFoundError(str(p))

    train_text = Path(cfg["train_text"]).read_text(encoding="utf-8")
    val_text   = Path(cfg["val_text"]).read_text(encoding="utf-8")

    tokenizer  = build_tokenizer(train_text)
    log.info(f"Tokenizer: {tokenizer.name}  vocab={tokenizer.vocab_size:,}")

    train_ids  = tokenizer.encode(train_text)
    val_ids    = tokenizer.encode(val_text)
    log.info(f"Tokens — train: {len(train_ids):,}  val: {len(val_ids):,}")

    train_ds   = TokenDataset(train_ids, cfg["seq_len"])
    val_ds     = TokenDataset(val_ids,   cfg["seq_len"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    device  = select_device()
    profile = RECOMMENDED_CONFIGS.get(cfg["hardware_profile"], {})
    params  = dict(profile.get(arch, {}))
    params["vocab_size"] = tokenizer.vocab_size
    if "max_seq_len" in params:
        params["max_seq_len"] = cfg["seq_len"]
    if arch == "prefix_gpt":
        params["cond_dim"] = cfg["cond_dim"]

    model   = build_model(arch, params).to(device)
    total   = sum(p.numel() for p in model.parameters())
    log.info(f"Device: {device}  |  Parameters: {total:,} ({total/1e6:.1f} M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler    = torch.cuda.amp.GradScaler(enabled=(cfg["use_amp"] and device.type == "cuda"))

    # ── Metric writers ────────────────────────────────────────────────────────
    step_writer  = MetricsWriter(out_dir / "step_metrics.csv",  ["epoch", "global_step", "train_loss", "grad_norm"])
    epoch_writer = MetricsWriter(out_dir / "epoch_metrics.csv", ["epoch", "val_loss", "val_ppl"])

    best_val  = math.inf
    best_path = out_dir / "best_model.pt"
    summary: Dict[str, Any] = {
        "run_id":      run_id,
        "arch":        arch,
        "task":        "dialogue_lm_from_scratch",
        "hyperparams": {k: v for k, v in cfg.items() if k not in ("run_id",)},
        "model_params": total,
        "tokenizer":   tokenizer.name,
        "data":        {"train_tokens": len(train_ids), "val_tokens": len(val_ids)},
        "epochs":      [],
        "best":        {},
    }

    global_step  = 0
    running_loss = 0.0
    running_n    = 0

    for epoch in range(1, cfg["epochs"] + 1):
        log.info(f"── Epoch {epoch}/{cfg['epochs']} ──────────────────────────────────")
        model.train()
        optimizer.zero_grad(set_to_none=True)

        for bi, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device), y.to(device)

            with amp_ctx(device, cfg["use_amp"]):
                if isinstance(model, PrefixTinyGPTLM):
                    cond = torch.zeros(x.size(0), cfg["cond_dim"], device=device)
                    out  = model(x, cond, y)
                else:
                    out  = model(x, y)
                loss = out.loss / cfg["grad_accum"]

            if cfg["use_amp"] and device.type == "cuda":
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += out.loss.item()
            running_n    += 1

            if bi % cfg["grad_accum"] == 0:
                if cfg["use_amp"] and device.type == "cuda":
                    scaler.unscale_(optimizer)
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
                    scaler.step(optimizer); scaler.update()
                else:
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % cfg["log_every"] == 0:
                    avg  = running_loss / running_n
                    ppl  = math.exp(min(avg, 20))
                    log.info(f"  step {global_step:5d} | loss={avg:.4f}  ppl={ppl:.2f}  grad_norm={gn:.4f}")
                    step_writer.write({"epoch": epoch, "global_step": global_step, "train_loss": avg, "grad_norm": gn})
                    running_loss = 0.0; running_n = 0

                if global_step % cfg["eval_every_steps"] == 0:
                    vm = evaluate(model, val_loader, device, cfg["cond_dim"], cfg["use_amp"])
                    log.info(f"  [eval] val_loss={vm['val_loss']:.4f}  val_ppl={vm['val_ppl']:.2f}")
                    model.train()

        # ── End-of-epoch validation ───────────────────────────────────────────
        vm = evaluate(model, val_loader, device, cfg["cond_dim"], cfg["use_amp"])
        log.info(f"  epoch {epoch} end → val_loss={vm['val_loss']:.4f}  val_ppl={vm['val_ppl']:.2f}")
        epoch_writer.write({"epoch": epoch, **vm})
        summary["epochs"].append({"epoch": epoch, **vm})

        if vm["val_loss"] < best_val:
            best_val = vm["val_loss"]
            torch.save({
                "arch":   arch, "params": params,
                "state":  model.state_dict(),
                "epoch":  epoch, "val_loss": best_val,
            }, best_path)
            log.info(f"  ✓ Best checkpoint saved  (val_loss={best_val:.4f})")

    best_ep = min(summary["epochs"], key=lambda e: e["val_loss"])
    summary["best"] = {"epoch": best_ep["epoch"], "val_loss": best_val, "val_ppl": best_ep["val_ppl"]}

    with open(out_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("=" * 60)
    log.info(f"DONE  arch={arch}  best val_ppl={best_ep['val_ppl']:.2f}  (epoch {best_ep['epoch']})")
    log.info(f"Artifacts → {out_dir}")
    log.info("=" * 60)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a small LM for NPC dialogue")
    p.add_argument("--config",           type=str)
    p.add_argument("--arch",             type=str, choices=["gru", "awdlstm", "gpt", "prefix_gpt", "moe", "mamba_like"])
    p.add_argument("--run-id",           type=str,   dest="run_id")
    p.add_argument("--hardware-profile", type=str,   dest="hardware_profile",
                   choices=["m1_small", "rtx4070_small"])
    p.add_argument("--train-text",       type=str,   dest="train_text")
    p.add_argument("--val-text",         type=str,   dest="val_text")
    p.add_argument("--seq-len",          type=int,   dest="seq_len")
    p.add_argument("--batch-size",       type=int,   dest="batch_size")
    p.add_argument("--grad-accum",       type=int,   dest="grad_accum")
    p.add_argument("--lr",               type=float)
    p.add_argument("--epochs",           type=int)
    p.add_argument("--output-dir",       type=str,   dest="output_dir")
    p.add_argument("--log-every",        type=int,   dest="log_every")
    p.add_argument("--eval-every-steps", type=int,   dest="eval_every_steps")
    p.add_argument("--seed",             type=int)
    p.add_argument("--amp",              action="store_true", dest="use_amp")
    return p.parse_args()


if __name__ == "__main__":
    args      = parse_args()
    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    if overrides.get("use_amp") is False:
        overrides.pop("use_amp")
    cfg = load_config(args.config, overrides)
    train(cfg)
