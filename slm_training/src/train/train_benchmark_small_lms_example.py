#!/usr/bin/env python3
"""
train_benchmark_small_lms.py

Unified training / evaluation / benchmarking harness for the small language-model
architectures in `small_lm_architectures.py`.

What it includes
----------------
- One training loop for:
    * awdlstm
    * gpt
    * prefix_gpt
    * moe
    * mamba_like
    * gru
- Gradient accumulation
- CUDA mixed precision (AMP) when available
- MPS-safe training path for Apple Silicon
- Shared dataset split and comparable evaluation
- Sequential benchmark runner across multiple architectures

Typical usage
-------------
# 1) Train a single model on a local text file
python train_benchmark_small_lms.py \
  --train-text data/dialogue.txt \
  --arch gpt \
  --out-dir runs/gpt_run

# 2) Benchmark several architectures on the same split
python train_benchmark_small_lms.py \
  --train-text data/dialogue.txt \
  --benchmark awdlstm gpt prefix_gpt moe mamba_like \
  --out-dir runs/bench_01

# 3) Use a Hugging Face dataset + text field
python train_benchmark_small_lms.py \
  --hf-dataset microsoft/DialoGPT-medium \
  --hf-split train \
  --text-field text \
  --arch gpt \
  --max-chars 5000000

Notes
-----
- For PrefixTinyGPTLM, this script generates a placeholder conditioning vector
  of zeros by default so the benchmark stays comparable. You can later replace
  `build_cond_vec(...)` with your personality/affect vectors.
- This script uses a simple BPE tokenizer from `tiktoken` if available,
  otherwise falls back to character-level tokenization.
"""

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# local import from the architecture file already created
from small_lm_architectures import (
    AWDLSTMConfig,
    GPTConfig,
    GRUConfig,
    LMOutput,
    MambaLikeConfig,
    MoEConfig,
    PrefixGPTConfig,
    PrefixTinyGPTLM,
    RECOMMENDED_CONFIGS,
    SmallGRULM,
    AWDLSTMLM,
    TinyGPTLM,
    TinyMoELM,
    MambaLikeLM,
    build_model,
    select_device,
)

try:
    import tiktoken  # type: ignore
    HAS_TIKTOKEN = True
except Exception:
    HAS_TIKTOKEN = False

try:
    from datasets import load_dataset  # type: ignore
    HAS_DATASETS = True
except Exception:
    HAS_DATASETS = False


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------

class CharTokenizer:
    def __init__(self, text: str):
        vocab = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(vocab)
        self.name = "char"

    def encode(self, text: str) -> List[int]:
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids)


class TikTokenWrapper:
    def __init__(self, name: str = "gpt2"):
        enc = tiktoken.get_encoding(name)
        self.enc = enc
        self.vocab_size = enc.n_vocab
        self.name = f"tiktoken:{name}"

    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text)

    def decode(self, ids: List[int]) -> str:
        return self.enc.decode(ids)


def build_tokenizer(text: str, prefer_tiktoken: bool = True):
    if prefer_tiktoken and HAS_TIKTOKEN:
        return TikTokenWrapper("gpt2")
    return CharTokenizer(text)


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def read_text_sources(
    train_text: Optional[str],
    hf_dataset: Optional[str],
    hf_split: str,
    text_field: str,
    max_chars: Optional[int],
) -> str:
    chunks: List[str] = []

    if train_text:
        p = Path(train_text)
        if p.is_file():
            chunks.append(p.read_text(encoding="utf-8"))
        else:
            raise FileNotFoundError(f"train text not found: {train_text}")

    if hf_dataset:
        if not HAS_DATASETS:
            raise RuntimeError("huggingface datasets package is not installed")
        ds = load_dataset(hf_dataset, split=hf_split)
        for item in ds:
            if text_field in item and item[text_field]:
                chunks.append(str(item[text_field]))
        # keep one separator between documents
        chunks = [c.strip() for c in chunks if c and str(c).strip()]
        chunks = [c + "\n\n" for c in chunks]

    if not chunks:
        raise ValueError("No text source provided. Use --train-text and/or --hf-dataset")

    text = "".join(chunks)
    if max_chars is not None:
        text = text[:max_chars]
    return text


class NextTokenDataset(Dataset):
    def __init__(self, token_ids: List[int], seq_len: int):
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, (len(self.tokens) - 1) // self.seq_len)

    def __getitem__(self, idx: int):
        start = idx * self.seq_len
        x = self.tokens[start:start + self.seq_len]
        y = self.tokens[start + 1:start + self.seq_len + 1]
        if len(x) < self.seq_len:
            pad = self.seq_len - len(x)
            x = torch.cat([x, torch.zeros(pad, dtype=torch.long)], dim=0)
        if len(y) < self.seq_len:
            pad = self.seq_len - len(y)
            y = torch.cat([y, torch.full((pad,), -100, dtype=torch.long)], dim=0)
        return x, y


def build_splits(
    token_ids: List[int],
    seq_len: int,
    train_frac: float = 0.9,
    val_frac: float = 0.05,
) -> Tuple[Dataset, Dataset, Dataset]:
    n = len(token_ids)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_ds = NextTokenDataset(token_ids[:train_end], seq_len)
    val_ds = NextTokenDataset(token_ids[train_end:val_end], seq_len)
    test_ds = NextTokenDataset(token_ids[val_end:], seq_len)
    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------

def build_model_from_arch(
    arch: str,
    vocab_size: int,
    hardware_profile: str,
    seq_len: int,
    cond_dim: int = 128,
):
    profile = RECOMMENDED_CONFIGS.get(hardware_profile, {})
    params = dict(profile.get(arch, {}))

    # Override vocab/seq_len safely
    params["vocab_size"] = vocab_size
    if "max_seq_len" in params:
        params["max_seq_len"] = seq_len

    arch = arch.lower()
    if arch == "gru":
        return SmallGRULM(GRUConfig(**params))
    if arch == "awdlstm":
        return AWDLSTMLM(AWDLSTMConfig(**params))
    if arch == "gpt":
        return TinyGPTLM(GPTConfig(**params))
    if arch == "prefix_gpt":
        params["cond_dim"] = cond_dim
        return PrefixTinyGPTLM(PrefixGPTConfig(**params))
    if arch == "moe":
        return TinyMoELM(MoEConfig(**params))
    if arch == "mamba_like":
        return MambaLikeLM(MambaLikeConfig(**params))
    raise ValueError(f"Unknown arch: {arch}")


# ---------------------------------------------------------------------
# Conditioning hook for PrefixTinyGPTLM
# ---------------------------------------------------------------------

def build_cond_vec(batch_size: int, cond_dim: int, device: torch.device) -> torch.Tensor:
    # Placeholder: replace later with your cached personality + affect vectors
    return torch.zeros(batch_size, cond_dim, device=device)


# ---------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------

def maybe_amp(device: torch.device, enabled: bool):
    if not enabled:
        return torch.autocast(device_type="cpu", enabled=False)
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)
    # MPS autocast support is more limited; keep it simple/safe.
    return torch.autocast(device_type="cpu", enabled=False)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cond_dim: int = 128,
    amp: bool = False,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        x, y = [t.to(device) for t in batch]
        with maybe_amp(device, amp):
            if isinstance(model, PrefixTinyGPTLM):
                cond = build_cond_vec(x.size(0), cond_dim, device)
                out = model(x, cond, y)
            else:
                out = model(x, y)
        losses.append(float(out.loss.detach().cpu()))
    mean_loss = sum(losses) / max(1, len(losses))
    ppl = math.exp(mean_loss) if mean_loss < 20 else float("inf")
    return {"loss": mean_loss, "ppl": ppl}


def train_one_model(
    arch: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    epochs: int,
    lr: float,
    weight_decay: float,
    grad_accum: int,
    log_every: int,
    eval_every_steps: int,
    cond_dim: int = 128,
    use_amp: bool = False,
) -> Dict[str, float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=(use_amp and device.type == "cuda"))

    global_step = 0
    best_val = float("inf")
    best_path = out_dir / f"{arch}_best.pt"

    history: List[Dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        running = 0.0
        num_steps = 0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader, start=1):
            x, y = [t.to(device) for t in batch]

            with maybe_amp(device, use_amp):
                if isinstance(model, PrefixTinyGPTLM):
                    cond = build_cond_vec(x.size(0), cond_dim, device)
                    out = model(x, cond, y)
                else:
                    out = model(x, y)
                loss = out.loss / grad_accum

            if use_amp and device.type == "cuda":
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running += float(loss.detach().cpu()) * grad_accum
            num_steps += 1

            if batch_idx % grad_accum == 0:
                if use_amp and device.type == "cuda":
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % log_every == 0:
                    avg_loss = running / max(1, num_steps)
                    print(f"[{arch}] epoch {epoch} step {global_step} train_loss={avg_loss:.4f}")

                if global_step % eval_every_steps == 0:
                    val_metrics = evaluate(model, val_loader, device, cond_dim=cond_dim, amp=use_amp, max_batches=50)
                    entry = {
                        "epoch": epoch,
                        "step": global_step,
                        "train_loss": running / max(1, num_steps),
                        "val_loss": val_metrics["loss"],
                        "val_ppl": val_metrics["ppl"],
                    }
                    history.append(entry)
                    print(f"[{arch}] eval step {global_step}: val_loss={val_metrics['loss']:.4f} val_ppl={val_metrics['ppl']:.2f}")
                    if val_metrics["loss"] < best_val:
                        best_val = val_metrics["loss"]
                        torch.save({
                            "arch": arch,
                            "model_state": model.state_dict(),
                            "step": global_step,
                            "epoch": epoch,
                        }, best_path)

        epoch_time = time.time() - t0
        print(f"[{arch}] finished epoch {epoch} in {epoch_time:.1f}s")

    # reload best if available
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    final_val = evaluate(model, val_loader, device, cond_dim=cond_dim, amp=use_amp)
    final_test = evaluate(model, test_loader, device, cond_dim=cond_dim, amp=use_amp)

    summary = {
        "arch": arch,
        "best_val_loss": best_val,
        "final_val_loss": final_val["loss"],
        "final_val_ppl": final_val["ppl"],
        "final_test_loss": final_test["loss"],
        "final_test_ppl": final_test["ppl"],
        "num_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }

    with open(out_dir / f"{arch}_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    with open(out_dir / f"{arch}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


# ---------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------

def save_csv(rows: List[Dict], path: Path) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-text", type=str, default=None, help="Path to local text file")
    ap.add_argument("--hf-dataset", type=str, default=None, help="Hugging Face dataset name")
    ap.add_argument("--hf-split", type=str, default="train")
    ap.add_argument("--text-field", type=str, default="text")
    ap.add_argument("--max-chars", type=int, default=None)

    ap.add_argument("--arch", type=str, default="gpt",
                    choices=["gru", "awdlstm", "gpt", "prefix_gpt", "moe", "mamba_like"])
    ap.add_argument("--benchmark", nargs="*", default=None,
                    help="Run multiple architectures sequentially on the same split")

    ap.add_argument("--hardware-profile", type=str, default="rtx4070_small",
                    choices=["m1_small", "rtx4070_small"])
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-every-steps", type=int, default=100)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="runs/benchmark")
    ap.add_argument("--no-tiktoken", action="store_true")
    ap.add_argument("--amp", action="store_true", help="Enable CUDA mixed precision")
    args = ap.parse_args()

    set_seed(args.seed)
    device = select_device()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {device}")
    text = read_text_sources(
        train_text=args.train_text,
        hf_dataset=args.hf_dataset,
        hf_split=args.hf_split,
        text_field=args.text_field,
        max_chars=args.max_chars,
    )
    tokenizer = build_tokenizer(text, prefer_tiktoken=(not args.no_tiktoken))
    token_ids = tokenizer.encode(text)
    print(f"Tokenizer={tokenizer.name} vocab={tokenizer.vocab_size} num_tokens={len(token_ids)}")

    train_ds, val_ds, test_ds = build_splits(token_ids, args.seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    archs = args.benchmark if args.benchmark else [args.arch]
    results: List[Dict] = []

    run_meta = {
        "device": str(device),
        "hardware_profile": args.hardware_profile,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "tokenizer": tokenizer.name,
        "vocab_size": tokenizer.vocab_size,
    }
    with open(out_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    for arch in archs:
        print(f"\n=== Training {arch} ===")
        model = build_model_from_arch(
            arch=arch,
            vocab_size=tokenizer.vocab_size,
            hardware_profile=args.hardware_profile,
            seq_len=args.seq_len,
        )
        summary = train_one_model(
            arch=arch,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            out_dir=out_dir,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_accum=args.grad_accum,
            log_every=args.log_every,
            eval_every_steps=args.eval_every_steps,
            use_amp=args.amp,
        )
        results.append(summary)

        # free memory between runs
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_csv(results, out_dir / "benchmark_results.csv")
    with open(out_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nFinished. Results:")
    for row in results:
        print(row)


if __name__ == "__main__":
    main()
