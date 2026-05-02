#!/usr/bin/env python3
"""
Evaluate registered SLM checkpoints from a YAML registry.

This script is the repo-local entry point for the actual baseline checkpoints
stored under `slm/npc_backend_scaffold/runs/...`.

It supports GPT-like checkpoints that store only raw weights plus a small
metadata stub. Model shapes are inferred from the checkpoint state dict, so the
tool works even when a full training summary is not available.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "slm_training" / "src" / "train"))

from conditioning import build_condition_vector, infer_arch_config_from_checkpoint, load_checkpoint_payload, extract_state_dict
from small_lm_architectures import PrefixTinyGPTLM, PrefixGPTConfig, build_model

try:
    import tiktoken
    _TIKTOKEN_OK = True
except ImportError:
    _TIKTOKEN_OK = False


class CharTokenizer:
    def __init__(self, text: str) -> None:
        vocab = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(vocab)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(vocab)
        self.name = "char"

    def encode(self, text: str) -> List[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos.get(i, "") for i in ids)


def load_registry(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    return payload.get("models", payload)


def build_tokenizer(text: str, vocab_size_hint: int) -> Any:
    if vocab_size_hint <= 512 or not _TIKTOKEN_OK:
        return CharTokenizer(text)
    enc = tiktoken.get_encoding("gpt2")
    enc.name = "tiktoken:gpt2"  # type: ignore[attr-defined]
    enc.vocab_size = enc.n_vocab  # type: ignore[attr-defined]
    return enc


@torch.no_grad()
def compute_ppl(
    model: torch.nn.Module,
    token_ids: List[int],
    seq_len: int,
    device: torch.device,
    tokenizer: Any,
    condition_mode: str,
    cond_dim: int,
) -> float:
    if len(token_ids) < seq_len + 1:
        return float("nan")

    total_loss = 0.0
    n_batches = 0
    for start in range(0, len(token_ids) - seq_len, seq_len):
        chunk = token_ids[start : start + seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long, device=device).unsqueeze(0)
        y = torch.tensor(chunk[1:], dtype=torch.long, device=device).unsqueeze(0)
        if isinstance(model, PrefixTinyGPTLM):
            text = tokenizer.decode(chunk[:-1]) if hasattr(tokenizer, "decode") else ""
            cond = build_condition_vector([text], condition_mode, cond_dim, device=device)
            out = model(x, cond, y)
        else:
            out = model(x, y)
        if out.loss is not None:
            total_loss += float(out.loss.item())
            n_batches += 1
    if n_batches == 0:
        return float("nan")
    return math.exp(min(total_loss / n_batches, 20))


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    prompt_ids: List[int],
    tokenizer: Any,
    max_new: int,
    device: torch.device,
    condition_mode: str,
    cond_dim: int,
    temperature: float = 0.8,
    top_k: int = 50,
) -> List[int]:
    ids = list(prompt_ids[-128:])
    generated: List[int] = []
    for _ in range(max_new):
        x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        if isinstance(model, PrefixTinyGPTLM):
            text = tokenizer.decode(ids) if hasattr(tokenizer, "decode") else ""
            cond = build_condition_vector([text], condition_mode, cond_dim, device=device)
            out = model(x, cond)
        else:
            out = model(x)
        logits = out.logits[0, -1, :]
        if temperature > 0:
            logits = logits / temperature
        if top_k > 0:
            topk_vals, _ = torch.topk(logits, top_k)
            logits = logits.masked_fill(logits < topk_vals[-1], float("-inf"))
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        ids.append(next_id)
        generated.append(next_id)
    return generated


def distinct_n(token_ids: List[int], n: int) -> float:
    ngrams = list(zip(*[token_ids[i:] for i in range(n)]))
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def load_model(ckpt_path: Path, device: torch.device):
    arch, cfg, ckpt = infer_arch_config_from_checkpoint(ckpt_path)
    if arch not in {"gpt", "prefix_gpt"}:
        raise ValueError(f"unsupported checkpoint architecture: {arch}")
    model = build_model(arch, cfg).to(device)
    state = extract_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    return model, arch, cfg, ckpt, missing, unexpected


def eval_registry_entry(
    name: str,
    entry: Dict[str, Any],
    device: torch.device,
    val_text: Optional[Path],
    seq_len: int,
    gen_len: int,
) -> Dict[str, Any]:
    ckpt_path = Path(entry["path"])
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    if not ckpt_path.exists():
        return {"name": name, "error": f"missing checkpoint: {ckpt_path}"}

    try:
        model, arch, cfg, ckpt, missing, unexpected = load_model(ckpt_path, device)
    except ValueError as exc:
        return {
            "name": name,
            "path": str(ckpt_path),
            "status": entry.get("status", "unknown"),
            "warning": str(exc),
        }
    except Exception as exc:
        return {"name": name, "error": f"load failed: {exc}"}

    result: Dict[str, Any] = {
        "name": name,
        "arch": arch,
        "path": str(ckpt_path),
        "conditioning": entry.get("conditioning", "unknown"),
        "status": entry.get("status", "unknown"),
        "params": sum(p.numel() for p in model.parameters()),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
    }

    if val_text and val_text.exists():
        raw = val_text.read_text(encoding="utf-8")
        tokenizer = build_tokenizer(raw, cfg.get("vocab_size", 0))
        token_ids = tokenizer.encode(raw)
        condition_mode = cfg.get("condition_mode", "ocean_vad")
        cond_dim = int(cfg.get("cond_dim", 8))

        result["val_ppl"] = round(
            compute_ppl(model, token_ids, seq_len, device, tokenizer, condition_mode, cond_dim), 4
        )
        prompt = token_ids[:32]
        gen_ids = generate(model, prompt, tokenizer, gen_len, device, condition_mode, cond_dim)
        result["distinct_1"] = round(distinct_n(gen_ids, 1), 4)
        result["distinct_2"] = round(distinct_n(gen_ids, 2), 4)
        try:
            result["sample"] = tokenizer.decode(gen_ids[:200]) if hasattr(tokenizer, "decode") else ""
        except Exception:
            result["sample"] = ""
    else:
        result["warning"] = "val text not found; checkpoint loaded but metrics were skipped"

    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--registry", type=Path, default=ROOT / "slm_training" / "trained_models.yaml")
    p.add_argument("--val-text", type=Path, default=ROOT / "data" / "dialogue" / "val.txt")
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--gen-len", type=int, default=128)
    p.add_argument("--out-csv", type=Path, default=None)
    args = p.parse_args()

    registry = load_registry(args.registry)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results: List[Dict[str, Any]] = []
    for name, entry in registry.items():
        results.append(eval_registry_entry(name, entry, device, args.val_text, args.seq_len, args.gen_len))

    for row in results:
        if "error" in row:
            print(f"{row['name']}: ERROR: {row['error']}")
            continue
        msg = f"{row['name']}: arch={row.get('arch', 'n/a')}"
        if "params" in row:
            msg += f" params={int(row['params']):,}"
        if "val_ppl" in row:
            msg += f" val_ppl={row['val_ppl']}"
        if "distinct_1" in row:
            msg += f" distinct1={row['distinct_1']} distinct2={row['distinct_2']}"
        if "warning" in row:
            msg += f" warning={row['warning']}"
        print(msg)

    if args.out_csv:
        fields = ["name", "arch", "path", "conditioning", "status", "params", "val_ppl", "distinct_1", "distinct_2", "missing_keys", "unexpected_keys", "warning", "error"]
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(results)
        print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
