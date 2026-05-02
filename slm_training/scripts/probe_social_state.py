#!/usr/bin/env python3
"""
Train frozen linear probes on hidden states from SLM checkpoints.

The probe targets the social-state labels already used by the LLM pipeline,
but evaluates whether the compact SLM representation exposes them linearly.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "slm_training" / "src" / "train"))

from conditioning import build_condition_vector, extract_state_dict, infer_arch_config_from_checkpoint
from small_lm_architectures import PrefixTinyGPTLM, build_model

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


LABEL_MAPS: dict[str, list[Any]] = {
    "dialogue_act": ["ask", "accuse", "threaten", "flatter", "apologize", "negotiate", "joke", "confess", "probe", "command"],
    "tone": ["warm", "neutral", "confrontational", "sarcastic", "fearful", "evasive"],
    "risk_type": ["none", "secret-risk", "face-risk", "status-risk", "conflict-risk"],
    "valence": ["negative", "neutral", "positive"],
    "arousal": ["low", "medium", "high"],
    "threat": ["low", "medium", "high"],
    "control": ["low", "medium", "high"],
    "player_intent": ["seek-info", "trap", "bond", "manipulate", "test", "persuade", "intimidate", "probe", "negotiate"],
    "player_knowledge": ["unaware", "partial", "informed", "knows-secret"],
    "player_credibility": ["low", "medium", "high"],
    "duty_pressure": ["low", "medium", "high"],
    "secrecy_pressure": ["low", "medium", "high"],
    "face_pressure": ["low", "medium", "high"],
    "value_conflict": ["none", "mild", "strong"],
    "response_policy": ["answer", "partial", "withhold", "deflect", "challenge", "soothe", "test", "threaten", "negotiate", "clarify"],
    "reveal_decision": ["none", "hint", "partial", "full"],
    "repair_strategy": ["none", "soften", "apologize", "clarify", "redirect"],
    "trust_delta": ["--", "-", "0", "+", "++"],
    "respect_delta": ["--", "-", "0", "+", "++"],
    "dominance_delta": ["--", "-", "0", "+", "++"],
    "familiarity_delta": ["--", "-", "0", "+", "++"],
    "obligation_delta": ["--", "-", "0", "+", "++"],
    "affection_delta": ["--", "-", "0", "+", "++"],
    "trust_level": ["VL", "L", "N", "H", "VH"],
    "respect_level": ["VL", "L", "N", "H", "VH"],
    "dominance_level": ["VL", "L", "N", "H", "VH"],
    "familiarity_level": ["VL", "L", "N", "H", "VH"],
    "obligation_level": ["VL", "L", "N", "H", "VH"],
    "affection_level": ["VL", "L", "N", "H", "VH"],
}


def build_tokenizer(text: str, vocab_size_hint: int) -> Any:
    if vocab_size_hint <= 512 or not _TIKTOKEN_OK:
        return CharTokenizer(text)
    enc = tiktoken.get_encoding("gpt2")
    enc.name = "tiktoken:gpt2"  # type: ignore[attr-defined]
    enc.vocab_size = enc.n_vocab  # type: ignore[attr-defined]
    return enc


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_labels(record: dict) -> dict[str, Any]:
    labels = record.get("labels", {})
    out = dict(labels)
    for key in LABEL_MAPS:
        if key in record and key not in out:
            out[key] = record[key]
    return out


def _as_list(val: Any) -> List[Any]:
    if val is None:
        return []
    if isinstance(val, list):
        return val
    return [val]


def encode_label(field: str, value: Any) -> torch.Tensor:
    values = LABEL_MAPS[field]
    if field == "dialogue_act":
        items = _as_list(value)
        vec = torch.zeros(len(values), dtype=torch.float32)
        for item in items:
            if item in values:
                vec[values.index(item)] = 1.0
        return vec
    if isinstance(value, list):
        value = value[0] if value else None
    idx = values.index(value) if value in values else -1
    return torch.tensor(idx, dtype=torch.long)


def tokenize_texts(texts: List[str], tokenizer: Any, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded: List[List[int]] = []
    for text in texts:
        ids = tokenizer.encode(text)[:max_len]
        encoded.append(ids)
    width = max((len(ids) for ids in encoded), default=1)
    width = min(width, max_len)
    input_ids = []
    attention = []
    for ids in encoded:
        pad_len = width - len(ids)
        input_ids.append(ids + [0] * pad_len)
        attention.append([1] * len(ids) + [0] * pad_len)
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention, dtype=torch.long),
    )


@torch.no_grad()
def extract_hidden(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    condition_mode: str,
    cond_dim: int,
    tokenizer: Any,
) -> torch.Tensor:
    if isinstance(model, PrefixTinyGPTLM):
        texts = [tokenizer.decode(ids[: mask.sum().item()].tolist()) for ids, mask in zip(input_ids, attention_mask)]
        cond = build_condition_vector(texts, condition_mode, cond_dim, device=input_ids.device)
        B, T = input_ids.shape
        P = model.cfg.prefix_length
        tok_h = model.tok_emb(input_ids)
        prefix = model.prefix_proj(cond).view(B, P, model.cfg.n_embd)
        h = torch.cat([prefix, tok_h], dim=1)
        pos = torch.arange(P + T, device=input_ids.device)
        h = model.drop(h + model.pos_emb(pos))
        for block in model.blocks:
            h = block(h)
        return model.ln_f(h[:, P:, :])

    # GPT-like fallback
    tok_h = model.tok_emb(input_ids)
    pos = torch.arange(input_ids.size(1), device=input_ids.device)
    h = model.drop(tok_h + model.pos_emb(pos))
    for block in model.blocks:
        h = block(h)
    return model.ln_f(h)


class LinearProbe(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, multi_label: bool = False) -> None:
        super().__init__()
        self.multi_label = multi_label
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def build_features(
    records: List[dict],
    model: torch.nn.Module,
    tokenizer: Any,
    condition_mode: str,
    cond_dim: int,
    max_len: int,
    device: torch.device,
) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
    texts = [rec.get("context") or rec.get("input") or rec.get("text") or "" for rec in records]
    input_ids, attention_mask = tokenize_texts(texts, tokenizer, max_len)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    hidden = extract_hidden(model, input_ids, attention_mask, condition_mode, cond_dim, tokenizer)
    pooled = mean_pool(hidden, attention_mask)

    labels: dict[str, torch.Tensor] = {}
    for field in LABEL_MAPS:
        vals = []
        for record in records:
            lab = extract_labels(record)
            vals.append(encode_label(field, lab.get(field, None)))
        labels[field] = torch.stack(vals).to(device)
    return pooled, labels


def accuracy(preds: torch.Tensor, targets: torch.Tensor, multi_label: bool) -> float:
    if targets.numel() == 0:
        return float("nan")
    if multi_label:
        pred_bin = (preds.sigmoid() > 0.5).float()
        return float((pred_bin == targets.float()).all(dim=-1).float().mean().item())
    return float((preds.argmax(dim=-1) == targets).float().mean().item())


def macro_f1(preds: torch.Tensor, targets: torch.Tensor, multi_label: bool) -> float:
    eps = 1e-8
    if targets.numel() == 0:
        return float("nan")
    if multi_label:
        pred = (preds.sigmoid() > 0.5).float()
        tp = (pred * targets).sum(dim=0)
        fp = (pred * (1 - targets)).sum(dim=0)
        fn = ((1 - pred) * targets).sum(dim=0)
        f1 = (2 * tp) / (2 * tp + fp + fn + eps)
        return float(f1.mean().item())
    num_classes = preds.size(-1)
    pred_idx = preds.argmax(dim=-1)
    scores = []
    for c in range(num_classes):
        tp = ((pred_idx == c) & (targets == c)).sum().float()
        fp = ((pred_idx == c) & (targets != c)).sum().float()
        fn = ((pred_idx != c) & (targets == c)).sum().float()
        score = (2 * tp) / (2 * tp + fp + fn + eps)
        scores.append(score)
    return float(torch.stack(scores).mean().item())


def train_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    num_classes: int,
    multi_label: bool,
    epochs: int,
    lr: float,
) -> Dict[str, float]:
    probe = LinearProbe(train_x.size(-1), num_classes, multi_label=multi_label).to(train_x.device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    if multi_label:
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

        train_mask = train_y != -1
        val_mask = val_y != -1
        train_x = train_x[train_mask]
        train_y = train_y[train_mask]
        val_x = val_x[val_mask]
        val_y = val_y[val_mask]

    if train_x.numel() == 0 or val_x.numel() == 0:
        return {"val_acc": float("nan"), "val_f1": float("nan")}

    for _ in range(epochs):
        probe.train()
        opt.zero_grad(set_to_none=True)
        logits = probe(train_x)
        loss = loss_fn(logits, train_y.float() if multi_label else train_y)
        loss.backward()
        opt.step()

    probe.eval()
    with torch.no_grad():
        val_logits = probe(val_x)
    return {
        "val_acc": accuracy(val_logits, val_y, multi_label),
        "val_f1": macro_f1(val_logits, val_y, multi_label),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--train-file", type=Path, default=ROOT / "data" / "splits" / "train_heads.jsonl")
    p.add_argument("--val-file", type=Path, default=ROOT / "data" / "splits" / "val_heads.jsonl")
    p.add_argument("--fields", type=str, default="response_policy,reveal_decision,trust_delta,secrecy_pressure,player_knowledge")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    arch, cfg, ckpt = infer_arch_config_from_checkpoint(checkpoint_path)
    if arch not in {"gpt", "prefix_gpt"}:
        raise SystemExit(f"Unsupported checkpoint architecture for probing: {arch}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(arch, cfg).to(device)
    model.load_state_dict(extract_state_dict(ckpt), strict=False)
    model.eval()

    train_records = load_jsonl(args.train_file)
    val_records = load_jsonl(args.val_file)
    vocab_source = "\n".join(
        (rec.get("context") or rec.get("input") or rec.get("text") or "") for rec in train_records[:200]
    )
    tokenizer = build_tokenizer(vocab_source or "probe", cfg.get("vocab_size", 0))
    cond_mode = cfg.get("condition_mode", "ocean_vad")
    cond_dim = int(cfg.get("cond_dim", 8))

    train_x, train_labels = build_features(train_records, model, tokenizer, cond_mode, cond_dim, args.max_len, device)
    val_x, val_labels = build_features(val_records, model, tokenizer, cond_mode, cond_dim, args.max_len, device)

    fields = [f.strip() for f in args.fields.split(",") if f.strip()]
    results: Dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "arch": arch,
        "condition_mode": cond_mode,
        "fields": {},
    }

    for field in fields:
        if field not in LABEL_MAPS:
            continue
        multi_label = field == "dialogue_act"
        metrics = train_probe(
            train_x,
            train_labels[field],
            val_x,
            val_labels[field],
            len(LABEL_MAPS[field]),
            multi_label,
            args.epochs,
            args.lr,
        )
        results["fields"][field] = metrics
        print(f"{field}: acc={metrics['val_acc']:.4f} f1={metrics['val_f1']:.4f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
