#!/usr/bin/env python3
"""
smoke_latent.py — fast pre-flight for the 29-head latent predictor.

Catches the failures that would otherwise waste hours of GPU time, in ~2 minutes:

  1. the 4-bit + LoRA + heads model builds at all
  2. a forward pass produces all 29 head logits with the expected shapes
  3. MultiHeadLoss consumes a real batch and returns a finite scalar
  4. LoRA and head gradients are actually NON-ZERO after backward

(4) is the important one. The pipeline never calls prepare_model_for_kbit_training
and enables gradient_checkpointing_enable() without enable_input_require_grads().
With a frozen 4-bit base, the checkpointed segment's inputs can have
requires_grad=False, which silently detaches LoRA gradients inside those blocks —
training then runs to completion, logs a plausible loss, and learns nothing in the
backbone. Cheap to check, expensive to miss.

Usage (on a GPU node):
    python llm_finetuning/scripts/smoke_latent.py --config llm_finetuning/configs/train_latent.yaml
    python llm_finetuning/scripts/smoke_latent.py --config ... --grad-checkpointing
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

from src.training.dataset import HeadSupervisionDataset, collate_head_batch, LABEL_MAPS
from src.training.loss import MultiHeadLoss, compute_class_weights
from src.training.model import build_latent_predictor, HEAD_SPECS


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="llm_finetuning/configs/train_latent.yaml")
    ap.add_argument("--n", type=int, default=2, help="batch size for the smoke batch")
    ap.add_argument("--grad-checkpointing", action="store_true",
                    help="enable gradient checkpointing, as the real training run does")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    train_cfg = cfg["training"]
    lora_cfg = cfg.get("lora", {})

    print(f"[1/5] building {cfg['base_model']} (4bit + LoRA + {len(HEAD_SPECS)} heads) ...")
    predictor, tokenizer = build_latent_predictor(
        model_name=cfg["base_model"],
        quantization=cfg.get("quantization", "4bit"),
        lora_config={
            "r": lora_cfg.get("r", 16),
            "alpha": lora_cfg.get("alpha", 32),
            "dropout": lora_cfg.get("dropout", 0.05),
            "target_modules": lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        },
        torch_dtype=cfg.get("torch_dtype", "bfloat16"),
        pooling=cfg.get("pooling", "last"),
    )
    assert len(predictor.heads) == 29, f"expected 29 heads, got {len(predictor.heads)}"

    if args.grad_checkpointing:
        predictor.backbone.gradient_checkpointing_enable()
        print("      gradient checkpointing: ENABLED")

    print("[2/5] loading a real batch ...")
    ds = HeadSupervisionDataset(cfg["data"]["train_file"], tokenizer,
                               max_seq_len=train_cfg.get("max_seq_len", 512))
    batch = collate_head_batch([ds[i] for i in range(args.n)])
    device = predictor.backbone.device
    batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

    print("[3/5] forward ...")
    predictor.train()
    out = predictor(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out["logits"]
    assert len(logits) == 29, f"expected 29 logit tensors, got {len(logits)}"
    for name, spec in HEAD_SPECS.items():
        got = tuple(logits[name].shape)
        want = (args.n, spec["n_classes"])
        assert got == want, f"head {name}: shape {got} != {want}"
    print(f"      all 29 heads OK (e.g. response_policy {tuple(logits['response_policy'].shape)})")

    print("[4/5] loss ...")
    class_weights = compute_class_weights(cfg["data"]["train_file"], LABEL_MAPS)
    loss_fn = MultiHeadLoss(
        cfg.get("loss_weights", {}),
        class_weights=class_weights,
        label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
        focal_gamma=float(train_cfg.get("focal_gamma", 0.0)),
    )
    loss, detail = loss_fn(logits, batch)
    assert torch.isfinite(loss), f"loss is not finite: {loss}"
    print(f"      loss = {loss.item():.4f} over {len(detail)} terms")

    print("[5/5] backward — checking gradients actually flow ...")
    loss.backward()

    lora_params = [(n, p) for n, p in predictor.backbone.named_parameters()
                   if p.requires_grad and "lora" in n.lower()]
    head_params = [(n, p) for n, p in predictor.heads.named_parameters() if p.requires_grad]
    assert lora_params, "no trainable LoRA parameters found"
    assert head_params, "no trainable head parameters found"

    def report(label, params):
        with_grad = [(n, p) for n, p in params if p.grad is not None]
        nonzero = [(n, p) for n, p in with_grad if p.grad.abs().sum().item() > 0]
        total_norm = sum(p.grad.norm().item() ** 2 for _, p in with_grad) ** 0.5
        print(f"      {label}: {len(params)} trainable, {len(with_grad)} with .grad, "
              f"{len(nonzero)} non-zero, grad_norm={total_norm:.4e}")
        return nonzero, total_norm

    lora_nonzero, lora_norm = report("LoRA", lora_params)
    head_nonzero, head_norm = report("heads", head_params)

    assert head_nonzero, "HEAD gradients are all zero — heads are not training"
    if not lora_nonzero:
        print()
        print("  FAIL: LoRA gradients are all zero. The backbone is not being trained.")
        print("  Fix: call prepare_model_for_kbit_training(model) before get_peft_model(),")
        print("  or model.enable_input_require_grads() before gradient_checkpointing_enable().")
        return 1

    print()
    print(f"SMOKE PASS — 29 heads, finite loss, LoRA grad_norm={lora_norm:.3e}, "
          f"head grad_norm={head_norm:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
