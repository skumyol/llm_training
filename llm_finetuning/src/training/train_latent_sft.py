"""Train the generative latent-state predictor (see latent_sft.py for why).

    python -m src.training.train_latent_sft --config configs/lat_S1_genstate.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from src.training.latent_sft import LatentSFTDataset, collate
from src.training.model import load_backbone


def _seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


@torch.no_grad()
def _val_loss(model, loader, device) -> float:
    model.eval()
    tot, n = 0.0, 0
    for b in loader:
        out = model(input_ids=b["input_ids"].to(device),
                    attention_mask=b["attention_mask"].to(device),
                    labels=b["labels"].to(device))
        tot += float(out.loss); n += 1
    model.train()
    return tot / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    tcfg = cfg["training"]
    _seed(int(cfg.get("seed", 42)))

    model, tokenizer, _ = load_backbone(
        cfg.get("base_model", "Qwen/Qwen3-4B"),
        quantization=cfg.get("quantization", "4bit"),
        lora_config=cfg["lora"],
        torch_dtype=cfg.get("torch_dtype", "bfloat16"),
    )
    if tcfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    excl = cfg["data"].get("exclude_counterfactual", False)
    mk = lambda p: LatentSFTDataset(p, tokenizer, tcfg.get("max_seq_len", 1024), excl)
    train_ds, val_ds = mk(cfg["data"]["train_file"]), mk(cfg["data"]["val_file"])
    print(f"train={len(train_ds)} val={len(val_ds)} exclude_counterfactual={excl}")

    bs = tcfg.get("batch_size", 1)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, collate_fn=collate)

    accum = tcfg.get("grad_accum", 32)
    epochs = tcfg.get("epochs", 5)
    steps = max(1, math.ceil(len(train_loader) / accum)) * epochs
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=float(tcfg.get("lr", 2e-4)),
                            weight_decay=float(tcfg.get("weight_decay", 0.01)))
    sched = get_cosine_schedule_with_warmup(
        opt, int(steps * float(tcfg.get("warmup_ratio", 0.05))), steps)
    print(f"{steps} optimizer steps over {epochs} epochs")

    device = next(model.parameters()).device
    best = float("inf")
    best_dir = Path(cfg["output"]["best_model_dir"])
    for ep in range(1, epochs + 1):
        model.train()
        run, seen = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, b in enumerate(train_loader, 1):
            out = model(input_ids=b["input_ids"].to(device),
                        attention_mask=b["attention_mask"].to(device),
                        labels=b["labels"].to(device))
            (out.loss / accum).backward()
            run += float(out.loss); seen += 1
            if i % accum == 0 or i == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    float(tcfg.get("max_grad_norm", 1.0)))
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        vl = _val_loss(model, val_loader, device)
        print(f"Epoch {ep}: train_loss={run/max(seen,1):.4f} | val_loss={vl:.4f}", flush=True)
        if vl < best:
            best = vl
            best_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(best_dir)); tokenizer.save_pretrained(str(best_dir))
            (best_dir / "train_meta.json").write_text(json.dumps(
                {"epoch": ep, "val_loss": vl, "config": a.config}, indent=2))
            print(f"  → new best (val_loss={vl:.4f}) → {best_dir}", flush=True)
    print(f"DONE best val_loss={best:.4f}")


if __name__ == "__main__":
    main()
