#!/usr/bin/env python3
"""
SLM Latent Predictor Trainer
=============================
Trains from-scratch SLM backbones (GPT, Mamba-like) as social-state latent
predictors. Supports optional JEPA auxiliary loss with shuffled-future placebo.

Usage:
  python -m slm_training.src.train.train_latent_slm --config configs/latent_gpt_base.yaml
  python -m slm_training.src.train.train_latent_slm --config configs/latent_mamba_jepa.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Project root resolution ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent.parent  # llm_training/
sys.path.insert(0, str(ROOT))

from slm_training.src.train.small_lm_architectures import (
    GPTConfig, TinyGPTLM,
    MambaLikeConfig, MambaLikeLM,
    MoEConfig, TinyMoELM,
    RECOMMENDED_CONFIGS, LMOutput,
)
from llm_finetuning.src.training.dataset import (
    HeadSupervisionDataset, collate_head_batch,
    LABEL_MAPS, LABEL_TO_IDX,
)
from llm_finetuning.src.training.model import ClassificationHead
from llm_finetuning.src.training.loss import MultiHeadLoss, compute_class_weights, GROUP_FIELDS
from llm_finetuning.src.training.jepa import (
    SocialJEPAHead, SocialJEPAPredictorConfig,
    SocialStateEmbeddingConfig, social_jepa_loss, JEPA_FIELDS,
)
from slm_training.src.train.mlflow_tracker import MLflowTracker


# ═══════════════════════════════════════════════════════════════════════════════
# SLM Latent Predictor Wrapper
# ═══════════════════════════════════════════════════════════════════════════════

class SLMLatentPredictor(nn.Module):
    """Wraps an SLM backbone with pooling + classification heads."""

    def __init__(
        self,
        slm: nn.Module,
        hidden_dim: int,
        pooling: str = "last",
    ):
        super().__init__()
        self.slm = slm
        self.hidden_dim = hidden_dim
        self.pooling = pooling

        # Classification heads (same spec as llm_finetuning LatentStatePredictor)
        self.heads = nn.ModuleDict()
        for field, idx_map in LABEL_TO_IDX.items():
            if field == "dialogue_act":
                continue  # skip multi-label
            n_classes = len(idx_map)
            self.heads[field] = ClassificationHead(hidden_dim, n_classes)

        if pooling == "attention":
            self.attn_vector = nn.Parameter(torch.randn(hidden_dim))

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        out: LMOutput = self.slm(input_ids, return_hidden=True)
        h = out.hidden_states  # (B, T, hidden_dim)

        # Pooling
        if self.pooling == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                pooled = h.mean(dim=1)
        elif self.pooling == "attention":
            scores = torch.matmul(h, self.attn_vector)
            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask == 0, -1e9)
            weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
            pooled = (h * weights).sum(dim=1)
        else:  # last
            if attention_mask is not None:
                lengths = attention_mask.sum(dim=1) - 1
                pooled = h[torch.arange(h.size(0), device=h.device), lengths.clamp(min=0)]
            else:
                pooled = h[:, -1, :]

        logits = {field: head(pooled) for field, head in self.heads.items()}
        return {"logits": logits, "pooled": pooled}


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _batch_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _future_label_ids_from_batch(
    batch: dict, horizons: list[int], fields: list[str]
) -> dict[int, dict[str, torch.Tensor]]:
    result: dict[int, dict[str, torch.Tensor]] = {}
    for horizon in horizons:
        result[horizon] = {}
        for field in fields:
            key = f"future_{horizon}_{field}"
            if key in batch:
                result[horizon][field] = batch[key]
    return result


def _evaluate_slm(
    predictor: SLMLatentPredictor,
    val_loader: DataLoader,
    loss_fn: MultiHeadLoss,
    jepa_head: SocialJEPAHead | None = None,
    jepa_cfg: dict | None = None,
    jepa_horizons: list[int] | None = None,
    jepa_fields: list[str] | None = None,
) -> tuple[float, dict]:
    predictor.eval()
    if jepa_head is not None:
        jepa_head.eval()
    total_loss = 0.0
    total_jepa_loss = 0.0
    jepa_batches = 0
    all_preds: dict[str, list] = {}
    all_golds: dict[str, list] = {}

    for batch in tqdm(val_loader, desc="Evaluating", leave=False):
        device = predictor.device
        batch = _batch_to_device(batch, device)

        out = predictor(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"))
        loss, _detail = loss_fn(out["logits"], batch)

        if jepa_head is not None and jepa_cfg and jepa_horizons and jepa_fields:
            future_ids = _future_label_ids_from_batch(batch, jepa_horizons, jepa_fields)
            jepa_out = jepa_head(out["pooled"], future_ids)
            loss_jepa = social_jepa_loss(
                jepa_out,
                horizon_weights={int(k): float(v) for k, v in jepa_cfg.get("horizon_weights", {}).items()},
                var_weight=float(jepa_cfg.get("var_weight", 0.0)),
            )
            total_jepa_loss += loss_jepa.item()
            jepa_batches += 1

        total_loss += loss.item()

        for field, logit_tensor in out["logits"].items():
            label_key = f"label_{field}"
            if label_key not in batch:
                continue
            gold = batch[label_key]
            if not isinstance(gold, torch.Tensor):
                continue
            valid = gold != -1
            if not valid.any():
                continue
            preds = logit_tensor.argmax(dim=-1)
            if field not in all_preds:
                all_preds[field] = []
                all_golds[field] = []
            all_preds[field].extend(preds[valid].cpu().tolist())
            all_golds[field].extend(gold[valid].cpu().tolist())

    n_batches = max(len(val_loader), 1)
    metrics: dict[str, Any] = {
        "val_loss": total_loss / n_batches,
        "n_evaluated": len(next(iter(all_golds.values()), [])),
    }
    if jepa_batches > 0:
        metrics["val_jepa_loss"] = total_jepa_loss / jepa_batches

    # Per-head accuracy
    per_head = {}
    for field in all_golds:
        correct = sum(1 for g, p in zip(all_golds[field], all_preds[field]) if g == p)
        per_head[field] = correct / max(len(all_golds[field]), 1)

    # Group aggregates
    groups: dict[str, dict] = {}
    for field, acc in per_head.items():
        g = GROUP_FIELDS.get(field, "other")
        if g not in groups:
            groups[g] = {"correct": 0, "total": 0}
        groups[g]["correct"] += int(acc * len(all_golds.get(field, [])))
        groups[g]["total"] += len(all_golds.get(field, []))

    for g, gm in groups.items():
        gm["accuracy"] = gm["correct"] / max(gm["total"], 1)

    macro_acc = sum(per_head.values()) / max(len(per_head), 1)
    metrics["val_macro_accuracy"] = macro_acc

    # Policy F1 (binary: is response_policy correct?)
    if "response_policy" in per_head:
        metrics["val_response_policy_f1"] = per_head["response_policy"]

    return metrics.get("val_loss", 0.0), {
        "summary": metrics,
        "fields": per_head,
        "groups": groups,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main training
# ═══════════════════════════════════════════════════════════════════════════════

def train_slm_latent(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    arch = cfg.get("architecture", "gpt")
    profile = cfg.get("hardware_profile", "rtx4070_small")
    arch_cfg_dict = RECOMMENDED_CONFIGS.get(profile, {}).get(arch)
    if arch_cfg_dict is None:
        raise ValueError(f"Unknown arch/profile: {arch}/{profile}")

    train_cfg = cfg.get("training", {})
    jepa_cfg = cfg.get("jepa", {})
    jepa_enabled = bool(jepa_cfg.get("enabled", False))
    jepa_fields = jepa_cfg.get("fields", JEPA_FIELDS)
    jepa_horizons = [int(k) for k in jepa_cfg.get("horizons", [1])]
    jepa_shuffle = bool(jepa_cfg.get("shuffle_future_labels", False))
    pooling = cfg.get("pooling", "last")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Arch: {arch}  |  Profile: {profile}  |  Pooling: {pooling}")

    # ── Build SLM ────────────────────────────────────────────────────────────
    if arch == "gpt":
        slm_cfg = GPTConfig(**arch_cfg_dict)
        slm = TinyGPTLM(slm_cfg).to(device)
        hidden_dim = slm_cfg.n_embd
    elif arch == "mamba_like":
        slm_cfg = MambaLikeConfig(**arch_cfg_dict)
        slm = MambaLikeLM(slm_cfg).to(device)
        hidden_dim = slm_cfg.n_embd
    elif arch == "moe":
        slm_cfg = MoEConfig(**arch_cfg_dict)
        slm = TinyMoELM(slm_cfg).to(device)
        hidden_dim = slm_cfg.n_embd
    else:
        raise ValueError(f"Unsupported architecture for latent training: {arch}")

    predictor = SLMLatentPredictor(slm, hidden_dim, pooling=pooling).to(device)
    total_params = sum(p.numel() for p in predictor.parameters())
    trainable_params = sum(p.numel() for p in predictor.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total | {trainable_params:,} trainable")

    # ── Datasets ─────────────────────────────────────────────────────────────
    # SLMs use GPT-2 tokenizer (vocab=50257 matches default SLM config)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_ds = HeadSupervisionDataset(
        cfg["data"]["train_file"],
        tokenizer,
        max_seq_len=train_cfg.get("max_seq_len", 256),
        jepa_fields=jepa_fields if jepa_enabled else None,
        jepa_horizons=jepa_horizons if jepa_enabled else None,
        shuffle_future_labels=jepa_shuffle if jepa_enabled else False,
    )
    val_ds = HeadSupervisionDataset(
        cfg["data"]["val_file"],
        tokenizer,
        max_seq_len=train_cfg.get("max_seq_len", 256),
        jepa_fields=jepa_fields if jepa_enabled else None,
        jepa_horizons=jepa_horizons if jepa_enabled else None,
        shuffle_future_labels=False,  # never shuffle validation
    )

    batch_size = train_cfg.get("batch_size", 4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_head_batch)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_head_batch)

    print(f"Train: {len(train_ds):,} examples | Val: {len(val_ds):,} examples")

    # ── Loss ─────────────────────────────────────────────────────────────────
    class_weights = compute_class_weights(cfg["data"]["train_file"], LABEL_MAPS)
    loss_weights = train_cfg.get("loss_weights", {
        "lambda_C": 1.0, "lambda_A": 1.0, "lambda_M": 1.5,
        "lambda_R": 2.0, "lambda_N": 1.0, "lambda_D": 2.0,
    })
    loss_fn = MultiHeadLoss(loss_weights, class_weights=class_weights)
    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    if label_smoothing > 0:
        # Patch loss_fn for label smoothing
        loss_fn.label_smoothing = label_smoothing
        print(f"Label smoothing: {label_smoothing}")

    # ── JEPA head ────────────────────────────────────────────────────────────
    jepa_head = None
    if jepa_enabled:
        label_vocab_sizes = {field: len(LABEL_MAPS[field]) for field in jepa_fields}
        jepa_head = SocialJEPAHead(
            SocialJEPAPredictorConfig(
                hidden_dim=hidden_dim,
                target_dim=int(jepa_cfg.get("target_dim", 128)),
                predictor_dim=int(jepa_cfg.get("predictor_dim", 256)),
                horizons=jepa_horizons,
                dropout=float(jepa_cfg.get("dropout", 0.1)),
            ),
            label_vocab_sizes=label_vocab_sizes,
            state_emb_cfg=SocialStateEmbeddingConfig(
                emb_dim=int(jepa_cfg.get("emb_dim", 64)),
                out_dim=int(jepa_cfg.get("target_dim", 128)),
                dropout=float(jepa_cfg.get("dropout", 0.1)),
            ),
        ).to(device)
        print(f"JEPA enabled: fields={jepa_fields}, horizons={jepa_horizons}, shuffle={jepa_shuffle}")

    # ── Optimizer ────────────────────────────────────────────────────────────
    lr = float(train_cfg.get("lr", 3e-4))
    optimizer = AdamW(predictor.parameters(), lr=lr, weight_decay=float(train_cfg.get("weight_decay", 0.01)))
    if jepa_head is not None:
        optimizer.add_param_group({"params": jepa_head.parameters(), "lr": lr})

    epochs = train_cfg.get("epochs", 10)
    grad_accum = train_cfg.get("grad_accum", 4)
    total_steps = math.ceil(len(train_loader) / grad_accum) * epochs
    warmup_steps = int(total_steps * float(train_cfg.get("warmup_ratio", 0.05)))
    # Simple cosine schedule via PyTorch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # ── Output dirs ──────────────────────────────────────────────────────────
    output_dir = Path(cfg.get("output", {}).get("checkpoint_dir", f"checkpoints/slm_latent_{arch}"))
    best_dir = Path(cfg.get("output", {}).get("best_model_dir", f"checkpoints/slm_latent_{arch}_best"))
    output_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    # ── MLflow ───────────────────────────────────────────────────────────────
    tracker = MLflowTracker(
        experiment=cfg.get("mlflow", {}).get("experiment_name", "slm_latent"),
        enabled=cfg.get("mlflow", {}).get("enabled", True),
    )
    tracker.start_run(run_name=cfg.get("mlflow", {}).get("run_name", f"slm_latent_{arch}"))
    tracker.log_params(cfg)

    best_metric_value: float | None = None
    best_metric_name = train_cfg.get("metric_for_best_model", "val_response_policy_f1")
    higher_is_better = "loss" not in best_metric_name
    global_step = 0

    print(f"Training: {epochs} epochs, batch={batch_size}×{grad_accum}, lr={lr}, warmup={warmup_steps}")
    print(f"Best metric: {best_metric_name} (higher_is_better={higher_is_better})")

    for epoch in range(1, epochs + 1):
        predictor.train()
        if jepa_head is not None:
            jepa_head.train()
        epoch_loss = 0.0
        epoch_jepa_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            device = predictor.device
            batch = _batch_to_device(batch, device)

            out = predictor(input_ids=batch["input_ids"], attention_mask=batch.get("attention_mask"))
            total_loss, _detail = loss_fn(out["logits"], batch)

            if jepa_head is not None:
                future_ids = _future_label_ids_from_batch(batch, jepa_horizons, jepa_fields)
                jepa_out = jepa_head(out["pooled"], future_ids)
                loss_jepa = social_jepa_loss(
                    jepa_out,
                    horizon_weights={int(k): float(v) for k, v in jepa_cfg.get("horizon_weights", {}).items()},
                    var_weight=float(jepa_cfg.get("var_weight", 0.0)),
                )
                total_loss = total_loss + float(jepa_cfg.get("lambda_jepa", 0.05)) * loss_jepa
                epoch_jepa_loss += loss_jepa.item()

            total_loss = total_loss / grad_accum
            total_loss.backward()
            epoch_loss += total_loss.item() * grad_accum

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
                if jepa_head is not None:
                    torch.nn.utils.clip_grad_norm_(jepa_head.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        # ── Validation ──────────────────────────────────────────────────────
        val_loss, val_metrics = _evaluate_slm(
            predictor, val_loader, loss_fn,
            jepa_head=jepa_head, jepa_cfg=jepa_cfg,
            jepa_horizons=jepa_horizons, jepa_fields=jepa_fields,
        )
        tracker.log_metrics({"train_loss": epoch_loss / max(step + 1, 1)}, step=epoch)
        tracker.log_metrics({"val_loss": val_loss}, step=epoch)
        if val_metrics.get("summary"):
            tracker.log_metrics(val_metrics["summary"], step=epoch)
        if epoch_jepa_loss > 0:
            tracker.log_metrics({"train_jepa_loss": epoch_jepa_loss / max(step + 1, 1)}, step=epoch)

        summary = val_metrics.get("summary", {})
        current_metric = summary.get(best_metric_name, summary.get("val_loss", 999))
        print(f"  Epoch {epoch}: val_loss={val_loss:.4f}  {best_metric_name}={current_metric:.4f}"
              + (f"  jepa_loss={summary.get('val_jepa_loss', 0):.4f}" if jepa_head else ""))

        # ── Checkpoint ──────────────────────────────────────────────────────
        is_better = (best_metric_value is None or
                     (higher_is_better and current_metric > best_metric_value) or
                     (not higher_is_better and current_metric < best_metric_value))
        if is_better:
            best_metric_value = current_metric
            torch.save({
                "slm_state_dict": predictor.slm.state_dict(),
                "heads_state_dict": {k: v.state_dict() for k, v in predictor.heads.items()},
                "config": cfg,
                "epoch": epoch,
            }, best_dir / "checkpoint.pt")
            if jepa_head is not None:
                torch.save(jepa_head.state_dict(), best_dir / "jepa_head.pt")
            print(f"  ✓ Best model saved → {best_dir}")

    # ── Final save ───────────────────────────────────────────────────────────
    torch.save({
        "slm_state_dict": predictor.slm.state_dict(),
        "heads_state_dict": {k: v.state_dict() for k, v in predictor.heads.items()},
        "config": cfg,
    }, output_dir / "final.pt")
    tracker.end_run()
    print(f"Done. Best {best_metric_name}={best_metric_value:.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SLM Latent Predictor Trainer")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--run-id", default=None, help="Run identifier")
    args = parser.parse_args()
    train_slm_latent(args.config)


if __name__ == "__main__":
    main()
