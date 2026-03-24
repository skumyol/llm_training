import os
import math
from pathlib import Path
from typing import Optional

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.dataset import HeadSupervisionDataset, collate_head_batch
from src.training.loss import MultiHeadLoss
from src.training.model import build_latent_predictor, save_predictor


def _batch_to_device(batch: dict, device) -> dict:
    """Return a copy of batch with all tensors moved to device."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def train_latent(config_path: str, debug: bool = False) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    import mlflow
    from src.mlflow_utils import setup_mlflow, log_config

    setup_mlflow(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    model_name = cfg["debug_model"] if debug else cfg["base_model"]
    lora_cfg = cfg.get("lora", {})
    train_cfg = cfg["training"]
    loss_weights = cfg.get("loss_weights", {})

    print(f"Loading model: {model_name}")
    predictor, tokenizer = build_latent_predictor(
        model_name=model_name,
        quantization=cfg.get("quantization", "4bit"),
        lora_config={
            "r": lora_cfg.get("r", 16),
            "alpha": lora_cfg.get("alpha", 32),
            "dropout": lora_cfg.get("dropout", 0.05),
            "target_modules": lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
            "bias": lora_cfg.get("bias", "none"),
        },
        torch_dtype=cfg.get("torch_dtype", "bfloat16"),
    )

    if train_cfg.get("gradient_checkpointing", False):
        predictor.backbone.gradient_checkpointing_enable()

    train_ds = HeadSupervisionDataset(
        cfg["data"]["train_file"],
        tokenizer,
        max_seq_len=train_cfg.get("max_seq_len", 1024),
    )
    val_ds = HeadSupervisionDataset(
        cfg["data"]["val_file"],
        tokenizer,
        max_seq_len=train_cfg.get("max_seq_len", 1024),
    )

    batch_size = train_cfg.get("batch_size", 4)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_head_batch,
        num_workers=train_cfg.get("dataloader_num_workers", 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_head_batch,
        num_workers=0,
    )

    loss_fn = MultiHeadLoss(loss_weights)
    optimizer = torch.optim.AdamW(
        [p for p in predictor.parameters() if p.requires_grad],
        lr=float(train_cfg.get("lr", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    epochs = train_cfg.get("epochs", 5)
    grad_accum = train_cfg.get("grad_accum", 8)
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    total_steps = math.ceil(len(train_loader) / grad_accum) * epochs

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    output_dir = Path(cfg["output"]["checkpoint_dir"])
    best_dir = Path(cfg["output"]["best_model_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    run_name = cfg["mlflow"].get("run_name", "latent_train")

    with mlflow.start_run(run_name=run_name):
        log_config(cfg)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("n_train", len(train_ds))
        mlflow.log_param("n_val", len(val_ds))

        best_val_loss = float("inf")
        global_step = 0

        for epoch in range(1, epochs + 1):
            predictor.train()
            epoch_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
                device = predictor.backbone.device
                batch = _batch_to_device(batch, device)
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]

                out = predictor(input_ids=input_ids, attention_mask=attention_mask)
                total_loss, detail = loss_fn(out["logits"], batch)
                total_loss = total_loss / grad_accum
                total_loss.backward()

                epoch_loss += total_loss.item() * grad_accum

                if (step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % train_cfg.get("logging_steps", 20) == 0:
                        mlflow.log_metric("train/loss", epoch_loss / (step + 1), step=global_step)

            val_loss, val_metrics = _evaluate(predictor, val_loader, loss_fn)
            mlflow.log_metric("val/loss", val_loss, step=epoch)
            for k, v in val_metrics.items():
                mlflow.log_metric(f"val/{k}", v, step=epoch)

            print(f"Epoch {epoch}: train_loss={epoch_loss/len(train_loader):.4f}  val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_predictor(predictor, str(best_dir))
                mlflow.log_metric("val/best_loss", best_val_loss, step=epoch)
                print(f"  → New best model saved to {best_dir}")

        save_predictor(predictor, str(output_dir / "final"))
        mlflow.log_artifact(str(best_dir))
        print("Training complete.")


@torch.no_grad()
def _evaluate(predictor, val_loader, loss_fn) -> tuple[float, dict]:
    predictor.eval()
    total_loss = 0.0
    correct: dict = {}
    total: dict = {}

    for batch in tqdm(val_loader, desc="Evaluating", leave=False):
        device = predictor.backbone.device
        batch = _batch_to_device(batch, device)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        out = predictor(input_ids=input_ids, attention_mask=attention_mask)
        loss, detail = loss_fn(out["logits"], batch)
        total_loss += loss.item()

        for field, logit_tensor in out["logits"].items():
            label_key = f"label_{field}"
            if label_key not in batch:
                continue
            gold = batch[label_key]
            if not isinstance(gold, torch.Tensor):
                continue
            if field == "dialogue_act":
                continue
            valid = gold != -1
            if not valid.any():
                continue
            pred = logit_tensor.argmax(dim=-1)
            correct[field] = correct.get(field, 0) + (pred[valid] == gold[valid]).sum().item()
            total[field] = total.get(field, 0) + valid.sum().item()

    metrics = {}
    if total:
        per_field_acc = {f: correct[f] / total[f] for f in correct if total.get(f, 0) > 0}
        metrics["mean_accuracy"] = sum(per_field_acc.values()) / len(per_field_acc) if per_field_acc else 0.0
        metrics["response_policy_acc"] = per_field_acc.get("response_policy", 0.0)
        metrics["trust_delta_acc"] = per_field_acc.get("trust_delta", 0.0)
        metrics["reveal_decision_acc"] = per_field_acc.get("reveal_decision", 0.0)

    avg_loss = total_loss / max(1, len(val_loader))
    return avg_loss, metrics
