import os
import math
from pathlib import Path
from typing import Optional

import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from src.training.dataset import HeadSupervisionDataset, collate_head_batch, LABEL_MAPS, LABEL_TO_IDX
from src.training.loss import MultiHeadLoss, compute_class_weights
from src.training.model import build_latent_predictor, save_predictor
from src.metrics_report import compute_latent_metrics, log_metrics_to_mlflow, write_metrics_bundle


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
    pooling = cfg.get("pooling", "last")
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
        pooling=pooling,
    )
    print(f"Pooling strategy: {pooling}")

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

    # Compute class weights early (needed for sampler and loss)
    class_weights = compute_class_weights(cfg["data"]["train_file"], LABEL_MAPS)
    print(f"Computed class weights for {len(class_weights)} heads")

    batch_size = train_cfg.get("batch_size", 4)
    use_weighted_sampler = train_cfg.get("use_weighted_sampler", True)

    if use_weighted_sampler:
        # Compute per-sample weights based on rarest label present
        sample_weights = []
        for record in train_ds.records:
            labels = record.get("labels", {})
            max_weight = 1.0
            for field, idx_map in LABEL_TO_IDX.items():
                if field == "dialogue_act":
                    continue
                val = labels.get(field)
                if val is not None:
                    idx = idx_map.get(str(val), -1)
                    if idx != -1 and field in class_weights:
                        w = class_weights[field][idx].item()
                        max_weight = max(max_weight, w)
            sample_weights.append(max_weight)

        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            collate_fn=collate_head_batch,
            num_workers=train_cfg.get("dataloader_num_workers", 0),
        )
        print(f"Using WeightedRandomSampler (oversampling minority classes)")
    else:
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

    label_smoothing = float(train_cfg.get("label_smoothing", 0.0))
    loss_fn = MultiHeadLoss(loss_weights, class_weights=class_weights, label_smoothing=label_smoothing)
    if label_smoothing > 0:
        print(f"Using label smoothing: {label_smoothing}")
    base_lr = float(train_cfg.get("lr", 2e-4))
    head_lr = float(train_cfg.get("head_lr", base_lr * 2))  # 2x for heads
    use_progressive_lr = train_cfg.get("use_progressive_lr", True)

    if use_progressive_lr:
        param_groups = [
            {"params": predictor.heads.parameters(), "lr": head_lr, "weight_decay": float(train_cfg.get("weight_decay", 0.01))},
            {"params": [p for n, p in predictor.backbone.named_parameters() if p.requires_grad], "lr": base_lr, "weight_decay": float(train_cfg.get("weight_decay", 0.01))},
        ]
        print(f"Progressive LR: heads={head_lr}, backbone={base_lr}")
    else:
        param_groups = [p for p in predictor.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=base_lr,
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    epochs = train_cfg.get("epochs", 5)
    grad_accum = train_cfg.get("grad_accum", 8)
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    total_steps = math.ceil(len(train_loader) / grad_accum) * epochs
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.05))
    warmup_steps = int(total_steps * warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps,
    )
    print(f"Scheduler: cosine with {warmup_steps} warmup steps / {total_steps} total steps")

    output_dir = Path(cfg["output"]["checkpoint_dir"])
    best_dir = Path(cfg["output"]["best_model_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    run_name = cfg["mlflow"].get("run_name", "latent_train")

    with mlflow.start_run(run_name=run_name):
        log_config(cfg)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("n_train", len(train_ds))
        mlflow.log_param("n_val", len(val_ds))

        best_metric_value = None
        best_metric_name = train_cfg.get("metric_for_best_model", "val/response_policy_f1")
        # Determine if higher is better (True for acc/f1, False for loss)
        higher_is_better = "loss" not in best_metric_name
        global_step = 0
        last_val_metrics: dict = {}

        print(f"Best model selection: {best_metric_name} (higher_is_better={higher_is_better})")

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
            log_metrics_to_mlflow({"loss": val_loss}, prefix="val", step=epoch)
            log_metrics_to_mlflow(val_metrics.get("summary", {}), prefix="val", step=epoch)
            if val_metrics.get("groups"):
                log_metrics_to_mlflow(val_metrics["groups"], prefix="val/groups", step=epoch)
            if val_metrics.get("fields"):
                log_metrics_to_mlflow(val_metrics["fields"], prefix="val/fields", step=epoch)
            last_val_metrics = val_metrics

            metrics_dir = output_dir / "metrics"
            write_metrics_bundle(
                metrics_dir,
                f"epoch_{epoch:03d}_latent",
                {"train_loss": epoch_loss / max(1, len(train_loader)), "val_loss": val_loss, **val_metrics},
                title=f"Latent Training Metrics - Epoch {epoch}",
            )

            # Resolve the metric used for best-model selection
            metric_key = best_metric_name.replace("val/", "")
            current_metric = val_metrics.get("summary", {}).get(metric_key, val_loss)

            summary_parts = [f"train_loss={epoch_loss/max(1, len(train_loader)):.4f}", f"val_loss={val_loss:.4f}"]
            for key in ["response_policy_f1", "mean_accuracy", "trust_delta_f1"]:
                value = val_metrics.get("summary", {}).get(key)
                if value is not None:
                    summary_parts.append(f"{key}={value:.4f}")
            print(f"Epoch {epoch}: {' | '.join(summary_parts)}")

            is_better = (
                best_metric_value is None
                or (higher_is_better and current_metric > best_metric_value)
                or (not higher_is_better and current_metric < best_metric_value)
            )
            if is_better:
                best_metric_value = current_metric
                save_predictor(predictor, str(best_dir))
                mlflow.log_metric(f"val/best_{metric_key}", best_metric_value, step=epoch)
                print(f"  → New best model ({best_metric_name}={best_metric_value:.4f}) saved to {best_dir}")

        save_predictor(predictor, str(output_dir / "final"))
        final_summary = {
            "model_name": model_name,
            "stage": "latent",
            "best_metric_name": best_metric_name,
            "best_metric_value": best_metric_value,
            "epochs": epochs,
            "final_val_summary": last_val_metrics.get("summary", {}),
        }
        write_metrics_bundle(
            output_dir / "metrics",
            "latent_training_summary",
            final_summary,
            title="Latent Training Summary",
        )
        mlflow.log_artifact(str(best_dir))
        print("Training complete.")


@torch.no_grad()
def _evaluate(predictor, val_loader, loss_fn) -> tuple[float, dict]:
    predictor.eval()
    total_loss = 0.0
    all_preds: dict[str, list] = {}
    all_golds: dict[str, list] = {}

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
                valid = gold.sum(dim=1) > 0
                if not valid.any():
                    continue
                pred = (logit_tensor.sigmoid() >= 0.5).to(torch.long)
                if field not in all_preds:
                    all_preds[field] = []
                    all_golds[field] = []
                all_preds[field].extend(pred[valid].cpu().tolist())
                all_golds[field].extend(gold[valid].cpu().long().tolist())
                continue

            valid = gold != -1
            if not valid.any():
                continue
            pred = logit_tensor.argmax(dim=-1)
            if field not in all_preds:
                all_preds[field] = []
                all_golds[field] = []
            all_preds[field].extend(pred[valid].cpu().tolist())
            all_golds[field].extend(gold[valid].cpu().tolist())

    metrics = compute_latent_metrics(all_preds, all_golds)

    avg_loss = total_loss / max(1, len(val_loader))
    return avg_loss, metrics
