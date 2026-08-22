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
from src.training.jepa import (
    JEPA_FIELDS,
    SocialJEPAHead,
    SocialJEPAPredictorConfig,
    SocialStateEmbeddingConfig,
    social_jepa_loss,
)
from src.training.loss import MultiHeadLoss, compute_class_weights
from src.training.model import build_latent_predictor, save_predictor
from src.metrics_report import compute_latent_metrics, log_metrics_to_mlflow, write_metrics_bundle


def _batch_to_device(batch: dict, device) -> dict:
    """Return a copy of batch with all tensors moved to device."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def _future_label_ids_from_batch(batch: dict, horizons: list[int], fields: list[str]) -> dict[int, dict[str, torch.Tensor]]:
    future_label_ids: dict[int, dict[str, torch.Tensor]] = {}
    for horizon in horizons:
        future_label_ids[horizon] = {}
        for field in fields:
            key = f"future_{horizon}_{field}"
            if key in batch:
                future_label_ids[horizon][field] = batch[key]
    return future_label_ids


def _save_jepa_head(jepa_head: SocialJEPAHead | None, save_dir: Path, cfg: dict) -> None:
    if jepa_head is None:
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": jepa_head.state_dict(),
            "config": cfg,
        },
        save_dir / "jepa_head.pt",
    )


def _set_seed(seed: int) -> None:
    """Seed every RNG that affects a run.

    The configs have carried `seed: 42` since the beginning but nothing read it,
    so LoRA init, dropout, WeightedRandomSampler draws and shuffling were all
    unseeded — runs were not reproducible and repeated runs could not be used as
    error bars, because nothing distinguished a seed from a re-run.
    """
    import random as _random

    _random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as _np

        _np.random.seed(seed)
    except ImportError:
        pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_latent(config_path: str, debug: bool = False) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 42))
    _set_seed(seed)
    print(f"Seed: {seed}")

    import mlflow
    from src.mlflow_utils import setup_mlflow, log_config

    setup_mlflow(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    model_name = cfg["debug_model"] if debug else cfg["base_model"]
    lora_cfg = cfg.get("lora", {})
    train_cfg = cfg["training"]
    loss_weights = cfg.get("loss_weights", {})
    jepa_cfg = cfg.get("jepa", {})
    jepa_enabled = bool(jepa_cfg.get("enabled", False))
    jepa_fields = jepa_cfg.get("fields", JEPA_FIELDS)
    jepa_horizons = [int(k) for k in jepa_cfg.get("horizons", [1])]
    jepa_shuffle = bool(jepa_cfg.get("shuffle_future_labels", False))

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

    exclude_cf = cfg["data"].get("exclude_counterfactual", False)
    if exclude_cf:
        print("[data] counterfactual records EXCLUDED from train and val")

    train_ds = HeadSupervisionDataset(
        cfg["data"]["train_file"],
        tokenizer,
        max_seq_len=train_cfg.get("max_seq_len", 1024),
        jepa_fields=jepa_fields if jepa_enabled else None,
        jepa_horizons=jepa_horizons if jepa_enabled else None,
        shuffle_future_labels=jepa_shuffle if jepa_enabled else False,
        exclude_counterfactual=exclude_cf,
    )
    val_ds = HeadSupervisionDataset(
        cfg["data"]["val_file"],
        tokenizer,
        max_seq_len=train_cfg.get("max_seq_len", 1024),
        jepa_fields=jepa_fields if jepa_enabled else None,
        jepa_horizons=jepa_horizons if jepa_enabled else None,
        shuffle_future_labels=jepa_shuffle if jepa_enabled else False,
        exclude_counterfactual=exclude_cf,
    )

    # Compute class weights early (needed for sampler and loss)
    class_weights = compute_class_weights(cfg["data"]["train_file"], LABEL_MAPS,
                                          exclude_counterfactual=exclude_cf)
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
    focal_gamma = float(train_cfg.get("focal_gamma", 0.0))
    loss_fn = MultiHeadLoss(
        loss_weights,
        class_weights=class_weights,
        label_smoothing=label_smoothing,
        focal_gamma=focal_gamma,
    )
    if focal_gamma > 0.0:
        print(f"Using focal loss with gamma={focal_gamma}")
    jepa_head = None
    if jepa_enabled:
        label_vocab_sizes = {field: len(LABEL_MAPS[field]) for field in jepa_fields}
        jepa_head = SocialJEPAHead(
            SocialJEPAPredictorConfig(
                hidden_dim=predictor.hidden_size,
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
        ).to(next(predictor.heads.parameters()).device)
        print(f"Social-State JEPA enabled: fields={jepa_fields}, horizons={jepa_horizons}")
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
        if jepa_head is not None:
            param_groups.append({"params": jepa_head.parameters(), "lr": head_lr, "weight_decay": float(train_cfg.get("weight_decay", 0.01))})
        print(f"Progressive LR: heads={head_lr}, backbone={base_lr}")
    else:
        param_groups = [p for p in predictor.parameters() if p.requires_grad]
        if jepa_head is not None:
            param_groups += list(jepa_head.parameters())

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

    # Early stopping configuration
    early_stopping_patience = train_cfg.get("early_stopping_patience", 0)
    early_stopping_counter = 0

    # Resume from checkpoint if specified
    resume_from = cfg.get("resume_from")
    start_epoch = 1
    global_step = 0
    if resume_from and Path(resume_from).exists():
        print(f"Resuming from checkpoint: {resume_from}")
        save_predictor(predictor, str(resume_from), load=True)
        if jepa_head is not None:
            jepa_state_file = Path(resume_from) / "jepa_head.pt"
            if jepa_state_file.exists():
                jepa_state = torch.load(jepa_state_file)
                jepa_head.load_state_dict(jepa_state["state_dict"])
        # Try to load training state
        state_file = Path(resume_from) / "training_state.pt"
        if state_file.exists():
            state = torch.load(state_file)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            start_epoch = state["epoch"] + 1
            global_step = state["global_step"]
            best_metric_value = state.get("best_metric_value", None)
            early_stopping_counter = state.get("early_stopping_counter", 0)
            print(f"Resumed from epoch {start_epoch}, global_step {global_step}")
        else:
            best_metric_value = None
            early_stopping_counter = 0
    else:
        best_metric_value = None
        early_stopping_counter = 0

    with mlflow.start_run(run_name=run_name):
        log_config(cfg)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("n_train", len(train_ds))
        mlflow.log_param("n_val", len(val_ds))
        mlflow.log_param("early_stopping_patience", early_stopping_patience)

        best_metric_name = train_cfg.get("metric_for_best_model", "val/response_policy_f1")
        # Determine if higher is better (True for acc/f1, False for loss)
        higher_is_better = "loss" not in best_metric_name
        last_val_metrics: dict = {}

        print(f"Best model selection: {best_metric_name} (higher_is_better={higher_is_better})")

        for epoch in range(start_epoch, epochs + 1):
            predictor.train()
            if jepa_head is not None:
                jepa_head.train()
            epoch_loss = 0.0
            epoch_jepa_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
                device = predictor.backbone.device
                batch = _batch_to_device(batch, device)
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]

                out = predictor(input_ids=input_ids, attention_mask=attention_mask)
                total_loss, detail = loss_fn(out["logits"], batch)
                if jepa_head is not None:
                    future_label_ids = _future_label_ids_from_batch(batch, jepa_horizons, jepa_fields)
                    jepa_out = jepa_head(out["pooled"], future_label_ids)
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
                    clip_params = list(predictor.parameters())
                    if jepa_head is not None:
                        clip_params += list(jepa_head.parameters())
                    torch.nn.utils.clip_grad_norm_(clip_params, max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % train_cfg.get("logging_steps", 20) == 0:
                        mlflow.log_metric("train/loss", epoch_loss / (step + 1), step=global_step)
                        if jepa_head is not None:
                            mlflow.log_metric("train/jepa_loss", epoch_jepa_loss / (step + 1), step=global_step)

            val_loss, val_metrics = _evaluate(
                predictor,
                val_loader,
                loss_fn,
                jepa_head=jepa_head,
                jepa_cfg=jepa_cfg,
                jepa_horizons=jepa_horizons,
                jepa_fields=jepa_fields,
            )
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
            summary_metrics = val_metrics.get("summary", {})
            if metric_key not in summary_metrics:
                # Falling back to val_loss here would be silently wrong:
                # higher_is_better was derived from the metric NAME, so a
                # misspelled key selected the checkpoint with the WORST loss.
                raise KeyError(
                    f"metric_for_best_model={best_metric_name!r} not found in val summary. "
                    f"Available: {sorted(summary_metrics)}"
                )
            current_metric = summary_metrics[metric_key]

            summary_parts = [f"train_loss={epoch_loss/max(1, len(train_loader)):.4f}", f"val_loss={val_loss:.4f}"]
            if jepa_head is not None:
                summary_parts.append(f"train_jepa_loss={epoch_jepa_loss/max(1, len(train_loader)):.4f}")
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
                _save_jepa_head(jepa_head, best_dir, jepa_cfg)
                mlflow.log_metric(f"val/best_{metric_key}", best_metric_value, step=epoch)
                print(f"  → New best model ({best_metric_name}={best_metric_value:.4f}) saved to {best_dir}")
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_patience > 0 and early_stopping_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {early_stopping_counter} epochs without improvement")
                    break

            # Save checkpoint for resume
            checkpoint_dir = output_dir / f"checkpoint_epoch_{epoch}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            save_predictor(predictor, str(checkpoint_dir))
            _save_jepa_head(jepa_head, checkpoint_dir, jepa_cfg)
            torch.save({
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_metric_value": best_metric_value,
                "early_stopping_counter": early_stopping_counter,
            }, checkpoint_dir / "training_state.pt")

        save_predictor(predictor, str(output_dir / "final"))
        _save_jepa_head(jepa_head, output_dir / "final", jepa_cfg)
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
def _evaluate(
    predictor,
    val_loader,
    loss_fn,
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
        device = predictor.backbone.device
        batch = _batch_to_device(batch, device)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        out = predictor(input_ids=input_ids, attention_mask=attention_mask)
        loss, detail = loss_fn(out["logits"], batch)
        if jepa_head is not None and jepa_cfg is not None and jepa_horizons is not None and jepa_fields is not None:
            future_label_ids = _future_label_ids_from_batch(batch, jepa_horizons, jepa_fields)
            jepa_out = jepa_head(out["pooled"], future_label_ids)
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
    if jepa_batches > 0:
        metrics.setdefault("summary", {})["jepa_loss"] = total_jepa_loss / jepa_batches

    avg_loss = total_loss / max(1, len(val_loader))
    return avg_loss, metrics
