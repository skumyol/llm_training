import math
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.dataset import SFTDataset, collate_sft_batch
from src.training.model import load_backbone
from src.metrics_report import log_metrics_to_mlflow, write_metrics_bundle


def train_response(config_path: str, debug: bool = False) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    import mlflow
    from src.mlflow_utils import setup_mlflow, log_config

    setup_mlflow(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    model_name = cfg["debug_model"] if debug else cfg["base_model"]
    lora_cfg = cfg.get("lora", {})
    train_cfg = cfg["training"]

    quantization = cfg.get("quantization", "4bit")
    torch_dtype = cfg.get("torch_dtype", "bfloat16")

    print(f"Loading model: {model_name}")
    model, tokenizer, _ = load_backbone(
        model_name,
        quantization=quantization,
        lora_config={
            "r": lora_cfg.get("r", 32),
            "alpha": lora_cfg.get("alpha", 64),
            "dropout": lora_cfg.get("dropout", 0.05),
            "target_modules": lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
            "bias": lora_cfg.get("bias", "none"),
        },
        torch_dtype=torch_dtype,
    )

    if train_cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    max_seq_len = train_cfg.get("max_seq_len", 2048)
    mask_secret_spans = bool(train_cfg.get("mask_secret_spans", False))
    secret_strings_file = cfg.get("secret_strings_file") or cfg.get("data", {}).get("secret_strings_file")
    if mask_secret_spans:
        print(f"Secret-span masking ENABLED  (secret_strings_file={secret_strings_file})")
    train_ds = SFTDataset(
        cfg["data"]["train_file"], tokenizer, max_seq_len=max_seq_len,
        mask_secret_spans=mask_secret_spans, secret_strings_file=secret_strings_file,
    )
    val_ds = SFTDataset(
        cfg["data"]["val_file"], tokenizer, max_seq_len=max_seq_len,
        mask_secret_spans=mask_secret_spans, secret_strings_file=secret_strings_file,
    )

    batch_size = train_cfg.get("batch_size", 2)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_sft_batch,
        num_workers=train_cfg.get("dataloader_num_workers", 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_sft_batch, num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )

    epochs = train_cfg.get("epochs", 3)
    grad_accum = train_cfg.get("grad_accum", 16)
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    total_steps = math.ceil(len(train_loader) / grad_accum) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    output_dir = Path(cfg["output"]["checkpoint_dir"])
    best_dir   = Path(cfg["output"]["best_model_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    run_name = cfg["mlflow"].get("run_name", "response_sft")
    conditioning_mode = cfg.get("conditioning_mode", "gold")

    with mlflow.start_run(run_name=run_name):
        log_config(cfg)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("conditioning_mode", conditioning_mode)
        mlflow.log_param("n_train", len(train_ds))
        mlflow.log_param("n_val", len(val_ds))

        best_val_loss = float("inf")
        global_step = 0
        last_val_loss = float("nan")

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
                input_ids     = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                labels        = batch["labels"].to(model.device)

                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss / grad_accum
                loss.backward()
                epoch_loss += loss.item() * grad_accum

                if (step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % train_cfg.get("logging_steps", 20) == 0:
                        mlflow.log_metric("train/lm_loss", epoch_loss / (step + 1), step=global_step)

            val_loss = _evaluate_sft(model, val_loader)
            last_val_loss = val_loss
            mlflow.log_metric("val/lm_loss", val_loss, step=epoch)
            log_metrics_to_mlflow({"train_lm_loss": epoch_loss / max(1, len(train_loader)), "val_lm_loss": val_loss}, prefix="response", step=epoch)
            print(f"Epoch {epoch}: train_loss={epoch_loss/len(train_loader):.4f}  val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_pretrained(str(best_dir))
                tokenizer.save_pretrained(str(best_dir))
                mlflow.log_metric("val/best_lm_loss", best_val_loss, step=epoch)
                print(f"  → New best model saved to {best_dir}")

        model.save_pretrained(str(output_dir / "final"))
        tokenizer.save_pretrained(str(output_dir / "final"))
        mlflow.log_artifact(str(best_dir))
        write_metrics_bundle(
            output_dir / "metrics",
            "response_training_summary",
            {
                "summary": {
                    "model_name": model_name,
                    "conditioning_mode": conditioning_mode,
                    "epochs": epochs,
                    "best_val_lm_loss": best_val_loss,
                    "final_val_lm_loss": last_val_loss,
                    "n_train": len(train_ds),
                    "n_val": len(val_ds),
                }
            },
            title="Response Training Summary",
        )
        print("Response SFT training complete.")


@torch.no_grad()
def _evaluate_sft(model, val_loader) -> float:
    model.eval()
    total_loss = 0.0
    for batch in tqdm(val_loader, desc="Val", leave=False):
        input_ids      = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels         = batch["labels"].to(model.device)
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += out.loss.item()
    return total_loss / max(1, len(val_loader))
