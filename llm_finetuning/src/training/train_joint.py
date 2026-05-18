import math
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.dataset import JointDataset, collate_joint_batch
from src.training.loss import JointLoss
from src.training.model import load_predictor, save_predictor, LatentStatePredictor
from src.metrics_report import log_metrics_to_mlflow, write_metrics_bundle


def _batch_to_device(batch: dict, device) -> dict:
    """Return a copy of batch with all tensors moved to device."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def train_joint(config_path: str, debug: bool = False) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    import mlflow
    from src.mlflow_utils import setup_mlflow, log_config

    setup_mlflow(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    model_name = cfg.get("debug_model", cfg["base_model"]) if debug else cfg["base_model"]
    train_cfg  = cfg["training"]
    loss_weights = cfg.get("loss_weights", {})

    init_from = cfg.get("init_from", {})
    latent_ckpt = init_from.get("latent_checkpoint")
    response_ckpt = init_from.get("response_checkpoint")

    print(f"Initializing Joint Model...")
    
    # Logic:
    # 1. Base Model + Tokenizer (Always needed)
    # 2. Backbone Adapter: Prioritize Response Ckpt > Latent Ckpt > Random/LoraConfig
    # 3. Heads: Prioritize Latent Ckpt > Random

    from src.training.model import load_backbone, _move_heads_to_backbone_device
    from peft import PeftModel

    # 1. Load Base Model
    backbone, tokenizer, hidden_size = load_backbone(
        model_name,
        cfg.get("quantization", "4bit"),
        torch_dtype=cfg.get("torch_dtype", "bfloat16"),
    )
    
    # 2. Load Adapter
    if response_ckpt and Path(response_ckpt).exists():
        print(f"Loading backbone adapter from response checkpoint: {response_ckpt}")
        backbone = PeftModel.from_pretrained(backbone, response_ckpt)
    elif latent_ckpt and (Path(latent_ckpt) / "backbone").exists():
        print(f"Loading backbone adapter from latent checkpoint: {latent_ckpt}")
        backbone = PeftModel.from_pretrained(backbone, str(Path(latent_ckpt) / "backbone"))
    else:
        print("Initializing new LoRA adapter for backbone")
        from peft import LoraConfig, get_peft_model, TaskType
        lora_cfg = cfg.get("lora", {})
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_cfg.get("r", 32),
            lora_alpha=lora_cfg.get("alpha", 64),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
        )
        backbone = get_peft_model(backbone, peft_config)

    # 3. Build Predictor and Load Heads
    predictor = LatentStatePredictor(backbone, hidden_size)
    
    if latent_ckpt and (Path(latent_ckpt) / "heads.pt").exists():
        print(f"Loading heads from latent checkpoint: {latent_ckpt}")
        heads_path = Path(latent_ckpt) / "heads.pt"
        head_state = torch.load(heads_path, map_location="cpu")
        for name, state in head_state.items():
            if name in predictor.heads:
                predictor.heads[name].load_state_dict(state)
    else:
        print("Initializing heads from scratch")

    _move_heads_to_backbone_device(predictor)
    
    # Ensure gradients are enabled/disabled correctly
    if train_cfg.get("gradient_checkpointing", False):
        predictor.backbone.gradient_checkpointing_enable()
    
    # Make sure we are in training mode
    predictor.train()

    max_seq_len = train_cfg.get("max_seq_len", 2048)
    heads_seq_len = train_cfg.get("heads_max_seq_len", 1024)
    train_ds = JointDataset(
        cfg["data"]["train_file"], cfg["data"]["heads_train_file"],
        tokenizer, sft_max_seq_len=max_seq_len, heads_max_seq_len=heads_seq_len,
    )
    val_ds = JointDataset(
        cfg["data"]["val_file"], cfg["data"]["heads_val_file"],
        tokenizer, sft_max_seq_len=max_seq_len, heads_max_seq_len=heads_seq_len,
    )

    batch_size = train_cfg.get("batch_size", 2)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_joint_batch,
        num_workers=train_cfg.get("dataloader_num_workers", 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_joint_batch, num_workers=0,
    )

    loss_fn = JointLoss(
        loss_weights,
        label_smoothing=float(train_cfg.get("label_smoothing", 0.0)),
        focal_gamma=float(train_cfg.get("focal_gamma", 0.0)),
    )
    optimizer = torch.optim.AdamW(
        [p for p in predictor.parameters() if p.requires_grad],
        lr=float(train_cfg.get("lr", 5e-5)),
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

    run_name = cfg["mlflow"].get("run_name", "joint_train")

    with mlflow.start_run(run_name=run_name):
        log_config(cfg)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("n_train", len(train_ds))

        best_val_loss = float("inf")
        global_step = 0
        last_val_loss = float("nan")

        for epoch in range(1, epochs + 1):
            predictor.train()
            epoch_loss = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
                device = predictor.backbone.device
                batch = _batch_to_device(batch, device)
                
                # SFT inputs (Context + Latent + Response)
                input_ids      = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels         = batch["labels"]
                
                # Heads inputs (Context only)
                heads_input_ids = batch["heads_input_ids"]
                heads_attention_mask = batch["heads_attention_mask"]

                # 1. Latent Prediction (Heads) on Context only
                out_heads = predictor(
                    input_ids=heads_input_ids, 
                    attention_mask=heads_attention_mask
                )
                logits = out_heads["logits"]

                # 2. Response Generation (LM) on full sequence
                lm_out = predictor.backbone(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    labels=labels
                )
                lm_loss = lm_out.loss
                
                # loss_fn expects (logits_dict, lm_loss, batch)
                total_loss, detail = loss_fn(logits, lm_loss, batch)
                (total_loss / grad_accum).backward()
                epoch_loss += total_loss.item()

                if (step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % train_cfg.get("logging_steps", 20) == 0:
                        mlflow.log_metric("train/joint_loss", epoch_loss / (step + 1), step=global_step)
                        mlflow.log_metric("train/lm_loss", detail["lm_loss"].item(), step=global_step)

            val_loss = _eval_joint(predictor, val_loader, loss_fn)
            last_val_loss = val_loss
            mlflow.log_metric("val/joint_loss", val_loss, step=epoch)
            log_metrics_to_mlflow({"train_joint_loss": epoch_loss / max(1, len(train_loader)), "val_joint_loss": val_loss}, prefix="joint", step=epoch)
            print(f"Epoch {epoch}: train={epoch_loss/len(train_loader):.4f}  val={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_predictor(predictor, str(best_dir))
                print(f"  → New best saved to {best_dir}")

        save_predictor(predictor, str(output_dir / "final"))
        mlflow.log_artifact(str(best_dir))
        write_metrics_bundle(
            output_dir / "metrics",
            "joint_training_summary",
            {
                "summary": {
                    "model_name": model_name,
                    "epochs": epochs,
                    "best_val_joint_loss": best_val_loss,
                    "final_val_joint_loss": last_val_loss,
                    "n_train": len(train_ds),
                    "n_val": len(val_ds),
                }
            },
            title="Joint Training Summary",
        )
        print("Joint training complete.")


@torch.no_grad()
def _eval_joint(predictor, val_loader, loss_fn) -> float:
    predictor.eval()
    total = 0.0
    for batch in tqdm(val_loader, desc="Val", leave=False):
        device = predictor.backbone.device
        batch = _batch_to_device(batch, device)
        
        # SFT inputs
        input_ids      = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels         = batch.get("labels")

        # Heads inputs
        heads_input_ids = batch["heads_input_ids"]
        heads_attention_mask = batch["heads_attention_mask"]

        # 1. Latent Prediction (Heads)
        out_heads = predictor(
            input_ids=heads_input_ids, 
            attention_mask=heads_attention_mask
        )
        logits = out_heads["logits"]

        # 2. Response Generation (LM)
        if labels is not None:
            lm_out = predictor.backbone(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )
            lm_loss = lm_out.loss
        else:
            lm_loss = torch.tensor(0.0, device=predictor.backbone.device)

        loss, _ = loss_fn(logits, lm_loss, batch)
        total += loss.item()
    return total / max(1, len(val_loader))
