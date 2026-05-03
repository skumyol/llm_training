from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.config import AffectTrainConfig, ensure_dir
from src.data.dialogue_data import RegressionTextDataset
from src.models.affect import DistilBertRegressor


def run(cfg: AffectTrainConfig) -> None:
    train_ds = RegressionTextDataset(
        path=cfg.train_path,
        tokenizer_name=cfg.model_name,
        text_column=cfg.text_column,
        target_columns=cfg.target_columns,
        max_length=cfg.max_length,
    )
    val_ds = RegressionTextDataset(
        path=cfg.val_path,
        tokenizer_name=cfg.model_name,
        text_column=cfg.text_column,
        target_columns=cfg.target_columns,
        max_length=cfg.max_length,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistilBertRegressor(cfg.model_name, out_dim=len(cfg.target_columns)).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr)

    best_val = float("inf")
    out_dir = ensure_dir(cfg.output_dir)

    for epoch in range(cfg.epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"train affect epoch {epoch+1}"):
            batch = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask", "labels")}
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = F.mse_loss(out["preds"], batch["labels"])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask", "labels")}
                out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                total += F.mse_loss(out["preds"], batch["labels"], reduction="sum").item()
                count += batch["labels"].numel()
        val_loss = total / max(count, 1)
        if val_loss < best_val:
            best_val = val_loss
            model.encoder.save_pretrained(out_dir)
            torch.save(model.state_dict(), out_dir / "pytorch_model.bin")
            print(f"saved best affect encoder to {out_dir}, val_loss={val_loss:.6f}")


if __name__ == "__main__":
    cfg = AffectTrainConfig()
    run(cfg)
