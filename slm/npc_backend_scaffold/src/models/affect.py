from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from transformers import AutoModel


class DistilBertRegressor(nn.Module):
    """Continuous regressor for OCEAN/VAD style targets.

    Reference note for your architecture: OCEAN and VAD are continuous targets in the
    uploaded plan, so the head is a regressor, not a classifier. fileciteturn12file2L5-L9
    """

    def __init__(self, model_name: str, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def pooled(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        pooled = self.pooled(input_ids=input_ids, attention_mask=attention_mask)
        preds = self.head(pooled)
        return {"pooled": pooled, "preds": preds}
