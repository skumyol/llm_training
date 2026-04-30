from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from transformers import AutoModel


class DistilBertRegressor(nn.Module):
    """Continuous regressor for OCEAN/VAD style targets.

    Reference note for your architecture: OCEAN and VAD are continuous targets in the
    uploaded plan, so the head is a regressor, not a classifier. fileciteturn12file2L5-L9
    """

    def __init__(self, model_name: str, out_dim: int, dropout: float = 0.1,
                 multi_sample_dropout: int = 0) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.layer_norm = nn.LayerNorm(hidden)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
        self.multi_sample_dropout = multi_sample_dropout
        if multi_sample_dropout > 0:
            self._ms_dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(multi_sample_dropout)])

    def pooled(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        # Mean pooling (proven better than CLS for regression)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.layer_norm(pooled)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        pooled = self.pooled(input_ids=input_ids, attention_mask=attention_mask)
        if self.training and self.multi_sample_dropout > 0:
            preds = torch.stack([self.head(d(pooled)) for d in self._ms_dropouts], dim=0).mean(dim=0)
        else:
            preds = self.head(pooled)
        return {"pooled": pooled, "preds": preds}
