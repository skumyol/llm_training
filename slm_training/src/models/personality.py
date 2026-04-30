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

    def __init__(self, model_name: str, out_dim: int, dropout: float = 0.3,
                 use_sigmoid: bool = False, multi_sample_dropout: int = 0) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        # Mean+Max pooling → 2× hidden dim
        pool_dim = hidden * 2
        self.layer_norm = nn.LayerNorm(pool_dim)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(pool_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, out_dim),
        )
        self.use_sigmoid = use_sigmoid
        self.multi_sample_dropout = multi_sample_dropout
        if multi_sample_dropout > 0:
            self._ms_dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(multi_sample_dropout)])

    def pooled(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        # Mean pooling
        mean_pool = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        # Max pooling (mask out padding with large negative)
        max_pool = last_hidden.masked_fill(~mask.bool(), -1e9).max(dim=1).values
        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        return self.layer_norm(pooled)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        pooled = self.pooled(input_ids=input_ids, attention_mask=attention_mask)
        if self.training and self.multi_sample_dropout > 0:
            # Multi-sample dropout: average logits from multiple dropout masks
            logits = torch.stack([self.head(d(pooled)) for d in self._ms_dropouts], dim=0).mean(dim=0)
        else:
            logits = self.head(pooled)
        preds = torch.sigmoid(logits) if self.use_sigmoid else logits
        return {"pooled": pooled, "preds": preds, "logits": logits}
