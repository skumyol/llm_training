from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


JEPA_FIELDS = [
    "trust_delta",
    "respect_delta",
    "dominance_delta",
    "secrecy_pressure",
    "player_knowledge",
    "response_policy",
    "reveal_decision",
]


@dataclass
class SocialStateEmbeddingConfig:
    emb_dim: int = 64
    out_dim: int = 128
    dropout: float = 0.1


@dataclass
class SocialJEPAPredictorConfig:
    hidden_dim: int
    target_dim: int = 128
    predictor_dim: int = 256
    horizons: List[int] = field(default_factory=lambda: [1])
    dropout: float = 0.1


class SocialStateEmbedding(nn.Module):
    def __init__(self, label_vocab_sizes: Dict[str, int], cfg: SocialStateEmbeddingConfig):
        super().__init__()
        self.fields = list(label_vocab_sizes.keys())
        self.embeddings = nn.ModuleDict({
            field: nn.Embedding(num_classes, cfg.emb_dim)
            for field, num_classes in label_vocab_sizes.items()
        })
        in_dim = len(self.fields) * cfg.emb_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, cfg.out_dim),
            nn.LayerNorm(cfg.out_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.out_dim, cfg.out_dim),
        )

    def forward(self, label_ids: Dict[str, torch.Tensor]) -> torch.Tensor:
        embeddings = []
        for field in self.fields:
            if field not in label_ids:
                raise KeyError(f"Missing label field for JEPA target: {field}")
            embeddings.append(self.embeddings[field](label_ids[field].long()))
        return self.proj(torch.cat(embeddings, dim=-1))


class HorizonPredictor(nn.Module):
    def __init__(self, hidden_dim: int, predictor_dim: int, target_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, predictor_dim),
            nn.LayerNorm(predictor_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(predictor_dim, target_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class SocialJEPAHead(nn.Module):
    def __init__(
        self,
        cfg: SocialJEPAPredictorConfig,
        label_vocab_sizes: Dict[str, int],
        state_emb_cfg: SocialStateEmbeddingConfig,
    ):
        super().__init__()
        self.horizons = cfg.horizons
        self.target_embedder = SocialStateEmbedding(label_vocab_sizes, state_emb_cfg)
        self.predictors = nn.ModuleDict({
            str(k): HorizonPredictor(cfg.hidden_dim, cfg.predictor_dim, cfg.target_dim, cfg.dropout)
            for k in self.horizons
        })

    def forward(self, h_t: torch.Tensor, future_label_ids: Dict[int, Dict[str, torch.Tensor]]) -> Dict[int, Dict[str, torch.Tensor]]:
        outputs = {}
        for k in self.horizons:
            if k not in future_label_ids:
                continue
            target_labels = future_label_ids[k]
            pred = self.predictors[str(k)](h_t)
            valid = torch.ones(pred.size(0), dtype=torch.bool, device=pred.device)
            for values in target_labels.values():
                valid = valid & (values.to(pred.device) >= 0)
            safe_target_labels = {
                field: values.to(pred.device).clamp_min(0)
                for field, values in target_labels.items()
            }
            target = self.target_embedder(safe_target_labels)
            outputs[k] = {"pred": pred, "target": target, "valid": valid}
        return outputs


def cosine_jepa_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target.detach(), dim=-1)
    return 2.0 - 2.0 * (pred * target).sum(dim=-1).mean()


def variance_regularization(x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    # Guard against tensors with <2 samples (var() is undefined)
    if x.size(0) < 2:
        return torch.tensor(0.0, device=x.device)
    std = torch.sqrt(x.var(dim=0) + eps)
    return torch.mean(F.relu(1.0 - std))


def social_jepa_loss(
    jepa_outputs: Dict[int, Dict[str, torch.Tensor]],
    horizon_weights: Dict[int, float] | None = None,
    var_weight: float = 0.0,
) -> torch.Tensor:
    total = None
    norm = 0.0
    for k, pair in jepa_outputs.items():
        weight = 1.0 if horizon_weights is None else horizon_weights.get(k, 1.0)
        pred = pair["pred"]
        target = pair["target"]
        valid = pair.get("valid")
        if valid is not None:
            if not valid.any():
                continue
            pred = pred[valid]
            target = target[valid]
        loss = cosine_jepa_loss(pred, target)
        if var_weight > 0:
            loss = loss + var_weight * variance_regularization(pred)
        total = weight * loss if total is None else total + weight * loss
        norm += weight
    if total is None:
        device = next(iter(next(iter(jepa_outputs.values())).values())).device if jepa_outputs else torch.device("cpu")
        return torch.tensor(0.0, device=device)
    return total / max(norm, 1e-8)
