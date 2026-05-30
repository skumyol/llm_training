"""
Weighted multi-head training that prioritizes decision heads.

Usage:
    from src.training.weighted_loss_trainer import WeightedLossTrainer, HEAD_WEIGHTS

    trainer = WeightedLossTrainer(
        predictor, tokenizer,
        head_weights=HEAD_WEIGHTS['decision_priority'],
        focal_gamma={'reveal_decision': 2.0, 'secrecy_pressure': 2.0},
    )
    trainer.train(train_loader, epochs=3)

The loss becomes:
    L = sum_{heads} weight[head] * loss[head]

With optional focal loss on specified heads.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# Three preset weighting schemes
HEAD_WEIGHTS = {
    "uniform": {},  # all 1.0
    "decision_priority": {
        "response_policy": 1.0,
        "reveal_decision": 1.0,
        "value_conflict": 1.0,
        "secrecy_pressure": 1.0,
        "player_intent": 1.0,
        "threat": 1.0,
        # Designer-visible: reduced
        "valence": 0.3,
        "arousal": 0.3,
        "tone": 0.3,
        "face_pressure": 0.3,
        "repair_strategy": 0.3,
        "player_credibility": 0.3,
        # Relational: even lower
        "affection_level": 0.1,
        "respect_level": 0.1,
        "dominance_level": 0.1,
        "familiarity_level": 0.1,
        "trust_level": 0.1,
        "obligation_level": 0.1,
        # Deltas: minimal
        "affection_delta": 0.05,
        "respect_delta": 0.05,
        "dominance_delta": 0.05,
        "familiarity_delta": 0.05,
        "trust_delta": 0.05,
        "obligation_delta": 0.05,
        # Meta / other
        "dialogue_act": 0.2,
        "risk_type": 0.2,
        "control": 0.2,
        "player_knowledge": 0.2,
        "duty_pressure": 0.2,
    },
    "routing_only": {
        "response_policy": 1.0,
        "reveal_decision": 1.0,
        "value_conflict": 1.0,
        "secrecy_pressure": 1.0,
        # Everything else zeroed
    },
}


def _focal_loss(logits: torch.Tensor, labels: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """Focal loss for a single head."""
    ce = F.cross_entropy(logits, labels, reduction="none")
    probs = F.softmax(logits, dim=-1)
    p_t = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    focal_weight = (1.0 - p_t) ** gamma
    return (focal_weight * ce).mean()


class WeightedLossTrainer:
    def __init__(
        self,
        predictor: nn.Module,
        tokenizer,
        head_weights: dict[str, float] | str = "decision_priority",
        focal_gamma: dict[str, float] | None = None,
        lr: float = 2e-5,
        device: torch.device | None = None,
    ):
        self.predictor = predictor
        self.tokenizer = tokenizer
        self.focal_gamma = focal_gamma or {}

        if isinstance(head_weights, str):
            self.head_weights = HEAD_WEIGHTS.get(head_weights, {})
        else:
            self.head_weights = head_weights

        self.device = device or next(predictor.parameters()).device
        self.optimizer = torch.optim.AdamW(predictor.parameters(), lr=lr)

    def _compute_head_loss(
        self,
        field: str,
        logits: torch.Tensor,
        gold: torch.Tensor,
        weight: float,
    ) -> torch.Tensor:
        if weight == 0.0:
            return torch.tensor(0.0, device=logits.device)

        if field == "dialogue_act":
            valid = gold.sum(dim=1) > 0
            if not valid.any():
                return torch.tensor(0.0, device=logits.device)
            loss = F.binary_cross_entropy_with_logits(
                logits[valid], gold[valid].float()
            )
        else:
            valid = gold != -1
            if not valid.any():
                return torch.tensor(0.0, device=logits.device)
            if field in self.focal_gamma:
                loss = _focal_loss(logits[valid], gold[valid], self.focal_gamma[field])
            else:
                loss = F.cross_entropy(logits[valid], gold[valid])

        return weight * loss

    def train_epoch(self, dataloader, desc: str = "Training") -> float:
        self.predictor.train()
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(dataloader, desc=desc):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            out = self.predictor(input_ids=input_ids, attention_mask=attention_mask)

            losses = []
            for field in out["logits"]:
                label_key = f"label_{field}"
                if label_key not in batch:
                    continue
                gold = batch[label_key].to(self.device)
                logits = out["logits"][field]
                weight = self.head_weights.get(field, 1.0)
                loss = self._compute_head_loss(field, logits, gold, weight)
                if loss.item() > 0:
                    losses.append(loss)

            if not losses:
                continue

            loss = sum(losses)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(1, n_batches)

    def train(self, dataloader, epochs: int = 3) -> list[float]:
        epoch_losses = []
        for epoch in range(epochs):
            avg_loss = self.train_epoch(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
            epoch_losses.append(avg_loss)
            print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f}")
        return epoch_losses
