import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


GROUP_FIELDS: dict[str, list[str]] = {
    "C": ["dialogue_act", "tone", "risk_type"],
    "A": ["valence", "arousal", "threat", "control"],
    "M": ["player_intent", "player_knowledge", "player_credibility"],
    "R": [
        "affection_level", "affection_delta",
        "respect_level", "respect_delta",
        "dominance_level", "dominance_delta",
        "familiarity_level", "familiarity_delta",
        "trust_level", "trust_delta",
        "obligation_level", "obligation_delta",
    ],
    "N": ["duty_pressure", "secrecy_pressure", "face_pressure", "value_conflict"],
    "D": ["response_policy", "reveal_decision", "repair_strategy"],
}


def compute_class_weights(train_file: str, label_maps: dict) -> dict[str, torch.Tensor]:
    """Compute inverse-frequency class weights from training data."""
    from src.training.dataset import LABEL_TO_IDX
    records = []
    with open(train_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    weights = {}
    for field, idx_map in LABEL_TO_IDX.items():
        if field == "dialogue_act":
            continue
        n_classes = len(label_maps.get(field, []))
        if n_classes == 0:
            continue
        counts = Counter()
        for r in records:
            val = r.get("labels", {}).get(field)
            if val is not None:
                idx = idx_map.get(str(val), -1)
                if idx != -1:
                    counts[idx] += 1
        if not counts:
            continue
        total = sum(counts.values())
        w = torch.ones(n_classes, dtype=torch.float32)
        for cls_idx in range(n_classes):
            c = counts.get(cls_idx, 0)
            if c > 0:
                w[cls_idx] = total / (n_classes * c)
            else:
                w[cls_idx] = 1.0
        # Clamp to avoid extreme weights
        w = w.clamp(min=0.2, max=5.0)
        weights[field] = w
    return weights


def _focal_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma: float,
    weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Multi-class focal CE with optional class weights and label smoothing.

    Reduces to standard (weighted, smoothed) cross-entropy when gamma == 0.
    Targets are class indices; -1 entries must be filtered by the caller.
    """
    if gamma <= 0.0:
        return F.cross_entropy(logits, targets, weight=weight, label_smoothing=label_smoothing)
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    n_classes = logits.size(-1)
    with torch.no_grad():
        if label_smoothing > 0.0:
            true_dist = torch.full_like(log_probs, label_smoothing / max(1, n_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - label_smoothing)
        else:
            true_dist = F.one_hot(targets, num_classes=n_classes).to(log_probs.dtype)
    # focal weighting on the true-class probability
    pt = (probs * true_dist).sum(dim=-1).clamp(min=1e-8, max=1.0)
    focal = (1.0 - pt).pow(gamma)
    nll = -(true_dist * log_probs).sum(dim=-1)
    if weight is not None:
        # per-sample class weight applied via target class index
        w = weight.gather(0, targets)
        nll = nll * w
    return (focal * nll).mean()


class MultiHeadLoss(nn.Module):
    def __init__(
        self,
        loss_weights: dict[str, float],
        class_weights: dict[str, torch.Tensor] | None = None,
        label_smoothing: float = 0.0,
        focal_gamma: float = 0.0,
    ):
        super().__init__()
        self.weights = loss_weights
        self.class_weights = class_weights or {}
        self.label_smoothing = label_smoothing
        self.focal_gamma = float(focal_gamma)
        self.ce = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=label_smoothing)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: dict, batch: dict) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        group_losses: dict[str, torch.Tensor] = {}
        detail_losses: dict[str, torch.Tensor] = {}
        device = next(iter(logits.values())).device

        for group, fields in GROUP_FIELDS.items():
            group_loss = torch.tensor(0.0, device=device)
            count = 0
            for field in fields:
                label_key = f"label_{field}"
                if label_key not in batch or field not in logits:
                    continue
                pred = logits[field]
                gold = batch[label_key]
                if isinstance(gold, torch.Tensor):
                    gold = gold.to(device)

                if field == "dialogue_act":
                    valid_mask = gold.sum(dim=1) > 0
                    if valid_mask.any():
                        field_loss = self.bce(pred[valid_mask], gold[valid_mask].float())
                    else:
                        field_loss = torch.tensor(0.0, device=device)
                else:
                    valid_mask = gold != -1
                    if valid_mask.any():
                        cw = self.class_weights.get(field)
                        if cw is not None:
                            cw = cw.to(device)
                        field_loss = _focal_cross_entropy(
                            pred[valid_mask],
                            gold[valid_mask],
                            gamma=self.focal_gamma,
                            weight=cw,
                            label_smoothing=self.label_smoothing,
                        )
                    else:
                        field_loss = torch.tensor(0.0, device=device)

                detail_losses[field] = field_loss
                group_loss = group_loss + field_loss
                count += 1

            if count > 0:
                group_losses[group] = group_loss / count

        total = torch.tensor(0.0, device=device)
        for group, loss in group_losses.items():
            w = self.weights.get(f"lambda_{group}", 1.0)
            total = total + w * loss

        return total, {**group_losses, **detail_losses}


class ConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: dict,
    ) -> torch.Tensor:
        device = next(iter(logits.values())).device
        reveal_logits = logits.get("reveal_decision")
        secrecy_logits = logits.get("secrecy_pressure")

        if reveal_logits is None or secrecy_logits is None:
            return torch.tensor(0.0, device=device)

        reveal_probs = F.softmax(reveal_logits, dim=-1)
        # REVEAL_LABELS = ["none", "hint", "partial", "full"] -> index 3 is "full"
        full_reveal_prob = reveal_probs[:, 3]

        secrecy_probs = F.softmax(secrecy_logits, dim=-1)
        # THREE_LEVEL_LABELS = ["low", "medium", "high"] -> index 2 is "high"
        high_secrecy_prob = secrecy_probs[:, 2]

        penalty = (full_reveal_prob * high_secrecy_prob).mean()
        return penalty


class JointLoss(nn.Module):
    def __init__(
        self,
        loss_weights: dict[str, float],
        class_weights: dict[str, torch.Tensor] | None = None,
        label_smoothing: float = 0.0,
        focal_gamma: float = 0.0,
    ):
        super().__init__()
        self.multi_head_loss = MultiHeadLoss(
            loss_weights,
            class_weights=class_weights,
            label_smoothing=label_smoothing,
            focal_gamma=focal_gamma,
        )
        self.consistency_loss = ConsistencyLoss()
        self.lambda_Y = loss_weights.get("lambda_Y", 1.0)
        self.lambda_consistency = loss_weights.get("lambda_consistency", 0.5)

    def forward(
        self,
        logits: dict,
        lm_loss: torch.Tensor,
        batch: dict,
    ) -> tuple[torch.Tensor, dict]:
        head_loss, detail = self.multi_head_loss(logits, batch)
        consistency = self.consistency_loss(logits)
        
        total = head_loss + self.lambda_Y * lm_loss + self.lambda_consistency * consistency
        
        detail["lm_loss"] = lm_loss
        detail["head_loss"] = head_loss
        detail["consistency_loss"] = consistency
        return total, detail
