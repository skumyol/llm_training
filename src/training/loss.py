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


class MultiHeadLoss(nn.Module):
    def __init__(self, loss_weights: dict[str, float]):
        super().__init__()
        self.weights = loss_weights
        self.ce = nn.CrossEntropyLoss(ignore_index=-1)
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
                        field_loss = self.ce(pred[valid_mask], gold[valid_mask])
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
    def __init__(self, loss_weights: dict[str, float]):
        super().__init__()
        self.multi_head_loss = MultiHeadLoss(loss_weights)
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
