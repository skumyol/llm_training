"""
Lightweight relational memory for multi-turn NPC dialogue.

Only tracks trust/respect/affection/familiarity/dominance/obligation levels
across turns. Does NOT affect decision heads directly — memory is fed into
the router as an auxiliary signal.

Architecture:
    h_t       = backbone(context_t)           (frozen or fine-tuned)
    rel_t     = MLP(h_t)  → 6D relational features
    m_t       = GRU(m_{t-1}, rel_t)           (learned memory state)
    router    = concat(decision_probs, m_t)   (only if ablation proves helpful)

Usage:
    from src.training.relational_memory import RelationalMemory, RelationalMemoryPredictor

    mem = RelationalMemory(hidden_size=256, memory_size=32)
    # Or wrap existing predictor:
    wrapped = RelationalMemoryPredictor(predictor, memory_size=32)
"""
import torch
import torch.nn as nn
from typing import Optional


STANCE_DIMS = ["affection", "respect", "dominance", "familiarity", "trust", "obligation"]
RELATIONAL_HEADS = [f"{d}_level" for d in STANCE_DIMS]


class RelationalMemory(nn.Module):
    """GRU-based memory that carries relational state across dialogue turns."""

    def __init__(self, relational_feature_size: int = 6, memory_size: int = 32, num_layers: int = 1):
        super().__init__()
        self.memory_size = memory_size
        self.gru = nn.GRU(
            input_size=relational_feature_size,
            hidden_size=memory_size,
            num_layers=num_layers,
            batch_first=True,
        )
        # Project pooled backbone output to relational features
        self.relational_proj = nn.Sequential(
            nn.LazyLinear(128),
            nn.GELU(),
            nn.Linear(128, relational_feature_size),
        )
        # Optional: project memory back to something useful for routing
        self.memory_proj = nn.Sequential(
            nn.Linear(memory_size, 64),
            nn.GELU(),
            nn.Linear(64, relational_feature_size),
        )

    def forward(self, pooled: torch.Tensor, memory_state: Optional[torch.Tensor] = None):
        """
        Args:
            pooled: (batch, hidden_size) from backbone
            memory_state: (num_layers, batch, memory_size) or None
        Returns:
            relational_features: (batch, relational_feature_size)
            new_memory: (num_layers, batch, memory_size)
            projected_memory: (batch, relational_feature_size)
        """
        rel_features = self.relational_proj(pooled)  # (batch, 6)
        rel_features = rel_features.unsqueeze(1)  # (batch, 1, 6) for GRU

        if memory_state is None:
            # Initialize to zero
            memory_state = torch.zeros(
                self.gru.num_layers,
                pooled.size(0),
                self.memory_size,
                device=pooled.device,
                dtype=pooled.dtype,
            )

        _, new_memory = self.gru(rel_features, memory_state)  # new_memory: (layers, batch, mem)
        projected = self.memory_proj(new_memory[-1])  # (batch, relational_feature_size)

        return rel_features.squeeze(1), new_memory, projected

    def init_memory(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        return torch.zeros(self.gru.num_layers, batch_size, self.memory_size, device=device, dtype=dtype)


class RelationalMemoryPredictor(nn.Module):
    """
    Wraps an existing LatentStatePredictor with relational memory.

    The memory is updated per-turn and concatenated to the pooled representation
    for relational head prediction ONLY. Decision heads use pooled directly.
    """

    def __init__(self, base_predictor: nn.Module, memory_size: int = 32, relational_feature_size: int = 6):
        super().__init__()
        self.base = base_predictor
        self.memory = RelationalMemory(
            relational_feature_size=relational_feature_size,
            memory_size=memory_size,
        )
        # Rebuild relational heads to take pooled + memory projection
        hidden_size = base_predictor.hidden_size
        relational_input_size = hidden_size + relational_feature_size

        # Build relational heads that take pooled + memory projection
        import copy
        from src.training.model import ClassificationHead
        self.rel_heads = nn.ModuleDict()
        for name, head in base_predictor.heads.items():
            if name in RELATIONAL_HEADS:
                spec = base_predictor.head_specs[name]
                # Projection back to hidden_size so original head can consume it
                proj = nn.Linear(relational_input_size, hidden_size)
                # Initialize to approximately extract the pooled part (identity-like)
                with torch.no_grad():
                    eye = torch.eye(hidden_size, relational_input_size)
                    proj.weight.copy_(eye)
                    proj.bias.zero_()
                # Reuse original head weights so checkpoint knowledge is preserved
                self.rel_heads[name] = nn.Sequential(
                    proj,
                    copy.deepcopy(head),
                )
        self.relational_head_names = set(RELATIONAL_HEADS)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        memory_state: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        # Get base predictor output
        base_out = self.base(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        pooled = base_out["pooled"]  # (batch, hidden_size)

        # Update relational memory
        rel_features, new_memory, projected = self.memory(pooled, memory_state)

        # For relational heads, use pooled + projected memory
        enriched = torch.cat([pooled, projected], dim=-1)

        logits = dict(base_out["logits"])  # Start with base logits
        for name, head in self.rel_heads.items():
            logits[name] = head(enriched)

        return {
            "logits": logits,
            "pooled": pooled,
            "memory": new_memory,
            "relational_features": rel_features,
            "lm_loss": base_out.get("lm_loss", None),
        }
