import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, PeftModel


STANCE_DIMS = ["affection", "respect", "dominance", "familiarity", "trust", "obligation"]

HEAD_SPECS: dict[str, dict] = {
    "dialogue_act":       {"n_classes": 10, "multi_label": True},
    "tone":               {"n_classes": 6,  "multi_label": False},
    "risk_type":          {"n_classes": 5,  "multi_label": False},
    "valence":            {"n_classes": 3,  "multi_label": False},
    "arousal":            {"n_classes": 3,  "multi_label": False},
    "threat":             {"n_classes": 3,  "multi_label": False},
    "control":            {"n_classes": 3,  "multi_label": False},
    "player_intent":      {"n_classes": 9,  "multi_label": False},
    "player_knowledge":   {"n_classes": 4,  "multi_label": False},
    "player_credibility": {"n_classes": 3,  "multi_label": False},
    "duty_pressure":      {"n_classes": 3,  "multi_label": False},
    "secrecy_pressure":   {"n_classes": 3,  "multi_label": False},
    "face_pressure":      {"n_classes": 3,  "multi_label": False},
    "value_conflict":     {"n_classes": 3,  "multi_label": False},
    "response_policy":    {"n_classes": 10, "multi_label": False},
    "reveal_decision":    {"n_classes": 4,  "multi_label": False},
    "repair_strategy":    {"n_classes": 5,  "multi_label": False},
}

for dim in STANCE_DIMS:
    HEAD_SPECS[f"{dim}_level"] = {"n_classes": 5, "multi_label": False}
    HEAD_SPECS[f"{dim}_delta"] = {"n_classes": 5, "multi_label": False}


class ClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LatentStatePredictor(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, head_specs: dict = HEAD_SPECS, pooling: str = "last"):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.pooling = pooling
        self.heads = nn.ModuleDict({
            name: ClassificationHead(hidden_size, spec["n_classes"])
            for name, spec in head_specs.items()
        })
        self.head_specs = head_specs

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1]
        pooled = self._pool(last_hidden, attention_mask)
        pooled = pooled.to(torch.float32)

        logits = {name: head(pooled) for name, head in self.heads.items()}
        
        return {
            "logits": logits, 
            "pooled": pooled,
            "lm_loss": outputs.loss if labels is not None else None,
            "backbone_outputs": outputs
        }

    def _pool(self, last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
            sum_hidden = (last_hidden * mask_expanded).sum(dim=1)
            return sum_hidden / mask_expanded.sum(dim=1).clamp(min=1)
        elif self.pooling == "attention":
            # Learnable attention pooling (simple single-vector attention)
            if not hasattr(self, "attention_vector"):
                self.attention_vector = nn.Parameter(torch.randn(self.hidden_size))
            scores = torch.matmul(last_hidden, self.attention_vector)  # (batch, seq)
            scores = scores.masked_fill(~attention_mask.bool(), float('-inf'))
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (batch, seq, 1)
            return (last_hidden * weights).sum(dim=1)
        else:  # "last"
            seq_lengths = attention_mask.sum(dim=1) - 1
            return last_hidden[torch.arange(last_hidden.size(0), device=last_hidden.device), seq_lengths]


_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def load_backbone(
    model_name: str,
    quantization: str = "4bit",
    lora_config: Optional[dict] = None,
    torch_dtype: str = "bfloat16",
) -> tuple:
    bnb_config = None
    if quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif quantization == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    _dtype = _DTYPE_MAP.get(torch_dtype, torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=_dtype,
    )

    if lora_config is not None:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_config.get("r", 16),
            lora_alpha=lora_config.get("alpha", 32),
            lora_dropout=lora_config.get("dropout", 0.05),
            target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
            bias=lora_config.get("bias", "none"),
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Gemma 4 stores hidden_size under text_config
    if hasattr(model.config, 'text_config') and hasattr(model.config.text_config, 'hidden_size'):
        hidden_size = model.config.text_config.hidden_size
    else:
        hidden_size = model.config.hidden_size
    return model, tokenizer, hidden_size


def build_latent_predictor(
    model_name: str,
    quantization: str = "4bit",
    lora_config: Optional[dict] = None,
    torch_dtype: str = "bfloat16",
    pooling: str = "last",
) -> tuple:
    backbone, tokenizer, hidden_size = load_backbone(model_name, quantization, lora_config, torch_dtype)
    predictor = LatentStatePredictor(backbone, hidden_size, pooling=pooling)
    _move_heads_to_backbone_device(predictor)
    return predictor, tokenizer


def _move_heads_to_backbone_device(predictor: "LatentStatePredictor") -> None:
    """Move classification heads to the same device as the backbone."""
    try:
        device = next(predictor.backbone.parameters()).device
    except StopIteration:
        return
    predictor.heads.to(device)


def save_predictor(predictor: LatentStatePredictor, save_dir: str) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    predictor.backbone.save_pretrained(str(save_dir / "backbone"))
    head_state = {
        k: {pk: pv.detach().cpu() for pk, pv in v.state_dict().items()}
        for k, v in predictor.heads.items()
    }
    torch.save(head_state, save_dir / "heads.pt")
    print(f"Saved predictor to {save_dir}")


def load_predictor(
    save_dir: str,
    model_name: str,
    quantization: str = "4bit",
    torch_dtype: str = "bfloat16",
) -> tuple:
    save_dir = Path(save_dir)
    backbone_dir = save_dir / "backbone"

    # Always load base model and tokenizer from model_name
    backbone, tokenizer, hidden_size = load_backbone(
        model_name,
        quantization,
        lora_config=None,
        torch_dtype=torch_dtype,
    )
    
    if backbone_dir.exists():
        backbone = PeftModel.from_pretrained(backbone, str(backbone_dir))

    predictor = LatentStatePredictor(backbone, hidden_size)
    heads_path = save_dir / "heads.pt"
    if heads_path.exists():
        head_state = torch.load(heads_path, map_location="cpu")
        for name, state in head_state.items():
            if name in predictor.heads:
                predictor.heads[name].load_state_dict(state)
    _move_heads_to_backbone_device(predictor)
    return predictor, tokenizer
