from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class PersonalityTrainConfig:
    model_name: str = "distilbert-base-uncased"
    train_path: str = "data/personality/train.csv"
    val_path: str = "data/personality/val.csv"
    text_column: str = "text"
    target_columns: List[str] = field(
        default_factory=lambda: ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    )
    max_length: int = 256
    batch_size: int = 16
    lr: float = 2e-5
    epochs: int = 3
    output_dir: str = "artifacts/personality_encoder"


@dataclass
class AffectTrainConfig:
    model_name: str = "distilbert-base-uncased"
    train_path: str = "data/affect/train.csv"
    val_path: str = "data/affect/val.csv"
    text_column: str = "text"
    target_columns: List[str] = field(default_factory=lambda: ["valence", "arousal", "dominance"])
    max_length: int = 256
    batch_size: int = 16
    lr: float = 2e-5
    epochs: int = 3
    output_dir: str = "artifacts/affect_encoder"


@dataclass
class DialogueTrainConfig:
    base_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    train_path: str = "data/dialogue/train.jsonl"
    val_path: str = "data/dialogue/val.jsonl"
    personality_cache_path: str = "artifacts/personality_cache.jsonl"
    personality_encoder_path: str = "artifacts/personality_encoder"
    affect_encoder_path: str = "artifacts/affect_encoder"
    sentence_transformer_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    prefix_length: int = 8
    max_source_length: int = 768
    max_target_length: int = 192
    batch_size: int = 2
    grad_accum_steps: int = 8
    lr: float = 2e-4
    epochs: int = 1
    output_dir: str = "artifacts/dialogue_model"
    memory_top_k: int = 3
    randomize_vad: bool = False


@dataclass
class InferenceConfig:
    dialogue_model_dir: str = "artifacts/dialogue_model"
    personality_encoder_path: str = "artifacts/personality_encoder"
    affect_encoder_path: str = "artifacts/affect_encoder"
    personality_cache_path: str = "artifacts/personality_cache.jsonl"
    sentence_transformer_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_new_tokens: int = 120
    temperature: float = 0.8
    top_p: float = 0.95
    memory_top_k: int = 3
    device: Optional[str] = None


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
