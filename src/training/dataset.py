import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset


DIALOGUE_ACT_LABELS = ["ask", "accuse", "threaten", "flatter", "apologize", "negotiate", "joke", "confess", "probe", "command"]
TONE_LABELS         = ["warm", "neutral", "confrontational", "sarcastic", "fearful", "evasive"]
RISK_LABELS         = ["none", "secret-risk", "face-risk", "status-risk", "conflict-risk"]
VALENCE_LABELS      = ["negative", "neutral", "positive"]
THREE_LEVEL_LABELS  = ["low", "medium", "high"]
PLAYER_INTENT_LABELS = ["seek-info", "trap", "bond", "manipulate", "test", "persuade", "intimidate", "probe", "negotiate"]
PLAYER_KNOWLEDGE_LABELS = ["unaware", "partial", "informed", "knows-secret"]
STANCE_LEVEL_LABELS = ["VL", "L", "N", "H", "VH"]
STANCE_DELTA_LABELS = ["--", "-", "0", "+", "++"]
VALUE_CONFLICT_LABELS = ["none", "mild", "strong", ""]
RESPONSE_POLICY_LABELS = ["answer", "partial", "withhold", "deflect", "challenge", "soothe", "test", "threaten", "negotiate", "clarify"]
REVEAL_LABELS       = ["none", "hint", "partial", "full"]
REPAIR_LABELS       = ["none", "soften", "apologize", "clarify", "redirect"]
STANCE_DIMS         = ["affection", "respect", "dominance", "familiarity", "trust", "obligation"]

LABEL_MAPS: dict[str, list] = {
    "dialogue_act":       DIALOGUE_ACT_LABELS,
    "tone":               TONE_LABELS,
    "risk_type":          RISK_LABELS,
    "valence":            VALENCE_LABELS,
    "arousal":            THREE_LEVEL_LABELS,
    "threat":             THREE_LEVEL_LABELS,
    "control":            THREE_LEVEL_LABELS,
    "player_intent":      PLAYER_INTENT_LABELS,
    "player_knowledge":   PLAYER_KNOWLEDGE_LABELS,
    "player_credibility": THREE_LEVEL_LABELS,
    "duty_pressure":      THREE_LEVEL_LABELS,
    "secrecy_pressure":   THREE_LEVEL_LABELS,
    "face_pressure":      THREE_LEVEL_LABELS,
    "value_conflict":     ["none", "mild", "strong"],
    "response_policy":    RESPONSE_POLICY_LABELS,
    "reveal_decision":    REVEAL_LABELS,
    "repair_strategy":    REPAIR_LABELS,
}

for _dim in STANCE_DIMS:
    LABEL_MAPS[f"{_dim}_level"] = STANCE_LEVEL_LABELS
    LABEL_MAPS[f"{_dim}_delta"] = STANCE_DELTA_LABELS

LABEL_TO_IDX: dict[str, dict] = {
    field: {v: i for i, v in enumerate(values)}
    for field, values in LABEL_MAPS.items()
}


def _encode_labels(labels: dict) -> dict[str, torch.Tensor]:
    encoded: dict[str, torch.Tensor] = {}
    for field, idx_map in LABEL_TO_IDX.items():
        val = labels.get(field, None)
        if val is None:
            encoded[field] = torch.tensor(-1, dtype=torch.long)
            continue

        if field == "dialogue_act":
            if isinstance(val, str):
                val = [val]
            multi_hot = torch.zeros(len(DIALOGUE_ACT_LABELS), dtype=torch.float)
            for v in val:
                if v in idx_map:
                    multi_hot[idx_map[v]] = 1.0
            encoded[field] = multi_hot
        else:
            if isinstance(val, list):
                val = val[0] if val else ""
            idx = idx_map.get(str(val), -1)
            encoded[field] = torch.tensor(idx, dtype=torch.long)

    return encoded


class HeadSupervisionDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        max_seq_len: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.records: list[dict] = []
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        context = record["context"]

        encoding = self.tokenizer(
            context,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        label_tensors = _encode_labels(record.get("labels", {}))

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            **{f"label_{k}": v for k, v in label_tensors.items()},
            "episode_id": record.get("episode_id", ""),
            "scenario_type": record.get("scenario_type", ""),
        }


class SFTDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        max_seq_len: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.records: list[dict] = []
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        full_text = record["input"] + "\n" + record["target"]

        encoding = self.tokenizer(
            full_text,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_len = len(self.tokenizer(record["input"]).input_ids)
        labels = encoding["input_ids"].clone().squeeze(0)
        labels[:min(input_len, labels.size(0))] = -100

        # Calculate prompt length for evaluation (input + separator)
        # We want generation to start after "input" + "\n"
        prompt_text = record["input"] + "\n"
        prompt_ids = self.tokenizer(prompt_text, truncation=True, max_length=self.max_seq_len)["input_ids"]
        prompt_len = len(prompt_ids)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels,
            "prompt_len": torch.tensor(prompt_len, dtype=torch.long),
            "episode_id": record.get("episode_id", ""),
            "scenario_type": record.get("scenario_type", ""),
        }


def collate_head_batch(batch: list[dict]) -> dict:
    keys = [k for k in batch[0] if k not in ("episode_id", "scenario_type")]
    result: dict = {}
    for k in keys:
        tensors = [b[k] for b in batch]
        try:
            result[k] = torch.stack(tensors)
        except Exception:
            result[k] = tensors
    result["episode_ids"] = [b.get("episode_id", "") for b in batch]
    return result


class JointDataset(Dataset):
    """Merges SFT and HeadSupervision records by index for joint training.

    Both JSONL files are produced from the same validated turns in the same
    order, so zipping by index is safe.
    """

    def __init__(
        self,
        sft_jsonl_path: str,
        heads_jsonl_path: str,
        tokenizer,
        sft_max_seq_len: int = 2048,
        heads_max_seq_len: int = 1024,
    ):
        self.sft_ds   = SFTDataset(sft_jsonl_path,   tokenizer, sft_max_seq_len)
        self.heads_ds = HeadSupervisionDataset(heads_jsonl_path, tokenizer, heads_max_seq_len)
        assert len(self.sft_ds) == len(self.heads_ds), (
            f"SFT ({len(self.sft_ds)}) and heads ({len(self.heads_ds)}) record counts must match"
        )

    def __len__(self) -> int:
        return len(self.sft_ds)

    def __getitem__(self, idx: int) -> dict:
        sft_item   = self.sft_ds[idx]
        heads_item = self.heads_ds[idx]
        
        # Start with SFT item (input_ids, attention_mask, labels, etc.)
        merged = {**sft_item}
        
        # Add heads item with prefixed keys for inputs to avoid collision
        merged["heads_input_ids"] = heads_item["input_ids"]
        merged["heads_attention_mask"] = heads_item["attention_mask"]
        
        # Add all labels from heads item
        for k, v in heads_item.items():
            if k.startswith("label_"):
                merged[k] = v
                
        return merged


def collate_joint_batch(batch: list[dict]) -> dict:
    skip = {"episode_id", "scenario_type", "episode_ids"}
    result: dict = {}
    for k in batch[0]:
        if k in skip:
            continue
        tensors = [b[k] for b in batch]
        try:
            result[k] = torch.stack(tensors)
        except Exception:
            result[k] = tensors
    result["episode_ids"] = [b.get("episode_id", "") for b in batch]
    return result


def collate_sft_batch(batch: list[dict]) -> dict:
    keys = [k for k in batch[0] if k not in ("episode_id", "scenario_type")]
    result: dict = {}
    for k in keys:
        tensors = [b[k] for b in batch]
        try:
            result[k] = torch.stack(tensors)
        except Exception:
            result[k] = tensors
    return result
