"""Record-aware data utilities for the from-scratch dialogue language models.

The legacy training path treats a text file as one continuous token stream.  This
module provides the paper-oriented path: compact role tokens, record boundaries,
NPC-response-only labels, aligned conditioning vectors, and lightweight
deduplication.
"""
from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch.utils.data import Dataset

try:
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
except ImportError as exc:  # pragma: no cover - exercised by the CLI error path
    raise RuntimeError(
        "The JSONL scratch-LM pipeline requires `tokenizers>=0.19`."
    ) from exc


SPECIAL_TOKENS = (
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
    "<persona>",
    "<usr>",
    "<npc>",
    "<eot>",
)


def _iter_dialogue_text(paths: Iterable[Path]) -> Iterable[str]:
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                profile = str(record.get("npc_profile", "")).strip()
                if profile:
                    yield f"<persona> {profile} <eot>"
                for turn in record.get("dialogue_context", []):
                    role = "<npc>" if str(turn.get("speaker", "")).lower() == "npc" else "<usr>"
                    text = str(turn.get("text", "")).strip()
                    if text:
                        yield f"{role} {text} <eot>"
                response = str(record.get("target_response", "")).strip()
                if response:
                    yield f"<npc> {response} <eot>"


def train_dialogue_tokenizer(
    jsonl_paths: Sequence[str | Path],
    output_path: str | Path,
    *,
    vocab_size: int = 8192,
    min_frequency: int = 2,
) -> Path:
    """Train and save a byte-level BPE tokenizer from training JSONL only."""
    paths = [Path(path) for path in jsonl_paths]
    if not paths or any(not path.exists() for path in paths):
        raise FileNotFoundError("Every tokenizer-training JSONL path must exist.")
    if vocab_size < 320:
        raise ValueError("vocab_size must be at least 320 for byte fallback + specials")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    trainer = trainers.BpeTrainer(
        vocab_size=int(vocab_size),
        min_frequency=int(min_frequency),
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".txt",
        dir=output.parent,
        delete=False,
    ) as corpus:
        corpus_path = Path(corpus.name)
        for text in _iter_dialogue_text(paths):
            corpus.write(text.replace("\n", " ") + "\n")
    try:
        tokenizer.train([str(corpus_path)], trainer=trainer)
        tokenizer.save(str(output), pretty=True)
    finally:
        corpus_path.unlink(missing_ok=True)
    return output


class DialogueTokenizer:
    """Small compatibility wrapper around a saved Hugging Face tokenizer."""

    def __init__(self, tokenizer: Tokenizer, path: str | Path | None = None) -> None:
        self.tokenizer = tokenizer
        self.path = str(path) if path is not None else None
        self.name = f"byte_bpe:{Path(path).name}" if path is not None else "byte_bpe"
        self.vocab_size = tokenizer.get_vocab_size()
        vocab = tokenizer.get_vocab()
        missing = [token for token in SPECIAL_TOKENS if token not in vocab]
        if missing:
            raise ValueError(f"Tokenizer is missing required special tokens: {missing}")
        self.special_ids = {token: vocab[token] for token in SPECIAL_TOKENS}
        self.pad_id = self.special_ids["<pad>"]
        self.unk_id = self.special_ids["<unk>"]
        self.bos_id = self.special_ids["<bos>"]
        self.eos_id = self.special_ids["<eos>"]
        self.persona_id = self.special_ids["<persona>"]
        self.user_id = self.special_ids["<usr>"]
        self.npc_id = self.special_ids["<npc>"]
        self.eot_id = self.special_ids["<eot>"]

    @classmethod
    def from_file(cls, path: str | Path) -> "DialogueTokenizer":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        return cls(Tokenizer.from_file(str(path)), path)

    def encode(self, text: str) -> list[int]:
        normalized = unicodedata.normalize("NFC", str(text))
        return self.tokenizer.encode(normalized, add_special_tokens=False).ids

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(
            [int(token_id) for token_id in ids],
            skip_special_tokens=skip_special_tokens,
        )


def _normalized_key(record: dict[str, Any]) -> str:
    parts = [str(record.get("npc_profile", ""))]
    for turn in record.get("dialogue_context", []):
        parts.append(str(turn.get("speaker", "")))
        parts.append(str(turn.get("text", "")))
    parts.append(str(record.get("target_response", "")))
    normalized = " ".join(" ".join(parts).lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _float_vector(value: Any, expected: int) -> tuple[list[float], list[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != expected:
        return [0.0] * expected, [0.0] * expected
    try:
        return [float(x) for x in value], [1.0] * expected
    except (TypeError, ValueError):
        return [0.0] * expected, [0.0] * expected


def condition_from_record(record: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Read aligned OCEAN+VAD values, retaining a mask for missing dimensions."""
    conditioning = record.get("conditioning", {})
    ocean_raw = record.get("ocean", conditioning.get("ocean"))
    vad_raw = record.get("vad", conditioning.get("vad"))
    ocean, ocean_mask = _float_vector(ocean_raw, 5)
    vad, vad_mask = _float_vector(vad_raw, 3)

    # Generated records often carry categorical affect but no numeric OCEAN.
    if not any(vad_mask):
        metadata = record.get("metadata", {})
        valence_map = {"negative": 0.1, "neutral": 0.5, "positive": 0.9}
        arousal_map = {"low": 0.2, "medium": 0.5, "high": 0.8}
        valence = valence_map.get(str(metadata.get("valence", "")).lower())
        arousal = arousal_map.get(str(metadata.get("arousal", "")).lower())
        if valence is not None:
            vad[0], vad_mask[0] = valence, 1.0
        if arousal is not None:
            vad[1], vad_mask[1] = arousal, 1.0

    return (
        torch.tensor(ocean + vad, dtype=torch.float32),
        torch.tensor(ocean_mask + vad_mask, dtype=torch.float32),
    )


class DialogueRecordDataset(Dataset):
    """Fixed-length, response-masked dialogue examples from JSONL records."""

    def __init__(
        self,
        path: str | Path,
        tokenizer: DialogueTokenizer,
        seq_len: int,
        *,
        profile_max_tokens: int = 48,
        max_turns: int = 4,
        deduplicate: bool = True,
    ) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.seq_len = int(seq_len)
        self.profile_max_tokens = int(profile_max_tokens)
        self.max_turns = int(max_turns)
        if self.seq_len < 8:
            raise ValueError("seq_len must be at least 8")

        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON at {self.path}:{line_number}") from exc
                if not str(record.get("target_response", "")).strip():
                    continue
                key = _normalized_key(record)
                if deduplicate and key in seen:
                    continue
                seen.add(key)
                records.append(record)
        if not records:
            raise ValueError(f"No usable dialogue records found in {self.path}")
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def _prefix_and_response(
        self, record: dict[str, Any]
    ) -> tuple[list[int], list[int]]:
        tok = self.tokenizer
        profile_ids = tok.encode(record.get("npc_profile", ""))[: self.profile_max_tokens]
        head = [tok.bos_id, tok.persona_id, *profile_ids, tok.eot_id]

        history: list[int] = []
        turns = list(record.get("dialogue_context", []))[-self.max_turns :]
        for turn in turns:
            speaker = str(turn.get("speaker", "")).lower()
            role_id = tok.npc_id if speaker == "npc" else tok.user_id
            text_ids = tok.encode(str(turn.get("text", "")).strip())
            history.extend([role_id, *text_ids, tok.eot_id])

        response = tok.encode(str(record["target_response"]).strip())
        max_response = max(1, self.seq_len // 2)
        response = response[:max_response] + [tok.eot_id, tok.eos_id]

        prefix_budget = self.seq_len + 1 - len(response)
        min_prefix = [tok.bos_id, tok.npc_id]
        if prefix_budget < len(min_prefix):
            response = response[: max(1, self.seq_len - len(min_prefix))] + [tok.eos_id]
            prefix_budget = self.seq_len + 1 - len(response)

        suffix = [tok.npc_id]
        available = max(1, prefix_budget - len(suffix))
        # Preserve the most recent player/context tokens before spending the
        # whole budget on a repeated profile.
        history_budget = min(len(history), max(0, available // 2))
        head_budget = max(1, available - history_budget)
        if head_budget >= 3:
            compact_head = [
                tok.bos_id,
                tok.persona_id,
                *profile_ids[: head_budget - 3],
                tok.eot_id,
            ]
        else:
            compact_head = [tok.bos_id, tok.persona_id][:head_budget]
        prefix = (
            compact_head
            + (history[-history_budget:] if history_budget else [])
            + suffix
        )
        return prefix, response

    def encode_record(self, record: dict[str, Any]) -> dict[str, Any]:
        prefix, response = self._prefix_and_response(record)
        full = (prefix + response)[: self.seq_len + 1]
        input_ids = full[:-1]
        labels = full[1:]

        # y[prefix_len - 1] predicts the first response token.
        mask_until = max(0, len(prefix) - 1)
        labels[:mask_until] = [-100] * mask_until

        pad_needed = self.seq_len - len(input_ids)
        if pad_needed > 0:
            input_ids += [self.tokenizer.pad_id] * pad_needed
            labels += [-100] * pad_needed

        condition, condition_mask = condition_from_record(record)
        retained_response = [
            token_id
            for token_id in response
            if token_id not in (self.tokenizer.eot_id, self.tokenizer.eos_id)
        ]
        target_bytes = len(
            self.tokenizer.decode(retained_response).encode("utf-8")
        )
        source = str(record.get("metadata", {}).get("source", "unknown"))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "condition": condition,
            "condition_mask": condition_mask,
            "target_bytes": torch.tensor(target_bytes, dtype=torch.long),
            "source": source,
            "prompt_ids": prefix,
            "reference_ids": response,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.encode_record(self.records[index])
        # Variable-length generation fields are accessed directly by evaluators.
        item.pop("prompt_ids")
        item.pop("reference_ids")
        return item

    @property
    def target_token_count(self) -> int:
        return sum(
            int((self[index]["labels"] != -100).sum().item())
            for index in range(len(self))
        )

    @property
    def target_byte_count(self) -> int:
        return sum(
            int(self.encode_record(record)["target_bytes"].item())
            for record in self.records
        )

    @property
    def condition_coverage(self) -> dict[str, float]:
        masks = [condition_from_record(record)[1] for record in self.records]
        stacked = torch.stack(masks)
        names = [
            "openness",
            "conscientiousness",
            "extraversion",
            "agreeableness",
            "neuroticism",
            "valence",
            "arousal",
            "dominance",
        ]
        coverage = {
            name: float(stacked[:, index].mean().item())
            for index, name in enumerate(names)
        }
        coverage["all_dimensions"] = float(stacked.mean().item())
        coverage["fully_conditioned_records"] = float(
            (stacked.sum(dim=1) == stacked.size(1)).float().mean().item()
        )
        return coverage

    def easy_indices(
        self,
        *,
        max_target_tokens: int = 48,
        max_turns: int = 2,
    ) -> list[int]:
        """Indices for the optional short, high-confidence warm-up curriculum."""
        indices: list[int] = []
        for index, record in enumerate(self.records):
            response_tokens = len(
                self.tokenizer.encode(str(record.get("target_response", "")))
            )
            if response_tokens <= max_target_tokens and len(
                record.get("dialogue_context", [])
            ) <= max_turns:
                indices.append(index)
        return indices
