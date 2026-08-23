"""Generative structured prediction of the 29-field latent state.

The head-based predictor pools the backbone into one vector and reads 29 linear
probes off it. Every field is predicted independently from the same vector, so
nothing lets `reveal_decision` condition on `secrecy_pressure`, and the pretrained
model's actual competence — next-token prediction over structured text — goes
unused. Every variation tried on that design (pooling, context length, sampler,
head depth, head LR, label schema, epochs, counterfactual filtering) has landed
within a few points of the same ceiling.

This serialises the state as text and trains the backbone to emit it, so the
fields are predicted autoregressively and condition on each other.

Scoring goes through the same `compute_latent_metrics` as the head model, so the
numbers are directly comparable. A generated block that cannot be parsed is not
silently dropped — it is scored as wrong, and the parse rate is reported.

    python -m src.training.latent_sft        # self-check
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

from src.training.dataset import LABEL_MAPS, LABEL_TO_IDX

BEGIN, END = "<state>", "</state>"
# Deterministic order; dict insertion order is stable and matches the head specs.
FIELD_ORDER: list[str] = list(LABEL_MAPS.keys())


def serialize(labels: dict) -> str:
    """Render a flat label dict as the target text."""
    lines = [BEGIN]
    for f in FIELD_ORDER:
        v = labels.get(f, "")
        if f == "dialogue_act":
            v = ",".join(v) if isinstance(v, list) else str(v)
        elif isinstance(v, list):
            v = v[0] if v else ""
        lines.append(f"{f}={v}")
    lines.append(END)
    return "\n".join(lines)


def parse(text: str) -> dict[str, object]:
    """Inverse of `serialize`, tolerant of truncation and stray prose.

    Returns only the fields it could actually read, so a caller can tell the
    difference between "the model said something wrong" and "the model did not
    say anything for this field".
    """
    out: dict[str, object] = {}
    body = text.split(BEGIN, 1)[-1].split(END, 1)[0]
    for line in body.splitlines():
        line = line.strip()
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip()
        if k not in LABEL_TO_IDX:
            continue
        out[k] = [p for p in (s.strip() for s in v.split(",")) if p] if k == "dialogue_act" else v
    return out


def to_indices(parsed: dict) -> dict[str, int]:
    """Map parsed strings to class indices.

    A field that is missing or carries a label outside the schema is assigned
    `len(classes)` — an index that exists in no gold vector, so it counts as an
    error rather than vanishing from the metric.
    """
    idx: dict[str, int] = {}
    for f in FIELD_ORDER:
        if f == "dialogue_act":
            continue
        classes = LABEL_MAPS[f]
        v = parsed.get(f)
        if isinstance(v, list):
            v = v[0] if v else None
        idx[f] = LABEL_TO_IDX[f].get(str(v), len(classes))
    return idx


class LatentSFTDataset(Dataset):
    """(context -> serialized state) pairs with the prompt masked out of the loss."""

    def __init__(self, jsonl_path: str, tokenizer, max_seq_len: int = 1024,
                 exclude_counterfactual: bool = False) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.records: list[dict] = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if exclude_counterfactual and r.get("counterfactual"):
                    continue
                self.records.append(r)

    def __len__(self) -> int:
        return len(self.records)

    def prompt_for(self, record: dict) -> str:
        return record["context"] + f"\n\nPredict the latent state.\n{BEGIN}\n"

    def __getitem__(self, i: int) -> dict:
        r = self.records[i]
        prompt = self.prompt_for(r)
        # serialize() re-emits BEGIN, which the prompt already ends with; drop it
        # so the model continues the block instead of opening a second one.
        target = serialize(r["labels"]).split("\n", 1)[1] + self.tokenizer.eos_token

        p_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        t_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]

        # Truncate the PROMPT from the left, never the target: losing the end of
        # the state block would train the model to stop early.
        room = self.max_seq_len - len(t_ids)
        if room < 1:
            t_ids = t_ids[: self.max_seq_len - 1]
            room = 1
        p_ids = p_ids[-room:]

        ids = p_ids + t_ids
        labels = [-100] * len(p_ids) + t_ids[:]
        pad = self.max_seq_len - len(ids)
        attn = [1] * len(ids) + [0] * pad
        ids = ids + [self.tokenizer.pad_token_id] * pad
        labels = labels + [-100] * pad

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate(batch: list[dict]) -> dict:
    return {k: torch.stack([b[k] for b in batch]) for k in batch[0]}


def _self_check() -> None:
    labels = {f: (LABEL_MAPS[f][0] if f != "dialogue_act" else ["ask", "probe"])
              for f in FIELD_ORDER}
    s = serialize(labels)
    back = parse(s)
    assert back["dialogue_act"] == ["ask", "probe"], back["dialogue_act"]
    for f in FIELD_ORDER:
        if f == "dialogue_act":
            continue
        assert back[f] == LABEL_MAPS[f][0], (f, back[f])
    assert len(parse(s)) == len(FIELD_ORDER), "round trip lost a field"

    # truncated generation must degrade, not explode
    cut = parse(s[: len(s) // 2])
    assert len(cut) < len(FIELD_ORDER) and all(k in LABEL_TO_IDX for k in cut)

    # unknown / missing labels become an index no gold vector can contain
    idx = to_indices({"tone": "not-a-real-tone"})
    assert idx["tone"] == len(LABEL_MAPS["tone"]), idx["tone"]
    assert idx["valence"] == len(LABEL_MAPS["valence"]), "missing field must score as wrong"

    good = to_indices(back)
    assert good["tone"] == 0 and good["response_policy"] == 0
    print(f"ok: {len(FIELD_ORDER)} fields round-trip; "
          f"target is {len(s.splitlines())} lines / {len(s)} chars")


if __name__ == "__main__":
    _self_check()
