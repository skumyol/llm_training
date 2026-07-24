#!/usr/bin/env python3
"""
prepare_dialogue_data.py
=========================
Converts raw downloaded datasets into the two formats needed for A/B training:

  data/dialogue/train.jsonl  ← DialogueJsonlDataset (ConditionalDialogueModel)
  data/dialogue/val.jsonl
  data/dialogue/train.txt    ← plain text (small LM architectures)
  data/dialogue/val.txt

  data/personality/train.csv ← RegressionTextDataset (personality encoder)
  data/personality/val.csv
  data/affect/train.csv      ← RegressionTextDataset (affect encoder)
  data/affect/val.csv
  data/npc_profiles.csv      ← NPC profiles for personality cache

Source datasets (auto-downloaded via DataDownloader):
  personachat        – persona-conditioned chat  → primary NPC dialogue source
  crd3               – Critical Role D&D         → fantasy RPG dialogue
  empathetic_dialogues – emotion-grounded        → affect encoder training data
  dailydialog        – general clean multi-turn  → language model supplement

Usage:
  # 1. Download raw data first:
  python -m src.data.datasets --datasets personachat crd3 empathetic_dialogues dailydialog

  # 2. Convert to training-ready formats:
  python -m src.data.prepare_dialogue_data
  python -m src.data.prepare_dialogue_data --raw-dir data/raw --out-dir data --val-frac 0.05
  python -m src.data.prepare_dialogue_data --sources personachat crd3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Optional HuggingFace datasets ────────────────────────────────────────────
try:
    from datasets import load_from_disk, load_dataset
    HF_OK = True
except ImportError:
    HF_OK = False


# ── JSONL format ─────────────────────────────────────────────────────────────
# Each record matches DialogueJsonlDataset schema:
# {
#   "npc_id":           str,
#   "npc_profile":      str,
#   "dialogue_context": [{"speaker": "player"|"npc", "text": str}, ...],
#   "target_response":  str,
#   "metadata":         {source, split, ...}
# }


def _npc_id(persona_text: str) -> str:
    return "npc_" + hashlib.md5(persona_text.encode()).hexdigest()[:12]


def _persona_to_profile(sentences: List[str]) -> str:
    """Convert PersonaChat persona list to a natural-language NPC profile paragraph."""
    if not sentences:
        return "An unnamed character."
    caps = [s.strip().capitalize() for s in sentences if s.strip()]
    return " ".join(caps)


# =============================================================================
# Source converters
# =============================================================================

def convert_personachat(raw_dir: Path) -> Tuple[List[Dict], List[Dict], str]:
    """
    PersonaChat → dialogue JSONL records.

    Returns: (train_records, val_records, npc_profiles_text)
    Also produces rows for personality encoder: each persona → OCEAN proxy label.
    """
    src = raw_dir / "personachat"
    if not src.exists():
        print(f"  [SKIP] personachat not found at {src}")
        return [], [], ""

    if not HF_OK:
        print("  [SKIP] huggingface datasets not installed")
        return [], [], ""

    print("  Loading personachat …")
    try:
        ds = load_from_disk(str(src))
    except Exception:
        try:
            ds = load_dataset("bavard/personachat_truecased", cache_dir=str(src))
        except Exception as e:
            print(f"  [ERROR] personachat load failed: {e}")
            return [], [], ""

    def process_split(split_name: str) -> List[Dict]:
        records = []
        split   = ds[split_name] if split_name in ds else []
        for example in split:
            persona   = example.get("personality", [])
            if not persona:
                continue
            profile   = _persona_to_profile(persona)
            npc_id    = _npc_id(profile)
            history   = example.get("history", [])
            candidates = example.get("candidates", [])
            if not candidates:
                continue
            target    = candidates[-1]          # last candidate = ground truth
            ctx       = []
            for i, utt in enumerate(history):
                speaker = "player" if i % 2 == 0 else "npc"
                ctx.append({"speaker": speaker, "text": utt})
            records.append({
                "npc_id":           npc_id,
                "npc_profile":      profile,
                "dialogue_context": ctx,
                "target_response":  target,
                "metadata":         {"source": "personachat", "split": split_name},
            })
        return records

    train = process_split("train")
    val   = process_split("validation") or process_split("valid") or process_split("test")
    print(f"  personachat → {len(train):,} train  {len(val):,} val")
    return train, val, ""


def convert_crd3(raw_dir: Path) -> Tuple[List[Dict], List[Dict]]:
    """Critical Role D&D dataset → dialogue JSONL."""
    src = raw_dir / "crd3"
    if not src.exists():
        print(f"  [SKIP] crd3 not found at {src}")
        return [], []

    if not HF_OK:
        print("  [SKIP] huggingface datasets not installed")
        return [], []

    print("  Loading crd3 …")
    try:
        ds = load_from_disk(str(src))
    except Exception:
        try:
            ds = load_dataset("microsoft/crd3", cache_dir=str(src))
        except Exception as e:
            print(f"  [ERROR] crd3 load failed: {e}")
            return [], []

    FANTASY_NPC_PROFILES = [
        "A seasoned adventurer who has seen many dangers and speaks with measured caution.",
        "A mysterious wizard who guards ancient knowledge and reveals little about their past.",
        "A gruff fighter who values loyalty above all else and distrusts magic.",
        "A cunning rogue who operates in the shadows and trades in information.",
        "A pious cleric devoted to healing and the restoration of balance.",
    ]

    def process(split_name: str) -> List[Dict]:
        records = []
        split   = ds[split_name] if split_name in ds else []
        for example in split:
            chunk    = example.get("chunk", []) or example.get("turns", [])
            if len(chunk) < 2:
                continue
            profile  = random.choice(FANTASY_NPC_PROFILES)
            npc_id   = _npc_id(profile + str(random.random()))
            for i in range(len(chunk) - 1):
                ctx    = [{"speaker": "player" if j % 2 == 0 else "npc",
                           "text": str(chunk[j])} for j in range(i + 1)]
                target = str(chunk[i + 1])
                records.append({
                    "npc_id":           npc_id,
                    "npc_profile":      profile,
                    "dialogue_context": ctx,
                    "target_response":  target,
                    "metadata":         {"source": "crd3", "split": split_name},
                })
        return records

    train = process("train")
    val   = process("validation") or process("test")
    print(f"  crd3 → {len(train):,} train  {len(val):,} val")
    return train, val


def convert_empathetic_dialogues(raw_dir: Path) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    EmpathyDialogues → affect encoder CSV rows + dialogue JSONL.

    Returns: (train_dialogue, val_dialogue, affect_rows)
    Affect rows: {"text": ..., "valence": ..., "arousal": ..., "dominance": ...}
    Note: affect labels are approximated from emotion categories.
    """
    EMOTION_VAD: Dict[str, Tuple[float, float, float]] = {
        "joy":        (0.90, 0.70, 0.65),
        "sadness":    (0.10, 0.30, 0.20),
        "anger":      (0.15, 0.85, 0.75),
        "fear":       (0.10, 0.80, 0.10),
        "surprise":   (0.65, 0.80, 0.50),
        "disgust":    (0.10, 0.60, 0.55),
        "anticipation": (0.70, 0.60, 0.60),
        "trust":      (0.80, 0.40, 0.55),
        "neutral":    (0.50, 0.50, 0.50),
        "caring":     (0.80, 0.45, 0.55),
        "excited":    (0.90, 0.85, 0.70),
        "terrified":  (0.05, 0.90, 0.05),
        "grateful":   (0.85, 0.50, 0.55),
    }

    src = raw_dir / "empathetic_dialogues"
    if not src.exists():
        print(f"  [SKIP] empathetic_dialogues not found at {src}")
        return [], [], []

    if not HF_OK:
        print("  [SKIP] huggingface datasets not installed")
        return [], [], []

    print("  Loading empathetic_dialogues …")
    try:
        ds = load_from_disk(str(src))
    except Exception:
        try:
            ds = load_dataset("facebook/empathetic_dialogues", cache_dir=str(src))
        except Exception as e:
            print(f"  [ERROR] empathetic_dialogues load failed: {e}")
            return [], [], []

    EMPATHETIC_PROFILE = (
        "A compassionate listener who responds with emotional intelligence, "
        "acknowledging feelings before offering perspective or advice."
    )

    def process(split_name: str) -> Tuple[List[Dict], List[Dict]]:
        dialogue_rows: List[Dict] = []
        affect_rows:   List[Dict] = []
        split = ds[split_name] if split_name in ds else []
        for ex in split:
            emotion = (ex.get("context") or ex.get("emotion") or "neutral").lower()
            vad     = EMOTION_VAD.get(emotion, EMOTION_VAD["neutral"])
            utterances = ex.get("utterances", [])
            if not utterances:
                continue
            ctx    = []
            for i, utt in enumerate(utterances[:-1]):
                t = (utt.get("utterance") or utt.get("text") or str(utt)).strip()
                affect_rows.append({"text": t, "valence": vad[0], "arousal": vad[1], "dominance": vad[2]})
                ctx.append({"speaker": "player" if i % 2 == 0 else "npc", "text": t})
            last = utterances[-1]
            target = (last.get("utterance") or last.get("text") or str(last)).strip()
            npc_id = _npc_id(EMPATHETIC_PROFILE)
            dialogue_rows.append({
                "npc_id":           npc_id,
                "npc_profile":      EMPATHETIC_PROFILE,
                "dialogue_context": ctx,
                "target_response":  target,
                "metadata":         {"source": "empathetic_dialogues", "emotion": emotion, "split": split_name},
            })
        return dialogue_rows, affect_rows

    train_d, train_a = process("train")
    val_d,   val_a   = process("validation") or ([], [])
    if not val_d:
        val_d, val_a = process("test")
    print(f"  empathetic_dialogues → {len(train_d):,} dialogue  {len(train_a):,} affect rows")
    return train_d, val_d, train_a + val_a


DAILYDIALOG_PROFILE = (
    "A conversational NPC who responds clearly, naturally, and concisely."
)


def records_from_dailydialog_example(
    example: Dict[str, Any],
    *,
    split_name: str,
    dialogue_id: str,
) -> List[Dict]:
    """Expand one two-party DailyDialog conversation into response records."""
    raw_turns = example.get("dialog") or example.get("utterances") or []
    turns: List[str] = []
    for raw_turn in raw_turns:
        if isinstance(raw_turn, dict):
            text = raw_turn.get("text") or raw_turn.get("utterance") or ""
        else:
            text = raw_turn
        text = str(text).strip()
        if text:
            turns.append(text)
    records: List[Dict] = []
    if len(turns) < 2:
        return records
    npc_id = _npc_id(f"{DAILYDIALOG_PROFILE}:{dialogue_id}")
    for target_index in range(1, len(turns)):
        context = []
        for context_index in range(target_index):
            distance_from_target = target_index - 1 - context_index
            speaker = "player" if distance_from_target % 2 == 0 else "npc"
            context.append({"speaker": speaker, "text": turns[context_index]})
        records.append(
            {
                "npc_id": npc_id,
                "npc_profile": DAILYDIALOG_PROFILE,
                "dialogue_context": context,
                "target_response": turns[target_index],
                "metadata": {
                    "source": "dailydialog",
                    "split": split_name,
                    "dialogue_id": dialogue_id,
                    "turn_idx": target_index,
                },
            }
        )
    return records


def convert_dailydialog(raw_dir: Path) -> Tuple[List[Dict], List[Dict]]:
    """DailyDialog → compact two-speaker response records."""
    src = raw_dir / "dailydialog"
    if not src.exists():
        print(f"  [SKIP] dailydialog not found at {src}")
        return [], []
    if not HF_OK:
        print("  [SKIP] huggingface datasets not installed")
        return [], []
    print("  Loading dailydialog …")
    try:
        ds = load_from_disk(str(src))
    except Exception:
        try:
            ds = load_dataset("ConvLab/dailydialog", cache_dir=str(src))
        except Exception as exc:
            print(f"  [ERROR] dailydialog load failed: {exc}")
            return [], []

    def process(split_name: str) -> List[Dict]:
        rows: List[Dict] = []
        split = ds[split_name] if split_name in ds else []
        for index, example in enumerate(split):
            rows.extend(
                records_from_dailydialog_example(
                    example,
                    split_name=split_name,
                    dialogue_id=f"{split_name}_{index}",
                )
            )
        return rows

    train = process("train")
    val = process("validation") or process("test")
    print(f"  dailydialog → {len(train):,} train  {len(val):,} val")
    return train, val


# =============================================================================
# Writers
# =============================================================================

def write_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records):,} records → {path}")


def write_plain_text(path: Path, records: List[Dict]) -> None:
    """Convert JSONL records to plain text for small LM training."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for r in records:
        lines.append(f"PROFILE: {r['npc_profile']}")
        for turn in r.get("dialogue_context", []):
            lines.append(f"{turn['speaker'].upper()}: {turn['text']}")
        lines.append(f"NPC: {r['target_response']}")
        lines.append("")       # blank line between exchanges
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Wrote {len(lines):,} lines → {path}")


def write_csv(path: Path, rows: List[Dict], columns: List[str]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})
    print(f"  Wrote {len(rows):,} rows → {path}")


def write_npc_profiles(path: Path, records: List[Dict]) -> None:
    import csv
    seen    = {}
    for r in records:
        seen[r["npc_id"]] = r["npc_profile"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["npc_id", "profile_text"])
        w.writeheader()
        for nid, profile in seen.items():
            w.writerow({"npc_id": nid, "profile_text": profile})
    print(f"  Wrote {len(seen):,} NPC profiles → {path}")


# =============================================================================
# Orchestration
# =============================================================================

def run(raw_dir: Path, out_dir: Path, val_frac: float, sources: List[str], seed: int) -> None:
    random.seed(seed)

    train_records: List[Dict] = []
    val_records:   List[Dict] = []
    all_affect:    List[Dict] = []

    if "personachat" in sources:
        tr, va, _ = convert_personachat(raw_dir)
        train_records += tr; val_records += va

    if "crd3" in sources:
        tr, va = convert_crd3(raw_dir)
        train_records += tr; val_records += va

    if "empathetic_dialogues" in sources:
        tr, va, affect_rows = convert_empathetic_dialogues(raw_dir)
        train_records += tr; val_records += va
        all_affect    += affect_rows

    if "dailydialog" in sources:
        tr, va = convert_dailydialog(raw_dir)
        train_records += tr; val_records += va

    if not train_records:
        print("\n[ERROR] No records produced. Run data download first:")
        print("  python -m src.data.datasets --datasets personachat crd3 empathetic_dialogues")
        return

    # If no explicit val split, carve val_frac out of train
    if not val_records:
        random.shuffle(train_records)
        cut         = max(1, int(len(train_records) * (1 - val_frac)))
        val_records = train_records[cut:]
        train_records = train_records[:cut]

    random.shuffle(train_records)
    random.shuffle(val_records)

    print(f"\nTotal: {len(train_records):,} train  {len(val_records):,} val")

    # ── Dialogue JSONL ────────────────────────────────────────────────────────
    write_jsonl(out_dir / "dialogue" / "train.jsonl", train_records)
    write_jsonl(out_dir / "dialogue" / "val.jsonl",   val_records)

    # ── Plain text for small LMs ──────────────────────────────────────────────
    write_plain_text(out_dir / "dialogue" / "train.txt", train_records)
    write_plain_text(out_dir / "dialogue" / "val.txt",   val_records)

    # ── NPC profiles ─────────────────────────────────────────────────────────
    write_npc_profiles(out_dir / "npc_profiles.csv", train_records + val_records)

    # ── Affect CSV ────────────────────────────────────────────────────────────
    if all_affect:
        random.shuffle(all_affect)
        cut_a = max(1, int(len(all_affect) * (1 - val_frac)))
        write_csv(out_dir / "affect" / "train.csv", all_affect[:cut_a],  ["text", "valence", "arousal", "dominance"])
        write_csv(out_dir / "affect" / "val.csv",   all_affect[cut_a:],  ["text", "valence", "arousal", "dominance"])

    # ── Personality proxy CSV (PersonaChat personas as OCEAN proxies) ─────────
    # Without ground-truth labels, personality training needs external datasets.
    # See configs/personality.yaml — point train_path at your labelled OCEAN data.
    # PersonaChat personas are useful for building npc_profiles.csv; not for OCEAN labels.

    print("\nDone. Next steps:")
    print("  1. python -m src.data.build_caches  --profiles-path data/npc_profiles.csv \\")
    print("                                       --encoder-dir   artifacts/personality_encoder/best_model \\")
    print("                                       --out-path      artifacts/personality_cache.jsonl")
    print("  2. ./train_all.sh --run-id exp_01")


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare NPC dialogue training data")
    p.add_argument("--raw-dir",  type=Path, default=Path("data/raw"),
                   help="Directory containing raw downloaded datasets")
    p.add_argument("--out-dir",  type=Path, default=Path("data"),
                   help="Output directory for processed training files")
    p.add_argument("--val-frac", type=float, default=0.05,
                   help="Fraction of data to use as validation if no explicit val split")
    p.add_argument("--sources",  nargs="+",
                   default=["personachat", "crd3", "empathetic_dialogues", "dailydialog"],
                   choices=["personachat", "crd3", "empathetic_dialogues", "dailydialog"],
                   help="Which source datasets to convert")
    p.add_argument("--seed",     type=int, default=42)
    args = p.parse_args()
    run(args.raw_dir, args.out_dir, args.val_frac, args.sources, args.seed)


if __name__ == "__main__":
    main()
