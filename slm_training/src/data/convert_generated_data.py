#!/usr/bin/env python3
"""
convert_generated_data.py
==========================
Converts the upper-level data generation pipeline's output into the scaffold's
training formats for affect encoder, dialogue model, and small LM benchmarking.

Source (upper-level repo):
  data/validated_turns/*.jsonl   – per-turn records from the generation pipeline
  data/splits/full_trace.jsonl   – (optional) packaged full trace

Produced (scaffold):
  data/affect/from_gen_train.csv     }  RegressionTextDataset for affect encoder
  data/affect/from_gen_val.csv       }  VAD labels derived from A_t (real labels!)

  data/dialogue/from_gen_train.jsonl }  DialogueJsonlDataset for ConditionalDialogueModel
  data/dialogue/from_gen_val.jsonl   }

  data/dialogue/from_gen_train.txt   }  plain text for small LM architectures
  data/dialogue/from_gen_val.txt     }

  data/npc_profiles_generated.csv   –  NPC profiles for personality cache

Affect label mapping (from prompts/label_A_M.txt):
  valence  : negative→0.1  neutral→0.5  positive→0.9
  arousal  : low→0.2       medium→0.5   high→0.8
  dominance: (from A_t.control)  low→0.2  medium→0.5  high→0.8

Usage:
  # From the scaffold root, point at the upper-level repo's data directory:
  python -m src.data.convert_generated_data \\
      --source-dir ../../data \\
      --out-dir    data \\
      --val-frac   0.05

  # Or specify the validated_turns directory directly:
  python -m src.data.convert_generated_data \\
      --validated-turns-dir ../../data/validated_turns \\
      --out-dir data
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Affect label → float mapping ──────────────────────────────────────────────

_VALENCE_MAP: Dict[str, float] = {
    "negative": 0.1, "neg": 0.1, "low": 0.1,
    "neutral":  0.5, "med": 0.5, "medium": 0.5,
    "positive": 0.9, "pos": 0.9, "high": 0.9,
}
_LEVEL_MAP: Dict[str, float] = {
    "vl": 0.1, "very low":  0.1,
    "l":  0.3, "low":       0.2,
    "n":  0.5, "neutral":   0.5, "medium": 0.5, "med": 0.5,
    "h":  0.7, "high":      0.8,
    "vh": 0.9, "very high": 0.9,
}


def _valence(v: Optional[str]) -> float:
    return _VALENCE_MAP.get((v or "neutral").lower().strip(), 0.5)


def _level(v: Optional[str]) -> float:
    return _LEVEL_MAP.get((v or "medium").lower().strip(), 0.5)


# ── NPC profile builder ───────────────────────────────────────────────────────

def _npc_id_from_profile(text: str) -> str:
    return "npc_" + hashlib.md5(text.encode()).hexdigest()[:12]


def _build_profile(W: Dict[str, Any]) -> str:
    role    = W.get("role", "character").replace("_", " ")
    persona = ", ".join(W.get("persona_style", []))
    values  = ", ".join(W.get("values", []))
    goals   = ". ".join(str(g) for g in W.get("core_goals", []))
    parts   = [f"A {role}"]
    if persona:
        parts.append(f"with a {persona} demeanor")
    if values:
        parts.append(f"who values {values}")
    base    = " ".join(parts) + "."
    if goals:
        base += f" Goals: {goals}."
    return base


# ── Record parsers ────────────────────────────────────────────────────────────

def _build_affect_row(record: Dict[str, Any], history_text: str) -> Optional[Dict[str, float]]:
    """Extract A_t labels as a float VAD row."""
    A_t = record.get("A_t")
    if not A_t:
        return None
    return {
        "text":      history_text,
        "valence":   _valence(A_t.get("valence")),
        "arousal":   _level(A_t.get("arousal")),
        "dominance": _level(A_t.get("control")),
    }


def _build_dialogue_record(record: Dict[str, Any], profile: str, npc_id: str) -> Optional[Dict]:
    response = record.get("response", "").strip()
    if not response:
        return None

    history = record.get("dialogue_history", [])
    ctx = []
    for turn in history:
        player_utt = turn.get("player_utterance", "").strip()
        npc_resp   = turn.get("response", "").strip()
        if player_utt:
            ctx.append({"speaker": "player", "text": player_utt})
        if npc_resp:
            ctx.append({"speaker": "npc", "text": npc_resp})

    # Add the current player utterance (the one that triggered this response)
    current_input = record.get("input", "").strip()
    if current_input:
        ctx.append({"speaker": "player", "text": current_input})

    A_t = record.get("A_t", {})
    D_t = record.get("D_t", {})

    return {
        "npc_id":           npc_id,
        "npc_profile":      profile,
        "dialogue_context": ctx,
        "target_response":  response,
        "metadata": {
            "source":          "generated",
            "episode_id":      record.get("episode_id", ""),
            "turn_idx":        record.get("turn_idx", 0),
            "scenario_type":   record.get("scenario_type", ""),
            "setting":         record.get("setting", ""),
            "counterfactual":  record.get("counterfactual", False),
            "response_policy": D_t.get("response_policy", ""),
            "reveal_decision": D_t.get("reveal_decision", ""),
            "valence":         A_t.get("valence", ""),
            "arousal":         A_t.get("arousal", ""),
        },
    }


def _build_history_text(record: Dict[str, Any]) -> str:
    """Flatten dialogue history + current player utterance into plain text."""
    lines = []
    for t in record.get("dialogue_history", []):
        if t.get("player_utterance"):
            lines.append(f"player: {t['player_utterance']}")
        if t.get("response"):
            lines.append(f"npc: {t['response']}")
    if record.get("input"):
        lines.append(f"player: {record['input']}")
    return " ".join(lines) or "neutral"


# ── File readers ──────────────────────────────────────────────────────────────

def load_validated_turns(validated_turns_dir: Path) -> List[Dict]:
    records = []
    found   = list(sorted(validated_turns_dir.glob("*.jsonl")))
    if not found:
        # Also try flat full_trace.jsonl
        trace = validated_turns_dir.parent / "splits" / "full_trace.jsonl"
        if trace.exists():
            found = [trace]
        else:
            print(f"  [WARN] No .jsonl files found in {validated_turns_dir}")
            return []

    for path in found:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    print(f"  Loaded {len(records):,} validated turn records from {len(found)} file(s)")
    return records


# ── Writers ───────────────────────────────────────────────────────────────────

def write_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  {len(records):,} records → {path}")


def write_csv(path: Path, rows: List[Dict], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})
    print(f"  {len(rows):,} rows → {path}")


def write_plain_text(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for r in records:
        lines.append(f"PROFILE: {r['npc_profile']}")
        for turn in r.get("dialogue_context", []):
            lines.append(f"{turn['speaker'].upper()}: {turn['text']}")
        lines.append(f"NPC: {r['target_response']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  {len(lines):,} lines → {path}")


def write_npc_profiles(path: Path, records: List[Dict]) -> None:
    seen: Dict[str, str] = {}
    for r in records:
        seen[r["npc_id"]] = r["npc_profile"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["npc_id", "profile_text"])
        w.writeheader()
        for nid, txt in seen.items():
            w.writerow({"npc_id": nid, "profile_text": txt})
    print(f"  {len(seen):,} NPC profiles → {path}")


def split_dialogue_records_by_episode(
    records: List[Dict],
    *,
    val_frac: float,
    seed: int,
) -> tuple[List[Dict], List[Dict]]:
    """Deterministically split whole episodes, never individual turns."""
    groups: Dict[str, List[Dict]] = {}
    for index, record in enumerate(records):
        episode_id = str(record.get("metadata", {}).get("episode_id") or f"row_{index}")
        groups.setdefault(episode_id, []).append(record)
    episode_ids = sorted(groups)
    random.Random(seed).shuffle(episode_ids)
    if len(episode_ids) < 2:
        return list(records), []
    n_val = max(1, min(len(episode_ids) - 1, round(len(episode_ids) * val_frac)))
    val_ids = set(episode_ids[:n_val])
    train = [
        record
        for episode_id in episode_ids
        if episode_id not in val_ids
        for record in groups[episode_id]
    ]
    val = [
        record
        for episode_id in episode_ids
        if episode_id in val_ids
        for record in groups[episode_id]
    ]
    return train, val


# ── Main conversion ───────────────────────────────────────────────────────────

def convert(
    source_records: List[Dict],
    out_dir: Path,
    val_frac: float,
    seed: int,
) -> None:
    random.seed(seed)

    affect_rows:    List[Dict] = []
    dialogue_recs:  List[Dict] = []
    npc_profiles:   Dict[str, str] = {}   # npc_id → profile text
    skipped = 0

    for record in source_records:
        W       = record.get("W", {})
        profile = _build_profile(W)
        npc_id  = W.get("npc_id") or _npc_id_from_profile(profile)

        npc_profiles[npc_id] = profile

        hist_text = _build_history_text(record)

        # Affect row
        arow = _build_affect_row(record, hist_text)
        if arow:
            affect_rows.append(arow)

        # Dialogue record
        drec = _build_dialogue_record(record, profile, npc_id)
        if drec:
            dialogue_recs.append(drec)
        else:
            skipped += 1

    print(f"\n  Converted {len(dialogue_recs):,} dialogue records  ({skipped} skipped, no response)")
    print(f"  Affect rows: {len(affect_rows):,}")
    print(f"  Unique NPCs: {len(npc_profiles):,}")

    if not dialogue_recs:
        print("\n[WARN] No dialogue records produced — validated_turns may be empty.")
        print("       Run the upper-level data generation pipeline first:")
        print("       python run_data_gen.py --config configs/data_gen.yaml")
        return

    # ── Split ─────────────────────────────────────────────────────────────────
    # Sort for reproducibility, then split whole episodes to prevent turn leakage.
    dialogue_recs.sort(key=lambda r: (
        r["metadata"].get("episode_id", ""),
        r["metadata"].get("turn_idx", 0),
    ))
    train_d, val_d = split_dialogue_records_by_episode(
        dialogue_recs,
        val_frac=val_frac,
        seed=seed,
    )

    cut_a = max(1, int(len(affect_rows) * (1 - val_frac)))
    random.shuffle(affect_rows)
    train_a, val_a = affect_rows[:cut_a], affect_rows[cut_a:]

    # ── Write dialogue JSONL ──────────────────────────────────────────────────
    write_jsonl(out_dir / "dialogue" / "from_gen_train.jsonl", train_d)
    write_jsonl(out_dir / "dialogue" / "from_gen_val.jsonl",   val_d)

    # ── Write plain text for small LMs ───────────────────────────────────────
    write_plain_text(out_dir / "dialogue" / "from_gen_train.txt", train_d)
    write_plain_text(out_dir / "dialogue" / "from_gen_val.txt",   val_d)

    # ── Write affect CSVs ─────────────────────────────────────────────────────
    cols = ["text", "valence", "arousal", "dominance"]
    write_csv(out_dir / "affect" / "from_gen_train.csv", train_a, cols)
    write_csv(out_dir / "affect" / "from_gen_val.csv",   val_a,   cols)

    # ── Write NPC profiles ────────────────────────────────────────────────────
    all_profiles = [{"npc_id": k, "profile_text": v} for k, v in npc_profiles.items()]
    write_npc_profiles(out_dir / "npc_profiles_generated.csv",
                       [{"npc_id": k, "npc_profile": v} for k, v in npc_profiles.items()])

    # ── Stats ─────────────────────────────────────────────────────────────────
    scenario_counts: Dict[str, int] = {}
    for r in dialogue_recs:
        st = r["metadata"].get("scenario_type", "unknown")
        scenario_counts[st] = scenario_counts.get(st, 0) + 1

    print("\n  Scenario type distribution:")
    for st, cnt in sorted(scenario_counts.items(), key=lambda x: -x[1]):
        print(f"    {st:30s} {cnt:5d}")

    cf_count = sum(1 for r in dialogue_recs if r["metadata"].get("counterfactual"))
    print(f"\n  Counterfactual turns: {cf_count:,} / {len(dialogue_recs):,}")
    print("\nDone. Data ready for training.")
    print("  Next: point configs to from_gen_* files or merge with public data:")
    print("  python -m src.data.merge_datasets  (see README)")


def main() -> None:
    p = argparse.ArgumentParser(description="Convert generated NPC data to scaffold training formats")
    p.add_argument("--source-dir",
                   type=Path,
                   default=Path("../../data"),
                   help="Root of upper-level data dir (contains validated_turns/)")
    p.add_argument("--validated-turns-dir",
                   type=Path,
                   default=None,
                   help="Override: path directly to validated_turns/ directory")
    p.add_argument("--out-dir",
                   type=Path,
                   default=Path("data"),
                   help="Scaffold data output directory")
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--seed",     type=int,   default=42)
    args = p.parse_args()

    turns_dir = args.validated_turns_dir or (args.source_dir / "validated_turns")

    if not turns_dir.exists():
        print(f"[ERROR] validated_turns directory not found: {turns_dir}")
        print("  Run the data generation pipeline first:")
        print("  python run_data_gen.py --config configs/data_gen.yaml")
        raise SystemExit(1)

    print(f"Source: {turns_dir}")
    print(f"Output: {args.out_dir}")

    records = load_validated_turns(turns_dir)
    if records:
        convert(records, args.out_dir, args.val_frac, args.seed)
    else:
        print("[WARN] No records loaded. The validated_turns directory may be empty.")
        print("  Run: python run_data_gen.py --config configs/data_gen.yaml")


if __name__ == "__main__":
    main()
