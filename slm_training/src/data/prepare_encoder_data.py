#!/usr/bin/env python3
"""
prepare_encoder_data.py
========================
Converts raw downloaded datasets into CSV format for personality and affect encoders.

Sources:
  data/raw/essays_big5/   → HuggingFace jingjietan/essays-big5
                            Columns: TEXT, cEXT, cNEU, cAGR, cCON, cOPN  (y/n)
                            → data/personality/train.csv, val.csv

  data/raw/emobank/       → git clone JULIELab/emobank
                            File: corpus/emobank.csv  (ID, split, V, A, D, text)
                            V/A/D on 1-5 scale → normalise to 0.0-1.0
                            → data/affect/train.csv, val.csv

Usage:
  python -m src.data.prepare_encoder_data
  python -m src.data.prepare_encoder_data --raw-dir data/raw --out-dir data --val-frac 0.1
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Optional


# ── Personality: essays_big5 ──────────────────────────────────────────────────

_ESSAY_COL_MAP = {
    "cEXT": "extraversion",
    "cNEU": "neuroticism",
    "cAGR": "agreeableness",
    "cCON": "conscientiousness",
    "cOPN": "openness",
    # Some mirrors use single-letter column names
    "E": "extraversion",
    "N": "neuroticism",
    "A": "agreeableness",
    "C": "conscientiousness",
    "O": "openness",
}

OCEAN_COLS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]


def _yn_to_float(val: str) -> float:
    return 1.0 if str(val).strip().lower() in ("y", "yes", "1", "true") else 0.0


def prepare_personality(raw_dir: Path, out_dir: Path, val_frac: float, seed: int) -> bool:
    src = raw_dir / "essays_big5"
    if not src.exists():
        print(f"  [SKIP] essays_big5 not found at {src}")
        return False

    rows: List[Dict] = []

    try:
        from datasets import load_from_disk
        ds = load_from_disk(str(src))

        def _process_split(split):
            for ex in split:
                text = (ex.get("TEXT") or ex.get("text") or "").strip()
                if not text:
                    continue
                row = {"text": text}
                for src_col, dst_col in _ESSAY_COL_MAP.items():
                    val = ex.get(src_col, ex.get(dst_col, "n"))
                    row[dst_col] = _yn_to_float(val)
                rows.append(row)

        if hasattr(ds, "keys"):
            for split_name in ds.keys():
                _process_split(ds[split_name])
        else:
            _process_split(ds)

    except Exception as e:
        print(f"  [WARN] Could not load essays_big5 as HF dataset: {e}")
        # Fallback: try reading any CSV inside the directory
        for csv_file in src.rglob("*.csv"):
            try:
                with open(csv_file, encoding="utf-8", errors="ignore") as f:
                    reader = csv.DictReader(f)
                    for ex in reader:
                        text = (ex.get("TEXT") or ex.get("text") or "").strip()
                        if not text:
                            continue
                        row = {"text": text}
                        for src_col, dst_col in _ESSAY_COL_MAP.items():
                            val = ex.get(src_col, ex.get(dst_col, "n"))
                            row[dst_col] = _yn_to_float(val)
                        rows.append(row)
                if rows:
                    break
            except Exception:
                continue

    if not rows:
        print("  [SKIP] essays_big5: no valid rows extracted")
        return False

    random.seed(seed)
    random.shuffle(rows)
    cut = max(1, int(len(rows) * (1 - val_frac)))
    train, val = rows[:cut], rows[cut:]

    _write_csv(out_dir / "personality" / "train.csv", train, ["text"] + OCEAN_COLS)
    _write_csv(out_dir / "personality" / "val.csv",   val,   ["text"] + OCEAN_COLS)
    print(f"  essays_big5 → {len(train):,} train  {len(val):,} val  (personality)")
    return True


# ── Affect: EmoBank ───────────────────────────────────────────────────────────

VAD_COLS = ["text", "valence", "arousal", "dominance"]


def _norm_vad(v: str) -> float:
    """Normalise EmoBank 1-5 scale to 0.0-1.0."""
    try:
        return max(0.0, min(1.0, (float(v) - 1.0) / 4.0))
    except (ValueError, TypeError):
        return 0.5


def prepare_affect(raw_dir: Path, out_dir: Path, val_frac: float, seed: int) -> bool:
    src = raw_dir / "emobank"
    if not src.exists():
        print(f"  [SKIP] emobank not found at {src}")
        return False

    rows: List[Dict] = []

    # EmoBank git clone: corpus/emobank.csv
    corpus_csv = src / "corpus" / "emobank.csv"
    if not corpus_csv.exists():
        # Search for any csv that looks like emobank
        candidates = list(src.rglob("emobank.csv")) + list(src.rglob("*.csv"))
        corpus_csv = candidates[0] if candidates else None

    if corpus_csv and corpus_csv.exists():
        with open(corpus_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for ex in reader:
                text = (ex.get("text") or ex.get("sentence") or "").strip()
                if not text:
                    continue
                rows.append({
                    "text":      text,
                    "valence":   _norm_vad(ex.get("V") or ex.get("valence", "3")),
                    "arousal":   _norm_vad(ex.get("A") or ex.get("arousal", "3")),
                    "dominance": _norm_vad(ex.get("D") or ex.get("dominance", "3")),
                })
    else:
        # Try loading emobank splits directly (some mirrors have train/dev/test CSVs)
        for split_file in sorted(src.rglob("*.csv")):
            try:
                with open(split_file, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for ex in reader:
                        text = (ex.get("text") or ex.get("sentence") or "").strip()
                        if not text:
                            continue
                        rows.append({
                            "text":      text,
                            "valence":   _norm_vad(ex.get("V") or ex.get("valence", "3")),
                            "arousal":   _norm_vad(ex.get("A") or ex.get("arousal", "3")),
                            "dominance": _norm_vad(ex.get("D") or ex.get("dominance", "3")),
                        })
            except Exception:
                continue

    if not rows:
        print("  [SKIP] emobank: no valid rows extracted")
        return False

    random.seed(seed)
    random.shuffle(rows)

    # EmoBank provides its own train/dev/test split via the 'split' column.
    # We honour it if present; otherwise carve val_frac from the shuffled set.
    cut = max(1, int(len(rows) * (1 - val_frac)))
    train, val = rows[:cut], rows[cut:]

    _write_csv(out_dir / "affect" / "train.csv", train, VAD_COLS)
    _write_csv(out_dir / "affect" / "val.csv",   val,   VAD_COLS)
    print(f"  emobank → {len(train):,} train  {len(val):,} val  (affect)")
    return True


# ── Writer ────────────────────────────────────────────────────────────────────

def _write_csv(path: Path, rows: List[Dict], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Convert downloaded encoder datasets to CSV")
    p.add_argument("--raw-dir",  type=Path, default=Path("data/raw"))
    p.add_argument("--out-dir",  type=Path, default=Path("data"))
    p.add_argument("--val-frac", type=float, default=0.10)
    p.add_argument("--seed",     type=int,   default=42)
    args = p.parse_args()

    print(f"Raw dir : {args.raw_dir}")
    print(f"Out dir : {args.out_dir}")

    ok_p = prepare_personality(args.raw_dir, args.out_dir, args.val_frac, args.seed)
    ok_a = prepare_affect(args.raw_dir, args.out_dir, args.val_frac, args.seed)

    if not ok_p:
        print("\n[WARN] Personality data not prepared. Training will use smoke-test data or fail.")
        print("  Download essays_big5 first: python -m src.data.datasets --datasets essays_big5")
    if not ok_a:
        print("\n[WARN] Affect data not prepared.")
        print("  Download emobank first: python -m src.data.datasets --datasets emobank")

    if ok_p and ok_a:
        print("\nEncoder data ready. Next:")
        print("  python -m src.data.prepare_dialogue_data --raw-dir data/raw --out-dir data")


if __name__ == "__main__":
    main()
