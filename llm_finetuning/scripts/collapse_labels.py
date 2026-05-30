#!/usr/bin/env python3
"""
Collapse fine-grained labels to coarser categories for heads with low agreement.

Usage:
    # Collapse 5-class stance deltas → 3-class (neg / neutral / pos)
    PYTHONPATH=. python scripts/collapse_labels.py \
        --input data/splits/train_heads.jsonl \
        --output data/splits/train_heads_collapsed.jsonl \
        --collapse stance_deltas

    # Collapse both stance deltas and levels
    PYTHONPATH=. python scripts/collapse_labels.py \
        --input data/splits/train_heads.jsonl \
        --output data/splits/train_heads_collapsed.jsonl \
        --collapse stance_deltas stance_levels

Supported modes:
    stance_deltas   : "--","-","0","+","++" → "neg","neutral","pos"
    stance_levels   : "VL","L","N","H","VH" → "low","neutral","high"
    reveal_decision : "none","hint","partial","full" → "none","hint","reveal"
    tone            : 6-class → 3-class (warm/soothe vs neutral vs confrontational/threaten)
"""
import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Input heads JSONL")
    p.add_argument("--output", required=True, help="Output collapsed JSONL")
    p.add_argument("--collapse", nargs="+", required=True,
                   choices=["stance_deltas", "stance_levels", "reveal_decision", "tone"],
                   help="Which label groups to collapse")
    return p.parse_args()


# Mapping definitions
COLLAPSE_MAPS = {
    "stance_deltas": {
        "--": "neg",
        "-": "neg",
        "0": "neutral",
        "+": "pos",
        "++": "pos",
    },
    "stance_levels": {
        "VL": "low",
        "L": "low",
        "N": "neutral",
        "H": "high",
        "VH": "high",
    },
    "reveal_decision": {
        "none": "none",
        "hint": "hint",
        "partial": "reveal",
        "full": "reveal",
    },
    "tone": {
        "warm": "positive",
        "neutral": "neutral",
        "confrontational": "negative",
        "sarcastic": "negative",
        "fearful": "negative",
        "evasive": "negative",
    },
}

# Which label keys in the record to modify
FIELD_TARGETS = {
    "stance_deltas": [
        "affection_delta", "respect_delta", "dominance_delta",
        "familiarity_delta", "trust_delta", "obligation_delta",
    ],
    "stance_levels": [
        "affection_level", "respect_level", "dominance_level",
        "familiarity_level", "trust_level", "obligation_level",
    ],
    "reveal_decision": ["reveal_decision"],
    "tone": ["tone"],
}


def collapse_record(record: dict, collapse_modes: list[str]) -> dict:
    rec = dict(record)
    labels = rec.get("labels", {})
    for mode in collapse_modes:
        mapping = COLLAPSE_MAPS[mode]
        for field in FIELD_TARGETS[mode]:
            if field in labels:
                old = labels[field]
                if isinstance(old, list):
                    new = [mapping.get(v, v) for v in old]
                else:
                    new = mapping.get(str(old), old)
                labels[field] = new
    rec["labels"] = labels
    return rec


def main():
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {mode: {"changed": 0, "total": 0} for mode in args.collapse}

    with open(in_path) as f_in, open(out_path, "w") as f_out:
        for line in f_in:
            rec = json.loads(line.strip())
            rec = collapse_record(rec, args.collapse)

            # Stats
            for mode in args.collapse:
                for field in FIELD_TARGETS[mode]:
                    stats[mode]["total"] += 1
                    old_val = json.loads(line.strip()).get("labels", {}).get(field, "")
                    new_val = rec["labels"].get(field, "")
                    if old_val != new_val:
                        stats[mode]["changed"] += 1

            f_out.write(json.dumps(rec) + "\n")

    print(f"Wrote collapsed labels to {out_path}")
    print("Collapse statistics:")
    for mode, s in stats.items():
        print(f"  {mode}: {s['changed']}/{s['total']} labels changed ({s['changed']/max(1,s['total'])*100:.1f}%)")

    # Also write a mapping reference
    ref_path = out_path.with_suffix(".collapse_map.json")
    with open(ref_path, "w") as f:
        json.dump({m: COLLAPSE_MAPS[m] for m in args.collapse}, f, indent=2)
    print(f"Saved collapse map reference to {ref_path}")


if __name__ == "__main__":
    main()
