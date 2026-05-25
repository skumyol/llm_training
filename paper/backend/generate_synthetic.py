# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "scikit-learn>=1.3",
# ]
# ///
"""
Synthetic Annotator Generator

Generates synthetic human audit annotations with a configurable statistical
distribution somewhere between real human annotators and the teacher/LLM labels.

The generator models per-head confusion patterns from real human data and
produces annotations with a target agreement level against the teacher.

Usage:
    uv run generate_synthetic.py \
        --data ../audit_input_clean.jsonl \
        --human-a ../audit_results/audit_654cfad67f990b0393b85132.jsonl \
        --human-b ../audit_results/audit_67c87fc1b3ba111d0e1526a0.jsonl \
        --output ../audit_results/audit_synthetic.jsonl \
        --agreement-target 0.45 \
        --count 3 \
        --seed 42

This produces `count` synthetic annotator files, each with 150 turns.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


HEADS = [
    "valence", "arousal", "secrecy_pressure", "reveal_decision",
    "response_policy", "repair_strategy", "trust_level", "familiarity_level",
]


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def make_turn_id(rec: dict, fallback_index: int = 0) -> str:
    if "turn_id" in rec and rec["turn_id"]:
        return rec["turn_id"]
    ep = rec.get("episode_id", "unknown")
    tn = rec.get("turn_number", fallback_index)
    return f"{ep}_{tn}"


def index_by_turn_id(records: list[dict]) -> dict[str, dict]:
    return {make_turn_id(r, i): r for i, r in enumerate(records)}


# ---------------------------------------------------------------------------
# Per-head label distributions and confusion matrices
# ---------------------------------------------------------------------------
def compute_head_stats(
    teacher: dict[str, dict],
    human_a: dict[str, dict],
    human_b: dict[str, dict],
    common_ids: list[str],
) -> dict[str, dict]:
    """Compute per-head: teacher label distribution, human label distribution,
    and P(human_label | teacher_label) confusion matrix."""
    stats = {}
    for head in HEADS:
        t_vals = [teacher[tid]["labels"][head] for tid in common_ids]
        a_vals = [human_a[tid]["labels"].get(head) for tid in common_ids]
        b_vals = [human_b[tid]["labels"].get(head) for tid in common_ids]

        # Teacher label distribution
        t_dist = dict(Counter(t_vals))

        # Human label distribution (pooled from both annotators)
        human_pooled = [l for l in (a_vals + b_vals) if l is not None]
        h_dist = dict(Counter(human_pooled))

        # Confusion: for each teacher label, what do humans pick?
        confusion: dict[str, Counter] = defaultdict(Counter)
        for t, a, b_h in zip(t_vals, a_vals, b_vals):
            if a is not None:
                confusion[t][a] += 1
            if b_h is not None:
                confusion[t][b_h] += 1

        # Normalize to probabilities
        confusion_prob: dict[str, dict[str, float]] = {}
        for t_label, counter in confusion.items():
            total = sum(counter.values())
            confusion_prob[t_label] = {h: c / total for h, c in counter.items()}

        stats[head] = {
            "teacher_distribution": t_dist,
            "human_distribution": h_dist,
            "confusion": confusion_prob,
            "human_labels": sorted(set(human_pooled)),
        }
    return stats


# ---------------------------------------------------------------------------
# Generate synthetic labels
# ---------------------------------------------------------------------------
def generate_synthetic_turn(
    teacher_labels: dict[str, str],
    stats: dict[str, dict],
    agreement_target: float,
    rng: random.Random,
) -> dict[str, str]:
    """
    For each head, with probability `agreement_target` use the teacher label,
    otherwise sample from the human confusion distribution EXCLUDING the
    teacher label (so the disagreement is genuine).
    """
    synthetic = {}
    for head in HEADS:
        t_label = teacher_labels.get(head)
        if t_label is None:
            continue

        if rng.random() < agreement_target:
            synthetic[head] = t_label
        else:
            confusion = stats[head]["confusion"].get(t_label, {})
            # Exclude teacher label from disagreement sampling
            disagree_labels = {h: c for h, c in confusion.items() if h != t_label}
            if disagree_labels:
                labels, counts = zip(*disagree_labels.items())
                total = sum(counts)
                probs = [c / total for c in counts]
                synthetic[head] = rng.choices(list(labels), weights=probs, k=1)[0]
            else:
                # Fallback: sample from any human label except teacher
                h_dist = stats[head]["human_distribution"]
                fallback = {h: c for h, c in h_dist.items() if h != t_label}
                if fallback:
                    labels, counts = zip(*fallback.items())
                    synthetic[head] = rng.choices(list(labels), weights=list(counts), k=1)[0]
                else:
                    synthetic[head] = t_label
    return synthetic


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate synthetic audit annotations")
    parser.add_argument("--data", required=True, type=Path, help="Teacher labels JSONL (audit_input_clean.jsonl)")
    parser.add_argument("--human-a", required=True, type=Path, help="Human annotator A JSONL")
    parser.add_argument("--human-b", required=True, type=Path, help="Human annotator B JSONL")
    parser.add_argument("--output", default="./audit_results", type=Path, help="Output directory or file")
    parser.add_argument("--agreement-target", type=float, default=0.40,
                        help="Target agreement with teacher labels (0.0-1.0). Default 0.40 (~human level)")
    parser.add_argument("--count", type=int, default=1,
                        help="Number of synthetic annotators to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--min-time", type=int, default=45,
                        help="Minimum turn time for synthetic records (seconds)")
    parser.add_argument("--max-time", type=int, default=90,
                        help="Maximum turn time for synthetic records (seconds)")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Load data
    print(f"Loading teacher data from {args.data} ...")
    teacher_recs = index_by_turn_id(load_jsonl(args.data))
    print(f"  {len(teacher_recs)} teacher records")

    print(f"Loading human A from {args.human_a} ...")
    human_a = index_by_turn_id(load_jsonl(args.human_a))
    print(f"  {len(human_a)} human A records")

    print(f"Loading human B from {args.human_b} ...")
    human_b = index_by_turn_id(load_jsonl(args.human_b))
    print(f"  {len(human_b)} human B records")

    # Find common turn IDs
    common_ids = sorted(set(teacher_recs) & set(human_a) & set(human_b))
    if not common_ids:
        print("Error: no common turn IDs across all three files")
        return 1
    print(f"\nCommon turns: {len(common_ids)}")

    # Compute statistics from real human data
    print("\nComputing per-head distributions and confusion matrices ...")
    stats = compute_head_stats(teacher_recs, human_a, human_b, common_ids)

    for head in HEADS:
        s = stats[head]
        print(f"  {head}: teacher classes={len(s['teacher_distribution'])}, "
              f"human classes={len(s['human_distribution'])}")

    # Generate synthetic annotators
    args.output.mkdir(parents=True, exist_ok=True)

    for annotator_idx in range(args.count):
        annotator_id = f"synthetic_{annotator_idx + 1:02d}"
        out_path = args.output / f"audit_{annotator_id}.jsonl"
        meta_path = args.output / f"audit_{annotator_id}_meta.json"

        print(f"\nGenerating {annotator_id} (agreement_target={args.agreement_target}) ...")
        annotations: list[dict] = []

        for i, tid in enumerate(common_ids):
            teacher_labels = teacher_recs[tid]["labels"]
            synthetic_labels = generate_synthetic_turn(
                teacher_labels, stats, args.agreement_target, rng
            )

            turn_time = rng.randint(args.min_time, args.max_time)
            session_time = turn_time * (i + 1) + rng.randint(0, 300)

            record = {
                "turn_id": tid,
                "episode_id": teacher_recs[tid].get("episode_id"),
                "scenario_type": teacher_recs[tid].get("scenario_type"),
                "annotator": annotator_id,
                "labels": synthetic_labels,
                "notes": "",
                "recorded_at": datetime.now().isoformat(),
                "turn_elapsed_seconds": turn_time,
                "session_elapsed_seconds": session_time,
                "synthetic": True,
                "agreement_target": args.agreement_target,
            }
            annotations.append(record)

        # Write output
        with open(out_path, "w", encoding="utf-8") as f:
            for ann in annotations:
                f.write(json.dumps(ann, ensure_ascii=False) + "\n")

        # Compute actual agreement achieved
        agreement = {}
        for head in HEADS:
            matches = 0
            total = 0
            for ann in annotations:
                t_label = teacher_recs[ann["turn_id"]]["labels"].get(head)
                s_label = ann["labels"].get(head)
                if t_label is not None and s_label is not None:
                    matches += int(t_label == s_label)
                    total += 1
            agreement[head] = round(matches / total, 3) if total > 0 else 0.0

        meta = {
            "annotator": annotator_id,
            "synthetic": True,
            "agreement_target": args.agreement_target,
            "total_turns": len(annotations),
            "annotated_count": len(annotations),
            "seed": args.seed,
            "per_head_agreement": agreement,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"  Saved {len(annotations)} turns to {out_path}")
        print(f"  Per-head agreement:")
        for head, acc in agreement.items():
            diff = acc - args.agreement_target
            marker = "✓" if abs(diff) < 0.05 else ("↑" if diff > 0 else "↓")
            print(f"    {head:22s}  {acc:.3f}  ({marker} target={args.agreement_target:.2f})")

    print(f"\nDone. Generated {args.count} synthetic annotator(s).")


if __name__ == "__main__":
    main()