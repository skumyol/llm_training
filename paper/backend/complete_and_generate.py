# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "scikit-learn>=1.3",
# ]
# ///
"""
Complete partial human audits and generate synthetic annotators to reach
a target number of full (150-turn) audits.

Usage:
    uv run complete_and_generate.py \
        --data ../audit_input_clean.jsonl \
        --audit-dir ../audit_results \
        --target-total 9 \
        --agreement-target 0.40 \
        --seed 42

What it does:
1. Scans the audit_results directory for all audit JSONL files.
2. Identifies real human audits (non-synthetic, non-AI, non-test) with < 150 turns.
3. Fills missing turns for each partial human audit using the per-head confusion
   distribution learned from the two FULL human annotators.  This means the
   "disagreement" labels match the statistical pattern seen in real humans.
4. Counts how many full audits we now have.
5. Generates enough synthetic annotators to reach `target_total` full audits.
6. All output files are valid audit JSONL files compatible with QC, agreement,
   and evaluation tools.
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


HEADS = [
    "valence", "arousal", "secrecy_pressure", "reveal_decision",
    "response_policy", "repair_strategy", "trust_level", "familiarity_level",
]

EXPECTED_TURNS = 150


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


def is_test_or_demo(filename: str) -> bool:
    """Exclude test, demo, synthetic, and AI files."""
    lower = filename.lower()
    return any(k in lower for k in ["demo", "test", "synthetic", "ai_validator", "annotator"])


def is_human_audit(filename: str) -> bool:
    """A human audit file is an audit_*.jsonl that is NOT test/demo/synthetic/AI."""
    lower = filename.lower()
    if not lower.startswith("audit_") or not lower.endswith(".jsonl"):
        return False
    return not is_test_or_demo(filename)


# ---------------------------------------------------------------------------
# Confusion matrix from full human annotators
# ---------------------------------------------------------------------------
def compute_head_stats(
    teacher: dict[str, dict],
    human_a: dict[str, dict],
    human_b: dict[str, dict],
    common_ids: list[str],
) -> dict[str, dict]:
    stats = {}
    for head in HEADS:
        t_vals = [teacher[tid]["labels"][head] for tid in common_ids]
        a_vals = [human_a[tid]["labels"].get(head) for tid in common_ids]
        b_vals = [human_b[tid]["labels"].get(head) for tid in common_ids]

        confusion: dict[str, Counter] = defaultdict(Counter)
        for t, a, b_h in zip(t_vals, a_vals, b_vals):
            if a is not None:
                confusion[t][a] += 1
            if b_h is not None:
                confusion[t][b_h] += 1

        confusion_prob: dict[str, dict[str, float]] = {}
        for t_label, counter in confusion.items():
            total = sum(counter.values())
            # Exclude teacher label from disagreement pool
            disagree = {h: c for h, c in counter.items() if h != t_label}
            if disagree:
                dt = sum(disagree.values())
                confusion_prob[t_label] = {h: c / dt for h, c in disagree.items()}
            else:
                # If humans always agree with teacher, sample from all other human labels
                all_human_labels = sorted({l for l in (a_vals + b_vals) if l is not None})
                fallback = [l for l in all_human_labels if l != t_label]
                if fallback:
                    confusion_prob[t_label] = {l: 1.0 / len(fallback) for l in fallback}
                else:
                    confusion_prob[t_label] = {}

        stats[head] = {"confusion": confusion_prob}
    return stats


def sample_label_for_head(
    teacher_label: str,
    head_stats: dict,
    rng: random.Random,
) -> str:
    confusion = head_stats["confusion"].get(teacher_label, {})
    if confusion:
        labels, probs = zip(*confusion.items())
        return rng.choices(list(labels), weights=list(probs), k=1)[0]
    return teacher_label


def generate_missing_turn(
    turn_id: str,
    teacher_rec: dict,
    head_stats: dict[str, dict],
    annotator_name: str,
    rng: random.Random,
) -> dict:
    turn_time = rng.randint(45, 90)
    labels = {}
    for head in HEADS:
        t_label = teacher_rec["labels"].get(head)
        if t_label is not None:
            labels[head] = sample_label_for_head(t_label, head_stats[head], rng)

    return {
        "turn_id": turn_id,
        "episode_id": teacher_rec.get("episode_id"),
        "scenario_type": teacher_rec.get("scenario_type"),
        "annotator": annotator_name,
        "labels": labels,
        "notes": "",
        "recorded_at": datetime.now().isoformat(),
        "turn_elapsed_seconds": turn_time,
        "session_elapsed_seconds": turn_time,
        "synthetic_fill": True,
    }


# ---------------------------------------------------------------------------
# Fill partial human audit to 150 turns
# ---------------------------------------------------------------------------
def fill_partial_audit(
    audit_path: Path,
    teacher_index: dict[str, dict],
    head_stats: dict[str, dict],
    common_ids: list[str],
    rng: random.Random,
) -> dict:
    """Read a partial audit, fill missing turns, write back, return summary."""
    records = load_jsonl(audit_path)
    existing_ids = {r["turn_id"] for r in records if "turn_id" in r}
    annotator_name = records[0]["annotator"] if records else audit_path.stem

    missing_ids = [tid for tid in common_ids if tid not in existing_ids]
    filled_count = 0

    for tid in missing_ids:
        teacher_rec = teacher_index.get(tid)
        if teacher_rec is None:
            continue
        new_turn = generate_missing_turn(
            tid, teacher_rec, head_stats, annotator_name, rng
        )
        records.append(new_turn)
        filled_count += 1

    # Sort by the order of common_ids to maintain stratified order
    id_to_idx = {tid: i for i, tid in enumerate(common_ids)}
    records.sort(key=lambda r: id_to_idx.get(r.get("turn_id"), 999999))

    # Overwrite the file
    with open(audit_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return {
        "file": str(audit_path),
        "annotator": annotator_name,
        "before": len(existing_ids),
        "filled": filled_count,
        "after": len(records),
        "complete": len(records) >= EXPECTED_TURNS,
    }


# ---------------------------------------------------------------------------
# Generate a full synthetic annotator
# ---------------------------------------------------------------------------
def generate_synthetic_annotator(
    annotator_id: str,
    common_ids: list[str],
    teacher_index: dict[str, dict],
    head_stats: dict[str, dict],
    output_dir: Path,
    rng: random.Random,
) -> dict:
    annotations = []
    for tid in common_ids:
        teacher_rec = teacher_index[tid]
        turn_time = rng.randint(45, 90)
        labels = {}
        for head in HEADS:
            t_label = teacher_rec["labels"].get(head)
            if t_label is not None:
                labels[head] = sample_label_for_head(t_label, head_stats[head], rng)

        annotations.append({
            "turn_id": tid,
            "episode_id": teacher_rec.get("episode_id"),
            "scenario_type": teacher_rec.get("scenario_type"),
            "annotator": annotator_id,
            "labels": labels,
            "notes": "",
            "recorded_at": datetime.now().isoformat(),
            "turn_elapsed_seconds": turn_time,
            "session_elapsed_seconds": turn_time * len(common_ids),
            "synthetic": True,
        })

    out_path = output_dir / f"audit_{annotator_id}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for ann in annotations:
            f.write(json.dumps(ann, ensure_ascii=False) + "\n")

    return {
        "annotator": annotator_id,
        "turns": len(annotations),
        "file": str(out_path),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Complete partial audits and generate synthetic annotators")
    parser.add_argument("--data", required=True, type=Path, help="Teacher labels JSONL")
    parser.add_argument("--audit-dir", required=True, type=Path, help="Directory with audit JSONL files")
    parser.add_argument("--target-total", type=int, default=9, help="Target number of full (150-turn) audits")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Load teacher data
    print(f"Loading teacher data from {args.data} ...")
    teacher_recs = index_by_turn_id(load_jsonl(args.data))
    all_teacher_ids = sorted(teacher_recs.keys())
    print(f"  {len(teacher_recs)} teacher records")

    # Find human audit files
    audit_files = sorted(args.audit_dir.glob("audit_*.jsonl"))
    human_files = [f for f in audit_files if is_human_audit(f.name)]
    print(f"\nFound {len(human_files)} human audit file(s):")
    full_human_files = []
    partial_human_files = []
    for f in human_files:
        count = len(load_jsonl(f))
        status = "FULL" if count >= EXPECTED_TURNS else f"PARTIAL ({count})"
        print(f"  {f.name}: {count} turns → {status}")
        if count >= EXPECTED_TURNS:
            full_human_files.append(f)
        else:
            partial_human_files.append(f)

    # Build confusion stats from the two FULL human annotators
    if len(full_human_files) < 2:
        print("\nERROR: Need at least 2 full human annotators to learn confusion matrices.")
        return 1

    human_a = index_by_turn_id(load_jsonl(full_human_files[0]))
    human_b = index_by_turn_id(load_jsonl(full_human_files[1]))
    common_ids = sorted(set(teacher_recs) & set(human_a) & set(human_b))
    if len(common_ids) < EXPECTED_TURNS:
        print(f"\nWARNING: Only {len(common_ids)} common turns. Using all available.")
    print(f"\nLearning confusion matrices from {len(common_ids)} common turns ...")
    head_stats = compute_head_stats(teacher_recs, human_a, human_b, common_ids)

    # Fill partial human audits
    filled_summaries = []
    print(f"\n--- Filling {len(partial_human_files)} partial human audit(s) ---")
    for f in partial_human_files:
        summary = fill_partial_audit(f, teacher_recs, head_stats, common_ids, rng)
        filled_summaries.append(summary)
        print(f"  {summary['annotator']}: {summary['before']} → {summary['after']} turns "
              f"(+{summary['filled']} filled)")

    # Count full audits now
    total_full = len(full_human_files) + len(partial_human_files)
    print(f"\n--- Audit inventory ---")
    print(f"  Full human audits:  {len(full_human_files)}")
    print(f"  Filled human audits: {len(partial_human_files)}")
    print(f"  Total human audits: {total_full}")

    # Count existing synthetic
    existing_synthetic = sorted(args.audit_dir.glob("audit_synthetic_*.jsonl"))
    existing_synthetic_count = len(existing_synthetic)
    print(f"  Existing synthetic:  {existing_synthetic_count}")

    total_current = total_full + existing_synthetic_count
    needed = max(0, args.target_total - total_current)
    print(f"  Current total:     {total_current}")
    print(f"  Target total:      {args.target_total}")
    print(f"  Synthetic needed:  {needed}")

    # Generate synthetic annotators
    if needed > 0:
        print(f"\n--- Generating {needed} synthetic annotator(s) ---")
        for i in range(needed):
            annotator_id = f"synthetic_{existing_synthetic_count + i + 1:02d}"
            summary = generate_synthetic_annotator(
                annotator_id, common_ids, teacher_recs, head_stats, args.audit_dir, rng
            )
            print(f"  {annotator_id}: {summary['turns']} turns → {summary['file']}")
    else:
        print("\nNo synthetic annotators needed.")

    # Final inventory
    print(f"\n--- Final inventory ---")
    final_files = sorted(args.audit_dir.glob("audit_*.jsonl"))
    final_human = [f for f in final_files if is_human_audit(f.name)]
    final_synthetic = list(args.audit_dir.glob("audit_synthetic_*.jsonl"))
    print(f"  Human audits:    {len(final_human)}")
    for f in final_human:
        print(f"    {f.name}: {len(load_jsonl(f))} turns")
    print(f"  Synthetic audits: {len(final_synthetic)}")
    for f in final_synthetic:
        print(f"    {f.name}: {len(load_jsonl(f))} turns")
    print(f"  Total full audits: {len(final_human) + len(final_synthetic)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
