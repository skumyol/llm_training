# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "scikit-learn>=1.3",
# ]
# ///
"""
Quality-control diagnostics for a single annotator's audit output.

Flags random or careless responding by analysing timing, response patterns,
and agreement with teacher labels.

Usage:
    uv run audit_quality_control.py \
        --annotator ./audit_results/alice.jsonl \
        --teacher /path/to/test_heads.jsonl \
        --output qc_alice.json

Exit codes:
    0  = no critical flags
    1  = at least one critical flag raised (review before paying)
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

from compute_audit_agreement import HEADS, _make_turn_id, index_by_turn_id, load_jsonl


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------
CRITICAL_MEDIAN_TIME = 52      # median turn time <= 52s → rushed
CRITICAL_MIN_TIME_RATIO = 0.40 # ≥40% of turns at exactly the 50s minimum
CRITICAL_LONGEST_RUN = 8       # streak of 8+ identical answers → suspicious
CRITICAL_EMPTY_NOTES = 1.0     # 100% empty notes → suspicious
CRITICAL_ENTROPY_RATIO = 0.95  # entropy / max_entropy ≥ 0.95 → too uniform (only for N ≥ 20)
CRITICAL_TEACHER_ACC = 0.25    # teacher accuracy ≤ 0.25 → near chance or worse

WARN_MEDIAN_TIME = 55
WARN_MIN_TIME_RATIO = 0.25
WARN_LONGEST_RUN = 5
WARN_ENTROPY_RATIO = 0.90      # only for N ≥ 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _entropy(labels: list[str]) -> float:
    """Shannon entropy (base e) of a categorical distribution."""
    if not labels:
        return 0.0
    counts = Counter(labels)
    n = len(labels)
    h = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            h -= p * math.log(p)
    return h


def _max_entropy(labels: list[str]) -> float:
    """Maximum possible entropy for the observed number of distinct classes."""
    k = len(set(labels))
    if k <= 1:
        return 0.0
    p = 1.0 / k
    return -k * p * math.log(p)


def _longest_run(labels: list[str]) -> int:
    """Longest consecutive run of the same label."""
    if not labels:
        return 0
    best = 1
    curr = 1
    for i in range(1, len(labels)):
        if labels[i] == labels[i - 1]:
            curr += 1
            best = max(best, curr)
        else:
            curr = 1
    return best


def _min_time_ratio(times: list[int], minimum: int = 50) -> float:
    """Fraction of turns whose elapsed time is exactly the minimum."""
    if not times:
        return 0.0
    return sum(1 for t in times if t <= minimum) / len(times)


# ---------------------------------------------------------------------------
# Per-annotator QC
# ---------------------------------------------------------------------------
def qc_annotator(records: list[dict], teacher_recs: dict | None = None) -> dict:
    """Compute quality-control metrics for a single annotator."""
    n = len(records)
    if n == 0:
        return {"error": "No records found"}

    times = [r.get("turn_elapsed_seconds", 0) for r in records]
    notes = [r.get("notes", "") for r in records]
    median_time = sorted(times)[len(times) // 2] if times else 0
    min_ratio = _min_time_ratio(times)
    empty_notes_ratio = sum(1 for note in notes if not note.strip()) / n

    # Per-head diagnostics
    head_flags = {}
    for head in HEADS:
        labels = [r["labels"].get(head) for r in records if head in r.get("labels", {})]
        labels = [l for l in labels if l is not None]
        if not labels:
            continue

        ent = _entropy(labels)
        max_ent = _max_entropy(labels)
        ent_ratio = ent / max_ent if max_ent > 0 else 0.0
        run = _longest_run(labels)

        # Teacher agreement (if available)
        teacher_acc = None
        if teacher_recs:
            matched = 0
            total = 0
            for r in records:
                tid = r.get("turn_id")
                if tid and tid in teacher_recs:
                    t_label = teacher_recs[tid].get("labels", {}).get(head)
                    a_label = r.get("labels", {}).get(head)
                    if t_label is not None and a_label is not None:
                        matched += int(t_label == a_label)
                        total += 1
            teacher_acc = matched / total if total > 0 else None

        head_flags[head] = {
            "n": len(labels),
            "entropy": round(ent, 3),
            "entropy_ratio": round(ent_ratio, 3),
            "longest_run": run,
            "teacher_acc": round(teacher_acc, 3) if teacher_acc is not None else None,
            "distribution": dict(Counter(labels)),
        }

    # Composite flags
    flags = []
    if median_time <= CRITICAL_MEDIAN_TIME:
        flags.append(f"CRITICAL: median turn time {median_time}s (rushed)")
    elif median_time <= WARN_MEDIAN_TIME:
        flags.append(f"WARN: median turn time {median_time}s")

    if min_ratio >= CRITICAL_MIN_TIME_RATIO:
        flags.append(f"CRITICAL: {min_ratio:.0%} of turns at minimum time")
    elif min_ratio >= WARN_MIN_TIME_RATIO:
        flags.append(f"WARN: {min_ratio:.0%} of turns at minimum time")

    if empty_notes_ratio >= CRITICAL_EMPTY_NOTES:
        flags.append("CRITICAL: 100% empty notes")

    # Per-head pattern flags
    for head, diag in head_flags.items():
        if diag["longest_run"] >= CRITICAL_LONGEST_RUN:
            flags.append(f"CRITICAL: {head} has run of {diag['longest_run']} identical answers")
        elif diag["longest_run"] >= WARN_LONGEST_RUN:
            flags.append(f"WARN: {head} has run of {diag['longest_run']} identical answers")

        if diag["n"] >= 20:
            if diag["entropy_ratio"] >= CRITICAL_ENTROPY_RATIO:
                flags.append(f"CRITICAL: {head} responses too uniform (entropy_ratio={diag['entropy_ratio']:.2f})")
            elif diag["entropy_ratio"] >= WARN_ENTROPY_RATIO:
                flags.append(f"WARN: {head} responses unusually uniform (entropy_ratio={diag['entropy_ratio']:.2f})")

        if diag["teacher_acc"] is not None and diag["teacher_acc"] <= CRITICAL_TEACHER_ACC:
            flags.append(f"CRITICAL: {head} teacher accuracy {diag['teacher_acc']:.2f} (near chance)")

    overall_status = "PASS" if not flags else ("REVIEW" if not any(f.startswith("CRITICAL") for f in flags) else "FAIL")

    return {
        "n_turns": n,
        "median_turn_time_sec": median_time,
        "min_time_ratio": round(min_ratio, 3),
        "empty_notes_ratio": round(empty_notes_ratio, 3),
        "per_head": head_flags,
        "flags": flags,
        "overall_status": overall_status,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Audit quality control")
    parser.add_argument("--annotator", required=True, help="Annotator JSONL file")
    parser.add_argument("--teacher", default=None, help="Teacher labels JSONL for comparison")
    parser.add_argument("--output", default="qc_report.json", help="Output JSON file")
    args = parser.parse_args()

    records = load_jsonl(Path(args.annotator))
    teacher_recs = index_by_turn_id(load_jsonl(Path(args.teacher))) if args.teacher else None

    report = qc_annotator(records, teacher_recs)

    print(f"\n--- QC Report: {Path(args.annotator).name} ---")
    print(f"Turns:          {report['n_turns']}")
    print(f"Median time:    {report['median_turn_time_sec']}s")
    print(f"Min-time ratio: {report['min_time_ratio']:.2%}")
    print(f"Empty notes:    {report['empty_notes_ratio']:.2%}")
    print(f"Status:         {report['overall_status']}")
    if report["flags"]:
        print("\nFlags:")
        for f in report["flags"]:
            print(f"  - {f}")
    else:
        print("\nNo flags raised.")

    with open(args.output, "w", encoding="utf-8") as out:
        json.dump(report, out, indent=2, ensure_ascii=False)
    print(f"\nSaved report to {args.output}")

    sys.exit(0 if report["overall_status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
