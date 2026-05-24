# /// script
# requires-python = ">=3.10"
# ///
"""
Evaluate human audit quality: QC per annotator + pairwise agreement.
Pure Python, no sklearn dependency.
"""

import json
import math
import sys
from collections import Counter
from pathlib import Path

HEADS = [
    "valence", "arousal", "secrecy_pressure", "reveal_decision",
    "response_policy", "repair_strategy", "trust_level", "familiarity_level",
]

# ---------------------------------------------------------------------------
# Thresholds (from audit_quality_control.py)
# ---------------------------------------------------------------------------
CRITICAL_MEDIAN_TIME = 52
CRITICAL_MIN_TIME_RATIO = 0.40
CRITICAL_LONGEST_RUN = 8
CRITICAL_EMPTY_NOTES = 1.0
CRITICAL_ENTROPY_RATIO = 0.95

WARN_MEDIAN_TIME = 55
WARN_MIN_TIME_RATIO = 0.25
WARN_LONGEST_RUN = 5
WARN_ENTROPY_RATIO = 0.90

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def _make_turn_id(rec: dict, fallback_index: int = 0) -> str:
    if "turn_id" in rec and rec["turn_id"]:
        return rec["turn_id"]
    ep = rec.get("episode_id", "unknown")
    tn = rec.get("turn_number", fallback_index)
    return f"{ep}_{tn}"

def index_by_turn_id(records: list[dict]) -> dict[str, dict]:
    return {_make_turn_id(r, i): r for i, r in enumerate(records)}

def _entropy(labels: list[str]) -> float:
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
    k = len(set(labels))
    if k <= 1:
        return 0.0
    p = 1.0 / k
    return -k * p * math.log(p)

def _longest_run(labels: list[str]) -> int:
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
    if not times:
        return 0.0
    return sum(1 for t in times if t <= minimum) / len(times)

def accuracy(y_true, y_pred):
    if not y_true:
        return 1.0
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)

def cohen_kappa(y_true, y_pred):
    """Simple Cohen's kappa implementation."""
    n = len(y_true)
    if n == 0:
        return 1.0
    classes = sorted(set(y_true) | set(y_pred))
    if len(classes) <= 1:
        return 1.0

    # Confusion matrix
    matrix = {c1: {c2: 0 for c2 in classes} for c1 in classes}
    for a, b in zip(y_true, y_pred):
        matrix[a][b] += 1

    # Observed agreement
    po = sum(matrix[c][c] for c in classes) / n

    # Expected agreement
    pe = 0.0
    for c in classes:
        row_sum = sum(matrix[c][c2] for c2 in classes)
        col_sum = sum(matrix[c2][c] for c2 in classes)
        pe += (row_sum / n) * (col_sum / n)

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)

# ---------------------------------------------------------------------------
# QC per annotator
# ---------------------------------------------------------------------------
def qc_annotator(records: list[dict], name: str) -> dict:
    n = len(records)
    if n == 0:
        return {"name": name, "error": "No records"}

    times = [r.get("turn_elapsed_seconds", 0) for r in records]
    notes = [r.get("notes", "") for r in records]
    median_time = sorted(times)[len(times) // 2] if times else 0
    min_ratio = _min_time_ratio(times)
    empty_notes_ratio = sum(1 for note in notes if not note.strip()) / n

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
        head_flags[head] = {
            "n": len(labels),
            "entropy": round(ent, 3),
            "entropy_ratio": round(ent_ratio, 3),
            "longest_run": run,
            "distribution": dict(Counter(labels)),
        }

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

    overall = "PASS" if not flags else ("REVIEW" if not any(f.startswith("CRITICAL") for f in flags) else "FAIL")

    return {
        "name": name,
        "n_turns": n,
        "median_turn_time_sec": median_time,
        "min_time_ratio": round(min_ratio, 3),
        "empty_notes_ratio": round(empty_notes_ratio, 3),
        "per_head": head_flags,
        "flags": flags,
        "overall_status": overall,
    }

# ---------------------------------------------------------------------------
# Pairwise agreement
# ---------------------------------------------------------------------------
def pairwise_agreement(a_path: Path, b_path: Path, label_a: str, label_b: str):
    a_recs = index_by_turn_id(load_jsonl(a_path))
    b_recs = index_by_turn_id(load_jsonl(b_path))
    common_ids = sorted(set(a_recs.keys()) & set(b_recs.keys()))
    if not common_ids:
        return None, f"No common turn IDs between {label_a} and {label_b}"

    results = {}
    for head in HEADS:
        a_vals = [a_recs[tid]["labels"][head] for tid in common_ids]
        b_vals = [b_recs[tid]["labels"][head] for tid in common_ids]
        acc = accuracy(a_vals, b_vals)
        kappa = cohen_kappa(a_vals, b_vals)
        results[head] = {"acc": round(acc, 3), "kappa": round(kappa, 3), "n": len(common_ids)}

    return results, None

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    audit_dir = Path("audit_results")

    # Find all non-empty JSONL files
    jsonl_files = sorted(audit_dir.glob("audit_*.jsonl"))
    annotators = []
    for f in jsonl_files:
        recs = load_jsonl(f)
        if recs:
            annotators.append((f, recs))

    if not annotators:
        print("No annotator data found.")
        sys.exit(1)

    # ---- QC Reports ----
    print("=" * 70)
    print("PER-ANNOTATOR QUALITY CONTROL")
    print("=" * 70)

    for path, recs in annotators:
        name = path.stem
        report = qc_annotator(recs, name)
        print(f"\n--- {name} ---")
        print(f"  Turns:          {report['n_turns']}")
        print(f"  Median time:    {report['median_turn_time_sec']}s")
        print(f"  Min-time ratio: {report['min_time_ratio']:.2%}")
        print(f"  Empty notes:    {report['empty_notes_ratio']:.2%}")
        print(f"  Status:         {report['overall_status']}")
        if report.get("flags"):
            print(f"  Flags:")
            for fl in report["flags"]:
                print(f"    - {fl}")
        else:
            print(f"  Flags:          (none)")

        # Per-head distribution summary
        print(f"  Per-head summary:")
        for head, diag in report.get("per_head", {}).items():
            dist = diag.get("distribution", {})
            top = sorted(dist.items(), key=lambda x: -x[1])[:3]
            top_str = ", ".join(f"{k}:{v}" for k, v in top)
            print(f"    {head:20s}  n={diag['n']:>3}  run={diag['longest_run']:>2}  "
                  f"ent_ratio={diag['entropy_ratio']:.3f}  top=({top_str})")

    # ---- Pairwise Agreement ----
    print("\n" + "=" * 70)
    print("PAIRWISE HUMAN-HUMAN AGREEMENT")
    print("=" * 70)

    if len(annotators) >= 2:
        for i in range(len(annotators)):
            for j in range(i + 1, len(annotators)):
                path_a, recs_a = annotators[i]
                path_b, recs_b = annotators[j]
                label_a = path_a.stem
                label_b = path_b.stem
                results, err = pairwise_agreement(path_a, path_b, label_a, label_b)
                if err:
                    print(f"\n{label_a} vs {label_b}: {err}")
                    continue

                common_n = next(iter(results.values()))["n"] if results else 0
                print(f"\n{label_a}  vs  {label_b}  (common turns: {common_n})")
                print(f"{'Head':22s}  {'HH_acc':>7s}  {'HH_kappa':>9s}")
                print("-" * 42)
                for head, vals in results.items():
                    print(f"{head:22s}  {vals['acc']:>7.3f}  {vals['kappa']:>9.3f}")

                # Average across heads
                avg_acc = sum(v["acc"] for v in results.values()) / len(results)
                avg_kappa = sum(v["kappa"] for v in results.values()) / len(results)
                print("-" * 42)
                print(f"{'AVERAGE':22s}  {avg_acc:>7.3f}  {avg_kappa:>9.3f}")
    else:
        print("\nOnly one annotator — skipping pairwise agreement.")

    # ---- Final Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for path, recs in annotators:
        name = path.stem
        report = qc_annotator(recs, name)
        status = report["overall_status"]
        n = report["n_turns"]
        med = report["median_turn_time_sec"]
        print(f"  {name:45s}  turns={n:>4}  median_time={med:>5}s  status={status}")


if __name__ == "__main__":
    main()