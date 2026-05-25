# /// script
# requires-python = ">=3.10"
# ///
"""
Evaluate human audit quality: QC per annotator + pairwise agreement.
Pure Python, no sklearn dependency.

Handles:
  - Full annotators (150 turns)
  - Partial annotators (incomplete sessions)
  - AI validator as reference
  - Missing turn coverage analysis
"""

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

HEADS = [
    "valence", "arousal", "secrecy_pressure", "reveal_decision",
    "response_policy", "repair_strategy", "trust_level", "familiarity_level",
]

# Skip these — test/demo entries, not real annotators
SKIP_PREFIXES = ("audit_test", "audit_DEMO", "audit_annotator")

# ---------------------------------------------------------------------------
# Thresholds (from audit_quality_control.py)
# ---------------------------------------------------------------------------
CRITICAL_MEDIAN_TIME = 52
CRITICAL_MIN_TIME_RATIO = 0.40
CRITICAL_LONGEST_RUN = 8
CRITICAL_EMPTY_NOTES = 1.0
CRITICAL_ENTROPY_RATIO = 0.95
CRITICAL_TEACHER_ACC = 0.25

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

    matrix = {c1: {c2: 0 for c2 in classes} for c1 in classes}
    for a, b in zip(y_true, y_pred):
        matrix[a][b] += 1

    po = sum(matrix[c][c] for c in classes) / n
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
def qc_annotator(records: list[dict], name: str, teacher_recs: dict | None = None) -> dict:
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

        # Teacher agreement
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
        if diag["teacher_acc"] is not None and diag["teacher_acc"] <= CRITICAL_TEACHER_ACC:
            flags.append(f"CRITICAL: {head} teacher accuracy {diag['teacher_acc']:.2f} (near chance)")

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
def pairwise_agreement(a_recs: dict, b_recs: dict, label_a: str, label_b: str):
    common_ids = sorted(set(a_recs.keys()) & set(b_recs.keys()))
    if not common_ids:
        return None, f"No common turn IDs between {label_a} and {label_b}"

    results = {}
    for head in HEADS:
        a_vals = [a_recs[tid]["labels"].get(head) for tid in common_ids]
        b_vals = [b_recs[tid]["labels"].get(head) for tid in common_ids]
        # Filter out None values (missing labels)
        pairs = [(a, b) for a, b in zip(a_vals, b_vals) if a is not None and b is not None]
        if not pairs:
            results[head] = {"acc": None, "kappa": None, "n": 0}
            continue
        a_vals_f, b_vals_f = zip(*pairs)
        acc = accuracy(a_vals_f, b_vals_f)
        kappa = cohen_kappa(a_vals_f, b_vals_f)
        results[head] = {"acc": round(acc, 3), "kappa": round(kappa, 3), "n": len(pairs)}

    return results, None


# ---------------------------------------------------------------------------
# Missing turns analysis
# ---------------------------------------------------------------------------
def analyze_missing_turns(annotators: list[tuple[str, dict]], total_tids: set[str]):
    """Analyze which turns are missing from which annotators."""
    print(f"\nTotal unique turn_ids across all annotators: {len(total_tids)}")

    # Turn coverage: how many annotators labeled each turn
    turn_coverage = defaultdict(list)
    for name, recs in annotators:
        for tid in recs:
            turn_coverage[tid].append(name)

    # Turns labeled by all human annotators
    human_names = [n for n, _ in annotators if n != "ai_validator"]
    all_human_tids = set(tid for tid, names in turn_coverage.items()
                         if all(h in names for h in human_names))
    print(f"Turns labeled by ALL human annotators: {len(all_human_tids)}")

    # Coverage distribution
    cov_counts = Counter(len(names) for names in turn_coverage.values())
    print(f"Coverage distribution (num_annotators: count_of_turns):")
    for k in sorted(cov_counts):
        who = "human+AI" if k == len(annotators) else f"{k} annotators"
        print(f"  {k} annotators: {cov_counts[k]} turns ({who})")

    # Missing per annotator
    print(f"\nMissing turns per annotator (vs. total {len(total_tids)}):")
    for name, recs in annotators:
        covered = set(recs.keys())
        missing = total_tids - covered
        pct = len(covered) / len(total_tids) * 100 if total_tids else 0
        print(f"  {name:45s}  covered={len(covered):>4}  missing={len(missing):>4}  coverage={pct:.0f}%")

    return all_human_tids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    audit_dir = Path("audit_results")

    # Find all non-empty JSONL files (skip test/demo)
    jsonl_files = sorted(audit_dir.glob("audit_*.jsonl"))
    annotator_files = []
    for f in jsonl_files:
        if any(f.name.startswith(p) for p in SKIP_PREFIXES):
            continue
        recs = load_jsonl(f)
        if recs:
            annotator_files.append((f, recs))

    if not annotator_files:
        print("No annotator data found.")
        sys.exit(1)

    # Separate AI validator from humans
    ai_path = None
    ai_recs = None
    human_files = []
    for f, recs in annotator_files:
        name = f.stem.replace("audit_", "")
        pid = recs[0].get("prolific_pid", "")
        if name == "ai_validator" or pid == "":
            ai_path = f
            ai_recs = index_by_turn_id(recs)
        else:
            human_files.append((f, recs))

    # Index all human annotators by turn_id
    human_indexed = []
    all_tids = set()
    for f, recs in human_files:
        name = f.stem.replace("audit_", "")
        pid = recs[0].get("prolific_pid", name)
        indexed = index_by_turn_id(recs)
        human_indexed.append((pid, indexed))
        all_tids.update(indexed.keys())

    if ai_recs:
        all_tids.update(ai_recs.keys())

    # ---- Missing Turns Analysis ----
    print("=" * 70)
    print("MISSING TURNS ANALYSIS")
    print("=" * 70)
    all_indexed = [(pid, recs) for pid, recs in human_indexed]
    if ai_recs:
        all_indexed.append(("ai_validator", ai_recs))
    all_human_tids = analyze_missing_turns(all_indexed, all_tids)

    # ---- QC Reports ----
    print("\n" + "=" * 70)
    print("PER-ANNOTATOR QUALITY CONTROL")
    print("=" * 70)

    qc_reports = {}
    for f, recs in human_files:
        name = f.stem.replace("audit_", "")
        pid = recs[0].get("prolific_pid", name)
        report = qc_annotator(recs, pid, teacher_recs=ai_recs)
        qc_reports[pid] = report

        print(f"\n--- {pid} ---")
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
            ta = f"  HT_acc={diag['teacher_acc']:.3f}" if diag.get("teacher_acc") is not None else ""
            print(f"    {head:20s}  n={diag['n']:>3}  run={diag['longest_run']:>2}  "
                  f"ent_ratio={diag['entropy_ratio']:.3f}{ta}  top=({top_str})")

    # ---- Pairwise Human-Human Agreement ----
    print("\n" + "=" * 70)
    print("PAIRWISE HUMAN-HUMAN AGREEMENT")
    print("=" * 70)

    hh_pairs = []
    if len(human_indexed) >= 2:
        for i in range(len(human_indexed)):
            for j in range(i + 1, len(human_indexed)):
                pid_a, recs_a = human_indexed[i]
                pid_b, recs_b = human_indexed[j]
                results, err = pairwise_agreement(recs_a, recs_b, pid_a, pid_b)
                if err:
                    print(f"\n{pid_a} vs {pid_b}: {err}")
                    continue

                common_n = next(iter(results.values()))["n"] if results else 0
                print(f"\n{pid_a}  vs  {pid_b}  (common turns: {common_n})")
                print(f"{'Head':22s}  {'HH_acc':>7s}  {'HH_kappa':>9s}")
                print("-" * 42)
                acc_vals = []
                kappa_vals = []
                for head, vals in results.items():
                    acc_str = f"{vals['acc']:>7.3f}" if vals['acc'] is not None else "    --"
                    kappa_str = f"{vals['kappa']:>9.3f}" if vals['kappa'] is not None else "      --"
                    print(f"{head:22s}  {acc_str}  {kappa_str}")
                    if vals['acc'] is not None:
                        acc_vals.append(vals['acc'])
                    if vals['kappa'] is not None:
                        kappa_vals.append(vals['kappa'])

                # Average across heads
                avg_acc = sum(acc_vals) / len(acc_vals) if acc_vals else 0.0
                avg_kappa = sum(kappa_vals) / len(kappa_vals) if kappa_vals else 0.0
                print("-" * 42)
                print(f"{'AVERAGE':22s}  {avg_acc:>7.3f}  {avg_kappa:>9.3f}")
                hh_pairs.append((pid_a, pid_b, avg_acc, avg_kappa, common_n))

    # ---- Human vs AI Validator Agreement ----
    if ai_recs:
        print("\n" + "=" * 70)
        print("HUMAN vs AI VALIDATOR AGREEMENT (HT)")
        print("=" * 70)

        for pid, recs in human_indexed:
            results, err = pairwise_agreement(recs, ai_recs, pid, "ai_validator")
            if err:
                print(f"\n{pid} vs AI: {err}")
                continue

            common_n = next(iter(results.values()))["n"] if results else 0
            print(f"\n{pid}  vs  AI validator  (common turns: {common_n})")
            print(f"{'Head':22s}  {'HT_acc':>7s}  {'HT_kappa':>9s}")
            print("-" * 42)
            for head, vals in results.items():
                print(f"{head:22s}  {vals['acc']:>7.3f}  {vals['kappa']:>9.3f}")

            avg_acc = sum(v["acc"] for v in results.values()) / len(results)
            avg_kappa = sum(v["kappa"] for v in results.values()) / len(results)
            print("-" * 42)
            print(f"{'AVERAGE':22s}  {avg_acc:>7.3f}  {avg_kappa:>9.3f}")

    # ---- Final Summary ----
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"{'Annotator':45s}  {'Turns':>5s}  {'MedTime':>7s}  {'MinRatio':>8s}  {'EmptyNt':>7s}  {'Status':>6s}")
    print("-" * 85)
    for pid, report in sorted(qc_reports.items(), key=lambda x: -x[1]["n_turns"]):
        n = report["n_turns"]
        med = report["median_turn_time_sec"]
        mr = report["min_time_ratio"]
        en = report["empty_notes_ratio"]
        st = report["overall_status"]
        print(f"  {pid:43s}  {n:>5}  {med:>5}s  {mr:>7.1%}  {en:>6.1%}  {st}")

    # HH agreement summary
    if hh_pairs:
        print(f"\nHuman-Human Agreement Summary (sorted by avg kappa):")
        print(f"{'Pair':>50s}  {'Common':>7s}  {'AvgAcc':>7s}  {'AvgKappa':>9s}")
        print("-" * 78)
        for pid_a, pid_b, avg_acc, avg_kappa, common_n in sorted(hh_pairs, key=lambda x: -x[3]):
            pair_label = f"{pid_a} vs {pid_b}"
            print(f"  {pair_label:48s}  {common_n:>7}  {avg_acc:>7.3f}  {avg_kappa:>9.3f}")

    # Save full report
    output_path = audit_dir / "evaluation_report.json"
    full_report = {
        "qc_reports": qc_reports,
        "hh_pairs": [
            {"a": a, "b": b, "avg_acc": round(aa, 3), "avg_kappa": round(ak, 3), "common_turns": cn}
            for a, b, aa, ak, cn in hh_pairs
        ],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    print(f"\nFull report saved to {output_path}")


if __name__ == "__main__":
    main()
