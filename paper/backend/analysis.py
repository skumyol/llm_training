"""
Analysis utilities extracted from the paper's existing scripts.
Provides QC, agreement computation, evaluator, and result generation.
"""

import json
import math
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, cohen_kappa_score

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


def make_turn_id(rec: dict, fallback_index: int = 0) -> str:
    if "turn_id" in rec and rec["turn_id"]:
        return rec["turn_id"]
    ep = rec.get("episode_id", "unknown")
    tn = rec.get("turn_number", fallback_index)
    return f"{ep}_{tn}"


def index_by_turn_id(records: list[dict]) -> dict[str, dict]:
    return {make_turn_id(r, i): r for i, r in enumerate(records)}


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


def safe_accuracy(y_true: list[str], y_pred: list[str]) -> float:
    if not y_true:
        return 1.0
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


def safe_kappa(y_true: list[str], y_pred: list[str]) -> tuple[float, bool]:
    unique = set(y_true) | set(y_pred)
    degenerate = len(unique) <= 1
    if degenerate:
        return 1.0, True
    labels = sorted(unique)
    n = len(y_true)
    observed = safe_accuracy(y_true, y_pred)
    left_counts = Counter(y_true)
    right_counts = Counter(y_pred)
    expected = sum((left_counts[label] / n) * (right_counts[label] / n) for label in labels)
    if expected == 1.0:
        return 1.0, True
    return (observed - expected) / (1.0 - expected), False


def sklearn_safe_kappa(y_true, y_pred):
    unique = set(y_true) | set(y_pred)
    degenerate = len(unique) <= 1
    if degenerate:
        return 1.0, True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        try:
            return cohen_kappa_score(y_true, y_pred), False
        except Exception:
            return 1.0, True


def sklearn_safe_accuracy(y_true, y_pred):
    unique = set(y_true) | set(y_pred)
    if len(unique) <= 1:
        return 1.0
    return accuracy_score(y_true, y_pred)


# ---------------------------------------------------------------------------
# QC per annotator
# ---------------------------------------------------------------------------
def qc_annotator(records: list[dict], teacher_recs: dict | None = None) -> dict:
    n = len(records)
    if n == 0:
        return {"error": "No records found"}

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
# Agreement computation (sklearn version from compute_audit_agreement.py)
# ---------------------------------------------------------------------------
def compute_agreement(a_path: Path, b_path: Path, teacher_path: Path | None):
    a_recs = index_by_turn_id(load_jsonl(a_path))
    b_recs = index_by_turn_id(load_jsonl(b_path))
    teacher_recs = index_by_turn_id(load_jsonl(teacher_path)) if teacher_path else {}

    common_ids = sorted(set(a_recs.keys()) & set(b_recs.keys()))
    if not common_ids:
        raise ValueError("No common turn IDs between annotators.")

    results = {}
    for head in HEADS:
        a_vals = [a_recs[tid]["labels"][head] for tid in common_ids]
        b_vals = [b_recs[tid]["labels"][head] for tid in common_ids]

        hh_acc = sklearn_safe_accuracy(a_vals, b_vals)
        hh_kappa, hh_degenerate = sklearn_safe_kappa(a_vals, b_vals)

        ht_a_acc = None
        ht_a_kappa = None
        ht_a_degenerate = None
        if teacher_recs:
            teacher_vals_a = []
            a_vals_t = []
            for tid in common_ids:
                if tid in teacher_recs:
                    teacher_vals_a.append(teacher_recs[tid]["labels"][head])
                    a_vals_t.append(a_recs[tid]["labels"][head])
            if teacher_vals_a:
                ht_a_acc = sklearn_safe_accuracy(teacher_vals_a, a_vals_t)
                ht_a_kappa, ht_a_degenerate = sklearn_safe_kappa(teacher_vals_a, a_vals_t)

        ht_b_acc = None
        ht_b_kappa = None
        ht_b_degenerate = None
        if teacher_recs:
            teacher_vals_b = []
            b_vals_t = []
            for tid in common_ids:
                if tid in teacher_recs:
                    teacher_vals_b.append(teacher_recs[tid]["labels"][head])
                    b_vals_t.append(b_recs[tid]["labels"][head])
            if teacher_vals_b:
                ht_b_acc = sklearn_safe_accuracy(teacher_vals_b, b_vals_t)
                ht_b_kappa, ht_b_degenerate = sklearn_safe_kappa(teacher_vals_b, b_vals_t)

        ht_acc = None
        ht_kappa = None
        if ht_a_acc is not None and ht_b_acc is not None:
            ht_acc = (ht_a_acc + ht_b_acc) / 2
            ht_kappa = (ht_a_kappa + ht_b_kappa) / 2
        elif ht_a_acc is not None:
            ht_acc = ht_a_acc
            ht_kappa = ht_a_kappa

        results[head] = {
            "ht_acc": round(ht_acc, 3) if ht_acc is not None else None,
            "ht_kappa": round(ht_kappa, 3) if ht_kappa is not None else None,
            "ht_a_acc": round(ht_a_acc, 3) if ht_a_acc is not None else None,
            "ht_a_kappa": round(ht_a_kappa, 3) if ht_a_kappa is not None else None,
            "ht_a_degenerate": ht_a_degenerate,
            "ht_b_acc": round(ht_b_acc, 3) if ht_b_acc is not None else None,
            "ht_b_kappa": round(ht_b_kappa, 3) if ht_b_kappa is not None else None,
            "ht_b_degenerate": ht_b_degenerate,
            "hh_acc": round(hh_acc, 3),
            "hh_kappa": round(hh_kappa, 3),
            "hh_degenerate": hh_degenerate,
        }
    return results


# ---------------------------------------------------------------------------
# Pairwise agreement (pure Python from evaluate_auditors.py)
# ---------------------------------------------------------------------------
def pairwise_agreement(a_path: Path, b_path: Path):
    a_recs = index_by_turn_id(load_jsonl(a_path))
    b_recs = index_by_turn_id(load_jsonl(b_path))
    common_ids = sorted(set(a_recs.keys()) & set(b_recs.keys()))
    if not common_ids:
        return None, "No common turn IDs"

    results = {}
    for head in HEADS:
        a_vals = [a_recs[tid]["labels"][head] for tid in common_ids]
        b_vals = [b_recs[tid]["labels"][head] for tid in common_ids]
        acc = safe_accuracy(a_vals, b_vals)
        kappa, _ = safe_kappa(a_vals, b_vals)
        results[head] = {"acc": round(acc, 3), "kappa": round(kappa, 3), "n": len(common_ids)}
    return results, None


# ---------------------------------------------------------------------------
# Evaluate all auditors (from evaluate_auditors.py)
# ---------------------------------------------------------------------------
def evaluate_all(audit_dir: Path, teacher_path: Path | None = None):
    jsonl_files = sorted(audit_dir.glob("audit_*.jsonl"))
    annotators = []
    for f in jsonl_files:
        recs = load_jsonl(f)
        if recs:
            annotators.append((f, recs))

    if not annotators:
        raise ValueError("No annotator data found.")

    teacher_recs = index_by_turn_id(load_jsonl(teacher_path)) if teacher_path else None

    qc_reports = []
    for path, recs in annotators:
        name = path.stem
        report = qc_annotator(recs, teacher_recs)
        report["name"] = name
        qc_reports.append(report)

    pairwise = []
    if len(annotators) >= 2:
        for i in range(len(annotators)):
            for j in range(i + 1, len(annotators)):
                path_a, _ = annotators[i]
                path_b, _ = annotators[j]
                label_a = path_a.stem
                label_b = path_b.stem
                results, err = pairwise_agreement(path_a, path_b)
                if err:
                    continue
                avg_acc = sum(v["acc"] for v in results.values()) / len(results)
                avg_kappa = sum(v["kappa"] for v in results.values()) / len(results)
                pairwise.append({
                    "a": label_a,
                    "b": label_b,
                    "per_head": results,
                    "avg_acc": round(avg_acc, 3),
                    "avg_kappa": round(avg_kappa, 3),
                })

    return {"qc_reports": qc_reports, "pairwise": pairwise}


# ---------------------------------------------------------------------------
# Generate final results (from generate_audit_results.py)
# ---------------------------------------------------------------------------
def compare(left: dict[str, dict], right: dict[str, dict], ids: list[str], head: str):
    left_vals = []
    right_vals = []
    missing = 0
    for tid in ids:
        left_label = left.get(tid, {}).get("labels", {}).get(head)
        right_label = right.get(tid, {}).get("labels", {}).get(head)
        if left_label is None or right_label is None:
            missing += 1
            continue
        left_vals.append(left_label)
        right_vals.append(right_label)

    if not left_vals:
        return {"n": 0, "acc": None, "kappa": None, "degenerate": None, "missing": missing}

    kappa, degenerate = safe_kappa(left_vals, right_vals)
    return {
        "n": len(left_vals),
        "acc": round(safe_accuracy(left_vals, right_vals), 3),
        "kappa": round(kappa, 3),
        "degenerate": degenerate,
        "missing": missing,
    }


def mean_metric(items: list[dict], key: str):
    vals = [item[key] for item in items if item.get(key) is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 3)


def generate_final_results(human_a: Path, human_b: Path, ai: Path | None, teacher: Path):
    a = index_by_turn_id(load_jsonl(human_a))
    b = index_by_turn_id(load_jsonl(human_b))
    teacher_recs = index_by_turn_id(load_jsonl(teacher))
    ai_recs = index_by_turn_id(load_jsonl(ai)) if ai else {}

    common_human_ids = sorted(set(a) & set(b) & set(teacher_recs))
    if not common_human_ids:
        raise ValueError("No common turn IDs across human A, human B, and teacher packet.")

    common_ai_ids = sorted(set(ai_recs) & set(teacher_recs)) if ai_recs else []

    results = {
        "inputs": {
            "human_a": str(human_a),
            "human_b": str(human_b),
            "ai": str(ai) if ai else None,
            "teacher": str(teacher),
        },
        "n_common_human": len(common_human_ids),
        "n_common_ai": len(common_ai_ids),
        "heads": {},
    }
    latex_rows = []

    for head in HEADS:
        latex_head = head.replace("_", "\\_")
        ht_a = compare(teacher_recs, a, common_human_ids, head)
        ht_b = compare(teacher_recs, b, common_human_ids, head)
        hh = compare(a, b, common_human_ids, head)

        ht_acc = mean_metric([ht_a, ht_b], "acc")
        ht_kappa = mean_metric([ht_a, ht_b], "kappa")

        ai_teacher = compare(teacher_recs, ai_recs, common_ai_ids, head) if ai_recs else None
        ai_a = compare(a, ai_recs, sorted(set(a) & set(ai_recs)), head) if ai_recs else None
        ai_b = compare(b, ai_recs, sorted(set(b) & set(ai_recs)), head) if ai_recs else None
        ai_human_acc = mean_metric([x for x in [ai_a, ai_b] if x], "acc") if ai_recs else None
        ai_human_kappa = mean_metric([x for x in [ai_a, ai_b] if x], "kappa") if ai_recs else None

        results["heads"][head] = {
            "human_teacher": {
                "acc": ht_acc,
                "kappa": ht_kappa,
                "human_a": ht_a,
                "human_b": ht_b,
            },
            "human_human": hh,
            "ai_teacher": ai_teacher,
            "ai_human": {
                "acc": ai_human_acc,
                "kappa": ai_human_kappa,
                "ai_vs_human_a": ai_a,
                "ai_vs_human_b": ai_b,
            } if ai_recs else None,
        }

        def fmt(value):
            return "--" if value is None else f"{value:.2f}"

        latex_rows.append(
            f"{latex_head:20s} & "
            f"{fmt(ht_acc):>4s} & {fmt(ht_kappa):>4s} & "
            f"{fmt(hh.get('acc')):>4s} & {fmt(hh.get('kappa')):>4s} & "
            f"{fmt(ai_teacher.get('acc') if ai_teacher else None):>4s} & "
            f"{fmt(ai_teacher.get('kappa') if ai_teacher else None):>4s} \\"
        )

    return results, "\n".join(latex_rows) + "\n"
