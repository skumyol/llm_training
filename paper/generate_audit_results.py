# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Generate final audit agreement results after two human audits and the AI audit.

Usage:
    uv run generate_audit_results.py \
        --human-a ./audit_results/audit_alice.jsonl \
        --human-b ./audit_results/audit_bob.jsonl \
        --ai ./audit_results/audit_ai_validator.jsonl \
        --teacher ./audit_input_clean.jsonl \
        --output ./audit_results/audit_agreement_final.json \
        --latex-output ./audit_results/audit_table_rows.tex

The output contains:
  - human-teacher agreement, averaged across the two humans
  - human-human inter-annotator agreement
  - AI-teacher agreement
  - AI-human agreement, averaged across the two humans
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


HEADS = [
    "valence",
    "arousal",
    "secrecy_pressure",
    "response_policy",
    "reveal_decision",
    "trust_level",
    "familiarity_level",
    "repair_strategy",
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
    if rec.get("turn_id"):
        return str(rec["turn_id"])
    return f"{rec.get('episode_id', 'unknown')}_{rec.get('turn_number', fallback_index)}"


def index_by_turn_id(path: Path) -> dict[str, dict]:
    records = load_jsonl(path)
    indexed = {}
    duplicates = []
    for i, rec in enumerate(records):
        tid = make_turn_id(rec, i)
        if tid in indexed:
            duplicates.append(tid)
        indexed[tid] = rec
    if duplicates:
        dup_preview = ", ".join(duplicates[:5])
        raise ValueError(f"{path} contains duplicate turn_id values: {dup_preview}")
    return indexed


def safe_accuracy(y_true: list[str], y_pred: list[str]) -> float:
    if not y_true:
        return float("nan")
    return sum(int(a == b) for a, b in zip(y_true, y_pred)) / len(y_true)


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


def compare(
    left: dict[str, dict],
    right: dict[str, dict],
    ids: list[str],
    head: str,
) -> dict:
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


def mean_metric(items: list[dict], key: str) -> float | None:
    vals = [item[key] for item in items if item.get(key) is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 3)


def fmt(value: float | None) -> str:
    return "--" if value is None else f"{value:.2f}"


def compute(human_a: Path, human_b: Path, ai: Path | None, teacher: Path) -> tuple[dict, str]:
    a = index_by_turn_id(human_a)
    b = index_by_turn_id(human_b)
    teacher_recs = index_by_turn_id(teacher)
    ai_recs = index_by_turn_id(ai) if ai else {}

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

        latex_rows.append(
            f"{latex_head:20s} & "
            f"{fmt(ht_acc):>4s} & {fmt(ht_kappa):>4s} & "
            f"{fmt(hh.get('acc')):>4s} & {fmt(hh.get('kappa')):>4s} & "
            f"{fmt(ai_teacher.get('acc') if ai_teacher else None):>4s} & "
            f"{fmt(ai_teacher.get('kappa') if ai_teacher else None):>4s} \\\\"
        )

    return results, "\n".join(latex_rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final audit agreement results")
    parser.add_argument("--human-a", required=True, type=Path, help="Annotator A audit JSONL")
    parser.add_argument("--human-b", required=True, type=Path, help="Annotator B audit JSONL")
    parser.add_argument("--ai", type=Path, default=None, help="AI validator audit JSONL")
    parser.add_argument("--teacher", required=True, type=Path, help="Clean teacher audit packet JSONL")
    parser.add_argument("--output", type=Path, default=Path("audit_agreement_final.json"))
    parser.add_argument("--latex-output", type=Path, default=Path("audit_table_rows.tex"))
    args = parser.parse_args()

    try:
        results, latex_rows = compute(args.human_a, args.human_b, args.ai, args.teacher)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    args.latex_output.parent.mkdir(parents=True, exist_ok=True)
    args.latex_output.write_text(latex_rows, encoding="utf-8")

    print(f"Common human turns: {results['n_common_human']}")
    print(f"Common AI turns:    {results['n_common_ai']}")
    print(f"Saved JSON:         {args.output}")
    print(f"Saved LaTeX rows:   {args.latex_output}")
    print("\n--- LaTeX rows ---")
    print(latex_rows)


if __name__ == "__main__":
    main()
