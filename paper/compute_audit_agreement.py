# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "scikit-learn>=1.3",
# ]
# ///
"""
Compute human audit agreement statistics from two (or more) annotator JSONL files.

Usage:
    uv run compute_audit_agreement.py \
        --a ./audit_results/alice.jsonl \
        --b ./audit_results/bob.jsonl \
        --teacher /path/to/test_heads.jsonl \
        --output agreement.json

Outputs a JSON file with per-head HT and HH statistics, plus a LaTeX table snippet.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix


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


def _safe_kappa(y_true, y_pred, labels=None):
    """Cohen's kappa that returns 1.0 when all labels are identical (no variance)."""
    try:
        k = cohen_kappa_score(y_true, y_pred, labels=labels)
        if k != k:  # NaN check
            return 1.0
        return k
    except Exception:
        return 1.0


def index_by_turn_id(records: list[dict]) -> dict[str, dict]:
    return {r["turn_id"]: r for r in records}


def compute(a_path: Path, b_path: Path, teacher_path: Path | None):
    a_recs = index_by_turn_id(load_jsonl(a_path))
    b_recs = index_by_turn_id(load_jsonl(b_path))
    teacher_recs = index_by_turn_id(load_jsonl(teacher_path)) if teacher_path else {}

    common_ids = sorted(set(a_recs.keys()) & set(b_recs.keys()))
    if not common_ids:
        print("No common turn IDs between annotators.")
        sys.exit(1)

    results = {}
    latex_rows = []

    for head in HEADS:
        a_vals = [a_recs[tid]["labels"][head] for tid in common_ids]
        b_vals = [b_recs[tid]["labels"][head] for tid in common_ids]

        all_labels = sorted(set(a_vals) | set(b_vals))
        hh_acc = accuracy_score(a_vals, b_vals)
        hh_kappa = _safe_kappa(a_vals, b_vals, labels=all_labels)

        ht_acc = None
        ht_kappa = None
        if teacher_recs:
            teacher_vals = []
            a_vals_t = []
            for tid in common_ids:
                if tid in teacher_recs:
                    teacher_vals.append(teacher_recs[tid]["labels"][head])
                    a_vals_t.append(a_recs[tid]["labels"][head])
            if teacher_vals:
                all_labels_t = sorted(set(teacher_vals) | set(a_vals_t))
                ht_acc = accuracy_score(teacher_vals, a_vals_t)
                ht_kappa = _safe_kappa(teacher_vals, a_vals_t, labels=all_labels_t)

        results[head] = {
            "ht_acc": round(ht_acc, 3) if ht_acc is not None else None,
            "ht_kappa": round(ht_kappa, 3) if ht_kappa is not None else None,
            "hh_acc": round(hh_acc, 3),
            "hh_kappa": round(hh_kappa, 3),
        }

        ht_a = f"{ht_acc:.2f}" if ht_acc is not None else "--"
        ht_k = f"{ht_kappa:.2f}" if ht_kappa is not None else "--"
        latex_rows.append(
            f"{head:20s} & {ht_a:>6s} & {ht_k:>6s} & {hh_acc:.2f}  & {hh_kappa:.2f} \\\\"
        )

    print("\n--- Per-head agreement ---")
    for head, vals in results.items():
        print(f"{head:20s}  HT acc={vals['ht_acc']} k={vals['ht_kappa']}  |  HH acc={vals['hh_acc']} k={vals['hh_kappa']}")

    print("\n--- LaTeX table rows ---")
    print("\n".join(latex_rows))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="Annotator A JSONL")
    parser.add_argument("--b", required=True, help="Annotator B JSONL")
    parser.add_argument("--teacher", default=None, help="Teacher labels JSONL (test_heads.jsonl)")
    parser.add_argument("--output", default="agreement.json", help="Output JSON file")
    args = parser.parse_args()

    results = compute(Path(args.a), Path(args.b), Path(args.teacher) if args.teacher else None)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
