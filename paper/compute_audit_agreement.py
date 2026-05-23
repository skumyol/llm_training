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
import warnings
from pathlib import Path

from sklearn.metrics import accuracy_score, cohen_kappa_score


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


def _make_turn_id(rec: dict, fallback_index: int = 0) -> str:
    """Return turn_id if present, otherwise construct from episode_id + turn_number."""
    if "turn_id" in rec and rec["turn_id"]:
        return rec["turn_id"]
    ep = rec.get("episode_id", "unknown")
    tn = rec.get("turn_number", fallback_index)
    return f"{ep}_{tn}"


def index_by_turn_id(records: list[dict]) -> dict[str, dict]:
    return {_make_turn_id(r, i): r for i, r in enumerate(records)}


def _safe_kappa(y_true, y_pred):
    """Cohen's kappa that flags degenerate (single-class) cases."""
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


def _safe_accuracy(y_true, y_pred):
    """Accuracy that avoids sklearn warnings for single-class data."""
    unique = set(y_true) | set(y_pred)
    if len(unique) <= 1:
        return 1.0
    return accuracy_score(y_true, y_pred)


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

        hh_acc = _safe_accuracy(a_vals, b_vals)
        hh_kappa, hh_degenerate = _safe_kappa(a_vals, b_vals)

        # Human-teacher agreement for annotator A
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
                ht_a_acc = _safe_accuracy(teacher_vals_a, a_vals_t)
                ht_a_kappa, ht_a_degenerate = _safe_kappa(teacher_vals_a, a_vals_t)

        # Human-teacher agreement for annotator B
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
                ht_b_acc = _safe_accuracy(teacher_vals_b, b_vals_t)
                ht_b_kappa, ht_b_degenerate = _safe_kappa(teacher_vals_b, b_vals_t)

        # Average HT (only if both are available)
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

        ht_a = f"{ht_acc:.2f}" if ht_acc is not None else "--"
        ht_k = f"{ht_kappa:.2f}" if ht_kappa is not None else "--"
        deg_mark = "*" if (hh_degenerate or ht_a_degenerate or ht_b_degenerate) else " "
        latex_rows.append(
            f"{head:20s} & {ht_a:>6s} & {ht_k:>6s} & {hh_acc:.2f}  & {hh_kappa:.2f}{deg_mark} \\"
        )

    print("\n--- Per-head agreement ---")
    for head, vals in results.items():
        print(f"{head:20s}  HT_acc={vals['ht_acc']}  HT_k={vals['ht_kappa']}  |  "
              f"HH_acc={vals['hh_acc']}  HH_k={vals['hh_kappa']}")
        if vals["hh_degenerate"]:
            print(f"  {'':20s}  NOTE: HH kappa is degenerate (single class).")

    print("\n--- LaTeX table rows ---")
    print("\n".join(latex_rows))
    print("\n* = degenerate (single-class) kappa")

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
