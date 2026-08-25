"""Rescore OpenRouter zero-shot eval on a subset of the test split.

Re-runs the full metric computation (all 28 fields) on only the records
that are NOT counterfactual, so the denominator matches the fine-tuned
models which were evaluated on 358 originals.

Usage:
    python llm_finetuning/scripts/rescore_origins.py \
        --per-record eval_results/test_openrouter_qwen38_27b/per_record.jsonl \
        --test-file data/splits/test_heads.jsonl \
        --output-dir eval_results/test_openrouter_qwen38_27b_orig
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Reuse the same label maps and metrics from eval_openrouter
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_openrouter import (  # noqa: E402
    LABEL_MAPS,
    LABEL_TO_IDX,
    FIELD_ORDER,
    parse,
    to_indices,
    compute_metrics,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-record", required=True, help="per_record.jsonl from the full eval")
    ap.add_argument("--test-file", default="data/splits/test_heads.jsonl")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--filter",
        default="originals",
        choices=["originals", "all"],
        help="originals = non-counterfactual only; all = all 884",
    )
    args = ap.parse_args()

    # Load test records
    records: list[dict] = []
    with open(args.test_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    # Load per-record results (has idx, parsed fields info)
    per_record: dict[int, dict] = {}
    with open(args.per_record) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            per_record[r["idx"]] = r

    # Filter to originals
    if args.filter == "originals":
        kept = [(i, r) for i, r in enumerate(records) if not r.get("counterfactual")]
    else:
        kept = list(enumerate(records))

    print(f"Scoring {len(kept)} records ({args.filter})")

    # We need the full parsed output, but per_record only has response_policy.
    # We need to re-parse from the raw samples — but we don't have raw for all.
    # Instead, we can only rescore response_policy from per_record.
    # For full field scoring, we need to re-run the eval.
    #
    # BUT: the per_record file has response_policy gold and pred for every
    # record. We can at least rescore response_policy on the 358 originals.
    # For the full 28-field mean_accuracy, we'd need all fields.
    #
    # Let's do what we can: rescore response_policy from per_record.

    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        balanced_accuracy_score,
        matthews_corrcoef,
    )

    rp_golds = []
    rp_preds = []
    for i, rec in kept:
        if i not in per_record:
            continue
        pr = per_record[i]
        gold = pr.get("response_policy_gold")
        pred = pr.get("response_policy_pred", "")

        # Map to indices using the same logic as the eval script
        g = LABEL_TO_IDX["response_policy"].get(str(gold), -1)
        if g == -1:
            continue  # skip invalid gold (e.g. "defect", "hint")

        # Pred: if empty or invalid, assign len(classes) = 10 (always wrong)
        p = LABEL_TO_IDX["response_policy"].get(str(pred), len(LABEL_MAPS["response_policy"]))

        rp_golds.append(g)
        rp_preds.append(p)

    n = len(rp_golds)
    gold_labels = sorted(set(rp_golds))

    rp_acc = accuracy_score(rp_golds, rp_preds)
    rp_macro_f1 = f1_score(rp_golds, rp_preds, average="macro", labels=gold_labels, zero_division=0)
    rp_weighted_f1 = f1_score(rp_golds, rp_preds, average="weighted", zero_division=0)
    rp_balanced_acc = balanced_accuracy_score(rp_golds, rp_preds)
    rp_mcc = matthews_corrcoef(rp_golds, rp_preds) if len(gold_labels) > 1 else 0.0

    print(f"\n{'='*60}")
    print(f"RESPONSE POLICY ({args.filter}, n={n})")
    print(f"{'='*60}")
    print(f"accuracy         {rp_acc:.4f}")
    print(f"macro_f1         {rp_macro_f1:.4f}")
    print(f"weighted_f1      {rp_weighted_f1:.4f}")
    print(f"balanced_acc     {rp_balanced_acc:.4f}")
    print(f"mcc              {rp_mcc:.4f}")

    # Per-class breakdown
    from collections import Counter
    gold_counts = Counter(rp_golds)
    pred_counts = Counter(rp_preds)
    classes = LABEL_MAPS["response_policy"]
    print(f"\n{'class':<15} {'gold':>5} {'pred':>5} {'F1':>7}")
    print("-" * 35)
    for ci, cname in enumerate(classes):
        g = gold_counts.get(ci, 0)
        p = pred_counts.get(ci, 0)
        # F1 for this class
        tp = sum(1 for gg, pp in zip(rp_golds, rp_preds) if gg == ci and pp == ci)
        prec = tp / p if p > 0 else 0
        rec = tp / g if g > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"{cname:<15} {g:>5} {p:>5} {f1:>7.4f}")

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "filter": args.filter,
        "n_records": n,
        "response_policy": {
            "accuracy": rp_acc,
            "macro_f1": rp_macro_f1,
            "weighted_f1": rp_weighted_f1,
            "balanced_accuracy": rp_balanced_acc,
            "mcc": rp_mcc,
        },
    }
    (out_dir / "rescore_rp.json").write_text(json.dumps(result, indent=2))
    print(f"\nWrote {out_dir}/rescore_rp.json")


if __name__ == "__main__":
    main()
