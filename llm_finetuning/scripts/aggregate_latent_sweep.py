#!/usr/bin/env python3
"""
aggregate_latent_sweep.py — collect latent sweep results into one table.

Reads each run's training summary (validation, which selected the checkpoint)
and, when present, its held-out test evaluation. Test is the number to quote;
validation is shown only to make the selection story visible.

Usage:
    python llm_finetuning/scripts/aggregate_latent_sweep.py \
        --runs L1_control L2_nosampler L3_meanpool L4_ctx1024 \
        --eval-dir eval_results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--ckpt-root", default="checkpoints")
    ap.add_argument("--eval-dir", default="eval_results")
    args = ap.parse_args()

    rows = []
    for run in args.runs:
        val = _load(Path(args.ckpt_root) / run / "metrics" / "latent_training_summary.json")
        test = _load(Path(args.eval_dir) / f"test_{run}" / "latent_eval_metrics.json")
        vs = (val or {}).get("final_val_summary", {}) or {}
        ts = (test or {}).get("summary", {}) or {}
        rows.append({
            "run": run,
            "val_macro_f1": vs.get("mean_macro_f1"),
            "val_acc": vs.get("mean_accuracy"),
            "val_rp": vs.get("response_policy_f1"),
            "best_metric": (val or {}).get("best_metric_value"),
            "test_macro_f1": ts.get("mean_macro_f1"),
            "test_acc": ts.get("mean_accuracy"),
            "test_rp": ts.get("response_policy_f1"),
            "test_stance": ts.get("stance_delta_accuracy"),
        })

    def fmt(v, w=8, p=4):
        return f"{v:{w}.{p}f}" if isinstance(v, (int, float)) else " " * (w - 3) + "n/a"

    print(f"\n{'run':20s} {'val_mF1':>8s} {'val_acc':>8s} {'val_rp':>8s} | "
          f"{'test_mF1':>8s} {'test_acc':>8s} {'test_rp':>8s} {'test_stance':>11s}")
    print("-" * 92)
    for r in rows:
        print(f"{r['run']:20s} {fmt(r['val_macro_f1'])} {fmt(r['val_acc'])} {fmt(r['val_rp'])} | "
              f"{fmt(r['test_macro_f1'])} {fmt(r['test_acc'])} {fmt(r['test_rp'])} {fmt(r['test_stance'], 11)}")

    have_test = [r for r in rows if isinstance(r.get("test_rp"), float)]
    if len(have_test) > 1:
        base = next((r for r in have_test if "L1" in r["run"]), have_test[0])
        print(f"\nrelative to {base['run']} on test:")
        for r in have_test:
            if r is base:
                continue
            for key, label in (("test_macro_f1", "mean macro-F1"), ("test_rp", "response_policy")):
                if isinstance(r.get(key), float) and isinstance(base.get(key), float) and base[key]:
                    d = (r[key] - base[key]) / base[key] * 100
                    print(f"  {r['run']:20s} {label:16s} {d:+6.1f}%")
    print()


if __name__ == "__main__":
    main()
