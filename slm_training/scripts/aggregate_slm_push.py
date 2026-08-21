#!/usr/bin/env python3
"""
aggregate_slm_push.py — collect slm_push run summaries into a comparison table.

Reports val (selection) and test (held-out) perplexity and bits-per-character,
aggregated across seeds as mean +/- population std.

val_ppl chose the checkpoint, so it is optimistically biased; test_ppl is the
number to quote. bpc is tokenizer-invariant and stays comparable if the
vocabulary ever changes.

Usage:
    python scripts/aggregate_slm_push.py [--dir artifacts/slm_push] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path


def load_runs(root: Path) -> list[dict]:
    runs = []
    for summary in sorted(root.glob("*/run_summary.json")):
        try:
            d = json.loads(summary.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        best, test = d.get("best") or {}, d.get("test") or {}
        if not best:
            continue                      # still training
        run_id = d.get("run_id", summary.parent.name)
        m = re.match(r"^(.*)_s(\d+)$", run_id)
        runs.append({
            "run_id": run_id,
            "config": m.group(1) if m else run_id,
            "seed": int(m.group(2)) if m else -1,
            "params": d.get("model_params", 0),
            "best_epoch": best.get("epoch"),
            "val_ppl": best.get("val_ppl"),
            "val_bpc": best.get("val_bpc"),
            "test_ppl": test.get("test_ppl"),
            "test_bpc": test.get("test_bpc"),
            "test_tokens": test.get("test_tokens"),
        })
    return runs


def _agg(vals: list[float]) -> str:
    vals = [v for v in vals if isinstance(v, (int, float))]
    if not vals:
        return "     n/a"
    if len(vals) == 1:
        return f"{vals[0]:8.2f}"
    return f"{st.mean(vals):8.2f}±{st.pstdev(vals):5.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("artifacts/slm_push"))
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    runs = load_runs(args.dir)
    if not runs:
        print(f"No completed runs under {args.dir}")
        return

    print(f"\n{'run':28s} {'ep':>3s} {'val_ppl':>9s} {'test_ppl':>9s} {'val_bpc':>8s} {'test_bpc':>9s}")
    print("-" * 72)
    for r in sorted(runs, key=lambda x: (x["config"], x["seed"])):
        def f(v, w, p=2):
            return f"{v:{w}.{p}f}" if isinstance(v, (int, float)) else " " * (w - 3) + "n/a"
        print(f"{r['run_id']:28s} {str(r['best_epoch']):>3s} "
              f"{f(r['val_ppl'],9)} {f(r['test_ppl'],9)} {f(r['val_bpc'],8,3)} {f(r['test_bpc'],9,3)}")

    configs: dict[str, list[dict]] = {}
    for r in runs:
        configs.setdefault(r["config"], []).append(r)

    print(f"\n{'config':28s} {'n':>2s} {'val_ppl':>15s} {'test_ppl':>15s} {'test_bpc':>15s}")
    print("-" * 80)
    rows = []
    for cfg, rs in sorted(configs.items()):
        row = {
            "config": cfg, "n_seeds": len(rs),
            "test_ppl_mean": st.mean([r["test_ppl"] for r in rs if r["test_ppl"]]) if any(r["test_ppl"] for r in rs) else None,
        }
        rows.append(row)
        print(f"{cfg:28s} {len(rs):>2d} {_agg([r['val_ppl'] for r in rs]):>15s} "
              f"{_agg([r['test_ppl'] for r in rs]):>15s} "
              f"{_agg([r['test_bpc'] for r in rs]):>15s}")

    # Headline: relative improvement over the control, on the held-out set.
    base = next((r for r in rows if "A_baseline" in r["config"]), None)
    if base and base["test_ppl_mean"]:
        print()
        for r in rows:
            if r is base or not r["test_ppl_mean"]:
                continue
            delta = (base["test_ppl_mean"] - r["test_ppl_mean"]) / base["test_ppl_mean"] * 100
            print(f"  {r['config']:26s} test_ppl {delta:+6.1f}% vs A_baseline")

    if args.json:
        args.json.write_text(json.dumps({"runs": runs, "configs": rows}, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
