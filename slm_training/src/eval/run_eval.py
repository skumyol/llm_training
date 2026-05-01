#!/usr/bin/env python3
"""
run_eval.py  —  NPC Backend Evaluation
========================================
Evaluates all trained components and prints a unified quality report.

Metrics:
  Personality encoder : MSE / MAE / R² per OCEAN dimension
  Affect encoder      : MSE / MAE / R² per VAD dimension, CCC
  Dialogue model      : Perplexity, BLEU-1/2, Distinct-1/2, avg response length
  Full pipeline       : Side-by-side sample conversations
  Summary bundle      : `evaluation/evaluation_summary.json` and `.md`

Usage:
  python -m src.eval.run_eval                               # auto-discover artifacts/
  python -m src.eval.run_eval --artifacts artifacts/        # explicit path
  python -m src.eval.run_eval --dialogue-only               # skip encoders
  python -m src.eval.run_eval --samples 5                   # show 5 sample convos
  python -m src.eval.run_eval --out-csv eval_results.csv    # save to CSV
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.train.metrics_report import write_metrics_bundle

# ── ANSI colour helpers (no extra deps) ────────────────────────────────────────
BOLD  = "\033[1m"
DIM   = "\033[2m"
GREEN = "\033[32m"
CYAN  = "\033[36m"
YELLOW= "\033[33m"
RED   = "\033[31m"
RESET = "\033[0m"

def _h1(s: str) -> None:
    print(f"\n{BOLD}{CYAN}{'━'*60}{RESET}")
    print(f"{BOLD}{CYAN}  {s}{RESET}")
    print(f"{BOLD}{CYAN}{'━'*60}{RESET}")

def _h2(s: str) -> None:
    print(f"\n{BOLD}{YELLOW}  {s}{RESET}")

def _ok(s: str) -> None:
    print(f"  {GREEN}✓{RESET}  {s}")

def _warn(s: str) -> None:
    print(f"  {YELLOW}⚠{RESET}  {s}")

def _err(s: str) -> None:
    print(f"  {RED}✗{RESET}  {s}")


# ── Concordance Correlation Coefficient ────────────────────────────────────────

def _ccc(y_true: List[float], y_pred: List[float]) -> float:
    import statistics
    if len(y_true) < 2:
        return 0.0
    mu_t = statistics.mean(y_true)
    mu_p = statistics.mean(y_pred)
    var_t = statistics.variance(y_true)
    var_p = statistics.variance(y_pred)
    n = len(y_true)
    cov = sum((t - mu_t) * (p - mu_p) for t, p in zip(y_true, y_pred)) / (n - 1)
    denom = var_t + var_p + (mu_t - mu_p) ** 2
    return (2 * cov / denom) if denom > 0 else 0.0


# ── Tokenise (simple whitespace) ───────────────────────────────────────────────

def _tokens(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


# ── BLEU-N (micro-averaged, simple) ────────────────────────────────────────────

def _bleu_n(references: List[str], hypotheses: List[str], n: int) -> float:
    clip_total = 0
    cand_total = 0
    for ref, hyp in zip(references, hypotheses):
        ref_ngrams  = Counter(zip(*[_tokens(ref)[i:]  for i in range(n)]))
        hyp_ngrams  = Counter(zip(*[_tokens(hyp)[i:]  for i in range(n)]))
        if not hyp_ngrams:
            continue
        clip = sum(min(c, ref_ngrams[g]) for g, c in hyp_ngrams.items())
        clip_total += clip
        cand_total += sum(hyp_ngrams.values())
    return clip_total / cand_total if cand_total else 0.0


# ── Distinct-N ────────────────────────────────────────────────────────────────

def _distinct_n(texts: List[str], n: int) -> float:
    all_ngrams: List[Tuple] = []
    for t in texts:
        toks = _tokens(t)
        all_ngrams.extend(zip(*[toks[i:] for i in range(n)]))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


# ── Encoder eval: read predictions CSV ────────────────────────────────────────

def _eval_encoder(summary_path: Path, dims: List[str]) -> Dict[str, Any]:
    run_dir = summary_path.parent

    # Find latest predictions CSV
    pred_files = sorted(run_dir.glob("predictions_epoch*.csv"))
    if not pred_files:
        return {"error": "no predictions CSV found"}

    pred_file = pred_files[-1]
    true_vals: Dict[str, List[float]] = {d: [] for d in dims}
    pred_vals: Dict[str, List[float]] = {d: [] for d in dims}

    with open(pred_file, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for d in dims:
                try:
                    true_vals[d].append(float(row.get(f"true_{d}", row.get(d, 0))))
                    pred_vals[d].append(float(row.get(f"pred_{d}", 0)))
                except (ValueError, KeyError):
                    pass

    results: Dict[str, Any] = {}
    for d in dims:
        if not true_vals[d]:
            continue
        n = len(true_vals[d])
        mse = sum((t - p) ** 2 for t, p in zip(true_vals[d], pred_vals[d])) / n
        mae = sum(abs(t - p)   for t, p in zip(true_vals[d], pred_vals[d])) / n
        mu_t = sum(true_vals[d]) / n
        ss_tot = sum((t - mu_t) ** 2 for t in true_vals[d])
        ss_res = sum((t - p) ** 2 for t, p in zip(true_vals[d], pred_vals[d]))
        r2  = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        ccc = _ccc(true_vals[d], pred_vals[d])
        results[d] = {"n": n, "mse": mse, "mae": mae, "r2": r2, "ccc": ccc}

    return results


# ── Dialogue eval: perplexity from epoch_metrics.csv + BLEU/distinct ──────────

def _eval_dialogue_metrics(run_dir: Path) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}

    epoch_csv = run_dir / "epoch_metrics.csv"
    if epoch_csv.exists():
        with open(epoch_csv) as f:
            rows = list(csv.DictReader(f))
        if rows:
            best = min(rows, key=lambda r: float(r.get("val_loss", "inf")))
            metrics["val_loss"] = float(best.get("val_loss", 0))
            metrics["val_ppl"]  = math.exp(min(float(best.get("val_loss", 0)), 20))
            metrics["best_epoch"] = best.get("epoch", "?")

    return metrics


def _eval_dialogue_generation(
    val_jsonl: Path,
    n_samples: int,
    service,
    npc_id: str,
) -> Dict[str, Any]:
    """Run model on val samples, compute BLEU + distinct."""
    references: List[str] = []
    hypotheses: List[str] = []
    lengths: List[int] = []

    with open(val_jsonl, encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    import random
    random.seed(42)
    sample_lines = random.sample(lines, min(n_samples, len(lines)))

    for line in sample_lines:
        rec = json.loads(line)
        ref = rec.get("target_response", "").strip()
        ctx = rec.get("dialogue_context", [])
        if not ref or not ctx:
            continue

        # Build conversation history in service
        history = [t["text"] for t in ctx if t.get("speaker") == "player"]
        last_player = history[-1] if history else "hello"

        try:
            hyp = service.respond(npc_id, last_player)
            references.append(ref)
            hypotheses.append(hyp)
            lengths.append(len(_tokens(hyp)))
        except Exception:
            continue

    if not hypotheses:
        return {"error": "no responses generated"}

    return {
        "n_samples":    len(hypotheses),
        "bleu1":        _bleu_n(references, hypotheses, 1),
        "bleu2":        _bleu_n(references, hypotheses, 2),
        "distinct1":    _distinct_n(hypotheses, 1),
        "distinct2":    _distinct_n(hypotheses, 2),
        "avg_len":      sum(lengths) / len(lengths),
    }


# ── Encoder report printer ─────────────────────────────────────────────────────

def _print_encoder_report(name: str, results: Dict[str, Any]) -> List[Dict]:
    _h2(name)
    if "error" in results:
        _warn(results["error"])
        return []

    rows = []
    header = f"  {'Dimension':<18}  {'MSE':>8}  {'MAE':>8}  {'R²':>7}  {'CCC':>7}"
    print(header)
    print("  " + "─" * 56)
    for dim, m in results.items():
        r2_color  = GREEN if m["r2"] > 0.3 else (YELLOW if m["r2"] > 0 else RED)
        ccc_color = GREEN if m["ccc"] > 0.3 else (YELLOW if m["ccc"] > 0 else RED)
        print(f"  {dim:<18}  {m['mse']:>8.4f}  {m['mae']:>8.4f}  "
              f"{r2_color}{m['r2']:>7.3f}{RESET}  "
              f"{ccc_color}{m['ccc']:>7.3f}{RESET}")
        rows.append({"component": name, "dimension": dim, **m})
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate trained NPC pipeline components")
    p.add_argument("--artifacts",      type=Path, default=Path("artifacts"))
    p.add_argument("--val-dialogue",   type=Path, default=Path("data/dialogue/from_persona_val.jsonl"))
    p.add_argument("--dialogue-only",  action="store_true")
    p.add_argument("--encoders-only",  action="store_true")
    p.add_argument("--samples",        type=int, default=20,
                   help="N dialogue val samples for BLEU/distinct (requires trained dialogue model)")
    p.add_argument("--out-csv",        type=Path, default=None)
    args = p.parse_args()

    all_rows: List[Dict] = []
    summary_bundle: Dict[str, Any] = {
        "summary": {},
        "encoders": {},
        "dialogue": {},
        "generation": {},
    }

    _h1("NPC Pipeline — Evaluation Report")

    # ── Discover run summaries ────────────────────────────────────────────────
    summaries = list(args.artifacts.glob("**/run_summary.json"))
    if not summaries:
        _warn(f"No run_summary.json found under {args.artifacts}")
        _warn("Run training first: ./train_all.sh or individual run_*.py scripts")
    else:
        print(f"\n  Found {len(summaries)} training run(s):")
        for s in summaries:
            d = json.loads(s.read_text())
            tag = f"{d.get('run_id','?')}  [{d.get('task','?')}]"
            best = d.get("best", {})
            best_str = "  ".join(f"{k}={v:.4f}" for k, v in best.items()
                                 if isinstance(v, (int, float)))
            print(f"    {DIM}{s.parent.name}/{s.parent.parent.name}{RESET}  "
                  f"{BOLD}{tag}{RESET}  {GREEN}{best_str}{RESET}")

    # ── Encoder evals ─────────────────────────────────────────────────────────
    if not args.dialogue_only:
        _h1("Encoder Quality")

        # Personality
        pers_summaries = list(args.artifacts.glob("personality_encoder/**/run_summary.json"))
        if pers_summaries:
            pers_dir = pers_summaries[-1].parent
            ocean = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
            res = _eval_encoder(pers_summaries[-1], ocean)
            rows = _print_encoder_report("Personality (OCEAN)", res)
            all_rows.extend(rows)
            summary_bundle["encoders"]["personality"] = res
        else:
            _warn("No personality encoder found — run: python -m src.train.run_personality")

        # Affect
        aff_summaries = list(args.artifacts.glob("affect_encoder/**/run_summary.json"))
        if aff_summaries:
            aff_dir = aff_summaries[-1].parent
            vad = ["valence", "arousal", "dominance"]
            res = _eval_encoder(aff_summaries[-1], vad)
            rows = _print_encoder_report("Affect (VAD)", res)
            all_rows.extend(rows)
            summary_bundle["encoders"]["affect"] = res
        else:
            _warn("No affect encoder found — run: python -m src.train.run_affect")

        # Interpretation guide
        _h2("R² / CCC interpretation")
        print(f"  {GREEN}> 0.3{RESET}  good signal for this task")
        print(f"  {YELLOW}0.0–0.3{RESET}  weak but positive correlation")
        print(f"  {RED}< 0.0{RESET}  model is worse than predicting the mean")
        print(f"\n  Note: personality R² is typically low (0.05–0.15) even in SOTA —")
        print(f"  text→OCEAN is genuinely hard. Affect VAD from EmoBank is easier.")

    # ── Dialogue eval ─────────────────────────────────────────────────────────
    if not args.encoders_only:
        _h1("Dialogue Model Quality")

        dlg_summaries = list(args.artifacts.glob("dialogue_model*/**/run_summary.json"))
        if not dlg_summaries:
            _warn("No dialogue model found yet — training may still be running")
            _warn("Check: tail -f artifacts/dialogue_run.log")
        else:
            for summary_path in dlg_summaries:
                d = json.loads(summary_path.read_text())
                run_id = d.get("run_id", "?")
                _h2(f"Run: {run_id}")

                # Training metrics
                m = _eval_dialogue_metrics(summary_path.parent)
                if m:
                    ppl_color = GREEN if m.get("val_ppl", 999) < 20 else \
                                YELLOW if m.get("val_ppl", 999) < 50 else RED
                    print(f"  val_loss  : {m.get('val_loss', 0):.4f}")
                    print(f"  val_ppl   : {ppl_color}{m.get('val_ppl', 0):.2f}{RESET}")
                    print(f"  best_epoch: {m.get('best_epoch', '?')}")
                    all_rows.append({"component": "dialogue", "run_id": run_id, **m})
                    summary_bundle["dialogue"][run_id] = m

                # Generation metrics (requires loaded model)
                if args.samples > 0 and args.val_dialogue.exists():
                    _h2("  Generation metrics (loading model...)")
                    try:
                        import sys
                        sys.path.insert(0, ".")
                        from src.common.config import InferenceConfig
                        from src.infer.service import NPCInferenceService

                        cfg_path = summary_path.parent / "config.json"
                        model_dir = summary_path.parent / "best_model"
                        if not model_dir.exists():
                            _warn("best_model not found — skipping generation metrics")
                        else:
                            cfg = InferenceConfig(dialogue_model_dir=str(model_dir))
                            svc = NPCInferenceService(cfg)
                            svc.register_npc("eval_npc",
                                "A pragmatic NPC who responds helpfully but stays in character.")
                            gen_m = _eval_dialogue_generation(
                                args.val_dialogue, args.samples, svc, "eval_npc")
                            if "error" not in gen_m:
                                bleu1_c = GREEN if gen_m["bleu1"] > 0.15 else YELLOW
                                print(f"  BLEU-1    : {bleu1_c}{gen_m['bleu1']:.3f}{RESET}")
                                print(f"  BLEU-2    : {gen_m['bleu2']:.3f}")
                                print(f"  Distinct-1: {gen_m['distinct1']:.3f}  "
                                      f"{DIM}(higher = more diverse){RESET}")
                                print(f"  Distinct-2: {gen_m['distinct2']:.3f}")
                                print(f"  Avg length: {gen_m['avg_len']:.1f} tokens")
                                all_rows.append({"component": "dialogue_gen", **gen_m})
                                summary_bundle["generation"][run_id] = gen_m
                            else:
                                _warn(gen_m["error"])
                    except Exception as e:
                        _warn(f"Could not run generation eval: {e}")

        # Perplexity guide
        _h2("Perplexity interpretation")
        print(f"  {GREEN}< 15{RESET}  strong fit — model is confident on val data")
        print(f"  {YELLOW}15–40{RESET}  reasonable for fine-tuned small LM on domain data")
        print(f"  {RED}> 40{RESET}  underfitting or data mismatch")
        print(f"\n  BLEU-1 > 0.15 and Distinct-2 > 0.7 is a practical target.")
        print(f"  High BLEU + low Distinct = mode collapse. High Distinct + low BLEU = hallucinating.")

    # ── Summary table ─────────────────────────────────────────────────────────
    if all_rows and args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        all_keys = sorted({k for r in all_rows for k in r.keys()})
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys)
            w.writeheader()
            for r in all_rows:
                w.writerow({k: r.get(k, "") for k in all_keys})
        _ok(f"Results saved → {args.out_csv}")

    if all_rows:
        summary_bundle["summary"] = {
            "num_rows": len(all_rows),
            "num_dialogue_runs": len(summary_bundle["dialogue"]),
            "num_generation_runs": len(summary_bundle["generation"]),
            "num_encoder_groups": len(summary_bundle["encoders"]),
        }
        bundle_dir = args.artifacts / "evaluation"
        write_metrics_bundle(bundle_dir, "evaluation_summary", summary_bundle, title="SLM Evaluation Summary")
        _ok(f"Summary bundle saved → {bundle_dir / 'evaluation_summary.json'}")

    _h1("Done")


if __name__ == "__main__":
    main()
