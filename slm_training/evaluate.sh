#!/usr/bin/env bash
# =============================================================================
# Evaluation — All Tracks
# =============================================================================
#
# Evaluates all trained models and prints a comparison table.
#
# Track A — Encoder metrics:
#   Personality: Macro F1, MSE per OCEAN dimension
#   Affect:      Mean CCC, MSE per VAD dimension
#
# Track B — Small LM metrics:
#   Val Perplexity (PPL), Distinct-1, Distinct-2
#   Picks the best checkpoint per architecture (lowest val_ppl)
#
# Track C — Dialogue LM:
#   Val Perplexity, BLEU-1/2, sample generations
#
# Output:
#   artifacts/eval_results_<timestamp>.csv   — full results table
#   stdout                                   — formatted summary table
#
# Usage:
#   bash evaluate.sh                    # evaluate everything
#   bash evaluate.sh --track b          # only Track B (Small LMs)
#   bash evaluate.sh --out results.csv  # custom output path
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT/../.venv/bin/python"
export PYTHONPATH="$ROOT"

# ── Defaults ──────────────────────────────────────────────────────────────────
TRACK="all"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_CSV="$ROOT/artifacts/eval_results_${TIMESTAMP}.csv"
LOG_DIR="$ROOT/logs"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --track)  TRACK="$2";   shift 2 ;;
    --out)    OUT_CSV="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR" "$(dirname "$OUT_CSV")"
LOG_FILE="$LOG_DIR/evaluate_${TIMESTAMP}.log"

echo "================================================================" | tee "$LOG_FILE"
echo "  Evaluation — NPC Backend Models" | tee -a "$LOG_FILE"
echo "  Track  : $TRACK" | tee -a "$LOG_FILE"
echo "  Output : $OUT_CSV" | tee -a "$LOG_FILE"
echo "  Log    : $LOG_FILE" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

# ── Track B: Small LMs ────────────────────────────────────────────────────────
if [[ "$TRACK" == "all" || "$TRACK" == "b" ]]; then
  echo "" | tee -a "$LOG_FILE"
  echo ">>> Track B — Small LM Evaluation" | tee -a "$LOG_FILE"
  echo "    Metrics: val PPL, Distinct-1/2, sample generations" | tee -a "$LOG_FILE"
  "$PYTHON" scripts/eval_small_lms.py \
    --out-csv "$OUT_CSV" \
    2>&1 | tee -a "$LOG_FILE"
fi

# ── Track A: Encoders ─────────────────────────────────────────────────────────
if [[ "$TRACK" == "all" || "$TRACK" == "a" ]]; then
  echo "" | tee -a "$LOG_FILE"
  echo ">>> Track A — Encoder Evaluation" | tee -a "$LOG_FILE"
  echo "    (Reading run_summary.json files from artifacts/)" | tee -a "$LOG_FILE"
  "$PYTHON" - 2>&1 | tee -a "$LOG_FILE" <<'PYEOF'
import json, glob
from pathlib import Path

ROOT = Path(".")

print("\n  Personality Encoder results:")
for f in sorted(glob.glob("artifacts/personality_encoder/*/run_summary.json")):
    d = json.load(open(f))
    run = d.get("run_id", Path(f).parent.name)
    best = d.get("best", {})
    print(f"    {run:40s}  F1={best.get('val_f1', best.get('val_macro_f1', 0)):.4f}  MSE={best.get('val_mse', 0):.4f}")

print("\n  Affect Encoder results:")
for f in sorted(glob.glob("artifacts/affect_encoder/*/run_summary.json")):
    d = json.load(open(f))
    run = d.get("run_id", Path(f).parent.name)
    best = d.get("best", {})
    print(f"    {run:40s}  CCC={best.get('val_ccc', 0):.4f}  MSE={best.get('val_mse', 0):.4f}")
PYEOF
fi

echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "  DONE — Results saved to $OUT_CSV" | tee -a "$LOG_FILE"
echo "  MLflow: mlflow ui --backend-store-uri ./mlruns" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
