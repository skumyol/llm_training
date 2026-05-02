#!/usr/bin/env bash
# =============================================================================
# Full SLM Training Pipeline with Parallel Execution
# =============================================================================
# Runs complete HPO + final training for all 6 architectures with optimized
# parallel execution on single or multiple GPUs.
#
# This script is now a thin wrapper around run_parallel_pipeline.py which
# manages job queuing, GPU monitoring, and concurrent execution.
#
# Phases:
#   1. Optuna HPO (20 trials × 5 epochs) per architecture — PARALLEL
#   2. Final training (3 seeds × 30 epochs) per architecture — PARALLEL
#   3. Evaluation (PPL + BLEU + Distinct)
#
# Usage:
#   bash run_full_slm_training.sh [arch]               # Full pipeline
#   bash run_full_slm_training.sh --hpo-only [arch]    # HPO phase only
#   bash run_full_slm_training.sh --train-only [arch]  # Skip HPO, use existing
#   bash run_full_slm_training.sh --dry-run            # Preview job list
#
# Examples:
#   bash run_full_slm_training.sh all                  # All 6 architectures
#   bash run_full_slm_training.sh prefix_gpt           # Single architecture
#   bash run_full_slm_training.sh --hpo-only gpt     # Just HPO for GPT
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ROOT}/../.venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="python3"
export PYTHONPATH="$ROOT"
cd "$ROOT"

# ── Configuration ──────────────────────────────────────────────────────────────
ARCH="${1:-all}"
N_TRIALS=20
HPO_EPOCHS=5
FINAL_EPOCHS=30
SEEDS=(42 43 44)

# Use external corpus if available
TRAIN_TEXT="${TRAIN_TEXT:-$ROOT/data/external/merged_dialogue.txt}"
VAL_TEXT="${VAL_TEXT:-$ROOT/data/dialogue/val.txt}"

# Workers: 1 for single GPU, higher for multi-GPU
WORKERS=1

# ── Parse flags ───────────────────────────────────────────────────────────────
MODE="full"
DRY_RUN=""

for arg in "${@:2}"; do
  case "$arg" in
    --hpo-only)   MODE="hpo" ;;
    --train-only) MODE="train" ;;
    --dry-run)    DRY_RUN="--dry-run" ;;
  esac
done

# Check for external data, fallback to internal
if [ ! -f "$TRAIN_TEXT" ]; then
  echo "WARNING: External data not found: $TRAIN_TEXT"
  echo "  Falling back to internal data..."
  TRAIN_TEXT="$ROOT/data/dialogue/train.txt"
  VAL_TEXT="$ROOT/data/dialogue/val.txt"
fi

echo "================================================================"
echo "  FULL SLM TRAINING PIPELINE (Parallel)"
echo "================================================================"
echo "  Run ID      : $(date +%Y%m%d_%H%M%S)"
echo "  Architecture: $ARCH"
echo "  Mode        : $MODE"
echo "  Workers     : $WORKERS"
echo "  Train data  : $TRAIN_TEXT"
echo "  Val data    : $VAL_TEXT"
echo ""
echo "  Phase breakdown:"
if [ "$MODE" == "full" ] || [ "$MODE" == "hpo" ]; then
  echo "    HPO   : $N_TRIALS trials × $HPO_EPOCHS epochs per arch"
fi
if [ "$MODE" == "full" ] || [ "$MODE" == "train" ]; then
  echo "    Train : ${#SEEDS[@]} seeds × $FINAL_EPOCHS epochs per arch"
fi
if [ "$MODE" == "full" ]; then
  echo "    Eval  : PPL + BLEU + Distinct metrics"
fi
echo "================================================================"
echo ""

# ── Run parallel pipeline ───────────────────────────────────────────────────
"$PYTHON" "$ROOT/scripts/run_parallel_pipeline.py" \
  --mode "$MODE" \
  --arch "$ARCH" \
  --workers "$WORKERS" \
  --trials "$N_TRIALS" \
  --hpo-epochs "$HPO_EPOCHS" \
  --epochs "$FINAL_EPOCHS" \
  --seeds "${SEEDS[@]}" \
  --train-text "$TRAIN_TEXT" \
  --val-text "$VAL_TEXT" \
  $DRY_RUN

RC=$?

echo ""
echo "================================================================"
if [ $RC -eq 0 ]; then
  echo "  ✓ PIPELINE COMPLETE"
  echo "  Results: artifacts/slm_parallel_eval_*.csv"
  echo "  Logs   : logs/parallel_*.log"
  echo "  MLflow : mlflow ui --backend-store-uri ./mlruns"
else
  echo "  ✗ PIPELINE FAILED (exit=$RC)"
  echo "  Check logs for errors"
fi
echo "================================================================"

exit $RC
