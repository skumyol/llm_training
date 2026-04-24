#!/usr/bin/env bash
# =============================================================================
# Full SLM Training Pipeline with External Datasets
# =============================================================================
# Runs complete HPO + final training for all 6 architectures on 107M token corpus
#
# Phases:
#   1. Optuna HPO (20 trials × 5 epochs) per architecture
#   2. Final training (3 seeds × 30 epochs) with best hyperparams
#   3. Evaluation (PPL + BLEU + Distinct)
#
# Usage:
#   bash run_full_slm_training.sh [arch]
#   bash run_full_slm_training.sh all        # All 6 architectures (default)
#   bash run_full_slm_training.sh prefix_gpt  # Single arch for testing
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT/.venv/bin/python"
export PYTHONPATH="$ROOT"

ARCH="${1:-all}"
RUN_ID="full_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/full_slm_${RUN_ID}.log"

# Use merged external corpus
TRAIN_TEXT="$ROOT/data/external/merged_dialogue.txt"
VAL_TEXT="$ROOT/data/dialogue/val.txt"

echo "================================================================" | tee "$LOG_FILE"
echo "  FULL SLM TRAINING PIPELINE" | tee -a "$LOG_FILE"
echo "  Run ID: $RUN_ID" | tee -a "$LOG_FILE"
echo "  Architecture(s): $ARCH" | tee -a "$LOG_FILE"
echo "  Data: $TRAIN_TEXT (107M tokens)" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

# Verify data exists
if [ ! -f "$TRAIN_TEXT" ]; then
    echo "ERROR: Training data not found. Run download_external_datasets.py first." | tee -a "$LOG_FILE"
    exit 1
fi

TOKENS=$(wc -c < "$TRAIN_TEXT")
echo "  Corpus size: ~$((TOKENS / 4)) tokens" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Phase 1: Optuna HPO
echo ">>> Phase 1: Optuna HPO (20 trials × 5 epochs)..." | tee -a "$LOG_FILE"
"$PYTHON" scripts/optuna_small_lm.py \
    --arch "$ARCH" \
    --n-trials 20 \
    --epochs 5 \
    --train-text "$TRAIN_TEXT" \
    --val-text "$VAL_TEXT" \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo ">>> Phase 2: Final Multi-seed Training (3 seeds × 30 epochs)..." | tee -a "$LOG_FILE"
"$PYTHON" scripts/train_final_small_lms.py \
    --arch "$ARCH" \
    --seeds 42 43 44 \
    --epochs 30 \
    --train-text "$TRAIN_TEXT" \
    --val-text "$VAL_TEXT" \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo ">>> Phase 3: Evaluation (PPL + BLEU + Distinct)..." | tee -a "$LOG_FILE"
"$PYTHON" scripts/eval_small_lms.py \
    --arch "$ARCH" \
    --seeds 42 43 44 \
    --val-text "$VAL_TEXT" \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "  FULL TRAINING COMPLETE" | tee -a "$LOG_FILE"
echo "  Results: artifacts/slm_final_eval_*.csv" | tee -a "$LOG_FILE"
echo "  Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
