#!/usr/bin/env bash
# =============================================================================
# TRACK A — Personality Encoder Training
# =============================================================================
#
# What this does:
#   1. (Optional) Optuna hyperparameter search: 30 trials of DistilBERT fine-tuning
#      Searches: lr, encoder_lr_factor, dropout, focal_gamma, token_drop_prob,
#                freeze_encoder_epochs
#   2. Final training with best found params × 3 seeds for stable reporting
#
# Model:  DistilBERT-base-uncased → [CLS] → Dropout → Linear(768→5) → Sigmoid
# Task:   Predict Big Five OCEAN personality dimensions from NPC profile text
# Loss:   Focal loss (handles class imbalance in generated profiles)
# Metric: Macro F1 per OCEAN dimension (reported as mean F1)
#
# Output: artifacts/personality_encoder/<run_id>/
#   run.log           — timestamped training log
#   step_metrics.csv  — train loss / lr / grad_norm per step
#   epoch_metrics.csv — val F1 / MSE per epoch
#   best_model/       — saved best checkpoint
#   run_summary.json  — all hyperparams + best metrics (for paper table)
#
# Best Optuna params cached at: artifacts/optuna/personality_best.json
# MLflow experiment: personality_encoder
#
# Usage:
#   bash train_personality_encoder.sh              # HPO + final training
#   bash train_personality_encoder.sh --skip-hpo   # use existing best params
#   bash train_personality_encoder.sh --hpo-only   # search only, no final run
#   bash train_personality_encoder.sh --run-id v2  # tag this run
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT/.venv/bin/python"
export PYTHONPATH="$ROOT"

# ── Defaults ──────────────────────────────────────────────────────────────────
RUN_ID="personality_$(date +%Y%m%d_%H%M%S)"
SKIP_HPO=false
HPO_ONLY=false
N_SEEDS=3
HPO_TRIALS=30
FINAL_EPOCHS=20
LOG_DIR="$ROOT/logs"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-hpo)   SKIP_HPO=true;  shift ;;
    --hpo-only)   HPO_ONLY=true;  shift ;;
    --run-id)     RUN_ID="$2";    shift 2 ;;
    --seeds)      N_SEEDS="$2";   shift 2 ;;
    --trials)     HPO_TRIALS="$2"; shift 2 ;;
    --epochs)     FINAL_EPOCHS="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/personality_encoder_${RUN_ID}.log"

echo "================================================================" | tee "$LOG_FILE"
echo "  TRACK A — Personality Encoder" | tee -a "$LOG_FILE"
echo "  Run ID : $RUN_ID" | tee -a "$LOG_FILE"
echo "  Log    : $LOG_FILE" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

# ── Phase 1: Hyperparameter search (Optuna) ───────────────────────────────────
BEST_JSON="$ROOT/artifacts/optuna/personality_best.json"

if [ "$SKIP_HPO" = false ] && [ ! -f "$BEST_JSON" ]; then
  echo "" | tee -a "$LOG_FILE"
  echo ">>> Phase 1: HPO — $HPO_TRIALS Optuna trials" | tee -a "$LOG_FILE"
  echo "    Searching: lr, dropout, focal_gamma, freeze_encoder_epochs" | tee -a "$LOG_FILE"
  "$PYTHON" scripts/hyperparam_search.py \
    --task personality \
    --n-trials "$HPO_TRIALS" \
    2>&1 | tee -a "$LOG_FILE"
  echo "    Best params saved → $BEST_JSON" | tee -a "$LOG_FILE"
elif [ -f "$BEST_JSON" ]; then
  echo "" | tee -a "$LOG_FILE"
  echo ">>> Phase 1: HPO — SKIPPED (best.json already exists)" | tee -a "$LOG_FILE"
  BEST_PPL=$(python3 -c "import json; d=json.load(open('$BEST_JSON')); print(f\"F1={d.get('best_value',0):.4f}\")" 2>/dev/null || echo "?")
  echo "    Cached result: $BEST_PPL" | tee -a "$LOG_FILE"
fi

if [ "$HPO_ONLY" = true ]; then
  echo "" | tee -a "$LOG_FILE"
  echo ">>> --hpo-only set — stopping after HPO." | tee -a "$LOG_FILE"
  exit 0
fi

# ── Phase 2: Final training with best params × N seeds ───────────────────────
echo "" | tee -a "$LOG_FILE"
echo ">>> Phase 2: Final training — $FINAL_EPOCHS epochs × $N_SEEDS seeds" | tee -a "$LOG_FILE"

for SEED in $(seq 42 $((41 + N_SEEDS))); do
  echo "    Seed $SEED ..." | tee -a "$LOG_FILE"
  "$PYTHON" -m src.train.run_personality \
    --config configs/personality.yaml \
    --run-id "${RUN_ID}_s${SEED}" \
    --epochs "$FINAL_EPOCHS" \
    --seed   "$SEED" \
    2>&1 | tee -a "$LOG_FILE"
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "  DONE — Personality Encoder" | tee -a "$LOG_FILE"
echo "  Artifacts : artifacts/personality_encoder/${RUN_ID}_s*/" | tee -a "$LOG_FILE"
echo "  MLflow    : mlflow ui --backend-store-uri ./mlruns" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
