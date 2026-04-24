#!/usr/bin/env bash
# =============================================================================
# run_full_experiment.sh
# Full Small-LM improvement pipeline: Optuna HPO → Final Training → Eval
#
# Usage:
#   bash scripts/run_full_experiment.sh            # full run (20 trials, 30 epochs)
#   bash scripts/run_full_experiment.sh --smoke    # smoke test (3 trials, 3 epochs)
#
# Logs: /tmp/experiment_<arch>.log
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="$ROOT/.venv/bin/python"

# ── Defaults ──────────────────────────────────────────────────────────────────
N_TRIALS=20
HPO_EPOCHS=5
FINAL_EPOCHS=30
SEEDS="42 43 44"
SMOKE=0

# ── Parse args ────────────────────────────────────────────────────────────────
for arg in "$@"; do
  case $arg in
    --smoke)   SMOKE=1; N_TRIALS=3; HPO_EPOCHS=3; FINAL_EPOCHS=5; SEEDS="42 43" ;;
    --trials=*) N_TRIALS="${arg#*=}" ;;
    --epochs=*) FINAL_EPOCHS="${arg#*=}" ;;
  esac
done

if [ "$SMOKE" -eq 1 ]; then
  echo "================================================================"
  echo "  SMOKE TEST MODE: $N_TRIALS trials × $HPO_EPOCHS epochs/trial"
  echo "  Final training: $FINAL_EPOCHS epochs × seeds: $SEEDS"
  echo "================================================================"
else
  echo "================================================================"
  echo "  FULL RUN: $N_TRIALS trials × $HPO_EPOCHS epochs/trial"
  echo "  Final training: $FINAL_EPOCHS epochs × seeds: $SEEDS"
  echo "================================================================"
fi

cd "$ROOT"

# ── Helper ────────────────────────────────────────────────────────────────────
run_optuna() {
  local arch=$1
  local logfile="/tmp/experiment_optuna_${arch}.log"
  echo ""
  echo ">>> [$(date '+%H:%M:%S')] Optuna HPO: $arch ($N_TRIALS trials × $HPO_EPOCHS epochs)"

  if [ -f "artifacts/optuna/small_lm_${arch}_best.json" ]; then
    echo "    SKIP — best.json already exists"
    return
  fi

  "$PYTHON" scripts/optuna_small_lm.py \
    --arch "$arch" \
    --n-trials "$N_TRIALS" \
    --epochs "$HPO_EPOCHS" \
    2>&1 | tee "$logfile"

  echo "    Done: $arch → artifacts/optuna/small_lm_${arch}_best.json"
}

# ── Phase 1: Wait for existing background jobs ────────────────────────────────
echo ""
echo ">>> [$(date '+%H:%M:%S')] Phase 1: Wait for background jobs"

# Wait for mamba_like s43 (if still running)
MAMBA_PID=$(pgrep -f "run_small_lm.*mamba_like.*s43" 2>/dev/null | head -1 || true)
if [ -n "$MAMBA_PID" ]; then
  echo "    Waiting for mamba_like s43 (PID=$MAMBA_PID)..."
  wait "$MAMBA_PID" 2>/dev/null || true
  echo "    mamba_like s43 done."
fi

# Wait for GRU Optuna (if still running)
GRU_OPT_PID=$(pgrep -f "optuna_small_lm.*gru" 2>/dev/null | head -1 || true)
if [ -n "$GRU_OPT_PID" ]; then
  echo "    Waiting for GRU Optuna (PID=$GRU_OPT_PID)..."
  wait "$GRU_OPT_PID" 2>/dev/null || true
  echo "    GRU Optuna done."
fi

# ── Phase 2: Optuna HPO for remaining archs ───────────────────────────────────
echo ""
echo ">>> [$(date '+%H:%M:%S')] Phase 2: Optuna HPO"

for ARCH in awdlstm gpt prefix_gpt moe mamba_like; do
  run_optuna "$ARCH"
done

# GRU might have just finished above, but check best.json was written
if [ ! -f "artifacts/optuna/small_lm_gru_best.json" ]; then
  run_optuna "gru"
fi

echo ""
echo ">>> [$(date '+%H:%M:%S')] Optuna HPO complete for all architectures"
echo ""
echo "    Summary of best PPL per arch:"
for ARCH in gru awdlstm gpt prefix_gpt moe mamba_like; do
  BEST_FILE="artifacts/optuna/small_lm_${ARCH}_best.json"
  if [ -f "$BEST_FILE" ]; then
    PPL=$(python3 -c "import json; d=json.load(open('$BEST_FILE')); print(f\"{d.get('best_val_ppl', 0):.2f}\")" 2>/dev/null || echo "?")
    echo "      $ARCH: val_ppl=$PPL"
  else
    echo "      $ARCH: MISSING"
  fi
done

# ── Phase 3: Final multi-seed training ────────────────────────────────────────
echo ""
echo ">>> [$(date '+%H:%M:%S')] Phase 3: Final training (${FINAL_EPOCHS} epochs × seeds: $SEEDS)"

# shellcheck disable=SC2086
"$PYTHON" scripts/train_final_small_lms.py \
  --seeds $SEEDS \
  --epochs "$FINAL_EPOCHS" \
  2>&1 | tee /tmp/experiment_final_training.log

# ── Phase 4: Evaluation ───────────────────────────────────────────────────────
echo ""
echo ">>> [$(date '+%H:%M:%S')] Phase 4: Evaluation"

"$PYTHON" scripts/eval_small_lms.py \
  --out-csv artifacts/slm_final_eval.csv \
  2>&1 | tee /tmp/experiment_eval.log

# ── Phase 5: Generate report ──────────────────────────────────────────────────
echo ""
echo ">>> [$(date '+%H:%M:%S')] Phase 5: Report generation"

"$PYTHON" scripts/comprehensive_training_report.py \
  --phase report-only \
  --out-dir artifacts/full_run \
  2>&1 | tee /tmp/experiment_report.log

echo ""
echo "================================================================"
echo "  ALL DONE at $(date '+%H:%M:%S')"
echo "  Eval CSV:  artifacts/slm_final_eval.csv"
echo "  Report:    artifacts/full_run/report/report.html"
echo "================================================================"
