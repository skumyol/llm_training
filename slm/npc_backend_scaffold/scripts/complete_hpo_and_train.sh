#!/usr/bin/env bash
# =============================================================================
# complete_hpo_and_train.sh
# Sequential completion of remaining HPO + full final training for all 6 archs.
#
# Why this script exists:
#   GRU, AWD-LSTM, Mamba-like are running in parallel (started earlier).
#   GPT, PrefixGPT, MoE need 20-trial re-searches (smoke had only 3 trials).
#   Running all 6 simultaneously causes GPU OOM, so this script:
#     1. Waits for GRU/AWD-LSTM/Mamba to finish
#     2. Runs GPT, PrefixGPT, MoE sequentially (one at a time)
#     3. Runs final training for ALL 6 architectures × 3 seeds × 30 epochs
#     4. Runs evaluation with PPL + BLEU + Distinct metrics
#
# Usage:
#   nohup bash scripts/complete_hpo_and_train.sh > /tmp/complete_pipeline.log 2>&1 &
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
export PYTHONPATH="$ROOT"
cd "$ROOT"

SEEDS="42 43 44"
FINAL_EPOCHS=30
HPO_TRIALS=20
HPO_EPOCHS=5

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Phase 1: Wait for parallel GRU / AWD-LSTM / Mamba searches ───────────────
log "Phase 1: Waiting for GRU, AWD-LSTM, Mamba-like HPO to complete..."
log "  (Polling for artifacts/optuna/small_lm_{arch}_best.json every 60s)"

for ARCH in gru awdlstm mamba_like; do
  BEST="$ROOT/artifacts/optuna/small_lm_${ARCH}_best.json"
  while [ ! -f "$BEST" ]; do
    log "  Waiting for $ARCH best.json..."
    sleep 60
  done
  PPL=$("$PYTHON" -c "import json; d=json.load(open('$BEST')); print(f\"{d.get('best_val_ppl',0):.2f}\")" 2>/dev/null || echo "?")
  log "  $ARCH done: val_ppl=$PPL"
done

log "Phase 1 complete. GPU freed by parallel searches."

# ── Phase 2: Sequential HPO for GPT, PrefixGPT, MoE ─────────────────────────
log ""
log "Phase 2: Running HPO for GPT, PrefixGPT, MoE (sequential, 20 trials each)"

for ARCH in gpt prefix_gpt moe; do
  BEST="$ROOT/artifacts/optuna/small_lm_${ARCH}_best.json"
  if [ -f "$BEST" ]; then
    PPL=$("$PYTHON" -c "import json; d=json.load(open('$BEST')); print(f\"{d.get('best_val_ppl',0):.2f}\")" 2>/dev/null || echo "?")
    log "  $ARCH: already has best.json (PPL=$PPL), skipping."
    continue
  fi
  log "  Starting HPO for $ARCH ($HPO_TRIALS trials × $HPO_EPOCHS epochs)..."
  "$PYTHON" scripts/optuna_small_lm.py \
    --arch "$ARCH" \
    --n-trials "$HPO_TRIALS" \
    --epochs "$HPO_EPOCHS"
  PPL=$("$PYTHON" -c "import json; d=json.load(open('$BEST')); print(f\"{d.get('best_val_ppl',0):.2f}\")" 2>/dev/null || echo "?")
  log "  $ARCH HPO done: val_ppl=$PPL"
done

log ""
log "All 6 HPO searches complete. Summary:"
for ARCH in gru awdlstm gpt prefix_gpt moe mamba_like; do
  BEST="$ROOT/artifacts/optuna/small_lm_${ARCH}_best.json"
  if [ -f "$BEST" ]; then
    PPL=$("$PYTHON" -c "import json; d=json.load(open('$BEST')); print(f\"{d.get('best_val_ppl',0):.2f}\")" 2>/dev/null || echo "?")
    log "  $ARCH: val_ppl=$PPL"
  else
    log "  $ARCH: MISSING best.json"
  fi
done

# ── Phase 3: Final multi-seed training for all 6 architectures ───────────────
log ""
log "Phase 3: Final training — $FINAL_EPOCHS epochs × seeds $SEEDS"
log "  This will train all 6 architectures with their Optuna-best configs."

# shellcheck disable=SC2086
"$PYTHON" scripts/train_final_small_lms.py \
  --arch all \
  --seeds $SEEDS \
  --epochs "$FINAL_EPOCHS" \
  2>&1 | tee /tmp/final_training.log

# ── Phase 4: Evaluation ───────────────────────────────────────────────────────
log ""
log "Phase 4: Evaluation (PPL + BLEU-1/2 + Distinct-1/2)"

EVAL_CSV="$ROOT/artifacts/slm_final_eval_$(date +%Y%m%d_%H%M%S).csv"
"$PYTHON" scripts/eval_small_lms.py \
  --out-csv "$EVAL_CSV" \
  2>&1 | tee /tmp/final_eval.log

log ""
log "======================================================="
log "  ALL DONE"
log "  Final results: $EVAL_CSV"
log "  MLflow: mlflow ui --backend-store-uri ./mlruns"
log "======================================================="
