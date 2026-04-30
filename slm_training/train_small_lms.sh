#!/usr/bin/env bash
# =============================================================================
# TRACK B — Small Language Model Training
# =============================================================================
#
# What this does:
#   1. (Optional) Per-architecture Optuna HPO:
#        20 trials × 5 epochs each, tailored search space per arch
#        GRU/AWD-LSTM: short seq_len (64-128), lower lr (RNN-specific)
#        GPT/PrefixGPT/MoE: transformer-scale search
#        Mamba-like: seq_len capped at 64 (O(L) Python scan constraint)
#   2. Final multi-seed training: best Optuna params × 3 seeds × 30 epochs
#   3. Evaluation: val PPL + Distinct-1/2 diversity metrics
#
# Architectures trained:
#   gru         SmallGRULM       ~4-8M params   Gated Recurrent Unit LM
#   awdlstm     AWDLSTMLM        ~8-20M params  LSTM + DropConnect + Variational Dropout
#   gpt         TinyGPTLM        ~2-10M params  Decoder-only causal Transformer
#   prefix_gpt  PrefixTinyGPTLM  ~2-10M params  GPT + personality conditioning prefix
#   moe         TinyMoELM        ~8-20M params  Sparse Mixture-of-Experts Transformer
#   mamba_like  MambaLikeLM      ~4-15M params  Selective State Space Model (pure PyTorch)
#
# Data: data/dialogue/train.txt + val.txt  (~545K tokens, ~2,183 dialogue turns)
# Tokeniser: GPT-2 BPE via tiktoken (vocab = 50,257)
# Metric: Validation perplexity (lower is better)
#
# Output: artifacts/small_lm/<run_id>/
#   run.log, step_metrics.csv, epoch_metrics.csv, best_model.pt, run_summary.json
#
# Optuna bests: artifacts/optuna/small_lm_<arch>_best.json
# Final results summary: artifacts/small_lm_final_results.json
# MLflow experiment: small_lm_final
#
# Usage:
#   bash train_small_lms.sh                       # full pipeline (HPO + final)
#   bash train_small_lms.sh --skip-hpo            # use existing Optuna bests
#   bash train_small_lms.sh --hpo-only            # run HPO only
#   bash train_small_lms.sh --arch gru            # single architecture
#   bash train_small_lms.sh --arch gru --hpo-only # HPO for one arch
#   bash train_small_lms.sh --trials 10           # quick HPO (10 trials/arch)
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT/../.venv/bin/python"
export PYTHONPATH="$ROOT"

# ── Defaults ──────────────────────────────────────────────────────────────────
ARCH="all"
SKIP_HPO=false
HPO_ONLY=false
N_SEEDS=3
HPO_TRIALS=20
HPO_EPOCHS=5
FINAL_EPOCHS=30
LOG_DIR="$ROOT/logs"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-hpo)    SKIP_HPO=true;      shift ;;
    --hpo-only)    HPO_ONLY=true;      shift ;;
    --arch)        ARCH="$2";          shift 2 ;;
    --seeds)       N_SEEDS="$2";       shift 2 ;;
    --trials)      HPO_TRIALS="$2";    shift 2 ;;
    --hpo-epochs)  HPO_EPOCHS="$2";    shift 2 ;;
    --epochs)      FINAL_EPOCHS="$2";  shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR"
RUN_TAG="slm_$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/small_lms_${RUN_TAG}.log"

echo "================================================================" | tee "$LOG_FILE"
echo "  TRACK B — Small Language Models" | tee -a "$LOG_FILE"
echo "  Architectures : $ARCH" | tee -a "$LOG_FILE"
echo "  HPO trials    : $HPO_TRIALS × $HPO_EPOCHS epochs/trial" | tee -a "$LOG_FILE"
echo "  Final training: $FINAL_EPOCHS epochs × $N_SEEDS seeds" | tee -a "$LOG_FILE"
echo "  Log           : $LOG_FILE" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

# ── Phase 1: Optuna HPO ───────────────────────────────────────────────────────
if [ "$SKIP_HPO" = false ]; then
  echo "" | tee -a "$LOG_FILE"
  echo ">>> Phase 1: Per-architecture Optuna HPO" | tee -a "$LOG_FILE"

  "$PYTHON" scripts/optuna_small_lm.py \
    --arch       "$ARCH" \
    --n-trials   "$HPO_TRIALS" \
    --epochs     "$HPO_EPOCHS" \
    --skip-existing \
    2>&1 | tee -a "$LOG_FILE"

  echo "" | tee -a "$LOG_FILE"
  echo "    HPO complete. Best configs:" | tee -a "$LOG_FILE"
  for F in "$ROOT"/artifacts/optuna/small_lm_*_best.json; do
    ANAME=$(basename "$F" | sed 's/small_lm_//;s/_best.json//')
    PPL=$("$PYTHON" -c "import json; d=json.load(open('$F')); print(f\"{d.get('best_val_ppl',0):.1f}\")" 2>/dev/null || echo "?")
    echo "      $ANAME: val_ppl=$PPL" | tee -a "$LOG_FILE"
  done
else
  echo "" | tee -a "$LOG_FILE"
  echo ">>> Phase 1: HPO — SKIPPED (--skip-hpo)" | tee -a "$LOG_FILE"
fi

if [ "$HPO_ONLY" = true ]; then
  echo "" | tee -a "$LOG_FILE"
  echo ">>> --hpo-only set — stopping after HPO." | tee -a "$LOG_FILE"
  exit 0
fi

# ── Phase 2: Final multi-seed training ───────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo ">>> Phase 2: Final training with Optuna-best params" | tee -a "$LOG_FILE"
echo "    $FINAL_EPOCHS epochs × $N_SEEDS seeds per architecture" | tee -a "$LOG_FILE"

SEEDS_STR=""
for S in $(seq 42 $((41 + N_SEEDS))); do SEEDS_STR="$SEEDS_STR $S"; done

"$PYTHON" scripts/train_final_small_lms.py \
  --arch   "$ARCH" \
  --seeds  $SEEDS_STR \
  --epochs "$FINAL_EPOCHS" \
  2>&1 | tee -a "$LOG_FILE"

# ── Phase 3: Evaluation ───────────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo ">>> Phase 3: Evaluation (PPL + Distinct-1/2)" | tee -a "$LOG_FILE"

"$PYTHON" scripts/eval_small_lms.py \
  --out-csv artifacts/slm_eval_${RUN_TAG}.csv \
  2>&1 | tee -a "$LOG_FILE"

# ── Summary ───────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "  DONE — Small Language Models" | tee -a "$LOG_FILE"
echo "  Checkpoints : artifacts/small_lm/final_*/" | tee -a "$LOG_FILE"
echo "  Results CSV : artifacts/slm_eval_${RUN_TAG}.csv" | tee -a "$LOG_FILE"
echo "  MLflow      : mlflow ui --backend-store-uri ./mlruns" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
