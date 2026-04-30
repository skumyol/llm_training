#!/usr/bin/env bash
# =============================================================================
# TRACK C — Large Language Model Fine-tuning
# =============================================================================
#
# What this does:
#   Fine-tunes a pre-trained LLM on NPC dialogue data using LoRA + soft-prefix
#   conditioning from the Track A personality and affect encoders.
#
# Model options:
#
#   --model tinyllama  (default, recommended)
#     Model : TinyLlama-1.1B-Chat-v1.0
#     Method: LoRA (rank=16, α=32) on attention projections (q,v)
#     VRAM  : ~8 GB (fits RTX 3070/4070)
#     Time  : ~2–4 hours on RTX 4070
#
#   --model gemma-e2b  (Gemma 4 with 2 active experts - RECOMMENDED)
#     Model : google/gemma-4-E2B (~2B active params, 16B total MoE)
#     Method: LoRA + 4-bit NF4 quantisation (QLoRA)
#     VRAM  : ~12 GB (fits RTX 3090/4090/2080Ti)
#     Time  : ~3–6 hours on RTX 4090
#
#   --model gemma-e4b  (Gemma 4 with 4 active experts)
#     Model : google/gemma-4-E4B (~4B active params, 16B total MoE)
#     Method: LoRA + 4-bit NF4 quantisation (QLoRA)
#     VRAM  : ~16 GB (fits RTX 4090 / A100)
#     Time  : ~4–8 hours on RTX 4090
#     Note  : E4B is 2× slower than E2B but higher quality
#
# Conditioning (from Track A):
#   The soft-prefix MLP maps [p_vec (5-dim OCEAN); a_vec (3-dim VAD)] → 8 prefix tokens
#   prepended to every prompt. This injects personality and emotional state
#   without modifying the LLM weights.
#
# Prerequisites:
#   - Trained personality encoder: artifacts/personality_encoder/*/best_model/
#   - Trained affect encoder:      artifacts/affect_encoder/*/best_model/
#   - Dialogue data:               data/dialogue/train.jsonl + val.jsonl
#
# Output: artifacts/dialogue/<run_id>/
#   run.log, step_metrics.csv, epoch_metrics.csv, best_model/, run_summary.json
#
# MLflow experiment: dialogue_lm
#
# Usage:
#   bash finetune_dialogue_lm.sh                     # TinyLlama (default)
#   bash finetune_dialogue_lm.sh --model tinyllama
#   bash finetune_dialogue_lm.sh --model gemma
#   bash finetune_dialogue_lm.sh --model tinyllama --epochs 5 --run-id exp_v2
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT/../.venv/bin/python"
export PYTHONPATH="$ROOT"

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL="tinyllama"
RUN_ID="dialogue_$(date +%Y%m%d_%H%M%S)"
EPOCHS=10
LOG_DIR="$ROOT/logs"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)   MODEL="$2";   shift 2 ;;
    --run-id)  RUN_ID="$2";  shift 2 ;;
    --epochs)  EPOCHS="$2";  shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/dialogue_lm_${MODEL}_${RUN_ID}.log"

echo "================================================================" | tee "$LOG_FILE"
echo "  TRACK C — Dialogue LM Fine-tuning" | tee -a "$LOG_FILE"
echo "  Model  : $MODEL" | tee -a "$LOG_FILE"
echo "  Run ID : $RUN_ID" | tee -a "$LOG_FILE"
echo "  Epochs : $EPOCHS" | tee -a "$LOG_FILE"
echo "  Log    : $LOG_FILE" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"

# ── Check prerequisites ───────────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo ">>> Checking prerequisites..." | tee -a "$LOG_FILE"

DATA_OK=true
for F in "data/dialogue/train.jsonl" "data/dialogue/val.jsonl"; do
  if [ ! -f "$ROOT/$F" ]; then
    echo "    MISSING: $F" | tee -a "$LOG_FILE"
    DATA_OK=false
  fi
done

if [ "$DATA_OK" = false ]; then
  echo "" | tee -a "$LOG_FILE"
  echo "    ERROR: dialogue data not found." | tee -a "$LOG_FILE"
  echo "    Generate data first:" | tee -a "$LOG_FILE"
  echo "      python -m src.data.prepare_dialogue_data" | tee -a "$LOG_FILE"
  exit 1
fi
echo "    Data: OK" | tee -a "$LOG_FILE"

# ── Launch fine-tuning ────────────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo ">>> Fine-tuning $MODEL..." | tee -a "$LOG_FILE"

if [ "$MODEL" = "tinyllama" ]; then
  "$PYTHON" -m src.train.run_dialogue \
    --config  configs/dialogue.yaml \
    --run-id  "$RUN_ID" \
    --epochs  "$EPOCHS" \
    2>&1 | tee -a "$LOG_FILE"

elif [ "$MODEL" = "gemma-e2b" ] || [ "$MODEL" = "gemma-e4b" ]; then
  # Gemma 4 models via native transformers + PEFT (no Unsloth needed)
  GEMMA_MODEL="google/gemma-4-E2B"
  if [ "$MODEL" = "gemma-e4b" ]; then
    GEMMA_MODEL="google/gemma-4-E4B"
  fi
  echo "    Using model: $GEMMA_MODEL" | tee -a "$LOG_FILE"
  "$PYTHON" -m src.train.run_gemma_unsloth \
    --config  configs/dialogue_gemma_unsloth.yaml \
    --run-id  "$RUN_ID" \
    --epochs  "$EPOCHS" \
    --base-model-name "$GEMMA_MODEL" \
    2>&1 | tee -a "$LOG_FILE"

else
  echo "Unknown model: $MODEL  (choose: tinyllama, gemma-e2b, gemma-e4b)" | tee -a "$LOG_FILE"
  exit 1
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
echo "  DONE — Dialogue LM ($MODEL)" | tee -a "$LOG_FILE"
echo "  Artifacts : artifacts/dialogue/${RUN_ID}/" | tee -a "$LOG_FILE"
echo "  MLflow    : mlflow ui --backend-store-uri ./mlruns" | tee -a "$LOG_FILE"
echo "================================================================" | tee -a "$LOG_FILE"
