#!/usr/bin/env bash
# Waits for Stage 1 latent training to finish, then kicks off:
#   1. 500-episode real data generation
#   2. Stage 2 response training (once data is ready)
set -euo pipefail
cd "$(dirname "$0")"

set -a; source .env; set +a

TRAIN_PID="${1:-}"
TRAIN_LOG="train_latent.log"
GEN_LOG="data_gen_500.log"

# ── Wait for training to finish ────────────────────────────────────────────────
if [[ -n "$TRAIN_PID" ]]; then
    echo "[launch_after_train] Waiting for training PID $TRAIN_PID to finish..."
    while kill -0 "$TRAIN_PID" 2>/dev/null; do
        PROGRESS=$(cat -v "$TRAIN_LOG" 2>/dev/null | tr '\r' '\n' | \
                   grep -oP 'Epoch \d+:\s+\d+%' | tail -1 || echo "running")
        echo "  training: $PROGRESS  $(date '+%H:%M:%S')"
        sleep 60
    done
    echo "[launch_after_train] Training finished."
fi

# ── Sanity-check best model was saved ─────────────────────────────────────────
if [[ ! -d "checkpoints/latent_predictor_best" ]]; then
    echo "[launch_after_train] WARNING: best model checkpoint not found at checkpoints/latent_predictor_best"
fi

# ── Clear stale dry-run data and re-generate 500 real episodes ────────────────
echo "[launch_after_train] Clearing dry-run data and starting 500-episode real generation..."
rm -rf data/raw_episodes data/validated_turns data/counterfactuals \
       data/merged_validated data/packaged data/splits

nohup env PYTHONUNBUFFERED=1 .venv/bin/python run_data_gen.py \
    --config configs/data_gen.yaml \
    --n-episodes 500 \
    --no-mlflow \
    --stage all > "$GEN_LOG" 2>&1 &
GEN_PID=$!
echo "[launch_after_train] Generation started PID=$GEN_PID  log=$GEN_LOG"

# ── Wait for generation to finish ─────────────────────────────────────────────
echo "[launch_after_train] Waiting for generation to finish..."
while kill -0 "$GEN_PID" 2>/dev/null; do
    VALID=$(grep -oP 'valid=\d+' "$GEN_LOG" | tail -1 || echo "valid=0")
    echo "  generation: $VALID  $(date '+%H:%M:%S')"
    sleep 120
done

if ! grep -q "Generation complete" "$GEN_LOG"; then
    echo "[launch_after_train] ERROR: generation did not complete cleanly — check $GEN_LOG"
    exit 1
fi
echo "[launch_after_train] Generation done."

# ── Stage 2: response generator training ──────────────────────────────────────
echo "[launch_after_train] Starting Stage 2 response training..."
env PYTHONUNBUFFERED=1 .venv/bin/python run_train.py \
    --stage response \
    --config configs/train_response.yaml \
    2>&1 | tee train_response.log
echo "[launch_after_train] Stage 2 done."
