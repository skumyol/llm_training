#!/usr/bin/env bash
# Orchestrates the remaining pipeline steps:
# 1. Waits for Data Generation to finish
# 2. Runs Stage 2 (Response SFT)
# 3. Runs Stage 3 (Joint Training)

set -euo pipefail
cd "$(dirname "$0")"

GEN_LOG="data_gen_500.log"
RESPONSE_LOG="train_response.log"
JOINT_LOG="train_joint.log"

# 1. Wait for Data Generation
echo "[Pipeline] Checking data generation status..."
# Find the PID of the running data gen process
GEN_PID=$(pgrep -f "run_data_gen.py" || echo "")

if [[ -n "$GEN_PID" ]]; then
    echo "[Pipeline] Data generation is running (PID $GEN_PID). Waiting for it to finish..."
    while kill -0 "$GEN_PID" 2>/dev/null; do
        VALID=$(grep -oP 'valid=\d+' "$GEN_LOG" | tail -1 || echo "valid=?")
        echo "  generation: $VALID  $(date '+%H:%M:%S')"
        sleep 60
    done
    echo "[Pipeline] Data generation process finished."
else
    echo "[Pipeline] No data generation process found. Assuming it completed or needs restart."
    # Check if we have data
    if [[ -f "data/splits/train_sft.jsonl" ]]; then
        echo "[Pipeline] Found packaged data. Proceeding to training."
    else
        echo "[Pipeline] ERROR: No running generation and no packaged data found."
        echo "Please check $GEN_LOG or restart generation."
        exit 1
    fi
fi

# Double check generation success by looking for "Packaging" in log if we just waited
if [[ -n "$GEN_PID" ]] && ! grep -q "Packaging" "$GEN_LOG"; then
    echo "[Pipeline] WARNING: Generation log doesn't show 'Packaging'. Checking file existence..."
    if [[ ! -f "data/splits/train_sft.jsonl" ]]; then
         echo "[Pipeline] ERROR: Data generation failed. See $GEN_LOG"
         exit 1
    fi
fi

# 2. Stage 2: Response SFT
echo "[Pipeline] Starting Stage 2: Response Training..."
env PYTHONUNBUFFERED=1 .venv/bin/python run_train.py \
    --stage response \
    --config configs/train_response.yaml \
    2>&1 | tee "$RESPONSE_LOG"

if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    echo "[Pipeline] Stage 2 Training failed. See $RESPONSE_LOG"
    exit 1
fi
echo "[Pipeline] Stage 2 completed."

# 3. Stage 3: Joint Training
echo "[Pipeline] Starting Stage 3: Joint Training..."
env PYTHONUNBUFFERED=1 .venv/bin/python run_train.py \
    --stage joint \
    --config configs/train_joint.yaml \
    2>&1 | tee "$JOINT_LOG"

if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    echo "[Pipeline] Stage 3 Training failed. See $JOINT_LOG"
    exit 1
fi

echo "[Pipeline] All stages completed successfully!"
