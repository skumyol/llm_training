#!/usr/bin/env bash
# =============================================================================
# Smoke Test with External Datasets
# =============================================================================
# Quick validation that training works with the expanded external corpus.
# Runs 2 epochs on a single small LM architecture with merged external data.
#
# Usage: bash smoke_test_external.sh [arch]
# Default arch: prefix_gpt (fastest convergence)
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT/../.venv/bin/python"
export PYTHONPATH="$ROOT"

ARCH="${1:-prefix_gpt}"
RUN_ID="smoke_ext_$(date +%H%M%S)"
LOG_FILE="$ROOT/logs/smoke_external_${ARCH}_${RUN_ID}.log"

echo "================================================================"
echo "  Smoke Test with External Datasets"
echo "  Architecture: $ARCH"
echo "  Data: data/external/merged_dialogue.txt"
echo "  Run ID: $RUN_ID"
echo "================================================================"

# Create directories
mkdir -p "$ROOT/logs" "$ROOT/artifacts/small_lm"

# Verify data exists
if [ ! -f "$ROOT/data/external/merged_dialogue.txt" ]; then
    echo "ERROR: merged_dialogue.txt not found. Run download_external_datasets.py first."
    exit 1
fi

TOKENS=$(wc -c < "$ROOT/data/external/merged_dialogue.txt")
echo "  Input tokens: ~$((TOKENS / 4)) (estimated)"
echo ""

# Create temp config
cat > "/tmp/smoke_ext_${ARCH}.yaml" <<EOF
arch: ${ARCH}
hardware_profile: rtx4070_small
train_text: ${ROOT}/data/external/merged_dialogue.txt
val_text: ${ROOT}/data/dialogue/val.txt
seq_len: 128
batch_size: 32
grad_accum: 2
lr: 3e-4
weight_decay: 0.1
epochs: 2
log_every: 50
eval_every_steps: 500
seed: 42
output_dir: artifacts/small_lm
cond_dim: 8
use_amp: false
embedding_model: null
embedding_cache: true
scheduler: cosine_warm_restarts
T_0: 2
T_mult: 1
eta_min: 1e-6
mlflow_experiment: smoke_test_external
mlflow_enabled: true
run_id: ${RUN_ID}
EOF

echo ">>> Starting training..."
"$PYTHON" -m src.train.run_small_lm \
    --config "/tmp/smoke_ext_${ARCH}.yaml" \
    --run-id "$RUN_ID" \
    --arch "$ARCH" \
    2>&1 | tee "$LOG_FILE"

# Check results
echo ""
echo ">>> Checking results..."
SUMMARY="$ROOT/artifacts/small_lm/${RUN_ID}/run_summary.json"
if [ -f "$SUMMARY" ]; then
    echo "  Run summary found:"
    cat "$SUMMARY" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f\"    Params: {d.get('model_params',0)/1e6:.1f}M\"); print(f\"    Best val PPL: {d.get('best',{}).get('val_ppl','N/A')}\")"
    echo ""
    echo "================================================================"
    echo "  ✓ Smoke test PASSED"
    echo "  Artifacts: artifacts/small_lm/${RUN_ID}/"
    echo "  Log: logs/smoke_external_${ARCH}_${RUN_ID}.log"
    echo "================================================================"
else
    echo "  ✗ Smoke test FAILED - no run_summary.json found"
    exit 1
fi
