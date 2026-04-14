#!/bin/bash
set -e
cd /home/serkan/llm_training/slm/npc_backend_scaffold

# Clean previous run
rm -rf artifacts/full_run
mkdir -p artifacts/full_run

# Set up environment
export PYTHONPATH=/home/serkan/llm_training/slm/npc_backend_scaffold

# Run comprehensive training with all phases, 3 seeds
/home/serkan/llm_training/.venv/bin/python -u scripts/comprehensive_training_report.py \
    --phase all \
    --n-seeds 3 \
    --out-dir artifacts/full_run \
    2>&1

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  TRAINING COMPLETE"
echo "═══════════════════════════════════════════════════════════════════"
echo "View results: artifacts/full_run/"
echo "View MLflow: mlflow ui --backend-store-uri ./mlruns"
