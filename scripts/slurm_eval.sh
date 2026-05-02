#!/usr/bin/env bash
# =============================================================================
# slurm_eval.sh — SLURM job: Evaluate trained models
# =============================================================================
# Runs evaluation on already-trained checkpoints and produces metrics reports.
#
# Usage:
#   sbatch scripts/slurm_eval.sh slm         # SLM eval (encoders + dialogue)
#   sbatch scripts/slurm_eval.sh slm --dialogue-only
#   sbatch scripts/slurm_eval.sh slm --artifacts /path/to/artifacts
#   sbatch scripts/slurm_eval.sh llm latent  # LLM latent predictor eval
#   sbatch scripts/slurm_eval.sh llm all     # LLM full eval (latent+response+routing)
#   sbatch scripts/slurm_eval.sh all         # Both SLM + LLM
# =============================================================================
#SBATCH --job-name=npc-eval
#SBATCH --output=/scratch/%u/logs/eval_%j_%x.out
#SBATCH --error=/scratch/%u/logs/eval_%j_%x.err
#SBATCH --account=xrimlab
#SBATCH --partition=gpu-l20
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@ust.hk

set -euo pipefail

WORK_BASE="${WORK_BASE:-/scratch/${USER}}"
REPO_DIR="${WORK_BASE}/npc"
LOG_DIR="${WORK_BASE}/logs"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${WORK_BASE}/checkpoints}"

mkdir -p "${LOG_DIR}"

# Source config
[ -f "${REPO_DIR}/scripts/mlflow_env.sh" ] && source "${REPO_DIR}/scripts/mlflow_env.sh"

# Parse args
SYSTEM="${1:-slm}"; shift 2>/dev/null || true
EXTRA_ARGS=("$@")

# Load modules + venv
module purge 2>/dev/null || true
module load cuda/12.4.0 2>/dev/null || {
    echo "ERROR: cuda/12.4.0 not found" >&2; exit 1
}

case "${SYSTEM}" in
    slm|all)
        VENV_DIR="${WORK_BASE}/venvs/slm_env"
        if [ -f "${VENV_DIR}/bin/activate" ]; then
            source "${VENV_DIR}/bin/activate"
        else
            echo "ERROR: SLM venv not found: ${VENV_DIR}" >&2; exit 1
        fi

        echo "================================================================"
        echo "  SLM Evaluation — $(hostname)"
        echo "  Job ID: ${SLURM_JOB_ID:-local}"
        echo "  GPU:    $(nvidia-smi -L 2>/dev/null | head -1 || echo 'none')"
        echo "================================================================"

        cd "${REPO_DIR}/slm_training"
        export PYTHONPATH="${REPO_DIR}/slm_training"

        # Run comprehensive eval
        python -m src.eval.run_eval "${EXTRA_ARGS[@]}" \
            2>&1 | tee "${LOG_DIR}/eval_slm_${SLURM_JOB_ID:-manual}.log"
        EVAL_EXIT=$?

        # Also run the metrics report if available
        if python -c "from src.train.metrics_report import write_metrics_bundle; print('ok')" 2>/dev/null; then
            echo ""
            echo "── Generating metrics bundle ──"
            python -c "
from src.train.metrics_report import write_metrics_bundle
from pathlib import Path
import json, glob

artifacts = Path('artifacts')
summaries = list(artifacts.glob('**/run_summary.json'))
if summaries:
    bundle = {'summary': {'num_runs': len(summaries)}}
    write_metrics_bundle(artifacts / 'evaluation', 'slurm_eval', bundle, title='SLURM Eval Report')
    print(f'  Bundle saved to artifacts/evaluation/slurm_eval.json')
else:
    print('  [SKIP] No run_summary.json files found')
" 2>/dev/null || true
        fi
        ;;
esac

case "${SYSTEM}" in
    llm|all)
        VENV_DIR="${WORK_BASE}/venvs/llm_env"
        if [ -f "${VENV_DIR}/bin/activate" ]; then
            source "${VENV_DIR}/bin/activate"
        else
            echo "ERROR: LLM venv not found: ${VENV_DIR}" >&2; exit 1
        fi

        STAGE="${1:-all}"
        echo ""
        echo "================================================================"
        echo "  LLM Evaluation — $(hostname)"
        echo "  Stage: ${STAGE}"
        echo "  GPU:   $(nvidia-smi -L 2>/dev/null | head -1 || echo 'none')"
        echo "================================================================"

        cd "${REPO_DIR}"
        export PYTHONPATH="${REPO_DIR}/llm_finetuning"

        python llm_finetuning/run_eval.py \
            --stage "${STAGE}" \
            --config llm_finetuning/configs/eval.yaml \
            2>&1 | tee "${LOG_DIR}/eval_llm_${SLURM_JOB_ID:-manual}.log"
        EVAL_EXIT=$?
        ;;
esac

echo "Done (exit=${EVAL_EXIT:-0})"
exit ${EVAL_EXIT:-0}
