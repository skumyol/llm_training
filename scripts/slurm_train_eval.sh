#!/usr/bin/env bash
# =============================================================================
# slurm_train_eval.sh — SLURM job: Train + Auto-Evaluate
# =============================================================================
# Trains a model and immediately runs evaluation when training finishes.
# Uses the new train_and_eval.sh orchestration.
#
# Usage:
#   sbatch scripts/slurm_train_eval.sh slm --arch gpt --epochs 5
#   sbatch scripts/slurm_train_eval.sh slm --arch mamba_like --epochs 10 --seed 42
#   sbatch scripts/slurm_train_eval.sh llm latent --debug
#   sbatch scripts/slurm_train_eval.sh llm all
#   sbatch scripts/slurm_train_eval.sh all
# =============================================================================
#SBATCH --job-name=npc-train-eval
#SBATCH --output=/scratch/%u/logs/te_%j_%x.out
#SBATCH --error=/scratch/%u/logs/te_%j_%x.err
#SBATCH --account=xrimlab
#SBATCH --partition=gpu-l20
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@ust.hk

set -euo pipefail

WORK_BASE="${WORK_BASE:-/scratch/${USER}}"
REPO_DIR="${WORK_BASE}/npc"
LOG_DIR="${WORK_BASE}/logs"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${WORK_BASE}/checkpoints}"
MLRUNS_DIR="${WORK_BASE}/mlruns"

mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}" "${MLRUNS_DIR}"

# Source config
[ -f "${REPO_DIR}/scripts/mlflow_env.sh" ] && source "${REPO_DIR}/scripts/mlflow_env.sh"

# Parse args
SYSTEM="${1:-}"; shift 2>/dev/null || true
EXTRA_ARGS=("$@")

# Load CUDA
module purge 2>/dev/null || true
module load cuda/12.4.0 2>/dev/null || {
    echo "ERROR: cuda/12.4.0 not found" >&2; exit 1
}

RUN_ID="te_${SLURM_JOB_ID:-manual}_${SYSTEM}_$(date +%Y%m%d_%H%M%S)"

echo "================================================================"
echo "  HKUST HPC — Train + Eval"
echo "================================================================"
echo "  Job ID:    ${SLURM_JOB_ID:-local}"
echo "  Node:      $(hostname)"
echo "  System:    ${SYSTEM}"
echo "  Run ID:    ${RUN_ID}"
echo "  GPU:       $(nvidia-smi -L 2>/dev/null | head -1 || echo 'none')"
echo "================================================================"

cd "${REPO_DIR}"

# ── SLM train + eval ──────────────────────────────────────────────────────────
if [ "${SYSTEM}" = "slm" ]; then
    VENV_DIR="${WORK_BASE}/venvs/slm_env"
    [ -f "${VENV_DIR}/bin/activate" ] || { echo "ERROR: SLM venv not found" >&2; exit 1; }
    source "${VENV_DIR}/bin/activate"

    echo ""
    echo "── Phase 1: Training ──────────────────────────────────────────"
    cd "${REPO_DIR}/slm_training"
    export PYTHONPATH="${REPO_DIR}/slm_training"

    python -m src.train.run_small_lm --run-id "${RUN_ID}" "${EXTRA_ARGS[@]}" \
        2>&1 | tee "${LOG_DIR}/${RUN_ID}_train.log"
    TRAIN_EXIT=$?

    if [ ${TRAIN_EXIT} -ne 0 ]; then
        echo "ERROR: Training failed (exit=${TRAIN_EXIT})" >&2
        exit ${TRAIN_EXIT}
    fi

    echo ""
    echo "── Phase 2: Evaluation ────────────────────────────────────────"
    python -m src.eval.run_eval \
        2>&1 | tee "${LOG_DIR}/${RUN_ID}_eval.log"
    EVAL_EXIT=$?

# ── LLM train + eval ──────────────────────────────────────────────────────────
elif [ "${SYSTEM}" = "llm" ]; then
    STAGE="${1:-latent}"
    VENV_DIR="${WORK_BASE}/venvs/llm_env"
    [ -f "${VENV_DIR}/bin/activate" ] || { echo "ERROR: LLM venv not found" >&2; exit 1; }
    source "${VENV_DIR}/bin/activate"

    export PYTHONPATH="${REPO_DIR}/llm_finetuning"
    cd "${REPO_DIR}"

    echo ""
    echo "── Phase 1: Training (${STAGE}) ────────────────────────────────"
    python llm_finetuning/run_train.py \
        --stage "${STAGE}" \
        --config "llm_finetuning/configs/train_${STAGE}.yaml" \
        2>&1 | tee "${LOG_DIR}/${RUN_ID}_train.log"
    TRAIN_EXIT=$?

    if [ ${TRAIN_EXIT} -ne 0 ]; then
        echo "ERROR: Training failed (exit=${TRAIN_EXIT})" >&2
        exit ${TRAIN_EXIT}
    fi

    echo ""
    echo "── Phase 2: Evaluation (${STAGE}) ──────────────────────────────"
    python llm_finetuning/run_eval.py \
        --stage "${STAGE}" \
        --config llm_finetuning/configs/eval.yaml \
        2>&1 | tee "${LOG_DIR}/${RUN_ID}_eval.log"
    EVAL_EXIT=$?

# ── Full pipeline (SLM + LLM) ─────────────────────────────────────────────────
elif [ "${SYSTEM}" = "all" ]; then
    echo "  Full pipeline not yet supported in single Slurm job."
    echo "  Use separate calls:"
    echo "    sbatch scripts/slurm_train_eval.sh slm --arch gpt --epochs 30"
    echo "    sbatch scripts/slurm_train_eval.sh llm latent"
    exit 1
else
    echo "Usage: $0 {slm|llm} [...]" >&2
    exit 1
fi

echo ""
echo "================================================================"
echo "  Train+Eval complete (exit=${EVAL_EXIT:-0})"
echo "  Run ID: ${RUN_ID}"
echo "================================================================"
exit ${EVAL_EXIT:-0}
