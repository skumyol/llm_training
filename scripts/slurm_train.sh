#!/usr/bin/env bash
# =============================================================================
# slurm_train.sh — SLURM job: LLM fine-tuning or SLM training
# =============================================================================
# Proper HPC cluster job with module loading, scratch workspace,
# and automatic MLflow logging to remote tracking server.
#
# Usage:
#   sbatch scripts/slurm_train.sh llm latent
#   sbatch scripts/slurm_train.sh llm response
#   sbatch scripts/slurm_train.sh llm joint
#   sbatch scripts/slurm_train.sh slm personality
#   sbatch scripts/slurm_train.sh slm affect
#   sbatch scripts/slurm_train.sh slm small_lm --arch gpt --seed 42
#   sbatch scripts/slurm_train.sh slm dialogue
#
# Customize GPU/time:
#   sbatch --gres=gpu:a100:2 --time=72:00:00 scripts/slurm_train.sh slm small_lm --arch gpt
#
# Environment (set in mlflow_env.sh or sbatch --export):
#   MLFLOW_TRACKING_URI    — remote MLflow server
#   WORK_BASE              — scratch directory (default: /scratch/$USER)
#   CHECKPOINT_DIR         — NFS path for checkpoints
# =============================================================================
#SBATCH --job-name=npc-train
#SBATCH --output=/scratch/%u/logs/slurm_%j_%x.out
#SBATCH --error=/scratch/%u/logs/slurm_%j_%x.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

# ── Cluster config ────────────────────────────────────────────────────────────
WORK_BASE="${WORK_BASE:-/scratch/${USER}}"
REPO_DIR="${WORK_BASE}/npc"                    # Clone repo here first
DATA_DIR="${WORK_BASE}/data"                   # Fast scratch I/O
LOG_DIR="${WORK_BASE}/logs"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${WORK_BASE}/checkpoints}"

mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}"

# ── Source MLflow config ──────────────────────────────────────────────────────
if [ -f "${REPO_DIR}/scripts/mlflow_env.sh" ]; then
    source "${REPO_DIR}/scripts/mlflow_env.sh"
fi

# ── Load modules + venv ───────────────────────────────────────────────────────
SYSTEM="${1:-}"
shift 2>/dev/null || true

case "${SYSTEM}" in
    llm)
        VENV_NAME="${VENV_NAME:-llm_env}"
        CUDA_MOD="${CUDA_MOD:-cuda/12.4.0}"
        PY_MOD="${PY_MOD:-python/3.12}"
        ;;
    slm)
        VENV_NAME="${VENV_NAME:-slm_env}"
        CUDA_MOD="${CUDA_MOD:-cuda/12.4.0}"
        PY_MOD="${PY_MOD:-python/3.12}"
        ;;
    *)
        echo "Usage: $0 {llm|slm} {stage} [...]" >&2
        exit 1
        ;;
esac

module purge 2>/dev/null || true
module load "${CUDA_MOD}" 2>/dev/null || true
module load "${PY_MOD}" 2>/dev/null || true

VENV_DIR="${WORK_BASE}/venvs/${VENV_NAME}"
if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
else
    echo "ERROR: Venv not found: ${VENV_DIR}" >&2
    echo "Run: bash ${REPO_DIR}/scripts/setup_${SYSTEM}_env.sh first" >&2
    exit 1
fi

STAGE="$1"
shift 1 2>/dev/null || true
EXTRA_ARGS=("$@")

RUN_ID="slurm_${SLURM_JOB_ID:-manual}_${SYSTEM}_${STAGE}_$(date +%Y%m%d_%H%M%S)"

echo "================================================================"
echo "  SLURM Training Job"
echo "================================================================"
echo "  Job ID:      ${SLURM_JOB_ID:-local}"
echo "  Node:        $(hostname)"
echo "  System:      ${SYSTEM}"
echo "  Stage:       ${STAGE}"
echo "  Run ID:      ${RUN_ID}"
echo "  GPU:         $(nvidia-smi -L 2>/dev/null | head -1 || echo 'none')"
echo "  MLflow:      ${MLFLOW_TRACKING_URI:-local}"
echo "  Checkpoints: ${CHECKPOINT_DIR}"
echo "================================================================"

# ── Copy data to scratch for fast I/O ─────────────────────────────────────────
if [ -d "${REPO_DIR}/data" ] && [ ! -d "${DATA_DIR}/scenario_bank" ]; then
    echo "  Copying data to scratch..."
    cp -r "${REPO_DIR}/data" "${DATA_DIR}" 2>/dev/null || true
fi

# ── Run training ──────────────────────────────────────────────────────────────
cd "${REPO_DIR}"

case "${SYSTEM}_${STAGE}" in
    llm_latent|llm_response|llm_joint)
        cd "${REPO_DIR}/llm_finetuning"
        export PYTHONPATH="${REPO_DIR}/llm_finetuning"
        python run_train.py --stage "${STAGE}" --config "configs/train_${STAGE}.yaml" "${EXTRA_ARGS[@]}" \
            2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"
        ;;
    slm_personality)
        cd "${REPO_DIR}/slm_training"
        export PYTHONPATH="${REPO_DIR}/slm_training"
        python -m src.train.run_personality --run-id "${RUN_ID}" "${EXTRA_ARGS[@]}" \
            2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"
        ;;
    slm_affect)
        cd "${REPO_DIR}/slm_training"
        export PYTHONPATH="${REPO_DIR}/slm_training"
        python -m src.train.run_affect --run-id "${RUN_ID}" "${EXTRA_ARGS[@]}" \
            2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"
        ;;
    slm_small_lm)
        cd "${REPO_DIR}/slm_training"
        export PYTHONPATH="${REPO_DIR}/slm_training"
        python -m src.train.run_small_lm --run-id "${RUN_ID}" "${EXTRA_ARGS[@]}" \
            2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"
        ;;
    slm_dialogue)
        cd "${REPO_DIR}/slm_training"
        export PYTHONPATH="${REPO_DIR}/slm_training"
        python -m src.train.run_dialogue --run-id "${RUN_ID}" "${EXTRA_ARGS[@]}" \
            2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"
        ;;
    *)
        echo "Unknown stage: ${SYSTEM}/${STAGE}" >&2
        exit 1
        ;;
esac

EXIT_CODE=$?

echo ""
echo "================================================================"
echo "  Done (exit=${EXIT_CODE})  Run: ${RUN_ID}"
echo "  Log:     ${LOG_DIR}/${RUN_ID}.log"
echo "  MLflow:  ${MLFLOW_TRACKING_URI:-local}"
echo "================================================================"
exit ${EXIT_CODE}
