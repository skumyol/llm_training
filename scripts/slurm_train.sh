#!/usr/bin/env bash
# =============================================================================
# slurm_train.sh — SLURM job: LLM or SLM training (HKUST HPC4 optimized)
# =============================================================================
# HKUST HPC4 conventions:
#   --account=xrimlab    (required)
#   --gpus-per-node=N    (NOT --gres=gpu)
#   --ntasks-per-node=1  (single task per node)
#   --cpus-per-task=N    (CPU cores for data loading)
#   NO --mem on GPU jobs (auto-allocated)
#
# Usage:
#   sbatch scripts/slurm_train.sh llm latent
#   sbatch scripts/slurm_train.sh slm small_lm --arch gpt --seed 42
#   sbatch scripts/slurm_train.sh slm small_lm --arch mamba_like --seed 44
#
#   # Custom GPU/time:
#   sbatch --partition=gpu-a100 --gpus-per-node=2 --time=72:00:00 \
#       scripts/slurm_train.sh slm small_lm --arch moe --seed 42
# =============================================================================
#SBATCH --job-name=npc-train
#SBATCH --output=/scratch/%u/logs/slurm_%j_%x.out
#SBATCH --error=/scratch/%u/logs/slurm_%j_%x.err
#SBATCH --account=xrimlab
#SBATCH --partition=gpu-l20
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@ust.hk

set -euo pipefail

# ── Cluster config ────────────────────────────────────────────────────────────
WORK_BASE="${WORK_BASE:-/scratch/${USER}}"
REPO_DIR="${WORK_BASE}/npc"
DATA_DIR="${WORK_BASE}/data"
LOG_DIR="${WORK_BASE}/logs"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${WORK_BASE}/checkpoints}"

mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}"

# ── Source config ─────────────────────────────────────────────────────────────
[ -f "${REPO_DIR}/scripts/mlflow_env.sh" ] && source "${REPO_DIR}/scripts/mlflow_env.sh"

# ── Parse args ────────────────────────────────────────────────────────────────
SYSTEM="${1:-}"; STAGE="${2:-}"; shift 2 2>/dev/null || true; EXTRA_ARGS=("$@")

case "${SYSTEM}" in
    llm) VENV_NAME="llm_env"; MOD="python/3.12" ;;
    slm) VENV_NAME="slm_env"; MOD="python/3.12" ;;
    *)   echo "Usage: $0 {llm|slm} {stage} [...]" >&2; exit 1 ;;
esac

# ── Load modules ──────────────────────────────────────────────────────────────
module purge 2>/dev/null || true
module load cuda/12.4.0 2>/dev/null || true
module load "${MOD}" 2>/dev/null || true

VENV_DIR="${WORK_BASE}/venvs/${VENV_NAME}"
if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
else
    echo "ERROR: Venv not found: ${VENV_DIR}" >&2
    echo "Run: bash ${REPO_DIR}/scripts/setup_${SYSTEM}_env.sh first" >&2
    exit 1
fi

RUN_ID="slurm_${SLURM_JOB_ID:-manual}_${SYSTEM}_${STAGE}_$(date +%Y%m%d_%H%M%S)"

# ── Log header ────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  HKUST HPC4 — NPC Training Job"
echo "================================================================"
echo "  Job ID:      ${SLURM_JOB_ID:-local}"
echo "  Node:        $(hostname)"
echo "  System:      ${SYSTEM} / ${STAGE}"
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
        export PYTHONPATH="${REPO_DIR}/llm_finetuning"
        python llm_finetuning/run_train.py --stage "${STAGE}" \
            --config "llm_finetuning/configs/train_${STAGE}.yaml" "${EXTRA_ARGS[@]}" \
            2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"
        ;;
    slm_personality)
        export PYTHONPATH="${REPO_DIR}/slm_training"
        python -m src.train.run_personality --run-id "${RUN_ID}" "${EXTRA_ARGS[@]}" \
            2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"
        ;;
    slm_affect)
        export PYTHONPATH="${REPO_DIR}/slm_training"
        python -m src.train.run_affect --run-id "${RUN_ID}" "${EXTRA_ARGS[@]}" \
            2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"
        ;;
    slm_small_lm)
        export PYTHONPATH="${REPO_DIR}/slm_training"
        python -m src.train.run_small_lm --run-id "${RUN_ID}" "${EXTRA_ARGS[@]}" \
            2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"
        ;;
    slm_dialogue)
        export PYTHONPATH="${REPO_DIR}/slm_training"
        python -m src.train.run_dialogue --run-id "${RUN_ID}" "${EXTRA_ARGS[@]}" \
            2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"
        ;;
    *) echo "Unknown stage: ${SYSTEM}/${STAGE}" >&2; exit 1 ;;
esac

EXIT_CODE=$?
echo "Done (exit=${EXIT_CODE})  Run: ${RUN_ID}  MLflow: ${MLFLOW_TRACKING_URI:-local}"
exit ${EXIT_CODE}
