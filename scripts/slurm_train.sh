#!/usr/bin/env bash
# =============================================================================
# slurm_train.sh — SLURM job: LLM or SLM training (HKUST HPC — Spack)
# =============================================================================
# HKUST HPC conventions:
#   --account=xrimlab       (REQUIRED)
#   --gpus-per-node=N       (NOT --gres=gpu)
#   --ntasks-per-node=1     (single task per node)
#   --cpus-per-task=N       (CPU cores for data loading)
#   NO --mem on GPU jobs    (auto-allocated)
#
# Available GPU partitions:
#   gpu-a30       — NVIDIA A30 (24GB), 15 nodes
#   gpu-l20       — NVIDIA L20 (48GB), 6 nodes
#   gpu-rtx4090d  — NVIDIA RTX 4090D (24GB), 2 nodes
#
# Usage:
#   sbatch scripts/slurm_train.sh llm latent
#   sbatch scripts/slurm_train.sh llm response
#   sbatch scripts/slurm_train.sh llm joint
#   sbatch scripts/slurm_train.sh slm small_lm --arch gpt --seed 42
#   sbatch scripts/slurm_train.sh slm small_lm --arch mamba_like --seed 42
#   sbatch scripts/slurm_train.sh slm personality
#   sbatch scripts/slurm_train.sh slm affect
#   sbatch scripts/slurm_train.sh slm dialogue
#
#   # Custom GPU/time:
#   sbatch --partition=gpu-l20 --gpus-per-node=2 --time=72:00:00 \
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
MLRUNS_DIR="${WORK_BASE}/mlruns"

mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}" "${MLRUNS_DIR}"

# ── Source cluster config ─────────────────────────────────────────────────────
if [ -f "${REPO_DIR}/scripts/mlflow_env.sh" ]; then
    source "${REPO_DIR}/scripts/mlflow_env.sh"
fi

# ── Parse args ────────────────────────────────────────────────────────────────
SYSTEM="${1:-}"; STAGE="${2:-}"; shift 2 2>/dev/null || true; EXTRA_ARGS=("$@")

case "${SYSTEM}" in
    llm) VENV_NAME="llm_env"; VENV_DIR="${WORK_BASE}/venvs/llm_env" ;;
    slm) VENV_NAME="slm_env"; VENV_DIR="${WORK_BASE}/venvs/slm_env" ;;
    *)   echo "Usage: $0 {llm|slm} {stage} [...]" >&2; exit 1 ;;
esac

# ── Load modules (from Spack) ─────────────────────────────────────────────────
# The HKUST cluster provides modules via Spack (lmod).
# cuda/12.4.0 is a Spack-generated module.
module purge 2>/dev/null || true
module load cuda/12.4.0 2>/dev/null || {
    echo "ERROR: Could not load cuda/12.4.0. Make sure Spack is set up." >&2
    echo "Try: source /opt/shared/spack/share/spack/setup-env.sh" >&2
    exit 1
}

# ── Activate venv ─────────────────────────────────────────────────────────────
if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
else
    echo "ERROR: Venv not found: ${VENV_DIR}" >&2
    echo "Run: bash ${REPO_DIR}/scripts/env_setup_spack.sh first" >&2
    exit 1
fi

# ── Run ID ────────────────────────────────────────────────────────────────────
RUN_ID="slurm_${SLURM_JOB_ID:-manual}_${SYSTEM}_${STAGE}_$(date +%Y%m%d_%H%M%S)"

# ── Log header ────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  HKUST HPC — NPC Training Job"
echo "================================================================"
echo "  Job ID:      ${SLURM_JOB_ID:-local}"
echo "  Job Name:    ${SLURM_JOB_NAME:-unknown}"
echo "  Node:        $(hostname)"
echo "  Partition:   ${SLURM_JOB_PARTITION:-unknown}"
echo "  System:      ${SYSTEM} / ${STAGE}"
echo "  Run ID:      ${RUN_ID}"
echo "  GPUs:        $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "  GPU:         $(nvidia-smi -L 2>/dev/null | head -1 || echo 'none')"
echo "  MLflow:      ${MLFLOW_TRACKING_URI:-file://${MLRUNS_DIR}}"
echo "  CUDA:        $(nvcc --version 2>/dev/null | grep release | head -1 || echo 'not found')"
echo "  Python:      $(python3 --version 2>/dev/null || echo 'not found')"
echo "  PyTorch:     $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo "================================================================"

# ── Sync data to scratch if needed ────────────────────────────────────────────
if [ -d "${REPO_DIR}/data" ] && [ ! -d "${DATA_DIR}/scenario_bank" ]; then
    echo "  Copying data to scratch..."
    cp -r "${REPO_DIR}/data" "${DATA_DIR}" 2>/dev/null || true
fi

# ── Run training ──────────────────────────────────────────────────────────────
cd "${REPO_DIR}"

case "${SYSTEM}_${STAGE}" in
    # --- LLM stages ---
    llm_latent|llm_response|llm_joint)
        export PYTHONPATH="${REPO_DIR}/llm_finetuning"
        python llm_finetuning/run_train.py --stage "${STAGE}" \
            --config "llm_finetuning/configs/train_${STAGE}.yaml" "${EXTRA_ARGS[@]}" \
            2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"
        ;;
    # --- SLM stages (cd into slm_training for relative data paths) ---
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
    slm_latent)
        cd "${REPO_DIR}"
        export PYTHONPATH="${REPO_DIR}:${REPO_DIR}/llm_finetuning"
        python -m slm_training.src.train.train_latent_slm --run-id "${RUN_ID}" "${EXTRA_ARGS[@]}" \
            2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"
        ;;
    *)
        echo "Unknown stage: ${SYSTEM}/${STAGE}" >&2
        echo "Valid combos:" >&2
        echo "  llm: latent | response | joint" >&2
        echo "  slm: personality | affect | small_lm | dialogue | latent" >&2
        exit 1
        ;;
esac

EXIT_CODE=$?
echo "Done (exit=${EXIT_CODE})  Run: ${RUN_ID}  MLflow: ${MLFLOW_TRACKING_URI:-file://${MLRUNS_DIR}}"
exit ${EXIT_CODE}
