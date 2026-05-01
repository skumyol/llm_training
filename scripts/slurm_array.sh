#!/usr/bin/env bash
# =============================================================================
# slurm_array.sh — SLURM job array for hyperparameter sweeps & multi-seed runs
# =============================================================================
# Submits array of training jobs. Each task = one (arch, seed) combo.
# All metrics logged to shared MLflow server on scratch.
#
# Usage:
#   # 6 archs × 3 seeds = 18 array tasks
#   sbatch --array=0-17 scripts/slurm_array.sh slm small_lm \
#       --archs gru,awdlstm,gpt,prefix_gpt,moe,mamba_like --seeds 42,43,44
#
#   # Single arch, 3 seeds = 3 array tasks
#   sbatch --array=0-2 scripts/slurm_array.sh slm small_lm \
#       --archs moe --seeds 42,43,44
#
#   # LLM 3-stage pipeline
#   sbatch --array=0-2 scripts/slurm_array.sh llm full_pipeline
#
#   # Custom partition
#   sbatch --partition=gpu-rtx4090d --array=0-2 scripts/slurm_array.sh ...
#
# Array mapping:
#   small_lm:  TASK 0 = (archs[0], seeds[0]), TASK 1 = (archs[0], seeds[1]), ...
#   full_pipeline: TASK 0 = latent, TASK 1 = response, TASK 2 = joint
# =============================================================================
#SBATCH --job-name=npc-array
#SBATCH --output=/scratch/%u/logs/slurm_%A_%a.out
#SBATCH --error=/scratch/%u/logs/slurm_%A_%a.err
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
LOG_DIR="${WORK_BASE}/logs"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${WORK_BASE}/checkpoints}"

mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}"

# ── Source config ─────────────────────────────────────────────────────────────
[ -f "${REPO_DIR}/scripts/mlflow_env.sh" ] && source "${REPO_DIR}/scripts/mlflow_env.sh"

# ── Parse args ────────────────────────────────────────────────────────────────
SYSTEM="${1:-}"; STAGE="${2:-}"; shift 2 2>/dev/null || true
ARCHS=""; SEEDS=""
for arg in "$@"; do
    case "$arg" in
        --archs=*) ARCHS="${arg#*=}" ;;
        --seeds=*) SEEDS="${arg#*=}" ;;
    esac
done
IFS=',' read -ra ARCH_LIST <<< "${ARCHS:-gpt}"
IFS=',' read -ra SEED_LIST <<< "${SEEDS:-42}"
N_SEEDS=${#SEED_LIST[@]}

# ── Load modules + venv ───────────────────────────────────────────────────────
case "${SYSTEM}" in
    llm) VENV_DIR="${WORK_BASE}/venvs/llm_env" ;;
    slm) VENV_DIR="${WORK_BASE}/venvs/slm_env" ;;
    *)   echo "Usage: $0 {llm|slm} {stage} [...]" >&2; exit 1 ;;
esac

module purge 2>/dev/null || true
module load cuda/12.4.0 2>/dev/null || {
    echo "ERROR: Could not load cuda/12.4.0" >&2
    exit 1
}

if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
else
    echo "ERROR: Venv not found: ${VENV_DIR}" >&2
    exit 1
fi

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

# ═══════════════════════════════════════════════════════════════════════════════
# SLM small LM array
# ═══════════════════════════════════════════════════════════════════════════════
if [ "${SYSTEM}" = "slm" ] && [ "${STAGE}" = "small_lm" ]; then
    ARCH_IDX=$(( TASK_ID / N_SEEDS ))
    SEED_IDX=$(( TASK_ID % N_SEEDS ))
    [ "$ARCH_IDX" -ge "${#ARCH_LIST[@]}" ] && { echo "TASK_ID=$TASK_ID out of range (${#ARCH_LIST[@]} archs × ${N_SEEDS} seeds)"; exit 0; }

    ARCH="${ARCH_LIST[$ARCH_IDX]}"
    SEED="${SEED_LIST[$SEED_IDX]}"
    RUN_ID="slurm_${SLURM_ARRAY_JOB_ID:-0}_${TASK_ID}_${ARCH}_s${SEED}"

    echo "================================================================"
    echo "  HKUST HPC — Array ${SLURM_ARRAY_JOB_ID:-?}[${TASK_ID}]"
    echo "  Arch: ${ARCH}  Seed: ${SEED}"
    echo "  Node: $(hostname)"
    echo "  GPU:  $(nvidia-smi -L 2>/dev/null | head -1 || echo 'none')"
    echo "  CUDA: $(nvcc --version 2>/dev/null | grep release | head -1 || echo '?')"
    echo "================================================================"

    export PYTHONPATH="${REPO_DIR}/slm_training"
    cd "${REPO_DIR}/slm_training"
    python -m src.train.run_small_lm --run-id "${RUN_ID}" --arch "${ARCH}" --seed "${SEED}" \
        2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"

# ═══════════════════════════════════════════════════════════════════════════════
# LLM pipeline array (3 stages)
# ═══════════════════════════════════════════════════════════════════════════════
elif [ "${SYSTEM}" = "llm" ] && [ "${STAGE}" = "full_pipeline" ]; then
    STAGES=("latent" "response" "joint")
    [ "$TASK_ID" -ge "${#STAGES[@]}" ] && exit 0
    STAGE_NAME="${STAGES[$TASK_ID]}"
    RUN_ID="slurm_${SLURM_ARRAY_JOB_ID:-0}_${TASK_ID}_${STAGE_NAME}"

    echo "================================================================"
    echo "  HKUST HPC — Array ${SLURM_ARRAY_JOB_ID:-?}[${TASK_ID}]"
    echo "  Stage: ${STAGE_NAME}"
    echo "  Node:  $(hostname)"
    echo "  GPU:   $(nvidia-smi -L 2>/dev/null | head -1 || echo 'none')"
    echo "================================================================"

    export PYTHONPATH="${REPO_DIR}/llm_finetuning"
    cd "${REPO_DIR}/llm_finetuning"
    python run_train.py --stage "${STAGE_NAME}" --config "configs/train_${STAGE_NAME}.yaml" \
        2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"

# ═══════════════════════════════════════════════════════════════════════════════
# SLM personality / affect / dialogue array (arch/seeds)
# ═══════════════════════════════════════════════════════════════════════════════
elif [ "${SYSTEM}" = "slm" ] && { [ "${STAGE}" = "personality" ] || [ "${STAGE}" = "affect" ] || [ "${STAGE}" = "dialogue" ]; }; then
    SEED_IDX=$(( TASK_ID % N_SEEDS ))
    SEED="${SEED_LIST[$SEED_IDX]}"
    RUN_ID="slurm_${SLURM_ARRAY_JOB_ID:-0}_${TASK_ID}_${STAGE}_s${SEED}"

    echo "================================================================"
    echo "  HKUST HPC — Array ${SLURM_ARRAY_JOB_ID:-?}[${TASK_ID}]"
    echo "  Stage: ${STAGE}  Seed: ${SEED}"
    echo "  Node:  $(hostname)"
    echo "================================================================"

    export PYTHONPATH="${REPO_DIR}/slm_training"
    cd "${REPO_DIR}/slm_training"
    python -m "src.train.run_${STAGE}" --run-id "${RUN_ID}" --seed "${SEED}" \
        2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"
else
    echo "Unknown combination: ${SYSTEM}/${STAGE}" >&2
    exit 1
fi

EXIT_CODE=$?
echo "Done (exit=${EXIT_CODE})  Run: ${RUN_ID}"
exit ${EXIT_CODE}
