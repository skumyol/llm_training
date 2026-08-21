#!/usr/bin/env bash
# =============================================================================
# slurm_slm_push.sh — SLM score-push sweep (config × seed job array)
# =============================================================================
# Unlike slurm_array.sh (which only varies --arch/--seed), this drives
# run_small_lm.py from YAML configs so recipe changes are versioned artifacts
# rather than command-line flags nobody can reconstruct later.
#
# Array mapping: TASK_ID = config_idx * N_SEEDS + seed_idx
#
# Usage:
#   # 4 configs x 3 seeds = 12 tasks
#   sbatch --array=0-11 scripts/slurm_slm_push.sh \
#       --configs=slm_A_baseline,slm_B_improved,slm_C_wide,slm_D_finetune \
#       --seeds=42,43,44
#
#   # single config, single seed (smoke)
#   sbatch --array=0-0 scripts/slurm_slm_push.sh --configs=slm_B_improved --seeds=42
# =============================================================================
#SBATCH --job-name=slm-push
#SBATCH --output=/scratch/%u/logs/slmpush_%A_%a.out
#SBATCH --error=/scratch/%u/logs/slmpush_%A_%a.err
#SBATCH --account=xrimlab
#SBATCH --partition=gpu-a30
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@ust.hk

set -euo pipefail

WORK_BASE="${WORK_BASE:-/scratch/${USER}}"
REPO_DIR="${WORK_BASE}/npc"
LOG_DIR="${WORK_BASE}/logs"
VENV_DIR="${WORK_BASE}/venvs/slm_env"
mkdir -p "${LOG_DIR}"

CONFIGS=""; SEEDS="42"
for arg in "$@"; do
    case "$arg" in
        --configs=*) CONFIGS="${arg#*=}" ;;
        --seeds=*)   SEEDS="${arg#*=}" ;;
    esac
done
[ -z "${CONFIGS}" ] && { echo "ERROR: --configs=a,b,c required" >&2; exit 1; }

IFS=',' read -ra CFG_LIST  <<< "${CONFIGS}"
IFS=',' read -ra SEED_LIST <<< "${SEEDS}"
N_SEEDS=${#SEED_LIST[@]}

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
CFG_IDX=$(( TASK_ID / N_SEEDS ))
SEED_IDX=$(( TASK_ID % N_SEEDS ))
if [ "${CFG_IDX}" -ge "${#CFG_LIST[@]}" ]; then
    echo "TASK_ID=${TASK_ID} out of range (${#CFG_LIST[@]} configs x ${N_SEEDS} seeds) — nothing to do"
    exit 0
fi
CFG="${CFG_LIST[$CFG_IDX]}"
SEED="${SEED_LIST[$SEED_IDX]}"
RUN_ID="${CFG}_s${SEED}"

module purge 2>/dev/null || true
module load cuda/12.4.0 2>/dev/null || { echo "ERROR: cuda/12.4.0 not loadable" >&2; exit 1; }
if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
else
    echo "ERROR: venv not found: ${VENV_DIR} (run scripts/setup_slm_env.sh)" >&2
    exit 1
fi

echo "================================================================"
echo "  SLM push — array ${SLURM_ARRAY_JOB_ID:-?}[${TASK_ID}]"
echo "  Config: ${CFG}   Seed: ${SEED}   Run: ${RUN_ID}"
echo "  Node:   $(hostname)"
echo "  GPU:    $(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "================================================================"

export PYTHONPATH="${REPO_DIR}/slm_training"
cd "${REPO_DIR}/slm_training"

# Config D warm-starts from the external-corpus pretrain, so that job must have
# finished first. Fail loudly rather than silently training from scratch and
# reporting the result as if it were a fine-tune.
INIT_CKPT="$(awk -F': *' '/^init_from:/{print $2}' "configs/${CFG}.yaml" 2>/dev/null | tr -d '"'"'"' ')"
if [ -n "${INIT_CKPT}" ] && [ "${INIT_CKPT}" != "null" ] && [ ! -f "${INIT_CKPT}" ]; then
    echo "ERROR: ${CFG} expects init_from=${INIT_CKPT} but it does not exist." >&2
    echo "       Run the pretrain config first." >&2
    exit 1
fi

python -m src.train.run_small_lm \
    --config "configs/${CFG}.yaml" \
    --run-id "${RUN_ID}" \
    --seed "${SEED}" \
    2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"

echo "DONE ${RUN_ID}"
