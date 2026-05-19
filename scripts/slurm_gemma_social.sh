#!/usr/bin/env bash
#SBATCH --job-name=gemma-social
#SBATCH --output=/scratch/%u/logs/gemma_social_%j_%x.out
#SBATCH --error=/scratch/%u/logs/gemma_social_%j_%x.err
#SBATCH --account=xrimlab
#SBATCH --partition=gpu-l20
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@ust.hk
set -euo pipefail
WORK_BASE="${WORK_BASE:-/scratch/${USER}}"
REPO_DIR="${WORK_BASE}/npc"
LOG_DIR="${WORK_BASE}/logs"
mkdir -p "${LOG_DIR}"
module purge 2>/dev/null || true
module load cuda/12.4.0 2>/dev/null || { echo "ERROR: cuda/12.4.0 not found" >&2; exit 1; }
VENV_DIR="${WORK_BASE}/venvs/slm_env"
if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
else
    echo "ERROR: SLM venv not found: ${VENV_DIR}" >&2; exit 1
fi
cd "${REPO_DIR}/slm_training"
export PYTHONPATH="${REPO_DIR}/slm_training"
python -m src.train.run_gemma_unsloth \
    --config configs/dialogue_gemma4_social_state.yaml \
    2>&1 | tee "${LOG_DIR}/gemma_social_${SLURM_JOB_ID:-manual}.log"
