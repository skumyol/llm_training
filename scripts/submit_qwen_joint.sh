#!/usr/bin/env bash
# Submit Qwen joint after latent + response checkpoints exist
# Usage: sbatch scripts/submit_qwen_joint.sh
#SBATCH --job-name=qwen-joint
#SBATCH --output=/scratch/%u/logs/qwen_joint_%j_%x.out
#SBATCH --error=/scratch/%u/logs/qwen_joint_%j_%x.err
#SBATCH --account=xrimlab
#SBATCH --partition=gpu-l20
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@ust.hk
set -euo pipefail
WORK_BASE="${WORK_BASE:-/scratch/${USER}}"
REPO_DIR="${WORK_BASE}/npc"
LOG_DIR="${WORK_BASE}/logs"
mkdir -p "${LOG_DIR}"

# Verify checkpoints exist before starting
if [ ! -d "${REPO_DIR}/checkpoints/latent_predictor_best" ]; then
    echo "ERROR: latent_predictor_best not found. Run latent training first." >&2
    exit 1
fi
if [ ! -d "${REPO_DIR}/checkpoints/response_generator_best" ]; then
    echo "ERROR: response_generator_best not found. Run response training first." >&2
    exit 1
fi

module purge 2>/dev/null || true
module load cuda/12.4.0 2>/dev/null || { echo "ERROR: cuda/12.4.0 not found" >&2; exit 1; }
VENV_DIR="${WORK_BASE}/venvs/llm_env"
if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
else
    echo "ERROR: LLM venv not found: ${VENV_DIR}" >&2; exit 1
fi

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/llm_finetuning"
python llm_finetuning/run_train.py --stage joint \
    --config llm_finetuning/configs/train_joint.yaml \
    2>&1 | tee "${LOG_DIR}/qwen_joint_${SLURM_JOB_ID:-manual}.log"
