#!/usr/bin/env bash
# slurm_lateval.sh — score one latent checkpoint.
#   sbatch scripts/slurm_lateval.sh eval_test_L5_nocf_orig
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --account=xrimlab
#SBATCH --partition=gpu-a30
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
set -euo pipefail

CFG="${1:?usage: sbatch scripts/slurm_lateval.sh <eval-config-stem>}"
cd "$HOME/llm_training"
mkdir -p slurm_logs
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PWD/llm_finetuning:${PYTHONPATH:-}"
exec python -u llm_finetuning/run_eval.py --stage latent \
    --config "llm_finetuning/configs/${CFG}.yaml"
