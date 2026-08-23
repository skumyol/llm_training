#!/usr/bin/env bash
# slurm_lat.sh — train one latent-predictor config.
#   sbatch scripts/slurm_lat.sh lat_L5_nocf
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --account=xrimlab
#SBATCH --partition=gpu-a30
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
set -euo pipefail

CFG="${1:?usage: sbatch scripts/slurm_lat.sh <config-stem>}"
cd "$HOME/llm_training"
mkdir -p slurm_logs
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# configs use repo-root-relative data paths, so run from the root with
# llm_finetuning on PYTHONPATH (run_train.py imports `src.training...`).
export PYTHONPATH="$PWD/llm_finetuning:${PYTHONPATH:-}"
exec python -u llm_finetuning/run_train.py --stage latent \
    --config "llm_finetuning/configs/${CFG}.yaml"
