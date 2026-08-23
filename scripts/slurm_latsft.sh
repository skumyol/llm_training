#!/usr/bin/env bash
# slurm_latsft.sh — generative latent-state predictor.
#   sbatch scripts/slurm_latsft.sh train lat_S1_genstate
#   sbatch scripts/slurm_latsft.sh eval  eval_S1_genstate
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --account=xrimlab
#SBATCH --partition=gpu-a30
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
set -euo pipefail
MODE="${1:?train|eval}"; CFG="${2:?config stem}"
cd "$HOME/llm_training"; mkdir -p slurm_logs
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Configs carry repo-root-relative data paths, so stay at the root and put
# llm_finetuning on PYTHONPATH rather than cd-ing into it.
export PYTHONPATH="$PWD/llm_finetuning:${PYTHONPATH:-}"
if [ "$MODE" = "train" ]; then
  exec python -u -m src.training.train_latent_sft --config "llm_finetuning/configs/${CFG}.yaml"
else
  exec python -u -m src.eval.eval_latent_sft --config "llm_finetuning/configs/${CFG}.yaml" "${@:3}"
fi
