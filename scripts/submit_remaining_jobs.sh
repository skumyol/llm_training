#!/usr/bin/env bash
# Submit remaining jobs after queue clears
set -euo pipefail
cd /scratch/skumyol/npc

# SLM architecture reruns - remaining seed 42
sbatch scripts/slurm_train.sh slm small_lm --arch mamba_like --seed 42 --epochs 20
sbatch scripts/slurm_train.sh slm small_lm --arch moe --seed 42 --epochs 20

# SLM architecture reruns - seed 43
sbatch scripts/slurm_train.sh slm small_lm --arch gpt --seed 43 --epochs 20
sbatch scripts/slurm_train.sh slm small_lm --arch prefix_gpt --seed 43 --epochs 20
sbatch scripts/slurm_train.sh slm small_lm --arch mamba_like --seed 43 --epochs 20
sbatch scripts/slurm_train.sh slm small_lm --arch moe --seed 43 --epochs 20

# SLM architecture reruns - seed 44
sbatch scripts/slurm_train.sh slm small_lm --arch gpt --seed 44 --epochs 20
sbatch scripts/slurm_train.sh slm small_lm --arch prefix_gpt --seed 44 --epochs 20
sbatch scripts/slurm_train.sh slm small_lm --arch mamba_like --seed 44 --epochs 20
sbatch scripts/slurm_train.sh slm small_lm --arch moe --seed 44 --epochs 20

# Gemma baselines
sbatch scripts/slurm_gemma_baseline.sh
sbatch scripts/slurm_gemma_social.sh

echo "Remaining jobs submitted."
