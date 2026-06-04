#!/usr/bin/env bash
# =============================================================================
# slurm_dry_run.sh — Quick validation that wrappers work on the cluster
# =============================================================================
# Submits minimal test jobs to verify environment, paths, and dispatch logic.
#
# Usage on login node:
#   bash scripts/slurm_dry_run.sh
#
# Each job runs in <2 minutes and logs its result.
# =============================================================================
set -euo pipefail

echo "Submitting dry-run test jobs..."

# 1. GPU wrapper dispatch test (no actual training — just import check)
GPU_JOB=$(sbatch --time=00:05:00 --output=/scratch/%u/logs/dry_gpu_%j.out \
    scripts/slurm_experiments.sh head_ablation_eval \
    --name dry_test --masking-mode 2>&1 | grep -oP '\d+')
echo "  GPU dry-run job: ${GPU_JOB}"

# 2. CPU wrapper dispatch test
echo ""
CPU_JOB=$(sbatch --time=00:05:00 --output=/scratch/%u/logs/dry_cpu_%j.out \
    scripts/slurm_cpu.sh analyze_head_utility \
    --heads-file data/splits/val_heads.jsonl \
    --output-dir /tmp/dry_head_utility 2>&1 | grep -oP '\d+')
echo "  CPU dry-run job: ${CPU_JOB}"

echo ""
echo "Monitor with:  squeue -j ${GPU_JOB},${CPU_JOB}"
echo "GPU log:       tail -f /scratch/\$USER/logs/dry_gpu_${GPU_JOB}.out"
echo "CPU log:       tail -f /scratch/\$USER/logs/dry_cpu_${CPU_JOB}.out"
