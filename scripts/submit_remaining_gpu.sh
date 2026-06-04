#!/usr/bin/env bash
# Submit remaining GPU jobs one at a time
set -euo pipefail

LOG_BASE="/scratch/${USER}/logs"
mkdir -p "${LOG_BASE}"

submit_one() {
    local name="$1"; shift
    local out="${LOG_BASE}/${name}_%j.out"
    local err="${LOG_BASE}/${name}_%j.err"
    sbatch \
        --job-name="${name}" \
        --output="${out}" \
        --error="${err}" \
        --partition=gpu-l20 \
        --gpus-per-node=1 \
        --cpus-per-task=8 \
        --time=04:00:00 \
        --ntasks-per-node=1 \
        --mail-type=END,FAIL \
        scripts/slurm_experiments.sh "$@" \
        < /dev/null
}

echo "Submitting rel_mem..."
submit_one rel_mem eval_relational_memory \
    --config llm_finetuning/configs/eval.yaml \
    --checkpoint checkpoints/latent_predictor_best \
    --test-heads data/splits/test_heads.jsonl \
    --output eval_results/relational_memory_eval.json

echo "All remaining jobs submitted."
