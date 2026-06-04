#!/usr/bin/env bash
# =============================================================================
# submit_all_experiments.sh — Submit all new experiment jobs in parallel
# =============================================================================
# Submits the full suite of new experiments (GPU + CPU) as independent Slurm
# jobs.  Each job runs on its own node; the cluster scheduler handles
# queuing.  No job dependencies are declared — all are fire-and-forget.
#
# Usage:
#   bash scripts/submit_all_experiments.sh
#
# To submit only a subset:
#   bash scripts/submit_all_experiments.sh gpu       # GPU jobs only
#   bash scripts/submit_all_experiments.sh cpu       # CPU jobs only
#   bash scripts/submit_all_experiments.sh ablation    # Head-ablation suite only
#
# After submission:
#   squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"
# =============================================================================
set -uo pipefail

SUBSET="${1:-all}"

# ── Defaults (override via env) ───────────────────────────────────────────────
GPU_PARTITION="${GPU_PARTITION:-gpu-l20}"
GPU_GPUS="${GPU_GPUS:-1}"
GPU_TIME="${GPU_TIME:-04:00:00}"
GPU_CPUS="${GPU_CPUS:-8}"

CPU_PARTITION="${CPU_PARTITION:-gpu-l20}"
CPU_CPUS="${CPU_CPUS:-4}"
CPU_TIME="${CPU_TIME:-01:00:00}"

JOBS_SUBMITTED=()
LOG_BASE="/scratch/${USER}/logs"
mkdir -p "${LOG_BASE}"

# ── Helpers ───────────────────────────────────────────────────────────────────
submit_gpu() {
    local name="$1"; shift
    local out="${LOG_BASE}/${name}_%j.out"
    local err="${LOG_BASE}/${name}_%j.err"
    local job_id
    job_id=$(sbatch \
        --job-name="${name}" \
        --output="${out}" \
        --error="${err}" \
        --partition="${GPU_PARTITION}" \
        --gpus-per-node="${GPU_GPUS}" \
        --cpus-per-task="${GPU_CPUS}" \
        --time="${GPU_TIME}" \
        --ntasks-per-node=1 \
        --mail-type=END,FAIL \
        scripts/slurm_experiments.sh "$@" \
        2>&1 | grep -oP '^Submitted batch job \K\d+' || true)
    if [ -n "${job_id}" ]; then
        echo "  [GPU] ${name} → job ${job_id}"
        JOBS_SUBMITTED+=("${job_id}")
    else
        echo "  [GPU] ${name} → FAILED to submit (QOS limit or sbatch error)"
    fi
}

submit_cpu() {
    local name="$1"; shift
    local out="${LOG_BASE}/${name}_%j.out"
    local err="${LOG_BASE}/${name}_%j.err"
    local job_id
    job_id=$(sbatch \
        --job-name="${name}" \
        --output="${out}" \
        --error="${err}" \
        --partition="${CPU_PARTITION}" \
        --cpus-per-task="${CPU_CPUS}" \
        --time="${CPU_TIME}" \
        --ntasks-per-node=1 \
        --mail-type=END,FAIL \
        scripts/slurm_cpu.sh "$@" \
        2>&1 | grep -oP '^Submitted batch job \K\d+' || true)
    if [ -n "${job_id}" ]; then
        echo "  [CPU] ${name} → job ${job_id}"
        JOBS_SUBMITTED+=("${job_id}")
    else
        echo "  [CPU] ${name} → FAILED to submit"
    fi
}

echo "================================================================"
echo "  Submitting experiment suite (subset: ${SUBSET})"
echo "  GPU partition: ${GPU_PARTITION}  |  CPU partition: ${CPU_PARTITION}"
echo "================================================================"

# ═══════════════════════════════════════════════════════════════════════════════
# GPU EXPERIMENTS — Phase 1 (independent jobs)
# ═══════════════════════════════════════════════════════════════════════════════
if [ "${SUBSET}" = "all" ] || [ "${SUBSET}" = "gpu" ] || [ "${SUBSET}" = "ablation" ]; then
    echo ""
    echo "--- Head Ablation Suite (GPU) ---"

    # Experiment A: routing-only minimal state (4 heads) — evaluate trained checkpoint
    submit_gpu "abl_a_eval" head_ablation_eval_trained \
        --heads response_policy reveal_decision value_conflict secrecy_pressure \
        --name exp_a_routing_only \
        --batch-size 2

    # Experiment B: +affect — evaluate trained checkpoint
    submit_gpu "abl_b_eval" head_ablation_eval_trained \
        --heads response_policy reveal_decision value_conflict secrecy_pressure valence threat control \
        --name exp_b_plus_affect \
        --batch-size 2

    # Experiment C: +relational — evaluate trained checkpoint
    submit_gpu "abl_c_eval" head_ablation_eval_trained \
        --heads response_policy reveal_decision value_conflict secrecy_pressure trust_level respect_level \
        --name exp_c_plus_relational \
        --batch-size 2

    # Experiment D: full 29-head baseline — evaluate trained checkpoint
    submit_gpu "abl_d_eval" head_ablation_eval_trained \
        --heads response_policy reveal_decision value_conflict secrecy_pressure valence threat control \
        trust_level respect_level affection_level familiarity_level dominance_level obligation_level \
        trust_delta respect_delta affection_delta familiarity_delta dominance_delta obligation_delta \
        tone dialogue_act risk_type repair_strategy player_intent player_knowledge player_credibility \
        duty_pressure face_pressure \
        --name exp_d_full_29head \
        --batch-size 2
fi

if [ "${SUBSET}" = "all" ] || [ "${SUBSET}" = "gpu" ]; then
    echo ""
    echo "--- Calibration (GPU) ---"

    # Temperature scaling
    submit_gpu "calib_temp" calibrate \
        --method temperature \
        --calib-heads-file data/splits/val_heads.jsonl \
        --output-dir calibrators/temperature

    # Isotonic regression
    submit_gpu "calib_iso" calibrate \
        --method isotonic \
        --calib-heads-file data/splits/val_heads.jsonl \
        --output-dir calibrators/isotonic
fi

if [ "${SUBSET}" = "all" ] || [ "${SUBSET}" = "gpu" ]; then
    echo ""
    echo "--- Leakage Classifier Training (GPU) ---"

    submit_gpu "leak_cls" train_leakage_classifier \
        --heads-file data/splits/train_heads.jsonl \
        --sft-file data/splits/train_sft.jsonl \
        --output-dir leakage_classifier \
        --hard-negatives 0.3
fi

if [ "${SUBSET}" = "all" ] || [ "${SUBSET}" = "gpu" ]; then
    echo ""
    echo "--- Threshold Sweep (GPU) ---"

    submit_gpu "thresh_sweep" sweep_selective_router \
        --config llm_finetuning/configs/eval.yaml \
        --predicted-zt eval_results/predicted_zt.jsonl \
        --test-trace data/splits/test_trace.jsonl \
        --leakage-file eval_results/sample_generations.json \
        --output eval_results/threshold_sweep.json
fi

if [ "${SUBSET}" = "all" ] || [ "${SUBSET}" = "gpu" ]; then
    echo ""
    echo "--- Relational Memory Eval (GPU) ---"

    submit_gpu "rel_mem" eval_relational_memory \
        --config llm_finetuning/configs/eval.yaml \
        --checkpoint checkpoints/latent_predictor_best \
        --test-heads data/splits/test_heads.jsonl \
        --output eval_results/relational_memory_eval.json
fi

# ═══════════════════════════════════════════════════════════════════════════════
# CPU EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════════════════
if [ "${SUBSET}" = "all" ] || [ "${SUBSET}" = "cpu" ]; then
    echo ""
    echo "--- CPU Analysis Suite ---"

    submit_cpu "head_util" analyze_head_utility \
        --heads-file data/splits/val_heads.jsonl \
        --output-dir eval_results/head_utility

    submit_cpu "agg_abl" aggregate_ablation \
        --results-dir eval_results/ablation \
        --output eval_results/ablation_matrix.md

    submit_cpu "dec_card" build_decision_card \
        --predicted-zt eval_results/predicted_zt.jsonl \
        --episode-id ep_001 \
        --turn-idx 3 \
        --output eval_results/decision_card_ep001_t3.txt

    submit_cpu "collapse" collapse_labels \
        --input data/splits/train_heads.jsonl \
        --output data/splits/train_heads_collapsed.jsonl \
        --collapse stance_deltas stance_levels

    # Phase 2 dependent jobs — submit after prerequisites finish:
    #   - aggregate_ablation: after ablation evals complete
    #   - validate_regenerate: after leakage_classifier/final exists
    #   - head_leak_corr: after validated_generations.json exists
    #   - plot_tradeoff_curves: after threshold_sweep.json exists
    #   - constraint_judge: requires running OpenAI-compatible API endpoint
    #   - dec_card_ab: requires sample_generations_full.json + sample_generations_card.json
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo "  Submission complete: ${#JOBS_SUBMITTED[@]} jobs queued"
echo "================================================================"
for jid in "${JOBS_SUBMITTED[@]}"; do
    echo "  ${jid}"
done
echo ""
echo "Monitor all jobs:"
echo "  squeue -u \$USER -o \"%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R\""
echo ""
echo "Cancel all these jobs:"
echo "  scancel ${JOBS_SUBMITTED[*]}"
echo ""
echo "Logs directory: ${LOG_BASE}"
