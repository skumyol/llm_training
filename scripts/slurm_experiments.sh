#!/usr/bin/env bash
# =============================================================================
# slurm_experiments.sh — SLURM wrapper for GPU experiment scripts
# =============================================================================
# Covers head ablation, calibration, and validate-and-regenerate experiments.
# For local/non-SLURM execution, prefer scripts/experiments.sh.
#
# Usage:
#   # Head ablation training (4-head minimal state)
#   sbatch scripts/slurm_experiments.sh head_ablation_train \
#       --heads response_policy reveal_decision value_conflict secrecy_pressure \
#       --name exp_a_routing_only
#
#   # Head ablation evaluation (masking mode, no retraining)
#   sbatch scripts/slurm_experiments.sh head_ablation_eval \
#       --name exp_a_routing_only
#
#   # Calibrate classifier heads with temperature scaling
#   sbatch scripts/slurm_experiments.sh calibrate \
#       --method temperature --calib-heads-file data/splits/val_heads.jsonl \
#       --output-dir calibrators/temperature
#
#   # Validate and regenerate responses with leakage detection
#   sbatch scripts/slurm_experiments.sh validate_regenerate \
#       --input eval_results/sample_generations.json \
#       --classifier-dir leakage_classifier/final \
#       --output eval_results/validated_generations.json
#
#   # Custom partition / time:
#   sbatch --partition=gpu-a30 --time=08:00:00 \
#       scripts/slurm_experiments.sh head_ablation_train --name exp_b_plus_affect \
#       --heads response_policy reveal_decision value_conflict secrecy_pressure valence threat control
# =============================================================================
#SBATCH --job-name=npc-experiment
#SBATCH --output=/scratch/%u/logs/experiment_%j_%x.out
#SBATCH --error=/scratch/%u/logs/experiment_%j_%x.err
#SBATCH --account=xrimlab
#SBATCH --partition=gpu-l20
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@ust.hk

set -euo pipefail

# ── Cluster config ────────────────────────────────────────────────────────────
WORK_BASE="${WORK_BASE:-/scratch/${USER}}"
REPO_DIR="${REPO_DIR:-${WORK_BASE}/npc}"
LOG_DIR="${LOG_DIR:-${WORK_BASE}/logs}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${WORK_BASE}/checkpoints}"

mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}"

# Source config
[ -f "${REPO_DIR}/scripts/mlflow_env.sh" ] && source "${REPO_DIR}/scripts/mlflow_env.sh"

# Parse args
EXPERIMENT="${1:-}"; shift 2>/dev/null || true
EXTRA_ARGS=("$@")

if [ -z "${EXPERIMENT}" ]; then
    echo "Usage: sbatch scripts/slurm_experiments.sh {experiment_type} [args...]" >&2
    echo "" >&2
    echo "Experiment types:" >&2
    echo "  head_ablation_train      — train ablated predictor (GPU)" >&2
    echo "  head_ablation_eval       — evaluate ablated predictor with masking (GPU)" >&2
    echo "  head_ablation_eval_trained — evaluate a trained ablation checkpoint (GPU)" >&2
    echo "  calibrate                — temperature / isotonic calibration (GPU)" >&2
    echo "  validate_regenerate      — leakage validation + regenerate (GPU)" >&2
    echo "  sweep_selective_router   — threshold sweep for F1/leakage vs slow-path (GPU)" >&2
    echo "  eval_relational_memory   — multi-turn memory vs single-turn baseline (GPU)" >&2
    echo "  train_leakage_classifier — train binary leakage classifier (GPU/CPU)" >&2
    echo "  response_eval            — response generation evaluation (GPU)" >&2
    echo "  latent_eval              — latent state prediction evaluation (GPU)" >&2
    exit 1
fi

# Load modules + LLM venv (all new experiments use llm_finetuning)
module purge 2>/dev/null || true
module load cuda/12.4.0 2>/dev/null || {
    echo "ERROR: cuda/12.4.0 not found" >&2; exit 1
}

VENV_DIR="${WORK_BASE}/venvs/llm_env"
if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
else
    echo "ERROR: LLM venv not found: ${VENV_DIR}" >&2
    exit 1
fi

RUN_ID="exp_${SLURM_JOB_ID:-manual}_${EXPERIMENT}_$(date +%Y%m%d_%H%M%S)"

# ── Log header ────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  HKUST HPC — Experiment Job"
echo "================================================================"
echo "  Job ID:      ${SLURM_JOB_ID:-local}"
echo "  Experiment:  ${EXPERIMENT}"
echo "  Node:        $(hostname)"
echo "  Partition:   ${SLURM_JOB_PARTITION:-unknown}"
echo "  GPU:         $(nvidia-smi -L 2>/dev/null | head -1 || echo 'none')"
echo "  Run ID:      ${RUN_ID}"
echo "  Python:      $(python3 --version 2>/dev/null || echo 'not found')"
echo "================================================================"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/llm_finetuning"
set -o pipefail

# Reduce OOM failures from memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ═══════════════════════════════════════════════════════════════════════════════
# Dispatch experiments
# ═══════════════════════════════════════════════════════════════════════════════
case "${EXPERIMENT}" in
    head_ablation_train)
        python llm_finetuning/scripts/run_head_ablation.py \
            --config llm_finetuning/configs/eval.yaml \
            --train \
            "${EXTRA_ARGS[@]}" \
            2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"
        ;;

    head_ablation_eval)
        python llm_finetuning/scripts/run_head_ablation.py \
            --config llm_finetuning/configs/eval.yaml \
            --masking-mode \
            "${EXTRA_ARGS[@]}" \
            2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"
        ;;

    head_ablation_eval_trained)
        python llm_finetuning/scripts/run_head_ablation.py \
            --config llm_finetuning/configs/eval.yaml \
            --test-trace-file data/splits/val_trace.jsonl \
            "${EXTRA_ARGS[@]}" \
            2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"
        ;;

    calibrate)
        python llm_finetuning/scripts/calibrate_head.py \
            --config llm_finetuning/configs/eval.yaml \
            "${EXTRA_ARGS[@]}" \
            2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"
        ;;

    validate_regenerate)
        python llm_finetuning/scripts/validate_and_regenerate.py \
            --config llm_finetuning/configs/eval.yaml \
            "${EXTRA_ARGS[@]}" \
            2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"
        ;;

    sweep_selective_router)
        python llm_finetuning/scripts/sweep_selective_router.py \
            --config llm_finetuning/configs/eval.yaml \
            "${EXTRA_ARGS[@]}" \
            2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"
        ;;

    eval_relational_memory)
        python llm_finetuning/scripts/eval_relational_memory.py \
            --config llm_finetuning/configs/eval.yaml \
            "${EXTRA_ARGS[@]}" \
            2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"
        ;;

    train_leakage_classifier)
        python llm_finetuning/scripts/train_leakage_classifier.py \
            "${EXTRA_ARGS[@]}" \
            2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"
        ;;

    response_eval)
        CONFIG_PATH="${EXTRA_ARGS[0]:-llm_finetuning/configs/eval.yaml}"
        python llm_finetuning/run_eval.py \
            --stage response \
            --config "${CONFIG_PATH}" \
            2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"
        ;;

    latent_eval)
        CONFIG_PATH="${EXTRA_ARGS[0]:-llm_finetuning/configs/eval.yaml}"
        python llm_finetuning/run_eval.py \
            --stage latent \
            --config "${CONFIG_PATH}" \
            2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"
        ;;

    *)
        echo "Unknown experiment: ${EXPERIMENT}" >&2
        echo "Valid: head_ablation_train | head_ablation_eval | head_ablation_eval_trained | calibrate | validate_regenerate | sweep_selective_router | eval_relational_memory | train_leakage_classifier | response_eval | latent_eval" >&2
        exit 1
        ;;
esac

EXIT_CODE=$?
echo "Done (exit=${EXIT_CODE})  Run: ${RUN_ID}"
exit ${EXIT_CODE}
