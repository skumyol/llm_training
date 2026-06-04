#!/usr/bin/env bash
# =============================================================================
# slurm_cpu.sh — SLURM wrapper for CPU-only experiment scripts
# =============================================================================
# Covers head utility analysis, ablation aggregation, decision card generation,
# and label collapse. No GPU required.
#
# Usage:
#   # Analyze per-head utility for routing decisions
#   sbatch scripts/slurm_cpu.sh analyze_head_utility \
#       --heads-file data/splits/val_heads.jsonl \
#       --output-dir eval_results/head_utility
#
#   # Aggregate ablation results into markdown table
#   sbatch scripts/slurm_cpu.sh aggregate_ablation \
#       --results-dir eval_results/ablation \
#       --output eval_results/ablation_matrix.md
#
#   # Build a decision card for a specific turn
#   sbatch scripts/slurm_cpu.sh build_decision_card \
#       --predicted-zt eval_results/predicted_zt.jsonl \
#       --episode-id ep_001 --turn-idx 3
#
#   # Collapse fine-grained labels to coarser categories
#   sbatch scripts/slurm_cpu.sh collapse_labels \
#       --input data/splits/train_heads.jsonl \
#       --output data/splits/train_heads_collapsed.jsonl \
#       --collapse stance_deltas stance_levels
# =============================================================================
#SBATCH --job-name=npc-cpu
#SBATCH --output=/scratch/%u/logs/cpu_%j_%x.out
#SBATCH --error=/scratch/%u/logs/cpu_%j_%x.err
#SBATCH --account=xrimlab
#SBATCH --partition=cpu
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@ust.hk

set -euo pipefail

# ── Cluster config ────────────────────────────────────────────────────────────
WORK_BASE="${WORK_BASE:-/scratch/${USER}}"
REPO_DIR="${REPO_DIR:-${WORK_BASE}/npc}"
LOG_DIR="${LOG_DIR:-${WORK_BASE}/logs}"

mkdir -p "${LOG_DIR}"

# Source config
[ -f "${REPO_DIR}/scripts/mlflow_env.sh" ] && source "${REPO_DIR}/scripts/mlflow_env.sh"

# Parse args
EXPERIMENT="${1:-}"; shift 2>/dev/null || true
EXTRA_ARGS=("$@")

if [ -z "${EXPERIMENT}" ]; then
    echo "Usage: sbatch scripts/slurm_cpu.sh {experiment_type} [args...]" >&2
    echo "" >&2
    echo "Experiment types:" >&2
    echo "  analyze_head_utility          — per-head mutual info / redundancy (CPU)" >&2
    echo "  aggregate_ablation            — ablation matrix aggregation (CPU)" >&2
    echo "  build_decision_card           — compressed decision card for one turn (CPU)" >&2
    echo "  collapse_labels               — label category collapsing (CPU)" >&2
    echo "  analyze_head_leakage_correlation — head-leakage correlation analysis (CPU)" >&2
    echo "  llm_constraint_judge          — LLM-as-judge for constraint adherence (CPU)" >&2
    echo "  eval_decision_card_ab         — decision card A/B evaluation (CPU)" >&2
    echo "  plot_tradeoff_curves          — plot leakage/F1 vs slow-path trade-offs (CPU)" >&2
    exit 1
fi

# Load LLM venv (has sklearn, pandas, numpy)
VENV_DIR="${WORK_BASE}/venvs/llm_env"
if [ -f "${VENV_DIR}/bin/activate" ]; then
    source "${VENV_DIR}/bin/activate"
else
    echo "ERROR: LLM venv not found: ${VENV_DIR}" >&2
    exit 1
fi

RUN_ID="cpu_${SLURM_JOB_ID:-manual}_${EXPERIMENT}_$(date +%Y%m%d_%H%M%S)"

# ── Log header ────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  HKUST HPC — CPU Experiment Job"
echo "================================================================"
echo "  Job ID:      ${SLURM_JOB_ID:-local}"
echo "  Experiment:  ${EXPERIMENT}"
echo "  Node:        $(hostname)"
echo "  CPUs:        ${SLURM_CPUS_PER_TASK:-4}"
echo "  Run ID:      ${RUN_ID}"
echo "  Python:      $(python3 --version 2>/dev/null || echo 'not found')"
echo "================================================================"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/llm_finetuning"

# ═══════════════════════════════════════════════════════════════════════════════
# Dispatch CPU experiments
# ═══════════════════════════════════════════════════════════════════════════════
case "${EXPERIMENT}" in
    analyze_head_utility)
        python llm_finetuning/scripts/analyze_head_utility.py \
            "${EXTRA_ARGS[@]}" \
            2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"
        ;;

    aggregate_ablation)
        python llm_finetuning/scripts/aggregate_ablation_results.py \
            "${EXTRA_ARGS[@]}" \
            2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"
        ;;

    build_decision_card)
        python llm_finetuning/scripts/build_decision_card.py \
            "${EXTRA_ARGS[@]}" \
            2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"
        ;;

    collapse_labels)
        python llm_finetuning/scripts/collapse_labels.py \
            "${EXTRA_ARGS[@]}" \
            2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"
        ;;

    analyze_head_leakage_correlation)
        python llm_finetuning/scripts/analyze_head_leakage_correlation.py \
            "${EXTRA_ARGS[@]}" \
            2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"
        ;;

    llm_constraint_judge)
        python llm_finetuning/scripts/llm_constraint_judge.py \
            "${EXTRA_ARGS[@]}" \
            2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"
        ;;

    eval_decision_card_ab)
        python llm_finetuning/scripts/eval_decision_card_ab.py \
            "${EXTRA_ARGS[@]}" \
            2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"
        ;;

    plot_tradeoff_curves)
        python llm_finetuning/scripts/plot_tradeoff_curves.py \
            "${EXTRA_ARGS[@]}" \
            2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"
        ;;

    *)
        echo "Unknown experiment: ${EXPERIMENT}" >&2
        echo "Valid: analyze_head_utility | aggregate_ablation | build_decision_card | collapse_labels | analyze_head_leakage_correlation | llm_constraint_judge | eval_decision_card_ab | plot_tradeoff_curves" >&2
        exit 1
        ;;
esac

EXIT_CODE=$?
echo "Done (exit=${EXIT_CODE})  Run: ${RUN_ID}"
exit ${EXIT_CODE}
