#!/bin/bash
# =============================================================================
# submit_hpc_jobs.sh — Submit all remaining training jobs to HKUST HPC
# =============================================================================
# Each job gets its own GPU allocation.
# Uses sbatch with correct HKUST conventions (--gpus-per-node, --account).
#
# Usage:
#   bash scripts/submit_hpc_jobs.sh
#
#   # Submit only specific architectures:
#   bash scripts/submit_hpc_jobs.sh mamba_like moe
#
#   # Custom partition:
#   HPC_PARTITION=gpu-a30 bash scripts/submit_hpc_jobs.sh
#   HPC_PARTITION=gpu-rtx4090d bash scripts/submit_hpc_jobs.sh
#
#   # Dry run:
#   DRY_RUN=1 bash scripts/submit_hpc_jobs.sh
# =============================================================================
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
ACCOUNT="${HPC_ACCOUNT:-xrimlab}"
PARTITION="${HPC_PARTITION:-gpu-l20}"
TIME="${HPC_TIME:-24:00:00}"
CPUS="${HPC_CPUS:-8}"
WORK_BASE="/scratch/${USER}"
REPO_DIR="${WORK_BASE}/npc"
VENV_DIR="${WORK_BASE}/venvs/slm_env"
LOG_DIR="${WORK_BASE}/logs"
EPOCHS="${HPC_EPOCHS:-20}"
DRY_RUN="${DRY_RUN:-0}"

# ── Define jobs: ARCH:SEED ────────────────────────────────────────────────────
# Override by passing args:  bash submit_hpc_jobs.sh arch1 arch2...
if [ $# -gt 0 ]; then
    JOBS=()
    for arch in "$@"; do
        for seed in 42 43 44; do
            JOBS+=("${arch}:${seed}")
        done
    done
else
    # Default: all remaining SLM architectures × seeds
    JOBS=(
        "mamba_like:42"
        "mamba_like:43"
        "mamba_like:44"
        "prefix_gpt:42"
        "prefix_gpt:43"
        "prefix_gpt:44"
        "moe:42"
        "moe:43"
        "moe:44"
    )
fi

mkdir -p "${LOG_DIR}"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Submit Training Jobs to HKUST HPC                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Account:    ${ACCOUNT}"
echo "  Partition:  ${PARTITION}"
echo "  Time limit: ${TIME}"
echo "  Jobs:       ${#JOBS[@]}"
echo "  Dry run:    ${DRY_RUN}"
echo ""

for job in "${JOBS[@]}"; do
    IFS=':' read -r ARCH SEED <<< "$job"
    JOB_NAME="slm_${ARCH}_s${SEED}"

    # Check if checkpoint already exists
    CHKPT_DIR="${WORK_BASE}/checkpoints/${JOB_NAME}"
    if [ -f "${CHKPT_DIR}/best_model.pt" ] 2>/dev/null; then
        echo "  [SKIP] ${JOB_NAME} — checkpoint exists at ${CHKPT_DIR}"
        continue
    fi

    if [ "${DRY_RUN}" = "1" ]; then
        echo "  [DRY] sbatch --job-name=${JOB_NAME} ... slm small_lm --arch ${ARCH} --seed ${SEED}"
        continue
    fi

    # Submit via sbatch, overriding defaults as needed
    sbatch \
        --job-name="${JOB_NAME}" \
        --partition="${PARTITION}" \
        --account="${ACCOUNT}" \
        --gpus-per-node=1 \
        --ntasks-per-node=1 \
        --cpus-per-task="${CPUS}" \
        --time="${TIME}" \
        --output="${LOG_DIR}/${JOB_NAME}_%j.out" \
        --error="${LOG_DIR}/${JOB_NAME}_%j.err" \
        "${REPO_DIR}/scripts/slurm_train.sh" slm small_lm --arch "${ARCH}" --seed "${SEED}"

    echo "  [OK]  ${JOB_NAME}"
    sleep 1  # Be nice to the scheduler
done

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  Done. Monitor with:"
echo "    squeue -u \$USER"
echo "    tail -f ${LOG_DIR}/slm_*.out"
echo "    ls ${WORK_BASE}/checkpoints/"
echo "══════════════════════════════════════════════════════════════"
