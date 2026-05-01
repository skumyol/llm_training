#!/usr/bin/env bash
# =============================================================================
# sync_to_scratch.sh — Sync repo + data to scratch for fast I/O on compute nodes
# =============================================================================
# HKUST HPC compute nodes mount /scratch with fast local NVMe — much faster
# than /home (NFS). This script rsyncs the repo and data over.
#
# Usage:
#   bash scripts/sync_to_scratch.sh              # Sync code + data
#   bash scripts/sync_to_scratch.sh --code-only  # Only sync code
#   bash scripts/sync_to_scratch.sh --data-only  # Only sync data
#   bash scripts/sync_to_scratch.sh --dry-run    # Show what would change
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_BASE="/scratch/${USER}"
SCRATCH_REPO="${WORK_BASE}/npc"
SCRATCH_DATA="${WORK_BASE}/data"

CODE_ONLY=false
DATA_ONLY=false
DRY_RUN=""

for arg in "$@"; do
    case "$arg" in
        --code-only) CODE_ONLY=true ;;
        --data-only) DATA_ONLY=true ;;
        --dry-run)   DRY_RUN="--dry-run" ;;
        *) echo "Unknown: $arg"; exit 1 ;;
    esac
done

echo "================================================================"
echo "  Sync to Scratch (${WORK_BASE})"
echo "================================================================"

sync_code() {
    echo ""
    echo "── Syncing code: ${ROOT} → ${SCRATCH_REPO} ──"

    # If symlink exists, nothing to do
    if [ -L "${SCRATCH_REPO}" ]; then
        echo "  Repo is a symlink — no sync needed"
        echo "  ${SCRATCH_REPO} → $(readlink "${SCRATCH_REPO}")"
        return
    fi

    mkdir -p "${SCRATCH_REPO}"
    rsync -av ${DRY_RUN} \
        --exclude='.git' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.venv' \
        --exclude='venvs' \
        --exclude='mlruns' \
        --exclude='logs' \
        --exclude='checkpoints' \
        --exclude='outputs' \
        --exclude='*.ckpt' \
        --exclude='*.pt' \
        --exclude='*.pth' \
        --exclude='*.safetensors' \
        --exclude='.pytest_cache' \
        --exclude='dist' \
        --exclude='build' \
        --exclude='*.egg-info' \
        "${ROOT}/" "${SCRATCH_REPO}/"

    echo "  Code synced."
}

sync_data() {
    echo ""
    echo "── Syncing data: ${ROOT}/data → ${SCRATCH_DATA} ──"

    mkdir -p "${SCRATCH_DATA}"

    if [ -d "${ROOT}/data" ]; then
        rsync -av ${DRY_RUN} \
            --exclude='raw_episodes' \
            --exclude='validated_turns' \
            --exclude='counterfactuals' \
            --exclude='packaged' \
            --exclude='merged_validated' \
            --exclude='splits' \
            --exclude='*.tmp' \
            "${ROOT}/data/" "${SCRATCH_DATA}/"
    else
        echo "  No data directory at ${ROOT}/data"
    fi

    echo "  Data synced."
}

# ── Run ──────────────────────────────────────────────────────────────────────
if [ "$DATA_ONLY" = true ]; then
    sync_data
elif [ "$CODE_ONLY" = true ]; then
    sync_code
else
    sync_code
    sync_data
fi

echo ""
echo "================================================================"
echo "  Sync complete."
echo ""
echo "  Scratch layout:"
echo "    ${WORK_BASE}/npc/          — code"
echo "    ${WORK_BASE}/data/         — training data"
echo "    ${WORK_BASE}/venvs/        — Python environments"
echo "    ${WORK_BASE}/logs/         — Slurm output"
echo "    ${WORK_BASE}/checkpoints/  — model weights"
echo "    ${WORK_BASE}/mlruns/       — MLflow tracking"
echo "================================================================"
