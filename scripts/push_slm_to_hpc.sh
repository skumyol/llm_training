#!/usr/bin/env bash
# =============================================================================
# push_slm_to_hpc.sh — sync the SLM score-push work to HKUST HPC and submit
# =============================================================================
# Requires the HKUST VPN to be connected (hpc4.ust.hk:22 must be reachable).
#
# Usage:
#   bash scripts/push_slm_to_hpc.sh              # sync + submit A/B, then C, then D
#   bash scripts/push_slm_to_hpc.sh --sync-only  # push files, submit nothing
#   bash scripts/push_slm_to_hpc.sh --status     # show queue + latest results
# =============================================================================
set -euo pipefail

REMOTE="${REMOTE:-skumyol@hpc4.ust.hk}"
WORK_BASE="/scratch/skumyol"
REPO_DIR="${WORK_BASE}/npc"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SYNC_ONLY=false; STATUS_ONLY=false
for a in "$@"; do
    case "$a" in
        --sync-only) SYNC_ONLY=true ;;
        --status)    STATUS_ONLY=true ;;
    esac
done

if ! nc -z -G 10 hpc4.ust.hk 22 2>/dev/null; then
    echo "ERROR: hpc4.ust.hk:22 unreachable — connect the HKUST VPN first." >&2
    exit 1
fi

# ── Status mode ───────────────────────────────────────────────────────────────
if [ "$STATUS_ONLY" = true ]; then
    ssh "${REMOTE}" "cd ${REPO_DIR}/slm_training && squeue -u \$USER -o '%.12i %.14j %.9T %.10M %R' && python3 scripts/aggregate_slm_push.py"
    exit 0
fi

# ── Sync code + the split/config files the sweep needs ────────────────────────
echo "==> syncing code to ${REMOTE}:${REPO_DIR}"
rsync -az \
    "${ROOT}/slm_training/src/" "${REMOTE}:${REPO_DIR}/slm_training/src/"
rsync -az \
    --include='slm_*.yaml' --exclude='*' \
    "${ROOT}/slm_training/configs/" "${REMOTE}:${REPO_DIR}/slm_training/configs/"
rsync -az \
    "${ROOT}/slm_training/scripts/make_val_test_split.py" \
    "${REMOTE}:${REPO_DIR}/slm_training/scripts/"
rsync -az "${ROOT}/scripts/slurm_slm_push.sh" "${REMOTE}:${REPO_DIR}/scripts/"

# The val/test split is derived, not raw data — regenerate it remotely so the
# cluster copy provably matches val.txt there rather than trusting a stale upload.
echo "==> regenerating val_sel/test split on cluster"
ssh "${REMOTE}" bash -lc "'cd ${REPO_DIR}/slm_training && python3 scripts/make_val_test_split.py'"

if [ "$SYNC_ONLY" = true ]; then
    echo "sync complete (nothing submitted)"; exit 0
fi

# ── Submit ────────────────────────────────────────────────────────────────────
# A/B are independent and run in one array. C (external pretrain) must finish
# before D (fine-tune) starts, so D is chained with --dependency=afterok.
echo "==> submitting (phase 1: C pretrain + A/B control)"
# a30_qos caps MaxSubmitJobsPerUser=10, so the sweep goes out in two phases:
# phase 1 = C (1) + A/B (6) = 7 jobs; phase 2 = D (3) once C has finished.
ssh "${REMOTE}" bash -lc "'
    set -e
    cd ${REPO_DIR}
    mkdir -p ${WORK_BASE}/logs

    C=\$(sbatch --parsable --array=0-0 --time=12:00:00 scripts/slurm_slm_push.sh \
        --configs=slm_C_pretrain --seeds=42)
    echo \"C pretrain: \$C\"

    AB=\$(sbatch --parsable --array=0-5 scripts/slurm_slm_push.sh \
        --configs=slm_A_baseline,slm_B_improved --seeds=42,43,44)
    echo \"A/B array: \$AB  (2 configs x 3 seeds)\"

    echo \"\$C\" > ${WORK_BASE}/logs/.slm_push_C_jobid

    echo; squeue -u \$USER -o \"%.10i %.14j %.9T %.10M %R\"
'"

echo
echo "Monitor with: bash scripts/push_slm_to_hpc.sh --status"
