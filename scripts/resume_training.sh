#!/usr/bin/env bash
# =============================================================================
# resume_training.sh — One-shot: sync, cancel stale jobs, submit optimized training
# =============================================================================
# Run after SSH-ing into the cluster:
#   ssh skumyol@hpc4.ust.hk
#   cd ~/llm_training
#   bash scripts/resume_training.sh
#
# This will:
#   1. Cancel any lingering Slurm jobs
#   2. Sync code from home to scratch
#   3. Verify data and venvs
#   4. Submit training (6 architectures × 1 seed, 20 epochs)
#   5. Submit eval to run after training
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_BASE="/scratch/${USER}"
REPO_LINK="${WORK_BASE}/npc"
ACCOUNT="xrimlab"
PARTITION="gpu-a30"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Resume Training on HKUST HPC                               ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# 1. Cancel stale jobs
echo ""
echo "── 1. Cancelling old jobs ──"
scancel -u "${USER}" 2>/dev/null && echo "  Cancelled." || echo "  No jobs to cancel."

# 2. Sync code to scratch
echo ""
echo "── 2. Syncing code ──"
if [ ! -L "${REPO_LINK}" ]; then
    mkdir -p "${WORK_BASE}"
    ln -sfn "${ROOT}" "${REPO_LINK}"
    echo "  Symlinked: ${REPO_LINK} → ${ROOT}"
else
    echo "  Symlink exists: ${REPO_LINK}"
fi

# 3. Verify venvs
echo ""
echo "── 3. Verifying environment ──"
for venv in llm_env slm_env; do
    if [ -f "${WORK_BASE}/venvs/${venv}/bin/activate" ]; then
        echo "  ✅ ${venv}"
    else
        echo "  ❌ ${venv} missing — run: bash scripts/env_setup_spack.sh"
    fi
done

# 4. Verify data
echo ""
echo "── 4. Verifying data ──"
if [ -f "${REPO_LINK}/slm_training/data/dialogue/train.txt" ]; then
    LINES=$(wc -l < "${REPO_LINK}/slm_training/data/dialogue/train.txt")
    echo "  ✅ train.txt: ${LINES} lines"
else
    echo "  ❌ Missing dialogue data — generating mock data..."
    source "${WORK_BASE}/venvs/slm_env/bin/activate" 2>/dev/null || true
    python3 -c "
import json, random
from pathlib import Path
d = Path('${REPO_LINK}/slm_training/data/dialogue')
d.mkdir(parents=True, exist_ok=True)
lines = [f'Player: hello NPC: hi there' for _ in range(2000)]
with open(d/'train.txt','w') as f: f.write('\n'.join(lines[:1600])+'\n')
with open(d/'val.txt','w') as f: f.write('\n'.join(lines[1600:])+'\n')
print('  Generated mock data')
"
fi

# 5. Submit training
echo ""
echo "── 5. Submitting training (6 archs × 20 epochs) ──"
ARCHS=(gpt mamba_like prefix_gpt moe)
JOB_IDS=()

for arch in "${ARCHS[@]}"; do
    JID=$(sbatch --parsable \
        --job-name="t-${arch}" \
        --partition="${PARTITION}" \
        --account="${ACCOUNT}" \
        --gpus-per-node=1 \
        --ntasks-per-node=1 \
        --cpus-per-task=8 \
        --time=08:00:00 \
        --output="${WORK_BASE}/logs/t_${arch}_%j.out" \
        --error="${WORK_BASE}/logs/t_${arch}_%j.err" \
        "${REPO_LINK}/scripts/slurm_train.sh" slm small_lm --arch "${arch}" --epochs 20 --seed 42)
    JOB_IDS+=("${JID}")
    echo "  ${arch}: ${JID}"
    sleep 1
done

# 6. Submit eval after all training completes
echo ""
echo "── 6. Submitting final evaluation (after training) ──"
DEP_STR=$(IFS=:; echo "${JOB_IDS[*]}")
EVAL_JID=$(sbatch --parsable \
    --job-name="eval-final" \
    --partition="${PARTITION}" \
    --account="${ACCOUNT}" \
    --gpus-per-node=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=4 \
    --time=02:00:00 \
    --output="${WORK_BASE}/logs/eval_final_%j.out" \
    --error="${WORK_BASE}/logs/eval_final_%j.err" \
    --dependency="afterok:${DEP_STR}" \
    "${REPO_LINK}/scripts/slurm_eval.sh" slm --out-csv "${WORK_BASE}/artifacts/eval_results.csv")
echo "  eval-final: ${EVAL_JID}"

# 7. Submit artifacts export (copy results to home for easy SCP)
echo ""
echo "── 7. Submitting export job (after eval) ──"
sbatch --parsable \
    --job-name="export-art" \
    --partition="${PARTITION}" \
    --account="${ACCOUNT}" \
    --ntasks-per-node=1 \
    --cpus-per-task=2 \
    --time=00:15:00 \
    --output="${WORK_BASE}/logs/export_%j.out" \
    --dependency="afterok:${EVAL_JID}" \
    --wrap="
mkdir -p ${ROOT}/eval_results ${ROOT}/slurm_logs
cp -r ${WORK_BASE}/npc/slm_training/artifacts/* ${ROOT}/artifacts/ 2>/dev/null || true
cp -r ${WORK_BASE}/logs/*.out ${ROOT}/slurm_logs/ 2>/dev/null || true
cp -r ${WORK_BASE}/artifacts/eval_results.csv ${ROOT}/eval_results/ 2>/dev/null || true
echo 'Artifacts exported to home dir'
"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Training submitted!                                        ║"
echo "║                                                             ║"
echo "║  Monitor:  squeue -u \$USER                                 ║"
echo "║  Logs:     tail -f ${WORK_BASE}/logs/t_gpt_*.out            ║"
echo "║  Export:   auto-exported to ~/llm_training/ after eval      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
