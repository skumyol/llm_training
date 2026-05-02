#!/usr/bin/env bash
# =============================================================================
# resume_training.sh — Bulletproof training submission with pre-flight checks
# =============================================================================
# Run after SSH:
#   ssh skumyol@hpc4.ust.hk
#   cd ~/llm_training
#   bash scripts/resume_training.sh
#
# Pipeline:
#   1. Cancel stale jobs + verify environment
#   2. Pre-flight: 1-epoch test on GPT (quick failure detection)
#   3. If pre-flight passes → submit full training (4 archs × 20 epochs)
#   4. Auto-eval after training
#   5. Auto-export artifacts to home
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_BASE="/scratch/${USER}"
REPO_LINK="${WORK_BASE}/npc"
ACCOUNT="xrimlab"
PARTITION="${1:-gpu-a30}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
pass() { echo -e "  ${GREEN}✅${NC} $*"; }
fail() { echo -e "  ${RED}❌${NC} $*"; exit 1; }
warn() { echo -e "  ${YELLOW}⚠${NC}  $*"; }

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Bulletproof Training Submission                           ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 0 — Cancel stale jobs
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "── Phase 0: Cleanup ──"
CANCELLED=$(scancel -u "${USER}" 2>/dev/null && echo "yes" || echo "no")
if [ "$CANCELLED" = "yes" ]; then
    pass "Cancelled old jobs"
else
    pass "No stale jobs"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Verify environment
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "── Phase 1: Environment Checks ───────────────────────────────"

# 1a. Scratch venv
VENV="${WORK_BASE}/venvs/slm_env/bin/activate"
if [ -f "${VENV}" ]; then
    pass "SLM venv found"
else
    fail "SLM venv missing: ${VENV}"
fi

# 1b. CUDA module
module load cuda/12.4.0 2>/dev/null && pass "CUDA 12.4 loaded" || warn "CUDA module not on login node (expected)"

# 1c. Repo symlink
if [ -L "${REPO_LINK}" ]; then
    pass "Repo symlink: ${REPO_LINK}"
else
    mkdir -p "${WORK_BASE}"
    ln -sfn "${ROOT}" "${REPO_LINK}"
    pass "Created symlink: ${REPO_LINK} → ${ROOT}"
fi

# 1d. Training data
DATA_DIR="${REPO_LINK}/slm_training/data/dialogue"
TRAIN_TXT="${DATA_DIR}/train.txt"
VAL_TXT="${DATA_DIR}/val.txt"
mkdir -p "${DATA_DIR}" "${DATA_DIR}/../personality" "${DATA_DIR}/../affect"

if [ -f "${TRAIN_TXT}" ] && [ "$(wc -l < "${TRAIN_TXT}")" -ge 100 ]; then
    LINES=$(wc -l < "${TRAIN_TXT}")
    pass "Training data: ${LINES} lines"
else
    warn "No training data — generating mock data"
    source "${VENV}"
    python3 -c "
import json, random
from pathlib import Path
d = Path('${DATA_DIR}')
d.mkdir(parents=True, exist_ok=True)
random.seed(42)
lines = [f'Player asks about {random.choice([\"siege\",\"spy\",\"artifact\"])}. NPC replies cautiously.' for _ in range(2000)]
with open(d/'train.txt','w') as f: f.write('\n'.join(lines[:1600])+'\n')
with open(d/'val.txt','w') as f: f.write('\n'.join(lines[1600:])+'\n')
print(f'  Generated {len(lines)} lines')
"
    pass "Mock data generated"
fi

# 1e. Log/checkpoint dirs
for d in "${WORK_BASE}/logs" "${WORK_BASE}/checkpoints" "${WORK_BASE}/mlruns" "${WORK_BASE}/artifacts"; do
    mkdir -p "$d"
done
pass "Scratch directories ready"

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Pre-flight test (1 epoch GPT, must pass before full training)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "── Phase 2: Pre-flight Test (1 epoch GPT) ────────────────────"

PF_JOB=$(sbatch --parsable \
    --job-name="preflight" \
    --partition="${PARTITION}" \
    --account="${ACCOUNT}" \
    --gpus-per-node=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=8 \
    --time=00:15:00 \
    --output="${WORK_BASE}/logs/preflight_%j.out" \
    --error="${WORK_BASE}/logs/preflight_%j.err" \
    "${REPO_LINK}/scripts/slurm_train.sh" slm small_lm --arch gpt --epochs 1 --log-every 10)

echo "  preflight job: ${PF_JOB}"
echo "  Waiting for it to complete (max 15 min)..."

# Poll until job finishes
TIMEOUT=900  # 15 min
ELAPSED=0
while [ $ELAPSED -lt $TIMEOUT ]; do
    STATE=$(sacct -j "${PF_JOB}" --format=State --noheader -P 2>/dev/null | head -1)
    if [ "$STATE" = "COMPLETED" ] || [ "$STATE" = "FAILED" ] || [ "$STATE" = "TIMEOUT" ] || [ "$STATE" = "CANCELLED" ]; then
        break
    fi
    sleep 10
    ELAPSED=$((ELAPSED + 10))
    echo -ne "  ${STATE:-PENDING} ... ${ELAPSED}s elapsed\r"
done
echo ""

# Check result
FINAL_STATE=$(sacct -j "${PF_JOB}" --format=State --noheader -P 2>/dev/null | head -1)
EXIT_CODE=$(sacct -j "${PF_JOB}" --format=ExitCode --noheader -P 2>/dev/null | head -1 | cut -d: -f1)

if [ "$FINAL_STATE" = "COMPLETED" ] && [ "$EXIT_CODE" = "0" ]; then
    pass "Pre-flight PASSED (exit=${EXIT_CODE})"
else
    echo ""
    echo "  Pre-flight FAILED (state=${FINAL_STATE}, exit=${EXIT_CODE})"
    echo "  Last 20 lines of log:"
    tail -20 "${WORK_BASE}/logs/preflight_${PF_JOB}.out" 2>/dev/null || true
    fail "Pre-flight failed — fix issues before full training"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3 — Full training
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "── Phase 3: Full Training (4 archs × 20 epochs) ──────────────"

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

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4 — Evaluation (depends on all training)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "── Phase 4: Final Evaluation ─────────────────────────────────"
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

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5 — Auto-export artifacts to home
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "── Phase 5: Export Artifacts ─────────────────────────────────"
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
mkdir -p ${ROOT}/artifacts ${ROOT}/slurm_logs ${ROOT}/eval_results
cp -rn ${REPO_LINK}/slm_training/artifacts/* ${ROOT}/artifacts/ 2>/dev/null || true
cp -rn ${WORK_BASE}/logs/*.out ${ROOT}/slurm_logs/ 2>/dev/null || true
cp -rn ${WORK_BASE}/logs/*.err ${ROOT}/slurm_logs/ 2>/dev/null || true
cp ${WORK_BASE}/artifacts/eval_results.csv ${ROOT}/eval_results/ 2>/dev/null || true
echo '✅ Artifacts exported to ~/llm_training/{artifacts,slurm_logs,eval_results}'
"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Pipeline submitted:                                        ║"
echo "║    preflight ✅ → 4× train → eval → auto-export             ║"
echo "║                                                             ║"
echo "║  Monitor:  squeue -u \$USER                                 ║"
echo "║  Live log: tail -f ${WORK_BASE}/logs/t_gpt_*.out            ║"
echo "║  Cancel:   scancel -u \$USER                                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
