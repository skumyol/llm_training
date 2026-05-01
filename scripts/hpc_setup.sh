#!/usr/bin/env bash
# =============================================================================
# hpc_setup.sh — Full HPC bootstrap: env + sync + submit
# =============================================================================
# Run this on hpc4.ust.hk after SSH-ing in:
#   ssh skumyol@hpc4.ust.hk
#   cd ~/llm_training
#   bash scripts/hpc_setup.sh
#
# Options:
#   --dry-run    Print what would happen without executing
#   --skip-spack Skip Spack package install (use existing)
#   --no-submit  Don't submit jobs at the end
# =============================================================================
set -euo pipefail

DRY_RUN=false
SKIP_SPACK=false
NO_SUBMIT=false

for arg in "$@"; do
    case "$arg" in
        --dry-run)     DRY_RUN=true ;;
        --skip-spack)  SKIP_SPACK=true ;;
        --no-submit)   NO_SUBMIT=true ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_BASE="/scratch/${USER}"

echo "================================================================"
echo "  HPC Setup & Resume Training"
echo "================================================================"

# 1. Pull latest code
echo ""
echo "── 1. Git pull ──"
git -C "${ROOT}" pull || echo "  [SKIP] git pull failed (no network?)"

# 2. Run full environment setup
echo ""
echo "── 2. Environment setup ──"
SPACK_ARGS=""
[ "$SKIP_SPACK" = true ] && SPACK_ARGS="--skip-spack"
bash "${ROOT}/scripts/env_setup_spack.sh" ${SPACK_ARGS}

# 3. Check what's already done
echo ""
echo "── 3. Existing checkpoints ──"
bash "${ROOT}/scripts/submit_missing.sh"

# 4. Submit missing runs
if [ "$NO_SUBMIT" = false ]; then
    echo ""
    echo "── 4. Submitting jobs ──"
    bash "${ROOT}/scripts/submit_missing.sh" --submit 2>&1 || {
        echo "  WARNING: sbatch may not be available or jobs failed to submit"
        echo "  Manually submit with: sbatch scripts/slurm_train.sh ..."
    }
else
    echo ""
    echo "── 4. Skipping submission (--no-submit) ──"
fi

# 5. Show status
echo ""
echo "── 5. Job queue ──"
squeue -u "${USER}" 2>/dev/null || echo "  squeue not available"

# 6. Environment check
echo ""
echo "── 6. Environment ──"
echo "  Host:      $(hostname)"
echo "  GPU:       $(nvidia-smi -L 2>/dev/null | head -1 || echo 'none (login node)')"
echo "  Python:    $(python3 --version 2>/dev/null || echo 'not found')"
echo "  Spack:     $(spack --version 2>/dev/null || echo 'not loaded')"
echo "  cuda mod:  $(module list 2>&1 | grep cuda || echo 'not loaded')"
echo "  LLM venv:  $([ -f "${WORK_BASE}/venvs/llm_env/bin/activate" ] && echo 'YES' || echo 'NO')"
echo "  SLM venv:  $([ -f "${WORK_BASE}/venvs/slm_env/bin/activate" ] && echo 'YES' || echo 'NO')"
echo ""
echo "================================================================"
echo "  Done. Monitor with:"
echo "    squeue -u \$USER"
echo "    tail -f /scratch/\$USER/logs/slurm_*.out"
echo ""
echo "  To submit manually:"
echo "    sbatch scripts/slurm_train.sh slm small_lm --arch gpt --seed 42"
echo "    sbatch scripts/slurm_array.sh slm small_lm --archs gpt,moe --seeds 42,43,44"
echo "================================================================"
