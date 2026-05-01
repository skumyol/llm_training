#!/usr/bin/env bash
# =============================================================================
# submit_test_jobs.sh — Submit short test jobs for every training task
# =============================================================================
# Submits lightweight test jobs (1 epoch, 15 min limit) to verify that
# Slurm scripts, environments, GPU access, and all training entry points
# work correctly before running full training.
#
# Usage:
#   bash scripts/submit_test_jobs.sh              # Submit ALL test jobs
#   bash scripts/submit_test_jobs.sh --dry-run    # Print what would submit
#   bash scripts/submit_test_jobs.sh slm          # SLM tests only
#   bash scripts/submit_test_jobs.sh llm          # LLM tests only
#
# After submitting, monitor with:
#   squeue -u $USER | grep test-
#   tail -f /scratch/$USER/logs/test_*.out
#
# Check results:
#   grep -l "Done (exit=0)" /scratch/$USER/logs/test_*.out
#   grep -l "ERROR\|FAIL\|exit=1" /scratch/$USER/logs/test_*.out
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_BASE="/scratch/${USER}"
LOG_DIR="${WORK_BASE}/logs"
VENV_LLM="${WORK_BASE}/venvs/llm_env"
VENV_SLM="${WORK_BASE}/venvs/slm_env"
REPO_DIR="${WORK_BASE}/npc"

ACCOUNT="${HPC_ACCOUNT:-xrimlab}"
PARTITION="${HPC_PARTITION:-gpu-l20}"
TEST_TIME="00:15:00"      # 15 minutes per test job
TEST_EPOCHS=1

DRY_RUN=false
SCOPE="all"               # all, slm, llm

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        slm|llm|all) SCOPE="$arg" ;;
        *) echo "Usage: $0 [all|slm|llm] [--dry-run]"; exit 1 ;;
    esac
done

mkdir -p "${LOG_DIR}"

# ── Helpers ───────────────────────────────────────────────────────────────────
JOB_COUNT=0

submit_test() {
    local job_name="$1"
    local system="$2"
    local stage="$3"
    shift 3
    local extra_args=("$@")

    JOB_COUNT=$((JOB_COUNT + 1))
    local full_name="test-${job_name}"

    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY] sbatch --job-name=${full_name} slurm_train.sh ${system} ${stage} ${extra_args[*]}"
        return
    fi

    sbatch \
        --job-name="${full_name}" \
        --partition="${PARTITION}" \
        --account="${ACCOUNT}" \
        --gpus-per-node=1 \
        --ntasks-per-node=1 \
        --cpus-per-task=4 \
        --time="${TEST_TIME}" \
        --output="${LOG_DIR}/test_${job_name}_%j.out" \
        --error="${LOG_DIR}/test_${job_name}_%j.err" \
        "${ROOT}/scripts/slurm_train.sh" "${system}" "${stage}" "${extra_args[@]}"

    echo "  [OK] ${full_name}"
}

# ── Check prerequisites ───────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Submit Test Jobs — Verify Training Pipeline                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Account:   ${ACCOUNT}"
echo "  Partition: ${PARTITION}"
echo "  Time limit:${TEST_TIME}"
echo "  Epochs:    ${TEST_EPOCHS}"
echo "  Scope:     ${SCOPE}"
echo "  Dry run:   ${DRY_RUN}"
echo ""

if [ "$DRY_RUN" = false ]; then
    # Verify venvs exist
    if [ ! -f "${VENV_SLM}/bin/activate" ]; then
        echo "  [WARN] SLM venv not found: ${VENV_SLM}"
        echo "  Run: bash scripts/env_setup_spack.sh first"
    fi
    if [ ! -f "${VENV_LLM}/bin/activate" ]; then
        echo "  [WARN] LLM venv not found: ${VENV_LLM}"
        echo "  Run: bash scripts/env_setup_spack.sh first"
    fi
    echo ""
fi

# ═══════════════════════════════════════════════════════════════════════════════
# SLM Test Jobs
# ═══════════════════════════════════════════════════════════════════════════════
if [ "$SCOPE" = "slm" ] || [ "$SCOPE" = "all" ]; then
    echo "── SLM Training Tests ────────────────────────────────────────"

    # 1. Small LM benchmark — GPT (fastest to test)
    submit_test "slm_gpt"     slm small_lm --arch gpt         --epochs "${TEST_EPOCHS}"
    # 2. Small LM — GRU
    submit_test "slm_gru"     slm small_lm --arch gru         --epochs "${TEST_EPOCHS}"
    # 3. Small LM — AWD-LSTM
    submit_test "slm_awdlstm" slm small_lm --arch awdlstm     --epochs "${TEST_EPOCHS}"
    # 4. Small LM — PrefixGPT
    submit_test "slm_pref"    slm small_lm --arch prefix_gpt  --epochs "${TEST_EPOCHS}"
    # 5. Small LM — Mamba-like
    submit_test "slm_mamba"   slm small_lm --arch mamba_like  --epochs "${TEST_EPOCHS}"
    # 6. Small LM — MoE
    submit_test "slm_moe"     slm small_lm --arch moe         --epochs "${TEST_EPOCHS}"

    # 7. Personality encoder
    submit_test "slm_pers"    slm personality --epochs "${TEST_EPOCHS}"

    # 8. Affect encoder
    submit_test "slm_aff"     slm affect      --epochs "${TEST_EPOCHS}"

    # 9. Dialogue model
    submit_test "slm_dial"    slm dialogue    --epochs "${TEST_EPOCHS}"

    echo "  9 SLM test jobs"
    echo ""
fi

# ═══════════════════════════════════════════════════════════════════════════════
# LLM Test Jobs
# ═══════════════════════════════════════════════════════════════════════════════
if [ "$SCOPE" = "llm" ] || [ "$SCOPE" = "all" ]; then
    echo "── LLM Fine-Tuning Tests ─────────────────────────────────────"

    # 10. Stage 1: Latent state predictor (Qwen3-0.6B + LoRA) — debug mode
    submit_test "llm_lat"     llm latent   --debug

    # 11. Stage 2: Response generator (Qwen3-4B + QLoRA) — debug mode
    submit_test "llm_resp"    llm response --debug

    # 12. Stage 3: Joint fine-tuning — debug mode
    submit_test "llm_joint"   llm joint    --debug

    echo "  3 LLM test jobs"
    echo ""
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
echo "══════════════════════════════════════════════════════════════"
if [ "$DRY_RUN" = true ]; then
    echo "  DRY RUN — ${JOB_COUNT} jobs would be submitted."
    echo "  Remove --dry-run to actually submit."
else
    echo "  Submitted ${JOB_COUNT} test jobs."
    echo ""
    echo "  Monitor queue:"
    echo "    squeue -u \$USER | grep test-"
    echo "    watch -n 5 'squeue -u \$USER | grep test-'"
    echo ""
    echo "  Check logs:"
    echo "    tail -f ${LOG_DIR}/test_slm_gpt_*.out"
    echo "    tail -f ${LOG_DIR}/test_llm_lat_*.out"
    echo ""
    echo "  After completion, verify:"
    echo "    # Count successful exits"
    echo "    grep -l 'Done (exit=0)' ${LOG_DIR}/test_*.out | wc -l"
    echo ""
    echo "    # Show any failures"
    echo "    grep -l 'ERROR\|FAIL\|exit=1' ${LOG_DIR}/test_*.out"
    echo ""
    echo "    # Detailed per-job status"
    echo "    for f in ${LOG_DIR}/test_*.out; do"
    echo "      printf \"%-30s %s\n\" \$(basename \$f) \"\$(tail -1 \$f)\""
    echo "    done"
    echo ""
    echo "  To cancel all test jobs:"
    echo "    scancel -u \$USER -n test-"
fi
echo "══════════════════════════════════════════════════════════════"
