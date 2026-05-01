#!/usr/bin/env bash
# =============================================================================
# submit_missing.sh — Submit SLURM jobs only for incomplete SLM runs
# =============================================================================
# Auto-detects which (arch, seed) combos are missing based on checkpoint
# files and submits only those. Safe to re-run — skips completed runs.
#
# Usage:
#   bash scripts/submit_missing.sh            # Dry-run: print what would submit
#   bash scripts/submit_missing.sh --submit   # Actually submit to SLURM
#
#   # Also check LLM stages:
#   bash scripts/submit_missing.sh --system llm --submit
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_BASE="/scratch/${USER}"
SUBMIT="${1:-}"
SYSTEM="${2:-slm}"

# ── SLM check ─────────────────────────────────────────────────────────────────
if [ "$SYSTEM" = "slm" ] || [ "$SYSTEM" = "all" ]; then
    ARTIFACTS="${ROOT}/slm_training/artifacts/small_lm"
    ARCHS=(mamba_like prefix_gpt moe)
    SEEDS=(42 43 44)

    MISSING=()
    EXISTING=()

    for arch in "${ARCHS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            found=""
            for d in "${ARTIFACTS}"/final_"${arch}"_s"${seed}"_*; do
                [ -d "$d" ] || continue
                if [ -f "$d/best_model.pt" ] || ls "$d"/*_best.pt >/dev/null 2>&1; then
                    found="$d"
                    break
                fi
            done
            # Also check scratch checkpoints
            if [ -z "$found" ]; then
                for d in "${WORK_BASE}/checkpoints"/slm_${arch}_s${seed}_*; do
                    [ -d "$d" ] || continue
                    if [ -f "$d/best_model.pt" ]; then
                        found="$d"
                        break
                    fi
                done
            fi

            if [ -n "$found" ]; then
                EXISTING+=("$arch $seed")
            else
                MISSING+=("$arch $seed")
            fi
        done
    done

    echo "================================================================"
    echo "  SLM Training — Resume Missing Runs"
    echo "================================================================"
    echo ""
    echo "  Completed (skipping):"
    for r in "${EXISTING[@]}"; do
        read -r a s <<< "$r"
        echo "    ✅ $a s$s"
    done
    echo ""
    echo "  Pending to run:"
    for r in "${MISSING[@]}"; do
        read -r a s <<< "$r"
        echo "    ❌ $a s$s"
    done
    echo ""

    if [ ${#MISSING[@]} -eq 0 ]; then
        echo "  All SLM runs complete! Nothing to submit."
    elif [ "$SUBMIT" = "--submit" ]; then
        echo "  Submitting ${#MISSING[@]} jobs..."
        for r in "${MISSING[@]}"; do
            read -r arch seed <<< "$r"
            JOB_NAME="slm_${arch}_s${seed}"
            echo "    ${JOB_NAME}..."
            sbatch \
                --job-name="${JOB_NAME}" \
                --output="/scratch/${USER}/logs/${JOB_NAME}_%j.out" \
                --error="/scratch/${USER}/logs/${JOB_NAME}_%j.err" \
                "${ROOT}/scripts/slurm_train.sh" slm small_lm --arch "${arch}" --seed "${seed}"
        done
        echo ""
        echo "  Submitted ${#MISSING[@]} jobs. Check: squeue -u \$USER"
    else
        echo "  DRY RUN — add --submit to actually queue jobs."
        echo "  Commands that would run:"
        for r in "${MISSING[@]}"; do
            read -r arch seed <<< "$r"
            echo "    sbatch ${ROOT}/scripts/slurm_train.sh slm small_lm --arch $arch --seed $seed"
        done
    fi
fi

# ── LLM check ─────────────────────────────────────────────────────────────────
if [ "$SYSTEM" = "llm" ] || [ "$SYSTEM" = "all" ]; then
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "  LLM Fine-Tuning — Stage Completion"
    echo "────────────────────────────────────────────────────────────────"
    STAGES=("latent" "response" "joint")
    CKPT_BASE="${WORK_BASE}/checkpoints"

    for stage in "${STAGES[@]}"; do
        stage_dir="${CKPT_BASE}/${stage}_model"
        if [ -d "${stage_dir}" ] && ls "${stage_dir}"/pytorch_model*.bin &>/dev/null 2>&1; then
            echo "    ✅ ${stage} — checkpoint exists"
        elif [ -d "${ROOT}/checkpoints/${stage}_model" ] && ls "${ROOT}/checkpoints/${stage}_model"/pytorch_model*.bin &>/dev/null 2>&1; then
            echo "    ✅ ${stage} — checkpoint exists (home dir)"
        else
            echo "    ❌ ${stage} — no checkpoint found"
            if [ "$SUBMIT" = "--submit" ]; then
                echo "      Submitting..."
                sbatch \
                    --job-name="llm_${stage}" \
                    --output="/scratch/${USER}/logs/llm_${stage}_%j.out" \
                    --error="/scratch/${USER}/logs/llm_${stage}_%j.err" \
                    "${ROOT}/scripts/slurm_train.sh" llm "${stage}"
            elif [ "$SUBMIT" != "--submit" ]; then
                echo "      Would run: sbatch ${ROOT}/scripts/slurm_train.sh llm ${stage}"
            fi
        fi
    done
fi

echo ""
echo "  Use --submit to queue missing jobs."
