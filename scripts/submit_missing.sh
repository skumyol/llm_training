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
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARTIFACTS="$ROOT/slm_training/artifacts/small_lm"
SUBMIT="${1:-}"

# All combos that should exist
ARCHS=(mamba_like prefix_gpt moe)
SEEDS=(42 43 44)

MISSING=()
EXISTING=()

for arch in "${ARCHS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        # Check for any final_ directory with a model checkpoint
        found=""
        for d in "$ARTIFACTS"/final_${arch}_s${seed}_*; do
            [ -d "$d" ] || continue
            if [ -f "$d/best_model.pt" ] || ls "$d"/*_best.pt >/dev/null 2>&1; then
                found="$d"
                break
            fi
        done
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
    echo "  All runs complete! Nothing to submit."
    exit 0
fi

if [ "$SUBMIT" != "--submit" ]; then
    echo "  DRY RUN — add --submit to actually queue jobs."
    echo "  Commands that would run:"
    for r in "${MISSING[@]}"; do
        read -r arch seed <<< "$r"
        echo "    sbatch scripts/slurm_train.sh slm small_lm --arch $arch --seed $seed"
    done
    exit 0
fi

# Submit individual jobs
for r in "${MISSING[@]}"; do
    read -r arch seed <<< "$r"
    echo "  Submitting: $arch s$seed..."
    sbatch \
        --job-name="slm_${arch}_s${seed}" \
        --partition=gpu-l20 \
        --gpus-per-node=1 \
        --account=xrimlab \
        --output="/scratch/${USER}/logs/slm_${arch}_s${seed}_%j.out" \
        --error="/scratch/${USER}/logs/slm_${arch}_s${seed}_%j.err" \
        "$ROOT/scripts/slurm_train.sh" slm small_lm --arch "$arch" --seed "$seed"
done

echo ""
echo "  Submitted ${#MISSING[@]} jobs. Check with: squeue -u \$USER"
