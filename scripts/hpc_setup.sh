#!/usr/bin/env bash
# =============================================================================
# Run this on hpc4.ust.hk after SSH-ing in:
#   ssh skumyol@hpc4.ust.hk
#   cd ~/llm_training
#   bash scripts/hpc_setup.sh
# =============================================================================
set -euo pipefail

echo "================================================================"
echo "  HPC Setup & Resume Training"
echo "================================================================"

# 1. Pull latest code
echo ""
echo "── 1. Git pull ──"
git pull

# 2. Check what's already done
echo ""
echo "── 2. Existing checkpoints ──"
bash scripts/submit_missing.sh

# 3. Submit missing runs
echo ""
echo "── 3. Submitting jobs ──"
bash scripts/submit_missing.sh --submit 2>&1 || echo "  (SLURM may not be available — run sbatch manually)"

# 4. Show status
echo ""
echo "── 4. Job queue ──"
squeue -u "$USER" 2>/dev/null || echo "  squeue not available"

# 5. Quick env check
echo ""
echo "── 5. Environment ──"
echo "  Host: $(hostname)"
echo "  GPU:  $(nvidia-smi -L 2>/dev/null | head -1 || echo 'none (login node)')"
echo "  Python: $(python3 --version 2>/dev/null || echo 'not found')"
echo ""
echo "================================================================"
echo "  Done. Monitor with:"
echo "    squeue -u \$USER"
echo "    tail -f /scratch/\$USER/logs/slm_*.out"
echo "================================================================"
