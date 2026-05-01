#!/bin/bash
# submit_hpc_jobs.sh - Submit all remaining training jobs to HKUST HPC in parallel
# Each job gets its own GPU allocation

set -euo pipefail

# Configuration
ACCOUNT="${HPC_ACCOUNT:-YOUR_ACCOUNT}"
PARTITION="${HPC_PARTITION:-gpu-l20}"
TIME="24:00:00"
WORK_BASE="/scratch/${USER}"
EPOCHS=20

# Remaining jobs after current progress
JOBS=(
    "mamba_like:43"
    "mamba_like:44"
    "prefix_gpt:42"
    "prefix_gpt:43"
    "moe:42"
    "moe:43"
    "moe:44"
)

echo "=== Submitting ${#JOBS[@]} jobs to HKUST HPC ==="
echo "Account: ${ACCOUNT}"
echo "Partition: ${PARTITION}"
echo ""

for job in "${JOBS[@]}"; do
    IFS=':' read -r ARCH SEED <<< "$job"
    JOB_NAME="slm_${ARCH}_s${SEED}"
    
    # Create temp sbatch script
    cat > "/tmp/${JOB_NAME}.sh" << EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --partition=${PARTITION}
#SBATCH --account=${ACCOUNT}
#SBATCH --time=${TIME}
#SBATCH --output=${WORK_BASE}/logs/${JOB_NAME}_%j.out
#SBATCH --error=${WORK_BASE}/logs/${JOB_NAME}_%j.err

# Setup
module load cuda/12.4 python/3.12 2>/dev/null || true
source ${WORK_BASE}/venvs/slm_env/bin/activate
export PYTHONPATH=${WORK_BASE}/code/slm_training/src:\$PYTHONPATH

cd ${WORK_BASE}/code/slm_training

# Run training
python scripts/train_final_small_lms.py \\
    --arch ${ARCH} \\
    --seeds ${SEED} \\
    --epochs ${EPOCHS} \\
    --timeout 86400 \\
    --train-text data/external/merged_dialogue.txt \\
    --val-text data/dialogue/val.txt \\
    --skip-existing

EOF
    
    # Submit
    sbatch "/tmp/${JOB_NAME}.sh"
    echo "  Submitted: ${JOB_NAME}"
    sleep 2
done

echo ""
echo "=== All jobs submitted ==="
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f ${WORK_BASE}/logs/slm_*.out"
