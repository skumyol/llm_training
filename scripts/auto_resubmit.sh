#!/usr/bin/env bash
# Auto-resubmit remaining jobs when SLURM queue drops below limit
# Runs on login node, polls every 5 minutes

set -euo pipefail

REPO_DIR="/scratch/skumyol/npc"
LOG="/scratch/skumyol/logs/auto_resubmit.log"
mkdir -p "$(dirname "$LOG")"

echo "[$(date)] Auto-resubmit daemon started. PID=$$" >> "$LOG"

# Jobs to submit (one per line, with job name)
JOBS=(
  "slm_mamba_like_42|sbatch ${REPO_DIR}/scripts/slurm_train.sh slm small_lm --arch mamba_like --seed 42 --epochs 20"
  "slm_moe_42|sbatch ${REPO_DIR}/scripts/slurm_train.sh slm small_lm --arch moe --seed 42 --epochs 20"
  "slm_gpt_43|sbatch ${REPO_DIR}/scripts/slurm_train.sh slm small_lm --arch gpt --seed 43 --epochs 20"
  "slm_prefix_gpt_43|sbatch ${REPO_DIR}/scripts/slurm_train.sh slm small_lm --arch prefix_gpt --seed 43 --epochs 20"
  "slm_mamba_like_43|sbatch ${REPO_DIR}/scripts/slurm_train.sh slm small_lm --arch mamba_like --seed 43 --epochs 20"
  "slm_moe_43|sbatch ${REPO_DIR}/scripts/slurm_train.sh slm small_lm --arch moe --seed 43 --epochs 20"
  "slm_gpt_44|sbatch ${REPO_DIR}/scripts/slurm_train.sh slm small_lm --arch gpt --seed 44 --epochs 20"
  "slm_prefix_gpt_44|sbatch ${REPO_DIR}/scripts/slurm_train.sh slm small_lm --arch prefix_gpt --seed 44 --epochs 20"
  "slm_mamba_like_44|sbatch ${REPO_DIR}/scripts/slurm_train.sh slm small_lm --arch mamba_like --seed 44 --epochs 20"
  "slm_moe_44|sbatch ${REPO_DIR}/scripts/slurm_train.sh slm small_lm --arch moe --seed 44 --epochs 20"
  "gemma_baseline|sbatch ${REPO_DIR}/scripts/slurm_gemma_baseline.sh"
  "gemma_social|sbatch ${REPO_DIR}/scripts/slurm_gemma_social.sh"
)

SUBMITTED_FILE="/scratch/skumyol/logs/auto_resubmit_submitted.txt"
touch "$SUBMITTED_FILE"

idx=0
while [ $idx -lt ${#JOBS[@]} ]; do
  # Count current jobs for this user
  JOB_COUNT=$(squeue -u skumyol -h 2>/dev/null | wc -l)
  
  if [ "$JOB_COUNT" -lt 4 ]; then
    IFS='|' read -r name cmd <<< "${JOBS[$idx]}"
    # Check if already submitted
    if grep -q "^$name$" "$SUBMITTED_FILE" 2>/dev/null; then
      echo "[$(date)] SKIP $name (already submitted)" >> "$LOG"
      idx=$((idx+1))
      continue
    fi
    
    echo "[$(date)] Submitting $name (queue=$JOB_COUNT)" >> "$LOG"
    OUTPUT=$(eval "$cmd" 2>&1) || true
    if echo "$OUTPUT" | grep -q "Submitted batch job"; then
      JOB_ID=$(echo "$OUTPUT" | grep -oP 'Submitted batch job \K[0-9]+')
      echo "$name" >> "$SUBMITTED_FILE"
      echo "[$(date)] SUCCESS $name -> $JOB_ID" >> "$LOG"
      idx=$((idx+1))
    else
      echo "[$(date)] FAILED $name: $OUTPUT" >> "$LOG"
    fi
  else
    echo "[$(date)] Queue full ($JOB_COUNT/4), waiting..." >> "$LOG"
  fi
  
  sleep 300  # 5 minutes
done

echo "[$(date)] All jobs submitted. Exiting." >> "$LOG"
