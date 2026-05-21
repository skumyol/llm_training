#!/usr/bin/env bash
# =============================================================================
# submit_camera_ready.sh — Submit remaining camera-ready evaluation + training jobs
# =============================================================================
# Submits:
#   1) Test-set eval (Qwen3-4B, 884 test turns) — ~4h on A30
#   2) VAD placebo array (TinyLlama, 3 seeds × real/random VAD = 6 jobs) — ~3h each
#   3) Param-matched GPT-22M (6-layer, 512-embed, ~22M params) — ~2h on L20
# =============================================================================
set -euo pipefail

WORK_BASE="/scratch/${USER}"
REPO_DIR="${WORK_BASE}/npc"
LOG_DIR="${WORK_BASE}/logs"
mkdir -p "${LOG_DIR}"

echo "============================================================"
echo "  Submitting Camera-Ready Jobs"
echo "  Node: $(hostname)"
echo "  $(date)"
echo "============================================================"
echo ""
echo "  GPU availability:"
sinfo -p gpu-a30,gpu-l20 -t idle -o "%P %.6D %c %G" 2>/dev/null || echo "  (sinfo unavailable — assuming GPUs free)"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# Job 1: Test-set evaluation
# ═══════════════════════════════════════════════════════════════════════════════
echo "=== Job 1: Test-Set Evaluation (884 test turns) ==="

cat > /tmp/slurm_test_eval.sh << 'EVALEOF'
#!/usr/bin/env bash
#SBATCH --job-name=test-eval
#SBATCH --output=/scratch/%u/logs/test_eval_%j.out
#SBATCH --error=/scratch/%u/logs/test_eval_%j.err
#SBATCH --account=xrimlab
#SBATCH --partition=gpu-a30
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@ust.hk

set -euo pipefail
WORK_BASE="/scratch/${USER}"
REPO_DIR="${WORK_BASE}/npc"
LOG_DIR="${WORK_BASE}/logs"
mkdir -p "${LOG_DIR}"

echo "=== Starting test-set eval ==="
module purge 2>/dev/null || true
module load cuda/12.4.0 2>/dev/null || { echo "ERROR: cuda/12.4.0 not found" >&2; exit 1; }
source "${WORK_BASE}/venvs/llm_env/bin/activate"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/llm_finetuning:${REPO_DIR}"

echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1 || echo 'none')"
echo "Config: llm_finetuning/configs/eval_test.yaml"

python llm_finetuning/run_eval.py --stage all --config llm_finetuning/configs/eval_test.yaml \
    2>&1 | tee "${LOG_DIR}/test_eval_${SLURM_JOB_ID}.log"

RC=$?
echo "Exit code: ${RC}"
exit ${RC}
EVALEOF

chmod +x /tmp/slurm_test_eval.sh
JOB1=$(sbatch --parsable /tmp/slurm_test_eval.sh)
echo "  → ${JOB1}"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# Job 2: VAD placebo array (6 jobs: 3 seeds × real/random VAD)
# Only real vs random VAD — full placebo (shuffled OCEAN, no-condition) 
# needs code changes to run_dialogue.py for --shuffle-ocean flag.
# ═══════════════════════════════════════════════════════════════════════════════
echo "=== Job 2: VAD Placebo Array (3 seeds × 2 variants = 6 jobs) ==="

cat > /tmp/slurm_placebo_array.sh << 'PLACEBOEOF'
#!/usr/bin/env bash
#SBATCH --job-name=placebo
#SBATCH --output=/scratch/%u/logs/placebo_%A_%a.out
#SBATCH --error=/scratch/%u/logs/placebo_%A_%a.err
#SBATCH --account=xrimlab
#SBATCH --partition=gpu-l20
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@ust.hk

set -euo pipefail
WORK_BASE="/scratch/${USER}"
REPO_DIR="${WORK_BASE}/npc"
LOG_DIR="${WORK_BASE}/logs"
mkdir -p "${LOG_DIR}"

module purge 2>/dev/null || true
module load cuda/12.4.0 2>/dev/null || { echo "ERROR: cuda/12.4.0 not found" >&2; exit 1; }
source "${WORK_BASE}/venvs/slm_env/bin/activate"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEEDS=(42 43 44)
RANDOMIZE=(false true)
LABELS=("real_vad" "random_vad")

SEED_IDX=$(( TASK_ID / 2 ))
RAND_IDX=$(( TASK_ID % 2 ))
SEED="${SEEDS[$SEED_IDX]}"
DO_RANDOMIZE="${RANDOMIZE[$RAND_IDX]}"
LABEL="${LABELS[$RAND_IDX]}"

RUN_ID="placebo_s${SEED}_${LABEL}_$(date +%Y%m%d_%H%M%S)"

echo "================================================================"
echo "  PLACEBO — Task ${TASK_ID}: seed=${SEED} variant=${LABEL}"
echo "  Node: $(hostname)  GPU: $(nvidia-smi -L 2>/dev/null | head -1 || echo 'none')"
echo "================================================================"

cd "${REPO_DIR}/slm_training"
export PYTHONPATH="${REPO_DIR}/slm_training"

ARGS="--run-id ${RUN_ID} --seed ${SEED} --epochs 3"
if [ "${DO_RANDOMIZE}" = "true" ]; then
    ARGS="${ARGS} --randomize-vad"
fi

echo "  Args: ${ARGS}"
python -m src.train.run_dialogue ${ARGS} \
    2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"

RC=$?
echo "Done (exit=${RC})  Run: ${RUN_ID}"
exit ${RC}
PLACEBOEOF

chmod +x /tmp/slurm_placebo_array.sh
JOB2=$(sbatch --parsable /tmp/slurm_placebo_array.sh)
echo "  → ${JOB2} (array 0-5)"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# Job 3: Param-matched GPT-22M
# ═══════════════════════════════════════════════════════════════════════════════
echo "=== Job 3: Param-Matched GPT (~22M) ==="

cat > /tmp/slurm_gpt22m.sh << 'GPTEOF'
#!/usr/bin/env bash
#SBATCH --job-name=gpt22m
#SBATCH --output=/scratch/%u/logs/gpt22m_%j.out
#SBATCH --error=/scratch/%u/logs/gpt22m_%j.err
#SBATCH --account=xrimlab
#SBATCH --partition=gpu-l20
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=${USER}@ust.hk

set -euo pipefail
WORK_BASE="/scratch/${USER}"
REPO_DIR="${WORK_BASE}/npc"
LOG_DIR="${WORK_BASE}/logs"
mkdir -p "${LOG_DIR}"

echo "=== Starting GPT-22M training ==="
module purge 2>/dev/null || true
module load cuda/12.4.0 2>/dev/null || { echo "ERROR: cuda/12.4.0 not found" >&2; exit 1; }
source "${WORK_BASE}/venvs/slm_env/bin/activate"

RUN_ID="gpt22m_s42_$(date +%Y%m%d_%H%M%S)"

echo "GPU: $(nvidia-smi -L 2>/dev/null | head -1 || echo 'none')"
echo "Run: ${RUN_ID}"

cd "${REPO_DIR}/slm_training"
export PYTHONPATH="${REPO_DIR}/slm_training"

# Generate config with scaled params: 6 layers × 512 embed × 8 heads ≈ 22M
python -c "
import yaml
cfg = {
    'arch': 'gpt',
    'hardware_profile': 'm1_small',
    'train_text': 'data/dialogue/train.txt',
    'val_text':   'data/dialogue/val.txt',
    'seq_len':    256,
    'batch_size': 16,
    'grad_accum': 4,
    'lr':         3.0e-4,
    'weight_decay': 0.1,
    'epochs':     20,
    'use_amp':    False,
    'log_every':  20,
    'eval_every_steps': 200,
    'condition_mode': 'zero',
    'seed':       42,
    'output_dir': 'artifacts/small_lm',
    'arch_params': {
        'n_embd': 512,
        'n_head': 8,
        'n_layer': 6,
        'dropout': 0.1,
        'max_seq_len': 256,
    },
}
with open('configs/small_lm_gpt22m.yaml', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False)
print('Wrote configs/small_lm_gpt22m.yaml')
"

python -m src.train.run_small_lm \
    --run-id "${RUN_ID}" \
    --arch gpt \
    --config configs/small_lm_gpt22m.yaml \
    --seed 42 \
    2>&1 | tee "${LOG_DIR}/${RUN_ID}.log"

RC=$?
echo "Exit code: ${RC}  Run: ${RUN_ID}"
exit ${RC}
GPTEOF

chmod +x /tmp/slurm_gpt22m.sh
JOB3=$(sbatch --parsable /tmp/slurm_gpt22m.sh)
echo "  → ${JOB3}"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
echo "============================================================"
echo "  ALL JOBS SUBMITTED"
echo "============================================================"
echo ""
echo "  Job 1  test-eval        ${JOB1}      (A30,  ~4h)"
echo "  Job 2  placebo array    ${JOB2}      (L20,  ~3h each × 6)"
echo "  Job 3  gpt22m           ${JOB3}      (L20,  ~2h)"
echo ""
echo "  Monitor:  squeue -u ${USER}"
echo "  Cancel:   scancel ${JOB1} ${JOB2} ${JOB3}"
echo "  Logs:     ls -lt ${LOG_DIR}/"
echo ""
echo "  After jobs complete, integrate results:"
echo "    - Test metrics → paper main.tex (replace val with test numbers)"
echo "    - Placebo results → add placebo table to paper"
echo "    - GPT-22M → param-matched comparison for Track A discussion"
echo "============================================================"
