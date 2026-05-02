#!/usr/bin/env bash
# =============================================================================
# Parallel SLM Training Pipeline
# =============================================================================
# Runs HPO and final training for ALL architectures CONCURRENTLY on a single
# GPU by time-sharing via subprocess parallelism. Small LMs (2-20M params) are
# small enough that rapid context switching between them keeps GPU utilization
# high while reducing wall-clock time vs sequential execution.
#
# Usage:
#   bash run_parallel_slm_pipeline.sh                    # full pipeline
#   bash run_parallel_slm_pipeline.sh --hpo-only        # HPO only
#   bash run_parallel_slm_pipeline.sh --train-only      # skip HPO, use existing bests
#   bash run_parallel_slm_pipeline.sh --arch gpt        # single arch
#   bash run_parallel_slm_pipeline.sh --dry-run       # print what would run
#
# Monitoring:
#   tail -f logs/parallel_slm_*.log
#   watch -n 5 'ls -lt logs/parallel_slm_*.log | head -20'
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ROOT}/../.venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="python3"
export PYTHONPATH="$ROOT"
cd "$ROOT"

# ── Configuration ──────────────────────────────────────────────────────────────
ARCHS="gru awdlstm gpt prefix_gpt moe mamba_like"
N_TRIALS=20
HPO_EPOCHS=5
FINAL_EPOCHS=30
SEEDS="42 43 44"
MAX_JOBS=6           # Run all 6 architectures concurrently
TRAIN_TEXT="${TRAIN_TEXT:-$ROOT/data/dialogue/train.txt}"
VAL_TEXT="${VAL_TEXT:-$ROOT/data/dialogue/val.txt}"

MODE="full"
SELECTED_ARCH="all"
DRY_RUN=false

# ── Parse args ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hpo-only)    MODE="hpo"     ; shift ;;
    --train-only)  MODE="train"   ; shift ;;
    --arch)        SELECTED_ARCH="$2"; shift 2 ;;
    --dry-run)     DRY_RUN=true   ; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

RUN_TAG="parallel_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs"
PID_DIR="$LOG_DIR/pids_${RUN_TAG}"
mkdir -p "$LOG_DIR" "$PID_DIR"

LOG_FILE="$LOG_DIR/parallel_slm_${RUN_TAG}.log"
JOBS_FILE="$PID_DIR/jobs.txt"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

# ── Select architectures ───────────────────────────────────────────────────────
if [[ "$SELECTED_ARCH" == "all" ]]; then
  ARCH_LIST=( $ARCHS )
else
  ARCH_LIST=( "$SELECTED_ARCH" )
fi

log "================================================================"
log "  PARALLEL SLM PIPELINE"
log "  Run ID : $RUN_TAG"
log "  Mode   : $MODE"
log "  Archs  : ${ARCH_LIST[*]}"
log "  Max concurrent jobs: $MAX_JOBS"
log "  Log    : $LOG_FILE"
log "================================================================"

if [[ "$DRY_RUN" == true ]]; then
  log "[DRY RUN] Would launch these jobs:"
fi

# =============================================================================
# PHASE 1: Parallel HPO for all architectures
# =============================================================================
launch_hpo_job() {
  local arch="$1"
  local job_log="$LOG_DIR/parallel_slm_hpo_${arch}_${RUN_TAG}.log"
  local best_json="$ROOT/artifacts/optuna/small_lm_${arch}_best.json"

  if [[ "$DRY_RUN" == true ]]; then
    log "  [DRY] HPO: $arch → $job_log"
    return
  fi

  if [[ -f "$best_json" && "$MODE" != "hpo" ]]; then
    log "  $arch: best.json exists, skipping HPO"
    echo "$arch:hpo:skipped" >> "$JOBS_FILE"
    return
  fi

  log "  Launching HPO job: $arch (trials=$N_TRIALS, epochs=$HPO_EPOCHS)"

  # Run optuna in background, redirect all output to its own log
  (
    echo "=== HPO START: $arch $(date) ===" > "$job_log"
    "$PYTHON" "$ROOT/scripts/optuna_small_lm.py" \
      --arch "$arch" \
      --n-trials "$N_TRIALS" \
      --epochs "$HPO_EPOCHS" \
      --train-text "$TRAIN_TEXT" \
      --val-text "$VAL_TEXT" \
      --timeout 1800 \
      >> "$job_log" 2>&1
    RC=$?
    echo "=== HPO DONE: $arch rc=$RC $(date) ===" >> "$job_log"
    echo "$arch:hpo:$RC" >> "$JOBS_FILE"
  ) &

  echo $! > "$PID_DIR/hpo_${arch}.pid"
}

run_hpo_phase() {
  log ""
  log ">>> Phase 1: Parallel HPO for ${#ARCH_LIST[@]} architectures"
  log "    Each: $N_TRIALS trials × $HPO_EPOCHS epochs"
  log "    Output: logs/parallel_slm_hpo_<arch>_${RUN_TAG}.log"

  > "$JOBS_FILE"

  for arch in "${ARCH_LIST[@]}"; do
    launch_hpo_job "$arch"
    sleep 2  # Small stagger to avoid simultaneous CUDA init
  done

  log "  All HPO jobs launched. Waiting for completion..."
  wait  # Wait for ALL background HPO jobs to finish
  log "  All HPO jobs complete."

  # Summarize results
  log ""
  log "  HPO Results Summary:"
  for arch in "${ARCH_LIST[@]}"; do
    local best_json="$ROOT/artifacts/optuna/small_lm_${arch}_best.json"
    if [[ -f "$best_json" ]]; then
      local ppl
      ppl=$("$PYTHON" -c "import json; d=json.load(open('$best_json')); print(f\"{d.get('best_val_ppl',0):.2f}\")" 2>/dev/null || echo "?")
      log "    $arch: val_ppl=$ppl"
    else
      log "    $arch: MISSING best.json — HPO may have failed"
    fi
  done
}

# =============================================================================
# PHASE 2: Parallel Final Training (all arch × all seeds concurrently)
# =============================================================================
launch_train_job() {
  local arch="$1"
  local seed="$2"
  local job_log="$LOG_DIR/parallel_slm_train_${arch}_s${seed}_${RUN_TAG}.log"

  if [[ "$DRY_RUN" == true ]]; then
    log "  [DRY] TRAIN: $arch seed=$seed → $job_log"
    return
  fi

  log "  Launching train job: $arch seed=$seed"

  (
    echo "=== TRAIN START: $arch seed=$seed $(date) ===" > "$job_log"
    "$PYTHON" "$ROOT/scripts/train_final_small_lms.py" \
      --arch "$arch" \
      --seeds "$seed" \
      --epochs "$FINAL_EPOCHS" \
      --train-text "$TRAIN_TEXT" \
      --val-text "$VAL_TEXT" \
      --skip-existing \
      >> "$job_log" 2>&1
    RC=$?
    echo "=== TRAIN DONE: $arch seed=$seed rc=$RC $(date) ===" >> "$job_log"
    echo "$arch:train:s${seed}:$RC" >> "$JOBS_FILE"
  ) &

  echo $! > "$PID_DIR/train_${arch}_s${seed}.pid"
}

run_train_phase() {
  log ""
  log ">>> Phase 2: Parallel Final Training"
  log "    $N_ARCHS architectures × $N_SEEDS seeds = $((N_ARCHS * N_SEEDS)) jobs"
  log "    Output: logs/parallel_slm_train_<arch>_s<seed>_${RUN_TAG}.log"

  for arch in "${ARCH_LIST[@]}"; do
    local best_json="$ROOT/artifacts/optuna/small_lm_${arch}_best.json"
    if [[ ! -f "$best_json" ]]; then
      log "  WARNING: $arch missing best.json — skipping final training"
      continue
    fi

    for seed in $SEEDS; do
      launch_train_job "$arch" "$seed"
      sleep 1  # Stagger to reduce GPU init contention
    done
  done

  log "  All training jobs launched. Waiting for completion..."
  wait
  log "  All training jobs complete."
}

# =============================================================================
# PHASE 3: Evaluation (sequential — single script handles all)
# =============================================================================
run_eval_phase() {
  log ""
  log ">>> Phase 3: Evaluation (PPL + BLEU + Distinct)"

  if [[ "$DRY_RUN" == true ]]; then
    log "  [DRY] Would run: python scripts/eval_small_lms.py"
    return
  fi

  local eval_log="$LOG_DIR/parallel_slm_eval_${RUN_TAG}.log"
  local eval_csv="$ROOT/artifacts/slm_parallel_eval_${RUN_TAG}.csv"

  "$PYTHON" "$ROOT/scripts/eval_small_lms.py" \
    --out-csv "$eval_csv" \
    >> "$eval_log" 2>&1

  log "  Evaluation complete → $eval_csv"
}

# =============================================================================
# Main
# =============================================================================
SEED_ARRAY=($SEEDS)
N_ARCHS=${#ARCH_LIST[@]}
N_SEEDS=${#SEED_ARRAY[@]}

if [[ "$MODE" == "full" || "$MODE" == "hpo" ]]; then
  run_hpo_phase
fi

if [[ "$MODE" == "full" || "$MODE" == "train" ]]; then
  run_train_phase
fi

if [[ "$MODE" == "full" ]]; then
  run_eval_phase
fi

# ── Final Summary ────────────────────────────────────────────────────────────
log ""
log "================================================================"
log "  PARALLEL PIPELINE COMPLETE"
log "  Run ID: $RUN_TAG"
log ""

if [[ "$DRY_RUN" == false ]]; then
  # Count completed/failed
  local completed=0
  local failed=0
  if [[ -f "$JOBS_FILE" ]]; then
    while IFS=: read -r _ _ status _; do
      if [[ "$status" == "0" || "$status" == "skipped" ]]; then
        completed=$((completed + 1))
      else
        failed=$((failed + 1))
      fi
    done < "$JOBS_FILE"
  fi
  log "  Jobs completed: $completed"
  log "  Jobs failed:    $failed"
fi

log "  Logs:   $LOG_DIR/parallel_slm_*_${RUN_TAG}.log"
log "  Artifacts: $ROOT/artifacts/"
log "  MLflow: mlflow ui --backend-store-uri ./mlruns"
log "================================================================"

# Cleanup PID dir
rm -rf "$PID_DIR"
