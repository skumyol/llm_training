#!/usr/bin/env bash
# =============================================================================
# NPC Backend — Full Training Orchestrator
#
# Pipeline:
#   Stage 1 [parallel]  → personality encoder  +  affect encoder
#   Stage 2 [depends 1] → build personality cache
#   Stage 3 [depends 2] → dialogue model (LoRA + prefix tuning)
#   Stage 4 [optional]  → Gemma 4 + Unsloth dialogue model
#
# Usage:
#   ./train_all.sh                          # default configs, auto hardware
#   ./train_all.sh --run-id exp_01          # tag this run
#   ./train_all.sh --sequential             # never parallelise (low RAM mode)
#   ./train_all.sh --skip-stage1            # restart from stage 2 onward
#   ./train_all.sh --only-stage1            # train encoders only
#   ./train_all.sh --with-gemma             # also train Gemma 4 with Unsloth
#   ./train_all.sh --config-dir configs/    # use a different config dir
#   ./train_all.sh --personality-config configs/personality.yaml \
#                  --affect-config     configs/affect.yaml    \
#                  --dialogue-config   configs/dialogue.yaml   \
#                  --gemma-config      configs/dialogue_gemma_unsloth.yaml
#
# Hardware auto-config:
#   CUDA ≥ 24 GB  → batch 32 / 4, parallel stage 1
#   CUDA 16-24 GB → batch 16 / 2, parallel stage 1
#   CUDA  8-16 GB → batch 8  / 1, parallel stage 1 (tight)
#   CUDA < 8 GB   → batch 4  / 1, sequential, AMP recommended
#   MPS           → batch 16 / 1, parallel stage 1
#   CPU           → batch 8  / 1, sequential (slow — expect hours)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$ROOT_DIR/.venv"

# ── Defaults ──────────────────────────────────────────────────────────────────
RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
CONFIG_DIR="$SCRIPT_DIR/configs"
PERSONALITY_CFG="$CONFIG_DIR/personality.yaml"
AFFECT_CFG="$CONFIG_DIR/affect.yaml"
DIALOGUE_CFG="$CONFIG_DIR/dialogue.yaml"
GEMMA_CFG="$CONFIG_DIR/dialogue_gemma_unsloth.yaml"
NPC_PROFILES="data/npc_profiles.csv"
SEQUENTIAL=false
SKIP_STAGE1=false
ONLY_STAGE1=false
SKIP_CACHE=false
WITH_GEMMA=false

# ── Parse CLI ─────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)              RUN_ID="$2";            shift 2 ;;
    --config-dir)          CONFIG_DIR="$2";
                           PERSONALITY_CFG="$CONFIG_DIR/personality.yaml"
                           AFFECT_CFG="$CONFIG_DIR/affect.yaml"
                           DIALOGUE_CFG="$CONFIG_DIR/dialogue.yaml"
                           GEMMA_CFG="$CONFIG_DIR/dialogue_gemma_unsloth.yaml"; shift 2 ;;
    --personality-config)  PERSONALITY_CFG="$2";   shift 2 ;;
    --affect-config)       AFFECT_CFG="$2";        shift 2 ;;
    --dialogue-config)     DIALOGUE_CFG="$2";      shift 2 ;;
    --gemma-config)        GEMMA_CFG="$2";         shift 2 ;;
    --npc-profiles)        NPC_PROFILES="$2";      shift 2 ;;
    --sequential)          SEQUENTIAL=true;         shift   ;;
    --skip-stage1)         SKIP_STAGE1=true;        shift   ;;
    --only-stage1)         ONLY_STAGE1=true;        shift   ;;
    --skip-cache)          SKIP_CACHE=true;         shift   ;;
    --with-gemma)          WITH_GEMMA=true;         shift   ;;
    *) echo "[ERROR] Unknown flag: $1" && exit 1 ;;
  esac
done

# ── Log dir ───────────────────────────────────────────────────────────────────
LOG_DIR="$SCRIPT_DIR/logs/${RUN_ID}"
mkdir -p "$LOG_DIR"

log()  { echo "$(date '+%Y-%m-%d %H:%M:%S') | INFO  | $*" | tee -a "$LOG_DIR/orchestrator.log"; }
warn() { echo "$(date '+%Y-%m-%d %H:%M:%S') | WARN  | $*" | tee -a "$LOG_DIR/orchestrator.log"; }
fail() { echo "$(date '+%Y-%m-%d %H:%M:%S') | ERROR | $*" | tee -a "$LOG_DIR/orchestrator.log"; exit 1; }

# ── Activate venv ─────────────────────────────────────────────────────────────
if [ ! -f "$VENV_DIR/bin/activate" ]; then
  fail "Virtual environment not found at $VENV_DIR. Run smoke_test.sh first to set it up."
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
PYTHON="$VENV_DIR/bin/python"

log "========================================================"
log "  NPC Backend — Training Orchestrator"
log "  Run ID  : $RUN_ID"
log "  Log dir : $LOG_DIR"
log "========================================================"

# ── Hardware detection ────────────────────────────────────────────────────────
DEVICE=$("$PYTHON" -c "
import torch
if torch.cuda.is_available():
    print('cuda')
elif hasattr(torch.backends,'mps') and torch.backends.mps.is_available():
    print('mps')
else:
    print('cpu')
" 2>/dev/null)

VRAM_GB=$("$PYTHON" -c "
import torch
if torch.cuda.is_available():
    gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(int(gb))
else:
    print(0)
" 2>/dev/null || echo 0)

RAM_GB=$("$PYTHON" -c "
import subprocess, sys
try:
    r = subprocess.run(['sysctl','-n','hw.memsize'], capture_output=True, text=True)
    print(int(int(r.stdout.strip()) / 1024**3))
except Exception:
    print(16)
" 2>/dev/null || echo 16)

log "Device     : $DEVICE  (VRAM=${VRAM_GB} GB, RAM=${RAM_GB} GB)"

# ── Memory-based hyperparameter profiles ─────────────────────────────────────
ENCODER_BATCH=16
DIALOGUE_BATCH=2
DIALOGUE_ACCUM=8
PARALLEL_STAGE1=true

if [ "$DEVICE" = "cuda" ]; then
  if   [ "$VRAM_GB" -ge 24 ]; then
    ENCODER_BATCH=32; DIALOGUE_BATCH=4; DIALOGUE_ACCUM=4
    log "Profile    : CUDA ≥ 24 GB — high throughput"
  elif [ "$VRAM_GB" -ge 16 ]; then
    ENCODER_BATCH=16; DIALOGUE_BATCH=2; DIALOGUE_ACCUM=8
    log "Profile    : CUDA 16-24 GB — balanced"
  elif [ "$VRAM_GB" -ge 8 ]; then
    ENCODER_BATCH=8;  DIALOGUE_BATCH=1; DIALOGUE_ACCUM=16
    log "Profile    : CUDA 8-16 GB — memory-conservative"
  else
    ENCODER_BATCH=4;  DIALOGUE_BATCH=1; DIALOGUE_ACCUM=16
    PARALLEL_STAGE1=false
    warn "Profile    : CUDA < 8 GB — sequential, low batch. Consider --sequential."
  fi
elif [ "$DEVICE" = "mps" ]; then
  ENCODER_BATCH=16; DIALOGUE_BATCH=1; DIALOGUE_ACCUM=16
  log "Profile    : Apple MPS — moderate throughput"
else
  ENCODER_BATCH=8; DIALOGUE_BATCH=1; DIALOGUE_ACCUM=16
  PARALLEL_STAGE1=false
  warn "Profile    : CPU — training will be slow. GPU recommended."
fi

[ "$SEQUENTIAL" = true ] && PARALLEL_STAGE1=false

log "Encoder batch : $ENCODER_BATCH"
log "Dialogue batch: $DIALOGUE_BATCH  (accum=$DIALOGUE_ACCUM, effective=$(( DIALOGUE_BATCH * DIALOGUE_ACCUM )))"
log "Parallel S1   : $PARALLEL_STAGE1"
log ""

# ── Helper: wait for PID with live status ─────────────────────────────────────
wait_job() {
  local pid=$1
  local name=$2
  local logfile=$3
  local spinner='|/-\'
  local i=0
  while kill -0 "$pid" 2>/dev/null; do
    local last_line
    last_line=$(tail -1 "$logfile" 2>/dev/null || echo "...")
    printf "\r  [%c] %-20s  %s" "${spinner:$((i%4)):1}" "$name" "${last_line:0:70}     "
    sleep 3
    i=$((i+1))
  done
  printf "\r%80s\r" ""   # clear line
  wait "$pid"
  return $?
}

START_TIME=$(date +%s)

# =============================================================================
# Stage 1 — Personality encoder + Affect encoder
# =============================================================================
if [ "$SKIP_STAGE1" = false ]; then
  log "── Stage 1: Encoder Training ────────────────────────────────────"

  # Config existence checks
  [ -f "$PERSONALITY_CFG" ] || fail "Missing config: $PERSONALITY_CFG"
  [ -f "$AFFECT_CFG"      ] || fail "Missing config: $AFFECT_CFG"

  PERSONALITY_LOG="$LOG_DIR/stage1_personality.log"
  AFFECT_LOG="$LOG_DIR/stage1_affect.log"

  if [ "$PARALLEL_STAGE1" = true ]; then
    log "Running personality + affect encoders in PARALLEL"

    cd "$SCRIPT_DIR"
    "$PYTHON" -m src.train.run_personality \
      --config "$PERSONALITY_CFG" \
      --run-id "${RUN_ID}_personality" \
      --batch-size "$ENCODER_BATCH" \
      > "$PERSONALITY_LOG" 2>&1 &
    PID_P=$!

    "$PYTHON" -m src.train.run_affect \
      --config "$AFFECT_CFG" \
      --run-id "${RUN_ID}_affect" \
      --batch-size "$ENCODER_BATCH" \
      > "$AFFECT_LOG" 2>&1 &
    PID_A=$!

    log "  personality PID=$PID_P  |  affect PID=$PID_A"
    log "  Logs: $PERSONALITY_LOG"
    log "        $AFFECT_LOG"

    wait_job "$PID_P" "personality" "$PERSONALITY_LOG"
    P_EXIT=$?
    if [ $P_EXIT -ne 0 ]; then
      fail "Personality encoder FAILED (exit $P_EXIT). Check $PERSONALITY_LOG"
    fi
    log "  ✓ Personality encoder complete"

    wait_job "$PID_A" "affect" "$AFFECT_LOG"
    A_EXIT=$?
    if [ $A_EXIT -ne 0 ]; then
      fail "Affect encoder FAILED (exit $A_EXIT). Check $AFFECT_LOG"
    fi
    log "  ✓ Affect encoder complete"

  else
    log "Running personality encoder (sequential mode)"
    cd "$SCRIPT_DIR"
    "$PYTHON" -m src.train.run_personality \
      --config "$PERSONALITY_CFG" \
      --run-id "${RUN_ID}_personality" \
      --batch-size "$ENCODER_BATCH" \
      2>&1 | tee "$PERSONALITY_LOG" || fail "Personality encoder FAILED"
    log "  ✓ Personality encoder complete"

    log "Running affect encoder (sequential mode)"
    "$PYTHON" -m src.train.run_affect \
      --config "$AFFECT_CFG" \
      --run-id "${RUN_ID}_affect" \
      --batch-size "$ENCODER_BATCH" \
      2>&1 | tee "$AFFECT_LOG" || fail "Affect encoder FAILED"
    log "  ✓ Affect encoder complete"
  fi
else
  log "── Stage 1: SKIPPED (--skip-stage1)"
fi

[ "$ONLY_STAGE1" = true ] && { log "Only-stage1 flag set — stopping here."; exit 0; }

# =============================================================================
# Stage 2 — Build personality cache
# =============================================================================
if [ "$SKIP_CACHE" = false ]; then
  log ""
  log "── Stage 2: Build Personality Cache ─────────────────────────────"

  # Determine personality encoder dir from config
  PERSONALITY_ENCODER_DIR=$("$PYTHON" -c "
import yaml, sys
with open('$PERSONALITY_CFG') as f:
    cfg = yaml.safe_load(f)
import os
enc_dir = os.path.join(cfg.get('output_dir','artifacts/personality_encoder'), '${RUN_ID}_personality', 'best_model')
print(enc_dir)
" 2>/dev/null || echo "artifacts/personality_encoder/${RUN_ID}_personality/best_model")

  CACHE_OUT=$("$PYTHON" -c "
import yaml
with open('$DIALOGUE_CFG') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('personality_cache_path','artifacts/personality_cache.jsonl'))
" 2>/dev/null || echo "artifacts/personality_cache.jsonl")

  log "  Encoder : $PERSONALITY_ENCODER_DIR"
  log "  Profiles: $NPC_PROFILES"
  log "  Output  : $CACHE_OUT"

  CACHE_LOG="$LOG_DIR/stage2_cache.log"
  cd "$SCRIPT_DIR"
  "$PYTHON" -m src.data.build_caches \
    --profiles-path "$NPC_PROFILES" \
    --encoder-dir   "$PERSONALITY_ENCODER_DIR" \
    --out-path      "$CACHE_OUT" \
    2>&1 | tee "$CACHE_LOG" || fail "Personality cache build FAILED. Check $CACHE_LOG"

  log "  ✓ Personality cache written → $CACHE_OUT"
else
  log "── Stage 2: SKIPPED (--skip-cache)"
fi

# =============================================================================
# Stage 3 — Dialogue model
# =============================================================================
log ""
log "── Stage 3: Dialogue Model Training ─────────────────────────────"

[ -f "$DIALOGUE_CFG" ] || fail "Missing config: $DIALOGUE_CFG"

DIALOGUE_LOG="$LOG_DIR/stage3_dialogue.log"
cd "$SCRIPT_DIR"

"$PYTHON" -m src.train.run_dialogue \
  --config      "$DIALOGUE_CFG" \
  --run-id      "${RUN_ID}_dialogue" \
  --batch-size  "$DIALOGUE_BATCH" \
  --grad-accum  "$DIALOGUE_ACCUM" \
  2>&1 | tee "$DIALOGUE_LOG" || fail "Dialogue model training FAILED. Check $DIALOGUE_LOG"

log "  ✓ Dialogue model complete"

# =============================================================================
# Stage 4 — Gemma 4 + Unsloth (optional)
# =============================================================================
if [ "$WITH_GEMMA" = true ]; then
  log ""
  log "── Stage 4: Gemma 4 + Unsloth Training ─────────────────────────"

  [ -f "$GEMMA_CFG" ] || fail "Missing config: $GEMMA_CFG"

  GEMMA_LOG="$LOG_DIR/stage4_gemma_unsloth.log"
  cd "$SCRIPT_DIR"

  "$PYTHON" -m src.train.run_gemma_unsloth \
    --config "$GEMMA_CFG" \
    --run-id "${RUN_ID}_gemma_unsloth" \
    2>&1 | tee "$GEMMA_LOG" || fail "Gemma Unsloth training FAILED. Check $GEMMA_LOG"

  log "  ✓ Gemma Unsloth model complete"
else
  log ""
  log "── Stage 4: SKIPPED (pass --with-gemma to enable Gemma 4 + Unsloth)"
fi

# =============================================================================
# Summary
# =============================================================================
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
ELAPSED_FMT=$(printf "%dh %02dm %02ds" $((ELAPSED/3600)) $((ELAPSED%3600/60)) $((ELAPSED%60)))

log ""
log "========================================================"
log "  TRAINING COMPLETE"
log "  Run ID   : $RUN_ID"
log "  Elapsed  : $ELAPSED_FMT"
log "  Logs     : $LOG_DIR/"
log "========================================================"
log ""
log "  Artifacts produced:"
log "    Personality: artifacts/personality_encoder/${RUN_ID}_personality/"
log "    Affect:      artifacts/affect_encoder/${RUN_ID}_affect/"
log "    Dialogue:    artifacts/dialogue_model/${RUN_ID}_dialogue/"
if [ "$WITH_GEMMA" = true ]; then
  log "    Gemma:       artifacts/gemma_unsloth/${RUN_ID}_gemma_unsloth/"
fi
log ""
log "  Run summaries (for ablation):"
"$PYTHON" -c "
import json, glob, sys
for f in sorted(glob.glob('artifacts/*/${RUN_ID}*/run_summary.json')):
    s = json.load(open(f))
    task = s.get('task','?')
    best = s.get('best',{})
    if 'val_mse' in best:
        print(f'    {task:25s} val_mse={best[\"val_mse\"]:.6f}  (epoch {best.get(\"epoch\")})')
    elif 'val_loss' in best:
        ppl = best.get('val_ppl','?')
        print(f'    {task:25s} val_loss={best[\"val_loss\"]:.4f}  ppl={ppl:.2f}  (epoch {best.get(\"epoch\")})')
" 2>/dev/null || true

deactivate 2>/dev/null || true
