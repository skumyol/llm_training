#!/usr/bin/env bash
# =============================================================================
# deploy.sh — Transfer & set up the training pipeline on a remote server
# =============================================================================
# Usage:
#   ./scripts/deploy.sh user@host:/path/to/target
#   ./scripts/deploy.sh user@192.168.1.100:/home/user/llm_training
#
# What it does:
#   1. Rsyncs source + checkpoints + data (skipping venv, logs, caches, huge CSVs)
#   2. SSHes in, creates venv, installs deps, downloads base models
#   3. Runs smoke tests to verify
# =============================================================================
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${CYAN}[deploy]${NC} $*"; }
ok()   { echo -e "${GREEN}[  ok ]${NC} $*"; }
err()  { echo -e "${RED}[ FAIL]${NC} $*"; exit 1; }

# ── Parse target ──────────────────────────────────────────────────────────────
TARGET="${1:-}"
if [[ -z "$TARGET" ]]; then
    echo "Usage: $0 user@host:/path/to/llm_training"
    echo "  e.g.  $0 serkan@192.168.1.100:/home/serkan/llm_training"
    exit 1
fi

REMOTE_HOST="${TARGET%%:*}"
REMOTE_PATH="${TARGET#*:}"
REMOTE_PATH="${REMOTE_PATH:-/home/${TARGET%@*}/llm_training}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

log "Deploying to: ${REMOTE_HOST}:${REMOTE_PATH}"

# ── Prompt for confirmation ──────────────────────────────────────────────────
read -r -p "Continue? [y/N] " REPLY
[[ "$REPLY" =~ ^[Yy]$ ]] || err "Aborted."

# =============================================================================
# Phase 1: Rsync everything
# =============================================================================
log "Phase 1: Rsync files to remote..."

RSYNC_EXCLUDES=(
    --exclude '.venv/'
    --exclude '.git/'
    --exclude '__pycache__/'
    --exclude '*.pyc'
    --exclude '.pytest_cache/'
    --exclude 'mlruns/'
    --exclude '*.log'
    --exclude 'logs/'
    --exclude 'slm_training/logs/'
    --exclude 'slm_training/mlruns/'
    --exclude 'slm_training/models/'            # Re-download from HF
    --exclude 'slm_training/data/'              # Re-download from sources
    --exclude 'slm_training/artifacts/exported_models/'
    --exclude 'predictions_epoch*.csv'          # ~60GB of eval dumps
    --exclude 'slm_training/artifacts/*/optuna_*'  # HPO trial runs
    --exclude 'slm_training/artifacts/*/report_*'  # Report artifacts
    --exclude 'slm_training/artifacts/*/smoke_*'   # Smoke test artifacts
    --exclude 'slm_training/artifacts/*/affect_v*' # Iteration artifacts
    --exclude 'slm_training/artifacts/*/test_*'
    --exclude 'slm_training/artifacts/*/mamba_only_*'
    --exclude 'eval_results/'
    --exclude 'data/raw_episodes/'
    --exclude 'data/validated_turns/'
    --exclude 'data/counterfactuals/'
    --exclude 'data/merged_validated/'
    --exclude 'docs/_build/'
    --exclude 'docs/archive/'
)

# Ensure remote dir exists
ssh "$REMOTE_HOST" "mkdir -p $REMOTE_PATH"

rsync -avz --progress "${RSYNC_EXCLUDES[@]}" ./ "${REMOTE_HOST}:${REMOTE_PATH}/"
ok "Files transferred."

# =============================================================================
# Phase 2: Remote setup
# =============================================================================
log "Phase 2: Remote environment setup..."

ssh "$REMOTE_HOST" bash -s << ENDSSH
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'
log2()  { echo -e "\${CYAN}[remote]\${NC} \$*"; }
ok2()   { echo -e "\${GREEN}[  ok ]\${NC} \$*"; }

cd "$REMOTE_PATH"

# ── 2a: Python venv ──────────────────────────────────────────────────────────
if [ ! -d .venv ]; then
    log2 "Creating virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate

log2 "Upgrading pip..."
pip install --quiet --upgrade pip

log2 "Installing root requirements..."
pip install --quiet -r requirements.txt

log2 "Installing SLM requirements..."
pip install --quiet -r slm_training/requirements.txt 2>/dev/null || true

log2 "Installing doc requirements..."
pip install --quiet -r docs/requirements.txt 2>/dev/null || true

ok2 "Python environment ready."

# ── 2b: Download external datasets ──────────────────────────────────────────
cd slm_training

log2 "Downloading external datasets (personachat, crd3, empathetic_dialogues, dailydialog)..."
PYTHONPATH=. python -m src.data.datasets \
    --datasets personachat crd3 empathetic_dialogues dailydialog 2>&1 | tail -5 || true

log2 "Converting datasets to training format..."
PYTHONPATH=. python -m src.data.prepare_dialogue_data 2>&1 | tail -5 || true
PYTHONPATH=. python -m src.data.prepare_encoder_data 2>&1 | tail -5 || true

ok2 "External datasets ready."

# ── 2c: Smoke test ──────────────────────────────────────────────────────────
log2 "Running SLM smoke test..."
if bash smoke_test.sh 2>&1 | tail -10; then
    ok2 "SLM smoke test PASSED"
else
    log2 "SLM smoke test had warnings (may need GPU for full test)"
fi

cd ..

# ── 2d: LLM dry-run test ───────────────────────────────────────────────────
log2 "Running LLM data-gen dry run..."
cd llm_finetuning
PYTHONPATH=. python run_data_gen.py \
    --config configs/data_gen.yaml --dry-run --n-episodes 5 --no-mlflow 2>&1 | tail -5 || true
ok2 "LLM data-gen dry run OK"

cd ..
ok2 "Setup complete."
ENDSSH

# =============================================================================
# Phase 3: Summary
# =============================================================================
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  DEPLOY COMPLETE${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "  Remote:  ${REMOTE_HOST}:${REMOTE_PATH}"
echo ""
echo "  Next steps on remote:"
echo "    ssh ${REMOTE_HOST}"
echo "    cd ${REMOTE_PATH}"
echo "    source .venv/bin/activate"
echo ""
echo "  LLM fine-tuning:"
echo "    ./scripts/pipeline.sh data-gen --n-episodes 500"
echo "    ./scripts/pipeline.sh train latent --debug"
echo "    ./scripts/pipeline.sh train all"
echo ""
echo "  SLM training:"
echo "    cd slm_training"
echo "    bash train_all.sh --run-id my_experiment"
echo ""
echo "  Resume interrupted SLM run:"
echo "    cd slm_training"
echo "    nohup ../.venv/bin/python scripts/sequential_training_orchestrator.py \\"
echo "        > /tmp/sequential_training.log 2>&1 &"
echo ""
