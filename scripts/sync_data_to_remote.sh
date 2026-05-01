#!/usr/bin/env bash
# =============================================================================
# sync_data_to_remote.sh — Transfer only training data to remote server
# =============================================================================
# Usage:
#   ./scripts/sync_data_to_remote.sh user@host:/path/to/target
#   ./scripts/sync_data_to_remote.sh user@192.168.1.100:/home/user/llm_training
#
# What it transfers:
#   - data/ (raw_episodes, validated_turns, counterfactuals, merged_validated)
#   - slm_training/data/ (processed training datasets)
#   - checkpoints/ (model checkpoints if they exist)
# =============================================================================
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${CYAN}[sync-data]${NC} $*"; }
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

log "Syncing data to: ${REMOTE_HOST}:${REMOTE_PATH}"

# ── Verify local data exists ─────────────────────────────────────────────────
LOCAL_DATA_DIRS=(
    "data/raw_episodes"
    "data/validated_turns"
    "data/counterfactuals"
    "data/merged_validated"
    "data/packaged"
    "data/splits"
    "data/scenario_bank"
    "data/world_contexts"
    "slm_training/data"
)

for dir in "${LOCAL_DATA_DIRS[@]}"; do
    if [[ -d "$dir" ]]; then
        size=$(du -sh "$dir" 2>/dev/null | cut -f1 || echo "unknown")
        log "Found: $dir (${size})"
    else
        log "Warning: $dir not found locally"
    fi
done

# ── Prompt for confirmation ──────────────────────────────────────────────────
read -r -p "Continue with sync? [y/N] " REPLY
[[ "$REPLY" =~ ^[Yy]$ ]] || err "Aborted."

# ── Sync all data in just 2 rsync calls ────────────────────────────────────
log "Creating remote directories and syncing data..."

# Create all dirs in one SSH, then sync data/ (which includes all subdirs)
ssh "$REMOTE_HOST" "mkdir -p ${REMOTE_PATH}/data ${REMOTE_PATH}/slm_training/data"

log "Syncing data/ (all subdirectories)..."
rsync -avz "${ROOT}/data/" "${REMOTE_HOST}:${REMOTE_PATH}/data/"

log "Syncing slm_training/data..."
rsync -avz "${ROOT}/slm_training/data/" "${REMOTE_HOST}:${REMOTE_PATH}/slm_training/data/"

ok "Data sync complete!"
echo ""
echo "  Remote: ${REMOTE_HOST}:${REMOTE_PATH}"
echo ""
echo "  Verify on remote:"
echo "    ssh ${REMOTE_HOST}"
echo "    du -sh ${REMOTE_PATH}/data/*"
echo "    du -sh ${REMOTE_PATH}/slm_training/data"
