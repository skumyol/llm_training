#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

PYTHON="${PROJECT_ROOT}/.venv/bin/python"
if [[ ! -f "$PYTHON" ]]; then
  PYTHON="python3"
fi

CONFIG="${1:-configs/data_gen.yaml}"
DRY_RUN="${2:-}"
N_EPISODES="${3:-}"

EXTRA_ARGS=""
if [[ "$DRY_RUN" == "--dry-run" ]]; then
  EXTRA_ARGS="$EXTRA_ARGS --dry-run"
fi
if [[ -n "$N_EPISODES" ]]; then
  EXTRA_ARGS="$EXTRA_ARGS --n-episodes $N_EPISODES"
fi

echo "========================================="
echo " LLM Training — Data Generation Pipeline"
echo "========================================="
echo " Config:      $CONFIG"
echo " Dry-run:     ${DRY_RUN:-no}"
echo " N episodes:  ${N_EPISODES:-from config}"
echo "-----------------------------------------"

"$PYTHON" run_data_gen.py --config "$CONFIG" $EXTRA_ARGS

echo ""
echo "[Done] Data generation complete."
echo "  Raw episodes:     data/raw_episodes/"
echo "  Validated turns:  data/validated_turns/"
echo "  Counterfactuals:  data/counterfactuals/"
echo "  Packaged:         data/packaged/"
echo "  Splits:           data/splits/"
