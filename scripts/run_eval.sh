#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

PYTHON="${PROJECT_ROOT}/.venv/bin/python"
if [[ ! -f "$PYTHON" ]]; then
  PYTHON="python3"
fi

STAGE="${1:-all}"
CONFIG="${2:-configs/eval.yaml}"

echo "========================================="
echo " LLM Training — Evaluation: $STAGE"
echo "========================================="
echo " Config: $CONFIG"
echo "-----------------------------------------"

"$PYTHON" run_eval.py --stage "$STAGE" --config "$CONFIG"

echo "[Done] Evaluation stage '$STAGE' complete."
echo "  Results: $(grep results_dir $CONFIG | awk '{print $2}')"
