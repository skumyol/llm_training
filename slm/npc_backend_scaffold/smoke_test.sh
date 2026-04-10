#!/usr/bin/env bash
# =============================================================================
# NPC Backend Pipeline — Smoke Test Runner
#
# Usage:
#   ./smoke_test.sh              # full run
#   ./smoke_test.sh --clean      # delete smoke_artifacts first, then run
#
# What it does:
#   1. Creates / activates a local Python venv (.venv)
#   2. Installs requirements.txt
#   3. Runs smoke_test.py (downloads ~300 MB of models on first run)
#
# Models used (auto-downloaded from HuggingFace Hub):
#   - distilbert-base-uncased          (66 MB)  personality & affect encoders
#   - distilgpt2                       (82 MB)  dialogue backbone
#   - paraphrase-MiniLM-L3-v2          (17 MB)  episodic memory embedder
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
SMOKE_ARTIFACTS="$SCRIPT_DIR/smoke_artifacts"

# ── Parse flags ───────────────────────────────────────────────────────────────
CLEAN=0
for arg in "$@"; do
  case "$arg" in
    --clean) CLEAN=1 ;;
    *) echo "Unknown flag: $arg" && exit 1 ;;
  esac
done

if [ "$CLEAN" -eq 1 ] && [ -d "$SMOKE_ARTIFACTS" ]; then
  echo "[clean] Removing $SMOKE_ARTIFACTS"
  rm -rf "$SMOKE_ARTIFACTS"
fi

# ── Python version check ──────────────────────────────────────────────────────
PYTHON_BIN="python3"
PY_VERSION=$("$PYTHON_BIN" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
REQUIRED="3.9"

python3 -c "
import sys
v = sys.version_info
if (v.major, v.minor) < (3, 9):
    print(f'ERROR: Python {v.major}.{v.minor} < 3.9 required')
    sys.exit(1)
print(f'[setup] Python {v.major}.{v.minor} OK')
"

# ── Virtual environment ───────────────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
  echo "[setup] Creating virtual environment at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "[setup] Upgrading pip..."
pip install --quiet --upgrade pip

echo "[setup] Installing requirements..."
pip install --quiet -r "$SCRIPT_DIR/requirements.txt"

# tiktoken is optional (used by train_benchmark_small_lms_example.py)
pip install --quiet tiktoken 2>/dev/null || true

# ── Run smoke test ────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Starting smoke test from: $SCRIPT_DIR"
echo "============================================================"
echo ""

cd "$SCRIPT_DIR"
PYTHONPATH="$SCRIPT_DIR" python smoke_test.py

EXIT_CODE=$?
deactivate

if [ $EXIT_CODE -eq 0 ]; then
  echo ""
  echo "[smoke_test.sh] SUCCESS"
else
  echo ""
  echo "[smoke_test.sh] FAILED (exit $EXIT_CODE)"
fi

exit $EXIT_CODE
