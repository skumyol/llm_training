#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${ROOT}/../.venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="python3"

cd "$ROOT"
exec "$PYTHON" scripts/run_paper_small_lms.py "$@"
