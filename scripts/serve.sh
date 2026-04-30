#!/usr/bin/env bash
# =============================================================================
# serve.sh — Launch the NPC Dialogue API server
# =============================================================================
# Usage:
#   ./scripts/serve.sh llm                          # LLM fine-tuned model
#   ./scripts/serve.sh slm --arch gpt               # SLM from scratch
#   ./scripts/serve.sh both                         # Both systems
#   ./scripts/serve.sh llm --backend vllm           # vLLM for high throughput
#   ./scripts/serve.sh llm --port 8080              # Custom port
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"
[ -f "$PYTHON" ] || PYTHON="python3"

# ── Auto-detect checkpoints ───────────────────────────────────────────────────
LLM_CKPT=""
if [ -d "$ROOT/checkpoints/joint_model_best" ]; then
    LLM_CKPT="$ROOT/checkpoints/joint_model_best"
elif [ -d "$ROOT/checkpoints/response_generator_best" ]; then
    LLM_CKPT="$ROOT/checkpoints/response_generator_best"
elif [ -d "$ROOT/checkpoints/latent_predictor_best" ]; then
    LLM_CKPT="$ROOT/checkpoints/latent_predictor_best"
fi

SLM_DIR=""
if [ -d "$ROOT/slm_training/artifacts/small_lm" ]; then
    SLM_DIR=$(find "$ROOT/slm_training/artifacts/small_lm" -maxdepth 1 -name 'final_*' -type d | sort | tail -1)
fi

# ── Parse system ──────────────────────────────────────────────────────────────
SYSTEM="${1:-llm}"
shift 2>/dev/null || true
EXTRA_ARGS=()

case "$SYSTEM" in
    llm)
        EXTRA_ARGS+=(--system llm)
        [ -n "$LLM_CKPT" ] && EXTRA_ARGS+=(--checkpoint "$LLM_CKPT")
        ;;
    slm)
        EXTRA_ARGS+=(--system slm)
        [ -n "$SLM_DIR" ] && EXTRA_ARGS+=(--model-dir "$SLM_DIR")
        ;;
    both)
        EXTRA_ARGS+=(--system both)
        [ -n "$LLM_CKPT" ] && EXTRA_ARGS+=(--checkpoint "$LLM_CKPT")
        [ -n "$SLM_DIR" ] && EXTRA_ARGS+=(--model-dir "$SLM_DIR")
        ;;
    *)
        echo "Usage: $0 {llm|slm|both} [--port PORT] [--backend vllm] [...]"
        exit 1
        ;;
esac

# Pass through remaining args
EXTRA_ARGS+=("$@")

echo "============================================="
echo "  NPC Dialogue API Server"
echo "============================================="
echo "  System:   $SYSTEM"
echo "  LLM ckpt: ${LLM_CKPT:-not found}"
echo "  SLM dir:  ${SLM_DIR:-not found}"
echo "============================================="
echo ""

exec "$PYTHON" "$ROOT/scripts/serve.py" "${EXTRA_ARGS[@]}"
