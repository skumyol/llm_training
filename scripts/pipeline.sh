#!/usr/bin/env bash
# =============================================================================
# Unified LLM + SLM Pipeline
# =============================================================================
# Usage:
#   ./scripts/pipeline.sh data-gen [--dry-run] [--n-episodes N]   # Generate training data
#   ./scripts/pipeline.sh train latent|response|joint|all [--debug] # Fine-tune LLM
#   ./scripts/pipeline.sh eval latent|response|routing|all          # Evaluate
#   ./scripts/pipeline.sh slm-train [arch]                         # SLM from scratch
#   ./scripts/pipeline.sh slm-eval [arch]                          # SLM evaluation
#   ./scripts/pipeline.sh full                                     # Full end-to-end
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${ROOT}/.venv/bin/python"
[ -f "$PYTHON" ] || PYTHON="python3"

LLM_DIR="$ROOT/llm_finetuning"
SLM_DIR="$ROOT/slm_training"

# ── helpers ───────────────────────────────────────────────────────────────────
_run_llm() {
    local kind="$1"; shift
    PYTHONPATH="$LLM_DIR" "$PYTHON" "$LLM_DIR/run_${kind}.py" "$@"
}

die() { echo "[ERROR] $*" >&2; exit 1; }

# ── data generation ───────────────────────────────────────────────────────────
do_data_gen() {
    local dry="" n_ep=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run) dry="--dry-run" ;;
            --n-episodes) n_ep="--n-episodes $2"; shift ;;
            --no-mlflow) dry="${dry} --no-mlflow" ;;
        esac
        shift
    done
    echo "=== Data Generation ==="
    _run_llm data_gen --config configs/data_gen.yaml $dry $n_ep
}

# ── LLM training ──────────────────────────────────────────────────────────────
do_train() {
    local stage="${1:-all}"; shift
    local debug="${1:-}"
    [[ "$debug" == "--debug" ]] && debug="--debug" || debug=""

    case "$stage" in
        latent|response|joint)
            echo "=== LLM Training: $stage ==="
            _run_llm train --stage "$stage" --config "configs/train_${stage}.yaml" $debug
            ;;
        all)
            echo "=== Full LLM Training Pipeline ==="
            for s in latent response joint; do
                echo "--- Stage: $s ---"
                _run_llm train --stage "$s" --config "configs/train_${s}.yaml" $debug
            done
            ;;
        *) die "Unknown stage: $stage (use latent|response|joint|all)" ;;
    esac
}

# ── LLM evaluation ────────────────────────────────────────────────────────────
do_eval() {
    local stage="${1:-all}"
    echo "=== LLM Evaluation: $stage ==="
    _run_llm eval --stage "$stage" --config configs/eval.yaml
}

# ── SLM training from scratch ─────────────────────────────────────────────────
do_slm_train() {
    local arch="${1:-all}"
    echo "=== SLM Training: $arch ==="
    cd "$SLM_DIR"
    PYTHONPATH="$SLM_DIR" bash run_full_slm_training.sh "$arch"
}

do_slm_eval() {
    local arch="${1:-all}"
    echo "=== SLM Evaluation: $arch ==="
    cd "$SLM_DIR"
    PYTHONPATH="$SLM_DIR" bash evaluate.sh "$arch"
}

# ── full end-to-end ───────────────────────────────────────────────────────────
do_full() {
    echo "=== Full End-to-End Pipeline ==="
    do_data_gen "$@"
    do_train all
    do_eval all
    echo "=== Pipeline Complete ==="
}

# ── main ──────────────────────────────────────────────────────────────────────
case "${1:-}" in
    data-gen)    shift; do_data_gen "$@" ;;
    train)       shift; do_train "$@" ;;
    eval)        shift; do_eval "$@" ;;
    slm-train)   shift; do_slm_train "$@" ;;
    slm-eval)    shift; do_slm_eval "$@" ;;
    full)        shift; do_full "$@" ;;
    *)
        echo "Usage: $0 {data-gen|train|eval|slm-train|slm-eval|full} [args...]"
        exit 1
        ;;
esac
