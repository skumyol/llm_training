#!/usr/bin/env bash
# =============================================================================
# Train + Auto-Evaluate Wrapper
# =============================================================================
# One entry point for training a model and immediately running the matching
# evaluation/report step.
#
# Usage:
#   ./scripts/train_and_eval.sh llm [latent|response|joint|all] [--debug]
#   ./scripts/train_and_eval.sh slm [train_small_lms.sh args...]
#   ./scripts/train_and_eval.sh all
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT}/.venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="python3"

PIPELINE="$ROOT/scripts/pipeline.sh"
LLM_DIR="$ROOT/llm_finetuning"
SLM_DIR="$ROOT/slm_training"

usage() {
    cat <<'EOF'
Usage:
  train_and_eval.sh llm [latent|response|joint|all] [--debug]
  train_and_eval.sh slm [train_small_lms.sh args...]
  train_and_eval.sh all

Examples:
  ./scripts/train_and_eval.sh llm all
  ./scripts/train_and_eval.sh llm latent --debug
  ./scripts/train_and_eval.sh slm --arch gpt --skip-hpo
  ./scripts/train_and_eval.sh all
EOF
}

die() {
    printf '[ERROR] %s\n' "$*" >&2
    exit 1
}

run_llm() {
    local stage="${1:-all}"
    shift || true

    local debug_flag=""
    if [[ "${1:-}" == "--debug" ]]; then
        debug_flag="--debug"
    fi

    case "$stage" in
        latent|response|joint)
            bash "$PIPELINE" train "$stage" $debug_flag
            bash "$PIPELINE" eval "$([[ "$stage" == "joint" ]] && printf all || printf '%s' "$stage")"
            ;;
        all)
            bash "$PIPELINE" train all $debug_flag
            bash "$PIPELINE" eval all
            ;;
        *)
            die "Unknown LLM stage: $stage (use latent|response|joint|all)"
            ;;
    esac
}

run_slm() {
    local args=("$@")
    local has_hpo_only=false
    for arg in "${args[@]}"; do
        if [[ "$arg" == "--hpo-only" ]]; then
            has_hpo_only=true
            break
        fi
    done

    cd "$SLM_DIR"
    bash train_small_lms.sh "${args[@]}"

    if [[ "$has_hpo_only" == false ]]; then
        "$PYTHON" scripts/comprehensive_training_report.py --phase report-only
    fi
}

main() {
    local mode="${1:-}"
    shift || true

    case "$mode" in
        llm)
            run_llm "${1:-all}" "${@:2}"
            ;;
        slm)
            run_slm "$@"
            ;;
        all)
            run_llm all
            run_slm
            ;;
        -h|--help|"")
            usage
            ;;
        *)
            die "Unknown mode: $mode (use llm|slm|all)"
            ;;
    esac
}

main "$@"
