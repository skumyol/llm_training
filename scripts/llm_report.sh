#!/usr/bin/env bash
# =============================================================================
# DeepSeek LLM report helper
# =============================================================================
# Sends a single chat-completions request to DeepSeek using the root .env file.
#
# Usage:
#   ./scripts/llm_report.sh
#   ./scripts/llm_report.sh --prompt "Write a concise training report."
#   ./scripts/llm_report.sh --model deepseek-v4-pro --reasoning-effort high
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$ROOT/.env"

MODEL="deepseek-v4-pro"
PROMPT="Hello!"
THINKING_TYPE="enabled"
REASONING_EFFORT="high"
STREAM="false"
API_URL="https://api.deepseek.com/chat/completions"

usage() {
    cat <<'EOF'
Usage: llm_report.sh [options]

Options:
  -p, --prompt TEXT            User prompt to send (default: Hello!)
  -m, --model NAME             DeepSeek model name (default: deepseek-v4-pro)
  -r, --reasoning-effort LVL   Reasoning effort (default: high)
  -t, --thinking-type TYPE     Thinking mode (default: enabled)
  -u, --api-url URL            DeepSeek API URL
  --stream BOOL                Stream response true|false (default: false)
  -h, --help                   Show this help
EOF
}

die() {
    printf '[ERROR] %s\n' "$*" >&2
    exit 1
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

load_env() {
    [[ -f "$ENV_FILE" ]] || die "Missing root .env at $ENV_FILE"
    # shellcheck disable=SC1090
    set -a
    # shellcheck source=/dev/null
    source "$ENV_FILE"
    set +a
    [[ -n "${DEEPSEEK_API_KEY:-}" ]] || die "DEEPSEEK_API_KEY is not set in $ENV_FILE"
}

json_payload() {
    python3 - "$MODEL" "$PROMPT" "$THINKING_TYPE" "$REASONING_EFFORT" "$STREAM" <<'PY'
import json
import sys

model, prompt, thinking_type, reasoning_effort, stream = sys.argv[1:]
payload = {
    "model": model,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ],
    "thinking": {"type": thinking_type},
    "reasoning_effort": reasoning_effort,
    "stream": stream.lower() == "true",
}
print(json.dumps(payload))
PY
}

main() {
    require_cmd curl
    require_cmd python3

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -p|--prompt)
                [[ $# -ge 2 ]] || die "Missing value for $1"
                PROMPT="$2"
                shift 2
                ;;
            -m|--model)
                [[ $# -ge 2 ]] || die "Missing value for $1"
                MODEL="$2"
                shift 2
                ;;
            -r|--reasoning-effort)
                [[ $# -ge 2 ]] || die "Missing value for $1"
                REASONING_EFFORT="$2"
                shift 2
                ;;
            -t|--thinking-type)
                [[ $# -ge 2 ]] || die "Missing value for $1"
                THINKING_TYPE="$2"
                shift 2
                ;;
            -u|--api-url)
                [[ $# -ge 2 ]] || die "Missing value for $1"
                API_URL="$2"
                shift 2
                ;;
            --stream)
                [[ $# -ge 2 ]] || die "Missing value for $1"
                STREAM="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                PROMPT="$*"
                break
                ;;
        esac
    done

    load_env

    curl "$API_URL" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${DEEPSEEK_API_KEY}" \
        -d "$(json_payload)"
}

main "$@"
