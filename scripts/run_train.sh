#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

PYTHON="${PROJECT_ROOT}/.venv/bin/python"
if [[ ! -f "$PYTHON" ]]; then
  PYTHON="python3"
fi

STAGE="${1:-latent}"
DEBUG="${2:-}"

EXTRA_ARGS=""
if [[ "$DEBUG" == "--debug" ]]; then
  EXTRA_ARGS="--debug"
fi

case "$STAGE" in
  latent)
    CONFIG="configs/train_latent.yaml"
    ;;
  response)
    CONFIG="configs/train_response.yaml"
    ;;
  joint)
    CONFIG="configs/train_joint.yaml"
    ;;
  all)
    echo "========================================="
    echo " LLM Training — Full Training Pipeline"
    echo "========================================="
    echo "Stage 1/3: Latent State Predictor"
    "$PYTHON" run_train.py --stage latent   --config configs/train_latent.yaml   $EXTRA_ARGS
    echo "Stage 2/3: Response Generator SFT"
    "$PYTHON" run_train.py --stage response --config configs/train_response.yaml $EXTRA_ARGS
    echo "Stage 3/3: Joint Fine-Tuning"
    "$PYTHON" run_train.py --stage joint    --config configs/train_joint.yaml    $EXTRA_ARGS
    echo "[Done] Full training pipeline complete."
    exit 0
    ;;
  *)
    echo "Unknown stage: $STAGE. Choose from: latent response joint all"
    exit 1
    ;;
esac

echo "========================================="
echo " LLM Training — Training: $STAGE"
echo "========================================="
echo " Config: $CONFIG"
echo "-----------------------------------------"

"$PYTHON" run_train.py --stage "$STAGE" --config "$CONFIG" $EXTRA_ARGS

echo "[Done] Training stage '$STAGE' complete."
