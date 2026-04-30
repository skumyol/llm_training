#!/usr/bin/env bash
# =============================================================================
# setup_llm_env.sh — One-time LLM environment setup on HPC cluster
# =============================================================================
# Creates a venv on scratch storage with full LLM fine-tuning packages.
# Much heavier than SLM (transformers, peft, bitsandbytes, QLoRA, vLLM).
#
# Usage:
#   bash scripts/setup_llm_env.sh
# =============================================================================
set -euo pipefail

CUDA_MODULE="${CUDA_MODULE:-cuda/12.4.0}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.12}"
ENV_NAME="${ENV_NAME:-llm_env}"
WORK_BASE="${WORK_BASE:-/scratch/${USER}}"
ENV_DIR="${WORK_BASE}/venvs/${ENV_NAME}"
REPO_DIR="${WORK_BASE}/npc"

echo "================================================================"
echo "  LLM Environment Setup (Qwen3 fine-tuning + vLLM serving)"
echo "================================================================"
echo "  CUDA:    ${CUDA_MODULE}"
echo "  Python:  ${PYTHON_MODULE}"
echo "  Env:     ${ENV_DIR}"
echo "================================================================"

mkdir -p "${WORK_BASE}/data" "$(dirname "${ENV_DIR}")"

module purge
module load "${CUDA_MODULE}" 2>/dev/null || echo "  [WARN] Could not load ${CUDA_MODULE}"
module load "${PYTHON_MODULE}" 2>/dev/null || echo "  [WARN] Could not load ${PYTHON_MODULE}"

if [ -d "${ENV_DIR}" ]; then
    echo "  [SKIP] Venv exists: ${ENV_DIR}"
else
    python3 -m venv "${ENV_DIR}" --system-site-packages
fi

source "${ENV_DIR}/bin/activate"
pip install --quiet --upgrade pip wheel

# ── Install LLM packages (heavy) ──────────────────────────────────────────────
echo "  Installing LLM packages (~2GB download)..."
pip install --quiet \
    torch \
    transformers \
    peft \
    bitsandbytes \
    accelerate \
    trl \
    mlflow \
    pyyaml \
    tqdm \
    scikit-learn \
    tenacity

# Optional: vLLM for serving (heavy — install only if needed)
# pip install vllm

# ── Verify ────────────────────────────────────────────────────────────────────
echo ""
echo "  Verifying..."
python -c "
import torch, transformers, peft
print(f'  PyTorch: {torch.__version__}')
print(f'  Transformers: {transformers.__version__}')
print(f'  PEFT: {peft.__version__}')
print(f'  CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPUs: {torch.cuda.device_count()}')
    try:
        import bitsandbytes as bnb
        print(f'  bitsandbytes: {bnb.__version__}')
    except:
        print('  bitsandbytes: OK (no __version__)')
"

echo ""
echo "================================================================"
echo "  ✅ LLM environment ready!"
echo ""
echo "  Activate:  source ${ENV_DIR}/bin/activate"
echo "  Train:     bash ${REPO_DIR}/scripts/slurm_train.sh llm latent"
echo "================================================================"
