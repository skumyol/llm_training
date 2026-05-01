#!/usr/bin/env bash
# =============================================================================
# setup_llm_env.sh — LLM environment setup using Spack + pip on HKUST HPC
# =============================================================================
# Creates a venv on scratch storage with full LLM fine-tuning packages.
# Heavy: transformers, peft, bitsandbytes, QLoRA, vLLM.
#
# Usage:
#   # Full setup (Spack + pip):
#   bash scripts/setup_llm_env.sh
#
#   # Pip-only (if Spack env already exists):
#   bash scripts/setup_llm_env.sh --pip-only
# =============================================================================
set -euo pipefail

CUDA_MODULE="${CUDA_MODULE:-cuda/12.4.0}"
PYTHON_CMD="${PYTHON_CMD:-python3}"
WORK_BASE="${WORK_BASE:-/scratch/${USER}}"
REPO_DIR="${REPO_DIR:-${WORK_BASE}/npc}"
ENV_DIR="${WORK_BASE}/venvs/llm_env"

PIP_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --pip-only) PIP_ONLY=true ;;
    esac
done

echo "================================================================"
echo "  LLM Environment Setup (Qwen3 fine-tuning + QLoRA)"
echo "================================================================"
echo "  CUDA mod: ${CUDA_MODULE}"
echo "  Python:   ${PYTHON_CMD}"
echo "  Env:      ${ENV_DIR}"
echo "  Repo:     ${REPO_DIR}"
echo "  Pip only: ${PIP_ONLY}"
echo "================================================================"

mkdir -p "${WORK_BASE}/data" "$(dirname "${ENV_DIR}")"

# ── Load CUDA ─────────────────────────────────────────────────────────────────
module load "${CUDA_MODULE}" 2>/dev/null || {
    echo "  [WARN] Could not load ${CUDA_MODULE}"
    echo "  If Spack is set up, try: source /opt/shared/spack/share/spack/setup-env.sh"
}

# ── Create venv ───────────────────────────────────────────────────────────────
if [ -d "${ENV_DIR}" ]; then
    echo "  [SKIP] Venv exists: ${ENV_DIR}"
else
    echo "  Creating venv..."
    ${PYTHON_CMD} -m venv "${ENV_DIR}" --system-site-packages
fi

source "${ENV_DIR}/bin/activate"
pip install --quiet --upgrade pip wheel

# ── Install LLM packages from PyPI ────────────────────────────────────────────
echo "  Installing LLM packages (~2 GB download)..."
echo "  This may take 10-20 minutes depending on network speed."

# PyTorch with CUDA 12.4
pip install --quiet \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Core ML libraries
pip install --quiet \
    transformers>=4.47.0 \
    peft>=0.14.0 \
    bitsandbytes>=0.43.0 \
    accelerate>=0.27.0 \
    datasets>=2.20.0 \
    tokenizers>=0.19.0 \
    trl>=0.8.6

# Evaluation
pip install --quiet \
    evaluate>=0.4.1 \
    rouge-score>=0.1.2 \
    bert-score>=0.3.13 \
    scikit-learn>=1.4.0 \
    scipy>=1.12.0

# Experiment tracking
pip install --quiet \
    mlflow>=2.13.0

# Data / config
pip install --quiet \
    pyyaml>=6.0.1 \
    jsonlines>=4.0.0 \
    tqdm>=4.66.4 \
    pandas>=2.2.0 \
    tabulate>=0.9.0

# LLM API (teacher model)
pip install --quiet \
    openai>=1.30.0 \
    tiktoken>=0.7.0 \
    tenacity>=8.3.0

# Utilities
pip install --quiet \
    click>=8.1.7 \
    python-dotenv>=1.0.1 \
    rich>=13.7.0 \
    psutil>=5.9.8

# Optional: vLLM for serving (very heavy — uncomment if needed)
# pip install vllm

# ── Verify ────────────────────────────────────────────────────────────────────
echo ""
echo "  Verifying installation..."
python -c "
import torch, transformers, peft
print(f'  PyTorch:      {torch.__version__}')
print(f'  Transformers: {transformers.__version__}')
print(f'  PEFT:         {peft.__version__}')
print(f'  CUDA:         {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPUs:         {torch.cuda.device_count()}')
    print(f'  GPU 0:        {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:         {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    try:
        import bitsandbytes as bnb
        print(f'  bitsandbytes: OK')
    except Exception as e:
        print(f'  bitsandbytes: ERROR — {e}')
else:
    print('  [WARN] CUDA not available — training will use CPU (slow)')
"

echo ""
echo "================================================================"
echo "  ✅ LLM environment ready!"
echo ""
echo "  Activate:  source ${ENV_DIR}/bin/activate"
echo "  Train:     sbatch ${REPO_DIR}/scripts/slurm_train.sh llm latent"
echo ""
echo "  Important: CUDA is only visible on GPU nodes (login nodes don't have GPUs)."
echo "  Submit via sbatch to see GPUs."
echo "================================================================"
