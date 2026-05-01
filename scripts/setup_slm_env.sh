#!/usr/bin/env bash
# =============================================================================
# setup_slm_env.sh — SLM environment setup using Spack + pip on HKUST HPC
# =============================================================================
# Creates a venv on scratch storage with only the packages needed for
# Small LM training (pure PyTorch — no transformers/peft/bitsandbytes).
#
# Usage:
#   # Full setup:
#   bash scripts/setup_slm_env.sh
#
#   # Pip-only (if Spack env already exists):
#   bash scripts/setup_slm_env.sh --pip-only
# =============================================================================
set -euo pipefail

CUDA_MODULE="${CUDA_MODULE:-cuda/12.4.0}"
PYTHON_CMD="${PYTHON_CMD:-python3}"
WORK_BASE="${WORK_BASE:-/scratch/${USER}}"
REPO_DIR="${REPO_DIR:-${WORK_BASE}/npc}"
ENV_DIR="${WORK_BASE}/venvs/slm_env"

PIP_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --pip-only) PIP_ONLY=true ;;
    esac
done

echo "================================================================"
echo "  SLM Environment Setup (Small LMs from scratch)"
echo "================================================================"
echo "  CUDA mod: ${CUDA_MODULE}"
echo "  Python:   ${PYTHON_CMD}"
echo "  Env:      ${ENV_DIR}"
echo "  Repo:     ${REPO_DIR}"
echo "  Pip only: ${PIP_ONLY}"
echo "================================================================"

# ── Create directories ────────────────────────────────────────────────────────
mkdir -p "${WORK_BASE}/data" "${WORK_BASE}/models" "$(dirname "${ENV_DIR}")"

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

# ── Install SLM packages from PyPI (lightweight — pure PyTorch) ───────────────
echo "  Installing SLM packages..."

# PyTorch with CUDA 12.4
pip install --quiet \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Core SLM deps
pip install --quiet \
    "numpy<2" \
    tiktoken \
    optuna \
    mlflow \
    pyyaml \
    tqdm \
    pandas

# Optional: for dataset download + evaluation
pip install --quiet \
    transformers \
    datasets \
    sentence-transformers \
    sacrebleu 2>/dev/null || true

# ── Verify ────────────────────────────────────────────────────────────────────
echo ""
echo "  Verifying installation..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPUs: {torch.cuda.device_count()}')
    print(f'  GPU 0: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:  {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('  [INFO] CUDA not visible on login node — submit via sbatch for GPU access')
"
python -c "import optuna; import mlflow; import tiktoken; print('  optuna, mlflow, tiktoken: OK')"
python -c "import yaml; import tqdm; print('  yaml, tqdm: OK')"

echo ""
echo "================================================================"
echo "  ✅ SLM environment ready!"
echo ""
echo "  Activate:  source ${ENV_DIR}/bin/activate"
echo "  Train:     sbatch ${REPO_DIR}/scripts/slurm_train.sh slm small_lm --arch gpt"
echo ""
echo "  Important: CUDA is only visible on GPU compute nodes."
echo "  Login nodes don't have GPUs — submit via sbatch."
echo "================================================================"
