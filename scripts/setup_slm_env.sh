#!/usr/bin/env bash
# =============================================================================
# setup_slm_env.sh — One-time SLM environment setup on HPC cluster
# =============================================================================
# Creates a venv on scratch storage with only the packages needed for
# Small LM training (pure PyTorch — no transformers/peft/bitsandbytes).
#
# Usage:
#   bash scripts/setup_slm_env.sh
#
# Customize CUDA_MODULE and WORK_BASE for your cluster.
# =============================================================================
set -euo pipefail

# ── Cluster-specific settings (EDIT THESE) ────────────────────────────────────
CUDA_MODULE="${CUDA_MODULE:-cuda/12.4.0}"       # Module name on your cluster
PYTHON_MODULE="${PYTHON_MODULE:-python/3.12}"     # Python module
ENV_NAME="${ENV_NAME:-slm_env}"
WORK_BASE="${WORK_BASE:-/scratch/${USER}}"        # Scratch/work directory
ENV_DIR="${WORK_BASE}/venvs/${ENV_NAME}"
REPO_DIR="${WORK_BASE}/npc"

echo "================================================================"
echo "  SLM Environment Setup (Small LMs from scratch)"
echo "================================================================"
echo "  CUDA:    ${CUDA_MODULE}"
echo "  Python:  ${PYTHON_MODULE}"
echo "  Env:     ${ENV_DIR}"
echo "  Repo:    ${REPO_DIR}"
echo "================================================================"

# ── Create directories ────────────────────────────────────────────────────────
mkdir -p "${WORK_BASE}/data" "${WORK_BASE}/models" "$(dirname "${ENV_DIR}")"

# ── Load modules ──────────────────────────────────────────────────────────────
module purge
module load "${CUDA_MODULE}" 2>/dev/null || echo "  [WARN] Could not load ${CUDA_MODULE}"
module load "${PYTHON_MODULE}" 2>/dev/null || echo "  [WARN] Could not load ${PYTHON_MODULE}"

# ── Create venv ───────────────────────────────────────────────────────────────
if [ -d "${ENV_DIR}" ]; then
    echo "  [SKIP] Venv exists: ${ENV_DIR}"
else
    echo "  Creating venv..."
    python3 -m venv "${ENV_DIR}" --system-site-packages
fi

source "${ENV_DIR}/bin/activate"
pip install --quiet --upgrade pip wheel

# ── Install SLM packages (lightweight — pure PyTorch) ─────────────────────────
echo "  Installing SLM packages..."
pip install --quiet \
    torch \
    "numpy<2" \
    tiktoken \
    optuna \
    mlflow \
    pyyaml \
    tqdm

# Optional: for dataset download + evaluation
pip install --quiet \
    pandas \
    transformers \
    datasets \
    sentence-transformers \
    sacrebleu 2>/dev/null || true

# ── Verify ────────────────────────────────────────────────────────────────────
echo ""
echo "  Verifying..."
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPUs: {torch.cuda.device_count()}')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
"
python -c "import optuna; import mlflow; import tiktoken; print('  optuna, mlflow, tiktoken: OK')"
python -c "import yaml; import tqdm; print('  yaml, tqdm: OK')"

# ── Clone repo (if not already there) ─────────────────────────────────────────
if [ ! -d "${REPO_DIR}" ]; then
    echo ""
    echo "  Repo not found at ${REPO_DIR}."
    echo "  Clone it manually:"
    echo "    git clone <repo_url> ${REPO_DIR}"
fi

echo ""
echo "================================================================"
echo "  ✅ SLM environment ready!"
echo ""
echo "  Activate:  source ${ENV_DIR}/bin/activate"
echo "  Train:     bash ${REPO_DIR}/scripts/slurm_train.sh slm small_lm --arch gpt"
echo "================================================================"
