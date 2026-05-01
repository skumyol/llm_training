#!/usr/bin/env bash
# =============================================================================
# env_setup_spack.sh — One-shot HPC environment bootstrap using Spack
# =============================================================================
# Run once on hpc4.ust.hk after SSH:
#   ssh skumyol@hpc4.ust.hk
#   cd ~/llm_training
#   bash scripts/env_setup_spack.sh
#
# What this does:
#   1. Creates /scratch/$USER directory structure
#   2. Symlinks repo to /scratch/$USER/npc (for fast I/O)
#   3. Creates Spack environments for LLM + SLM training
#   4. Installs everything (Spack packages + pip deps)
#   5. Verifies GPUs are visible
#
# Options:
#   --skip-spack      Skip Spack install (use existing spack envs)
#   --skip-pip        Skip pip install (reuse existing venvs)
#   --dry-run         Print what would be done
#   --skip-slm        Skip SLM environment
#   --skip-llm        Skip LLM environment
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN=false
SKIP_SPACK=false
SKIP_PIP=false
SKIP_SLM=false
SKIP_LLM=false

for arg in "$@"; do
    case "$arg" in
        --dry-run)      DRY_RUN=true ;;
        --skip-spack)   SKIP_SPACK=true ;;
        --skip-pip)     SKIP_PIP=true ;;
        --skip-slm)     SKIP_SLM=true ;;
        --skip-llm)     SKIP_LLM=true ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

# ── Config ────────────────────────────────────────────────────────────────────
ACCOUNT="${SLURM_ACCOUNT:-xrimlab}"
PARTITION="${SLURM_PARTITION:-gpu-l20}"
WORK_BASE="/scratch/${USER}"
REPO_SRC="${ROOT}"
REPO_LINK="${WORK_BASE}/npc"
CUDA_MOD="cuda/12.4.0"
GCC_MOD="gcc/13.2.0"

# Spack environment names and paths
LLM_SPACK_ENV="llm_env_spack"
SLM_SPACK_ENV="slm_env_spack"
LLM_SPACK_YAML="${ROOT}/spack_envs/llm_env/spack.yaml"
SLM_SPACK_YAML="${ROOT}/spack_envs/slm_env/spack.yaml"

VENV_LLM="${WORK_BASE}/venvs/llm_env"
VENV_SLM="${WORK_BASE}/venvs/slm_env"

run() {
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY] $*"
    else
        echo "  RUN: $*"
        eval "$@"
    fi
}

echo "================================================================"
echo "  HPC Environment Bootstrap (Spack + Slurm)"
echo "================================================================"
echo "  User:       ${USER}"
echo "  Account:    ${ACCOUNT}"
echo "  Partition:  ${PARTITION}"
echo "  Work base:  ${WORK_BASE}"
echo "  Repo src:   ${REPO_SRC}"
echo "  Dry run:    ${DRY_RUN}"
echo "================================================================"

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Scratch directory structure
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "── 1. Scratch directories ──────────────────────────────────────"
for d in "${WORK_BASE}/data" "${WORK_BASE}/logs" "${WORK_BASE}/checkpoints" \
         "${WORK_BASE}/mlruns" "${WORK_BASE}/venvs" "${WORK_BASE}/models"; do
    if [ ! -d "$d" ]; then
        run mkdir -p "$d"
    else
        echo "  [OK] $d"
    fi
done

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Symlink repo to scratch
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "── 2. Repo symlink ─────────────────────────────────────────────"
if [ ! -L "${REPO_LINK}" ] && [ ! -d "${REPO_LINK}" ]; then
    run ln -s "${REPO_SRC}" "${REPO_LINK}"
    echo "  ${REPO_LINK} → ${REPO_SRC}"
else
    echo "  [OK] ${REPO_LINK} already exists"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Load Spack (source setup-env.sh)
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "── 3. Loading Spack ────────────────────────────────────────────"
SPACK_SETUP="/opt/shared/spack/share/spack/setup-env.sh"
if [ -f "${SPACK_SETUP}" ]; then
    source "${SPACK_SETUP}"
    echo "  Spack: $(spack --version)"
else
    echo "  [ERROR] Spack not found at ${SPACK_SETUP}"
    echo "  Falling back to module-based python. Continuing..."
    SPACK_MODE=false
fi

SPACK_MODE="${SPACK_MODE:-true}"

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Create & install Spack environments
# ═══════════════════════════════════════════════════════════════════════════════
if [ "$SKIP_SPACK" = true ]; then
    echo ""
    echo "── 4. Skipping Spack install (--skip-spack) ────────────────────"
else
    echo ""
    echo "── 4. Spack environments ───────────────────────────────────────"

    # --- LLM environment ---
    if [ "$SKIP_LLM" = false ]; then
        echo ""
        echo "  [LLM Spack env: ${LLM_SPACK_ENV}]"
        if spack env list 2>/dev/null | grep -q "${LLM_SPACK_ENV}"; then
            echo "    [OK] Spack env already exists"
        else
            run spack env create "${LLM_SPACK_ENV}" "${LLM_SPACK_YAML}"
        fi

        echo "    Concretizing..."
        run spack -e "${LLM_SPACK_ENV}" concretize -f 2>&1 | tail -3 || true

        echo "    Installing (this may take 30-90 min on first run)..."
        run spack -e "${LLM_SPACK_ENV}" install --fail-fast 2>&1 | tail -20 || {
            echo "    [WARN] Some Spack packages failed. Will try to continue with available ones."
        }
    fi

    # --- SLM environment ---
    if [ "$SKIP_SLM" = false ]; then
        echo ""
        echo "  [SLM Spack env: ${SLM_SPACK_ENV}]"
        if spack env list 2>/dev/null | grep -q "${SLM_SPACK_ENV}"; then
            echo "    [OK] Spack env already exists"
        else
            run spack env create "${SLM_SPACK_ENV}" "${SLM_SPACK_YAML}"
        fi

        echo "    Concretizing..."
        run spack -e "${SLM_SPACK_ENV}" concretize -f 2>&1 | tail -3 || true

        echo "    Installing..."
        run spack -e "${SLM_SPACK_ENV}" install --fail-fast 2>&1 | tail -20 || {
            echo "    [WARN] Some Spack packages failed."
        }
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Create Python venvs and pip-install ML packages
# ═══════════════════════════════════════════════════════════════════════════════
if [ "$SKIP_PIP" = true ]; then
    echo ""
    echo "── 5. Skipping pip install (--skip-pip) ────────────────────────"
else
    echo ""
    echo "── 5. Python venvs + pip ───────────────────────────────────────"

    # Find a usable Python. Priority:
    #   1. Spack env python
    #   2. System python3
    #   3. miniconda3 module
    find_python() {
        # Try spack env python
        if [ "${SPACK_MODE:-true}" = "true" ]; then
            local spack_py
            spack_py=$(spack -e "${1}" location -i 2>/dev/null || true)
            if [ -n "${spack_py}" ] && [ -f "${spack_py}/bin/python3" ]; then
                echo "${spack_py}/bin/python3"
                return
            fi
        fi
        # Try system python3
        if command -v python3 &>/dev/null; then
            echo "$(command -v python3)"
            return
        fi
        # Try miniconda3
        module load miniconda3/24.3.0 2>/dev/null || true
        if command -v python3 &>/dev/null; then
            echo "$(command -v python3)"
            return
        fi
        echo ""
    }

    # --- LLM venv ---
    if [ "$SKIP_LLM" = false ]; then
        echo ""
        echo "  [LLM venv: ${VENV_LLM}]"

        PYTHON_BIN=$(find_python "${LLM_SPACK_ENV}")
        if [ -z "${PYTHON_BIN}" ]; then
            echo "    [ERROR] No Python found. Skipping LLM venv."
        else
            echo "    Using: ${PYTHON_BIN}"
            if [ ! -f "${VENV_LLM}/bin/activate" ]; then
                run "${PYTHON_BIN}" -m venv "${VENV_LLM}" --system-site-packages
            fi
            source "${VENV_LLM}/bin/activate"
            run pip install --quiet --upgrade pip wheel

            echo "    Installing LLM packages (~2 GB download)..."
            run pip install --quiet \
                torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 \
                transformers peft bitsandbytes accelerate trl \
                datasets tokenizers evaluate \
                mlflow pyyaml jsonlines tqdm pandas tabulate \
                scikit-learn scipy rouge-score bert-score \
                openai tiktoken tenacity \
                click python-dotenv rich psutil

            echo "    Verifying..."
            python -c "
import torch, transformers, peft
print(f'    PyTorch:     {torch.__version__}')
print(f'    CUDA:        {torch.cuda.is_available()}')
print(f'    Transformers:{transformers.__version__}')
print(f'    PEFT:        {peft.__version__}')
if torch.cuda.is_available():
    print(f'    GPU count:   {torch.cuda.device_count()}')
    print(f'    GPU 0:       {torch.cuda.get_device_name(0)}')
" || echo "    [WARN] Verification had issues — check GPU visibility"

            deactivate 2>/dev/null || true
        fi
    fi

    # --- SLM venv ---
    if [ "$SKIP_SLM" = false ]; then
        echo ""
        echo "  [SLM venv: ${VENV_SLM}]"

        PYTHON_BIN=$(find_python "${SLM_SPACK_ENV}")
        if [ -z "${PYTHON_BIN}" ]; then
            echo "    [ERROR] No Python found. Skipping SLM venv."
        else
            echo "    Using: ${PYTHON_BIN}"
            if [ ! -f "${VENV_SLM}/bin/activate" ]; then
                run "${PYTHON_BIN}" -m venv "${VENV_SLM}" --system-site-packages
            fi
            source "${VENV_SLM}/bin/activate"
            run pip install --quiet --upgrade pip wheel

            echo "    Installing SLM packages..."
            run pip install --quiet \
                torch --index-url https://download.pytorch.org/whl/cu124 \
                "numpy<2" tiktoken optuna mlflow pyyaml tqdm pandas

            echo "    Verifying..."
            python -c "
import torch
print(f'    PyTorch: {torch.__version__}')
print(f'    CUDA:    {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'    GPU:     {torch.cuda.get_device_name(0)}')
" || echo "    [WARN] Verification had issues"
            deactivate 2>/dev/null || true
        fi
    fi
fi

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Summary
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "================================================================"
echo "  Setup Complete"
echo "================================================================"
echo ""
echo "  Directory structure:"
echo "    Repo (symlink):  ${REPO_LINK}"
echo "    Data:            ${WORK_BASE}/data"
echo "    Logs:            ${WORK_BASE}/logs"
echo "    Checkpoints:     ${WORK_BASE}/checkpoints"
echo "    MLflow:          ${WORK_BASE}/mlruns"
echo "    LLM venv:        ${VENV_LLM}"
echo "    SLM venv:        ${VENV_SLM}"
echo ""
echo "  Quick start:"
echo "    # Interactive test"
echo "    ssh skumyol@hpc4.ust.hk"
echo "    cd /scratch/\$USER/npc"
echo "    source /scratch/\$USER/venvs/slm_env/bin/activate"
echo "    cd slm_training && bash smoke_test.sh"
echo ""
echo "    # Submit training jobs"
echo "    bash scripts/submit_missing.sh --submit"
echo ""
echo "    # Single job"
echo "    sbatch scripts/slurm_train.sh slm small_lm --arch gpt --seed 42"
echo ""
echo "  Monitor:"
echo "    squeue -u \$USER"
echo "    tail -f /scratch/\$USER/logs/*.out"
echo "================================================================"
