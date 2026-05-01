#!/usr/bin/env bash
# =============================================================================
# gpu_test.sh — Interactive GPU verification on a compute node
# =============================================================================
# Requests an interactive GPU session and tests the environment.
# Useful for debugging before submitting long training jobs.
#
# Usage:
#   bash scripts/gpu_test.sh        # Interactive session, 1 GPU, 1 hour
#   bash scripts/gpu_test.sh slm    # Test SLM environment only
#   bash scripts/gpu_test.sh llm    # Test LLM environment only
#   bash scripts/gpu_test.sh both 2 # Test both, request 2 GPUs
# =============================================================================
set -euo pipefail

SYSTEM="${1:-both}"
NGPUS="${2:-1}"
TIME="${3:-01:00:00}"
PARTITION="${HPC_PARTITION:-gpu-l20}"
ACCOUNT="${HPC_ACCOUNT:-xrimlab}"

echo "Requesting interactive GPU session..."
echo "  Partition: ${PARTITION}"
echo "  GPUs:      ${NGPUS}"
echo "  Time:      ${TIME}"
echo ""

# Run salloc to get an interactive allocation
# After allocation, run the verification commands
srun --pty \
    --account="${ACCOUNT}" \
    --partition="${PARTITION}" \
    --gpus-per-node="${NGPUS}" \
    --ntasks-per-node=1 \
    --cpus-per-task=8 \
    --time="${TIME}" \
    bash -c '
set -euo pipefail

WORK_BASE="/scratch/${USER}"

echo "================================================================"
echo "  GPU Verification — $(hostname)"
echo "================================================================"

# Load CUDA
module purge 2>/dev/null || true
module load cuda/12.4.0 2>/dev/null || echo "  [WARN] cuda module not found"

echo ""
echo "── GPU Info ──"
nvidia-smi 2>/dev/null || echo "  No nvidia-smi"
echo ""

# Test SLM env
if [ "'"${SYSTEM}"'" = "slm" ] || [ "'"${SYSTEM}"'" = "both" ]; then
    echo "── Testing SLM environment ──"
    if [ -f "${WORK_BASE}/venvs/slm_env/bin/activate" ]; then
        source "${WORK_BASE}/venvs/slm_env/bin/activate"
        python3 -c "
import torch
print(f'  PyTorch:     {torch.__version__}')
print(f'  CUDA:        {torch.cuda.is_available()}')
print(f'  Device count:{torch.cuda.device_count()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}:       {p.name} ({p.total_memory/1e9:.1f} GB)')
    # Quick tensor test
    x = torch.randn(1000,1000).cuda()
    y = x @ x.T
    print(f'  Matmul test:  OK ({y.shape})')
print(f'  NumPy:        {__import__(\"numpy\").__version__}')
"
        deactivate 2>/dev/null || true
    else
        echo "  SLM venv not found at ${WORK_BASE}/venvs/slm_env/"
    fi
fi

# Test LLM env
if [ "'"${SYSTEM}"'" = "llm" ] || [ "'"${SYSTEM}"'" = "both" ]; then
    echo ""
    echo "── Testing LLM environment ──"
    if [ -f "${WORK_BASE}/venvs/llm_env/bin/activate" ]; then
        source "${WORK_BASE}/venvs/llm_env/bin/activate"
        python3 -c "
import torch, transformers, peft
print(f'  PyTorch:      {torch.__version__}')
print(f'  Transformers: {transformers.__version__}')
print(f'  PEFT:         {peft.__version__}')
print(f'  CUDA:         {torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}:        {p.name} ({p.total_memory/1e9:.1f} GB)')
    # Test bitsandbytes
    try:
        import bitsandbytes
        print(f'  bitsandbytes: OK')
    except:
        print(f'  bitsandbytes: not installed')
"
        deactivate 2>/dev/null || true
    else
        echo "  LLM venv not found at ${WORK_BASE}/venvs/llm_env/"
    fi
fi

echo ""
echo "================================================================"
echo "  Verification complete!"
echo "================================================================"
'
