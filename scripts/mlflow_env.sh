# =============================================================================
# mlflow_env.sh — HPC Cluster Configuration (HKUST HPC4)
# =============================================================================
# Source this before submitting jobs, or set in your ~/.bashrc.
#
# HKUST HPC4 docs: https://itso.hkust.edu.hk/services/academic-teaching-support/high-performance-computing/hpc4/slurm

# ── SLURM account (REQUIRED) ──────────────────────────────────────────────────
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-xrimlab}"

# ── SLURM partition ───────────────────────────────────────────────────────────
# HKUST HPC4 partitions:
#   cpu-share     — CPU nodes, shared
#   gpu-l20       — L20 GPUs (48GB), max 1 GPU per job typically
#   gpu-a100      — A100 GPUs, max 1 GPU per job
#   gpu-2080ti    — RTX 2080 Ti GPUs
export SLURM_PARTITION="${SLURM_PARTITION:-gpu-l20}"

# ── GPU spec (HKUST uses --gpus-per-node, NOT --gres=gpu) ─────────────────────
export SLURM_GPUS="${SLURM_GPUS:-1}"

# ── CPU tasks (HKUST: --ntasks-per-node=1 --cpus-per-task=N) ──────────────────
export SLURM_CPUS="${SLURM_CPUS:-8}"

# ── Time limit ────────────────────────────────────────────────────────────────
export SLURM_TIME="${SLURM_TIME:-24:00:00}"

# ── Working directories ───────────────────────────────────────────────────────
export WORK_BASE="${WORK_BASE:-/scratch/${USER}}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${WORK_BASE}/checkpoints}"

# ── MLflow ────────────────────────────────────────────────────────────────────
# Set up a shared MLflow server on a login node, or use a shared filesystem.
# For HKUST: use a shared NFS path or start a tracking server on a port.
# export MLFLOW_TRACKING_URI=http://mlflow-server:5000
# Or use file-based on shared scratch:
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file://${WORK_BASE}/mlruns}"

# ── Python venv ───────────────────────────────────────────────────────────────
export LLM_VENV="${WORK_BASE}/venvs/llm_env"
export SLM_VENV="${WORK_BASE}/venvs/slm_env"
