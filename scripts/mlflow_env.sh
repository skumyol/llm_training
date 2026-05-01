# =============================================================================
# mlflow_env.sh — HPC Cluster Configuration (HKUST HPC4 — Spack)
# =============================================================================
# Source this before submitting jobs, or set in your ~/.bashrc.
#
# HKUST HPC4 docs:
#   https://hkust-hpc-docs.readthedocs.io/latest/software/python/index.html
#   https://hkust-hpc-docs.readthedocs.io/latest/software/software-support-overview.html

# ── SLURM account (REQUIRED) ──────────────────────────────────────────────────
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-xrimlab}"

# ── SLURM partition ───────────────────────────────────────────────────────────
# Available GPU partitions on HKUST HPC4 (check with: sinfo)
#   gpu-a30       — A30 GPUs (24GB), 15 nodes
#   gpu-l20       — L20 GPUs (48GB), 6 nodes
#   gpu-rtx4090d  — RTX 4090D GPUs (24GB), 2 nodes
export SLURM_PARTITION="${SLURM_PARTITION:-gpu-l20}"

# ── GPU spec (HKUST uses --gpus-per-node, NOT --gres=gpu) ─────────────────────
export SLURM_GPUS="${SLURM_GPUS:-1}"

# ── CPU tasks (HKUST: --ntasks-per-node=1 --cpus-per-task=N) ──────────────────
export SLURM_CPUS="${SLURM_CPUS:-8}"

# ── Time limit ────────────────────────────────────────────────────────────────
export SLURM_TIME="${SLURM_TIME:-24:00:00}"

# ── Working directories ───────────────────────────────────────────────────────
export WORK_BASE="${WORK_BASE:-/scratch/${USER}}"
export REPO_DIR="${REPO_DIR:-${WORK_BASE}/npc}"
export DATA_DIR="${DATA_DIR:-${WORK_BASE}/data}"
export LOG_DIR="${LOG_DIR:-${WORK_BASE}/logs}"
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-${WORK_BASE}/checkpoints}"

# ── MLflow ────────────────────────────────────────────────────────────────────
# File-based tracking on shared scratch (no server needed).
# For multi-user: set up a tracking server on a login node port.
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file://${WORK_BASE}/mlruns}"

# ── Python venvs (on scratch) ─────────────────────────────────────────────────
export LLM_VENV="${LLM_VENV:-${WORK_BASE}/venvs/llm_env}"
export SLM_VENV="${SLM_VENV:-${WORK_BASE}/venvs/slm_env}"

# ── CUDA module (from Spack) ──────────────────────────────────────────────────
export CUDA_MODULE="${CUDA_MODULE:-cuda/12.4.0}"

# ── Spack setup ───────────────────────────────────────────────────────────────
export SPACK_SETUP="${SPACK_SETUP:-/opt/shared/spack/share/spack/setup-env.sh}"
