# =============================================================================
# MLflow Remote Configuration
# =============================================================================
# Copy this to .env and customize for your SLURM cluster.
#
# The SLURM job scripts source this file, so all jobs share the same config.
#
# For a self-hosted MLflow server:
#   MLFLOW_TRACKING_URI=http://mlflow-server.cluster:5000
#
# For shared NFS filesystem (simplest):
#   MLFLOW_TRACKING_URI=file:///shared/projects/npc/mlruns
#
# For S3/MinIO artifact store:
#   MLFLOW_S3_ENDPOINT_URL=http://minio.cluster:9000
#   AWS_ACCESS_KEY_ID=your_access_key
#   AWS_SECRET_ACCESS_KEY=your_secret_key
#   MLFLOW_ARTIFACT_ROOT=s3://npc-models
# =============================================================================

# ── MLflow tracking server ────────────────────────────────────────────────────
# Where metrics, params, and tags are stored.
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-file://./mlruns}"

# ── Artifact store (model checkpoints, CSVs, etc.) ────────────────────────────
# Options:
#   1. Shared NFS (default if empty — uses tracking URI's artifact root)
#   2. S3-compatible object storage (MinIO, AWS S3, Ceph)
#
# For S3/MinIO, uncomment:
# export MLFLOW_S3_ENDPOINT_URL=http://minio.cluster:9000
# export AWS_ACCESS_KEY_ID=minioadmin
# export AWS_SECRET_ACCESS_KEY=minioadmin
# export MLFLOW_ARTIFACT_ROOT=s3://npc-models/artifacts

# ── Checkpoint storage (NFS path for direct file access) ──────────────────────
# SLURM jobs save checkpoints here so jobs can resume or evaluation can find them.
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}"

# ── SLURM defaults ────────────────────────────────────────────────────────────
# Partition to use (adjust for your cluster)
export SLURM_PARTITION="${SLURM_PARTITION:-gpu}"

# GPU type request
export SLURM_GPU_TYPE="${SLURM_GPU_TYPE:-a100:1}"  # e.g., a100:1, v100:2, rtx3090:1

# Time limit (HH:MM:SS)
export SLURM_TIME="${SLURM_TIME:-24:00:00}"

# Memory per node
export SLURM_MEM="${SLURM_MEM:-64G}"

# CPUs per task
export SLURM_CPUS="${SLURM_CPUS:-8}"
