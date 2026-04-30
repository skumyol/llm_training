# SLURM Cluster Migration Guide

How to migrate training, evaluation, and artifact storage to a SLURM-based HPC cluster with remote MLflow tracking.

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        SLURM CLUSTER                               │
│                                                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │ GPU Node │  │ GPU Node │  │ GPU Node │  │ GPU Node │         │
│  │  (job 1) │  │  (job 2) │  │  (job 3) │  │  (job 4) │         │
│  │          │  │          │  │          │  │          │         │
│  │ Train    │  │ Train    │  │ Train    │  │ Eval     │         │
│  │ Log→MLflow│ │ Log→MLflow│ │ Log→MLflow│ │ Log→MLflow│        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘         │
│       │              │              │              │               │
│       ▼              ▼              ▼              ▼               │
│  ┌──────────────────────────────────────────────────────┐        │
│  │              SHARED NFS (/shared/projects/npc)        │        │
│  │  ├── checkpoints/     ← model weights                │        │
│  │  ├── mlruns/          ← MLflow metrics (or remote)    │        │
│  │  ├── data/            ← datasets + splits             │        │
│  │  └── logs/            ← SLURM job logs                │        │
│  └──────────────────────────────────────────────────────┘        │
│                                                                    │
│  ┌──────────────────────────────────────────────────────┐        │
│  │           MLflow Tracking Server (port 5000)          │        │
│  │  Stores: metrics, params, tags per run               │        │
│  │  Optional S3/MinIO backend for artifacts              │        │
│  └──────────────────────────────────────────────────────┘        │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│                     YOUR WORKSTATION                               │
│                                                                    │
│  $ python scripts/query_results.py --experiment small_lm          │
│    → Queries remote MLflow server                                 │
│    → Shows comparison tables across all SLURM runs                │
│                                                                    │
│  $ python scripts/query_results.py --artifacts                    │
│    → Lists checkpoint URIs for download                           │
└────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: Set up MLflow tracking server

### Option A: Shared NFS (simplest)

All nodes mount the same NFS. MLflow uses a file-based backend on the shared drive.

```bash
# On any cluster node or login node:
mkdir -p /shared/projects/npc/mlruns

# Set in your .env or sbatch script:
export MLFLOW_TRACKING_URI=file:///shared/projects/npc/mlruns
```

**Pro:** Zero setup. No extra service.  
**Con:** File locking issues with many concurrent writers (MLflow 2.20+ deprecated filesystem backend).

### Option B: Remote MLflow server + SQLite/Postgres (recommended)

Run MLflow as a service on a dedicated node:

```bash
# On the tracking server node:
pip install mlflow
mlflow server \
    --backend-store-uri sqlite:///shared/projects/npc/mlflow.db \
    --default-artifact-root /shared/projects/npc/mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5000
```

Then all jobs use:
```bash
export MLFLOW_TRACKING_URI=http://mlflow-server:5000
```

### Option C: S3/MinIO artifact store (production)

For large checkpoints (multi-GB), store artifacts in object storage:

```bash
# Start MinIO on cluster (or use existing)
docker run -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=admin -e MINIO_ROOT_PASSWORD=password \
  minio/minio server /data --console-address :9001

# MLflow config
export MLFLOW_TRACKING_URI=http://mlflow-server:5000
export MLFLOW_S3_ENDPOINT_URL=http://minio:9000
export AWS_ACCESS_KEY_ID=admin
export AWS_SECRET_ACCESS_KEY=password
export MLFLOW_ARTIFACT_ROOT=s3://npc-models
```

---

## Step 2: One-time cluster setup

### SLM (Small LMs from scratch — lightweight)

```bash
# Clone repo to scratch
cd /scratch/$USER
git clone <repo_url> npc
cd npc

# Set up venv with SLM packages (PyTorch only, no transformers/peft)
bash scripts/setup_slm_env.sh
```

**Installs:** torch, numpy, tiktoken, optuna, mlflow, pyyaml, tqdm  
**Does NOT install:** transformers, peft, bitsandbytes (not needed for custom architectures)

### LLM (Qwen3 fine-tuning — heavy)

```bash
bash scripts/setup_llm_env.sh
```

**Installs:** torch, transformers, peft, bitsandbytes, accelerate, trl, mlflow  
**Download size:** ~2 GB (QLoRA 4-bit quantization included)

### Package comparison

| Package | SLM | LLM | Reason |
|---------|-----|-----|--------|
| `torch` | ✅ | ✅ | Core framework |
| `numpy` | ✅ | ✅ | Array operations |
| `tiktoken` | ✅ | ❌ | GPT-2 tokenizer for small LMs |
| `optuna` | ✅ | ❌ | Hyperparameter optimization |
| `mlflow` | ✅ | ✅ | Experiment tracking |
| `transformers` | ❌ | ✅ | Qwen3 model loading |
| `peft` | ❌ | ✅ | LoRA adapters |
| `bitsandbytes` | ❌ | ✅ | 4-bit QLoRA quantization |
| `accelerate` | ❌ | ✅ | Multi-GPU training |
| `trl` | ❌ | ✅ | SFTTrainer |
| `sentence-transformers` | ⚠️ opt | ❌ | Memory store (dialogue only) |

---

## Step 3: Configure MLflow + cluster settings

Edit `scripts/mlflow_env.sh` with your cluster settings:

```bash
# MLflow — all jobs source this
export MLFLOW_TRACKING_URI=http://mlflow-server:5000
export CHECKPOINT_DIR=/shared/projects/npc/checkpoints
export SLURM_PARTITION=gpu
export SLURM_GPU_TYPE=a100:1
export SLURM_TIME=24:00:00
export SLURM_MEM=64G
```

---

## Step 4: Submit training jobs

The SLURM scripts auto-detect which venv to use based on `--system`:

| System flag | Venv | Packages |
|-------------|------|----------|
| `llm` | `llm_env` | transformers, peft, bitsandbytes |
| `slm` | `slm_env` | torch, optuna, tiktoken |

### Single job

```bash
# LLM: 3-stage fine-tuning
sbatch scripts/slurm_train.sh llm latent
sbatch scripts/slurm_train.sh llm response
sbatch scripts/slurm_train.sh llm joint

# SLM: encoders
sbatch scripts/slurm_train.sh slm personality
sbatch scripts/slurm_train.sh slm affect

# SLM: small LM from scratch (custom architecture, pure PyTorch)
sbatch scripts/slurm_train.sh slm small_lm --arch gpt --seed 42

# SLM: conditional dialogue model
sbatch scripts/slurm_train.sh slm dialogue

# Custom GPU/time:
sbatch --gres=gpu:a100:2 --time=72:00:00 scripts/slurm_train.sh slm small_lm --arch mamba_like
```

### Job arrays (multi-seed, multi-arch)

```bash
# Train all 6 SLM architectures with 3 seeds each = 18 jobs
sbatch --array=0-17 scripts/slurm_array.sh slm small_lm \
    --archs gru,awdlstm,gpt,prefix_gpt,moe,mamba_like \
    --seeds 42,43,44

# Train GPT with 5 seeds = 5 jobs
sbatch --array=0-4 scripts/slurm_array.sh slm small_lm \
    --archs gpt --seeds 42,43,44,45,46

# LLM full pipeline (latent → response → joint) = 3 sequential jobs
sbatch --array=0-2 scripts/slurm_array.sh llm full_pipeline
```

### How array task mapping works

| TASK_ID | arch[0] seeds[0] | arch[0] seeds[1] | arch[1] seeds[0] | arch[1] seeds[1] |
|---------|-------------------|-------------------|-------------------|-------------------|
| 0 | ✅ | | | |
| 1 | | ✅ | | |
| 2 | | | ✅ | |
| 3 | | | | ✅ |

For LLM pipeline: TASK 0 = latent, TASK 1 = response, TASK 2 = joint.

---

## Step 5: Query results from anywhere

```bash
# Point to remote MLflow server
export MLFLOW_TRACKING_URI=http://mlflow-server:5000

# List all experiments
python scripts/query_results.py --list

# Compare SLM architectures by PPL
python scripts/query_results.py --experiment small_lm --metric val_ppl --mode min

# Compare affect encoders by CCC
python scripts/query_results.py --experiment affect_encoder --metric val_ccc --mode max

# Show best LLM runs by F1
python scripts/query_results.py --experiment latent_state_prediction \
    --metric val/response_policy_f1 --mode max

# Get detailed run history
python scripts/query_results.py --run-id abc123

# List checkpoint artifact URIs
python scripts/query_results.py --experiment small_lm --artifacts

# Export to CSV
python scripts/query_results.py --experiment small_lm --csv results.csv
```

### Example output

```
Experiment: small_lm  (sorted by val_ppl, min)

  run_id   │ run_name              │ metric_name │ metric_value │ seed │ arch
  ─────────┼───────────────────────┼─────────────┼──────────────┼──────┼────────
  3a7b2c1d │ slurm_12345_0_gpt_s42 │ val_ppl     │ 18.42        │ 42   │ gpt
  5d8e9f0a │ slurm_12345_1_gpt_s43 │ val_ppl     │ 19.17        │ 43   │ gpt
  2c4b6a8e │ slurm_12345_3_moe_s42 │ val_ppl     │ 22.31        │ 42   │ moe

  18 runs
```

---

## Step 6: Evaluate across all runs

```bash
# After training completes, run evaluation for each checkpoint
for ckpt in $(python scripts/query_results.py --experiment small_lm --artifacts 2>/dev/null | grep -oP '/shared/.*\.pt'); do
    sbatch scripts/slurm_eval.sh slm --checkpoint "$ckpt"
done
```

Evaluation results are also logged to MLflow under `slm_eval`, so you can compare:

```bash
python scripts/query_results.py --experiment slm_eval --metric bleu_1 --mode max
```

---

## Artifact Storage Model

Every run's artifacts are stored in MLflow with the run ID as the key:

```
mlflow-artifacts/
├── {experiment_id}/
│   └── {run_id}/
│       └── artifacts/
│           ├── run_summary.json      ← hyperparams + final metrics
│           ├── epoch_metrics.csv     ← per-epoch val metrics
│           ├── step_metrics.csv      ← per-step train metrics
│           ├── config.json           ← full training config
│           ├── best_model/           ← LLM: LoRA adapter weights
│           └── {arch}_best.pt        ← SLM: model weights
```

To download a checkpoint:

```bash
# If using NFS — direct path
cp /shared/projects/npc/mlflow-artifacts/{exp_id}/{run_id}/artifacts/best_model.pt .

# If using S3
aws s3 cp s3://npc-models/{exp_id}/{run_id}/artifacts/best_model.pt .
```

---

## Evaluation Mapping

Every evaluation is linked to its training run via MLflow tags:

| Tag | Value | Example |
|-----|-------|---------|
| `task` | Training task | `affect`, `small_lm`, `dialogue` |
| `seed` | Random seed | `42` |
| `arch` | Architecture name | `gpt`, `awdlstm`, `mamba_like` |
| `model` | Model identifier | `Qwen/Qwen3-4B` |
| `lora_r` | LoRA rank | `16` |
| `slurm_job_id` | SLURM job ID | `12345` |
| `slurm_array_task` | Array task ID | `3` |

Query by tag:
```bash
python scripts/query_results.py --experiment small_lm --metric val_ppl --mode min
# Automatically groups by arch and seed from tags
```

---

## Monitoring

```bash
# Check job status
squeue -u $USER

# Watch job output
tail -f logs/slurm_12345_0.out

# Cancel job array
scancel 12345

# MLflow UI
mlflow ui --backend-store-uri http://mlflow-server:5000
# Then open http://mlflow-server:5000
```

---

## Resume interrupted runs

Each job logs its state to MLflow. To resume:

```bash
# Check which runs didn't finish
python scripts/query_results.py --experiment small_lm --list | grep -v FINISHED

# Resubmit only incomplete seeds
sbatch scripts/slurm_train.sh slm small_lm --arch mamba_like --seed 43
sbatch scripts/slurm_train.sh slm small_lm --arch mamba_like --seed 44
```

---

## Quick reference

| Task | Command |
|------|---------|
| Submit single training | `sbatch scripts/slurm_train.sh {llm\|slm} {stage}` |
| Submit array (18 jobs) | `sbatch --array=0-17 scripts/slurm_array.sh ...` |
| Check job status | `squeue -u $USER` |
| Cancel job | `scancel <job_id>` |
| View MLflow dashboard | `mlflow ui` → http://server:5000 |
| Compare architectures | `python scripts/query_results.py --experiment small_lm` |
| Export results CSV | `python scripts/query_results.py --experiment X --csv out.csv` |
| Download checkpoint | `cp /shared/.../mlflow-artifacts/{exp_id}/{run_id}/...` |
