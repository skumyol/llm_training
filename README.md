# NPC Dialogue with Structured Social State

**Architecture Benchmarks · Pretrained Baselines · Full Training Pipeline**

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Stack](#architecture-stack)
3. [Environment Setup](#environment-setup)
4. [Data Preparation](#data-preparation)
5. [Training Pipeline](#training-pipeline)
6. [Evaluation](#evaluation)
7. [Slurm / HPC Operations](#slurm--hpc-operations)
8. [Results & Paper](#results--paper)
9. [Code Fixes Applied](#code-fixes-applied)
10. [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# 1. SSH into the cluster
ssh skumyol@hpc4.ust.hk
cd ~/llm_training

# 2. Run smoke test (verifies everything works)
cd slm_training && bash smoke_test.sh

# 3. Submit full training pipeline
bash scripts/resume_training.sh

# 4. Monitor
squeue -u $USER
tail -f /scratch/$USER/logs/t_gpt_*.out
```

---

## Architecture Stack

| Track | Models | Parameters | Role |
|-------|--------|-----------|------|
| **A: From-scratch SLMs** | GPT, PrefixGPT, MoE, Mamba-like | 15–16M | Architecture benchmark |
| **B: Conditioning Encoders** | Personality (OCEAN), Affect (VAD) | 66M (DistilBERT) | Social state perception |
| **C: Response Generation** | ConditionalDialogue, TinyLlama+LoRA, Gemma 4 E2B | 1.1B–16B | Response quality |
| **D: Structured LLM** | Qwen3 + 29-head predictor | 0.6B–4B | Full social-state model |

### Key Results

| Model | Metric | Value |
|-------|--------|-------|
| **MoE** (best from-scratch) | val_ppl | 42.07 |
| **ConditionalDialogue** | val_ppl | 2.90 |
| **TinyLlama 1.1B + LoRA** | val_ppl | 3.30 |
| **Qwen3 latent (29-head)** | f1 | 0.474 |
| **Personality encoder** | val_f1 | 0.678 |
| **Affect encoder** | val_ccc | 0.559 |
| **Gemma 4 E2B** | val_ppl | 16.24 (1 epoch) |

---

## Environment Setup

### Local (smoke testing)

```bash
cd ~/llm_training
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### HPC Cluster (for training)

The cluster uses `/scratch/$USER/` for fast I/O with a symlink from `/scratch/$USER/npc` → `~/llm_training`.

```bash
# One-time setup
bash scripts/env_setup_spack.sh

# Or manual:
mkdir -p /scratch/$USER/{data,logs,checkpoints,mlruns,venvs,artifacts}
ln -sfn ~/llm_training /scratch/$USER/npc

# Create SLM venv (lightweight, for small LMs + encoders + Gemma)
module load miniconda3/24.3.0 cuda/12.4.0
python3 -m venv /scratch/$USER/venvs/slm_env --system-site-packages
source /scratch/$USER/venvs/slm_env/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers peft bitsandbytes accelerate trl datasets
pip install optuna mlflow pyyaml tqdm pandas faiss-cpu sentence-transformers

# Create LLM venv (heavier, for Qwen3 fine-tuning)
python3 -m venv /scratch/$USER/venvs/llm_env --system-site-packages
source /scratch/$USER/venvs/llm_env/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install transformers peft bitsandbytes accelerate trl datasets tokenizers
pip install mlflow pyyaml tqdm pandas scikit-learn evaluate
```

---

## Data Preparation

### Data locations

| Data | Path | Format |
|------|------|--------|
| SLM training | `slm_training/data/dialogue/train.txt` | Plain text |
| SLM validation | `slm_training/data/dialogue/val.txt` | Plain text |
| Dialogue SFT | `data/splits/train_sft.jsonl` | JSONL |
| Dialogue validation | `data/splits/val_sft.jsonl` | JSONL |
| NPC profiles | `slm_training/data/npc_profiles.csv` | CSV |
| Personality labels | `slm_training/data/personality/train.csv` | CSV |
| Affect labels | `slm_training/data/affect/train.csv` | CSV |
| Generated dialogues | `slm_training/data/dialogue/from_gen_train.jsonl` | JSONL |

### Generate mock data (if real data unavailable)

```bash
source /scratch/$USER/venvs/slm_env/bin/activate
cd /scratch/$USER/npc/slm_training

python3 -c "
import json, random
from pathlib import Path

# Small LM text data
d = Path('data/dialogue'); d.mkdir(parents=True, exist_ok=True)
lines = [f'Player asks about {random.choice([\"siege\",\"spy\",\"artifact\"])}. NPC replies.' for _ in range(2000)]
with open(d/'train.txt','w') as f: f.write('\n'.join(lines[:1600])+'\n')
with open(d/'val.txt','w') as f: f.write('\n'.join(lines[1600:])+'\n')

# SFT JSONL for LLM response training
sft_lines = []
for i in range(100):
    rec = {
        'episode_id': f'ep{i:04d}',
        'turn_idx': i,
        'scenario_type': 'mock',
        'input': f'<scene>Setting: tavern\nNPC: innkeeper\n</scene>\n\nPlayer: Tell me about the {random.choice([\"siege\",\"spy\"])}.',
        'target': f'The innkeeper responds about the situation.',
        'counterfactual': False
    }
    sft_lines.append(json.dumps(rec))
Path('../../data/splits').mkdir(parents=True, exist_ok=True)
with open('../../data/splits/train_sft.jsonl','w') as f: f.write('\n'.join(sft_lines[:80])+'\n')
with open('../../data/splits/val_sft.jsonl','w') as f: f.write('\n'.join(sft_lines[80:])+'\n')
print('Mock data generated')
"
```

---

## Training Pipeline

### Quick test (1 epoch, 15 min)

```bash
bash scripts/submit_test_jobs.sh          # Submit all 12 test jobs
bash scripts/submit_test_jobs.sh slm       # SLM tests only
bash scripts/submit_test_jobs.sh llm       # LLM tests only
bash scripts/submit_test_jobs.sh --dry-run # See what would submit
```

### Full training (all tracks)

```bash
# One command to rule them all:
bash scripts/resume_training.sh

# This runs:
#  1. Cancel stale jobs
#  2. Verify environment + data
#  3. Pre-flight test (1-epoch GPT → catches failures early)
#  4. Full training: 4 SLM architectures × 20 epochs
#  5. Auto-evaluation after training
#  6. Auto-export artifacts to home
```

### Individual model training

```bash
# --- Track A: From-scratch SLMs ---

# Single architecture
sbatch scripts/slurm_train.sh slm small_lm --arch gpt --epochs 20 --seed 42
sbatch scripts/slurm_train.sh slm small_lm --arch mamba_like --epochs 20
sbatch scripts/slurm_train.sh slm small_lm --arch prefix_gpt --epochs 20
sbatch scripts/slurm_train.sh slm small_lm --arch moe --epochs 20

# Job array (6 archs × 3 seeds = 18 jobs)
sbatch --array=0-17 scripts/slurm_array.sh slm small_lm \
    --archs gpt,prefix_gpt,moe,mamba_like --seeds 42,43,44

# --- Track B: Conditioning Encoders ---
sbatch scripts/slurm_train.sh slm personality --epochs 15
sbatch scripts/slurm_train.sh slm affect --epochs 15

# Build personality cache (required for dialogue model)
source /scratch/$USER/venvs/slm_env/bin/activate
cd /scratch/$USER/npc/slm_training
python -m src.data.build_caches \
    --profiles-path data/npc_profiles.csv \
    --encoder-dir artifacts/personality_encoder/<run_id>/best_model \
    --out-path artifacts/personality_cache.jsonl

# --- Track C: Response Generation ---
sbatch scripts/slurm_train.sh slm dialogue --epochs 5
sbatch scripts/slurm_train.sh slm small_lm --arch gpt --epochs 3 \
    --base-model-name TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Gemma 4 E2B (needs HF token + expandable_segments)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m src.train.run_gemma_unsloth \
    --config configs/dialogue_gemma_unsloth.yaml \
    --epochs 3 --max-train-samples 2000

# --- Track D: Structured LLM (Qwen3) ---
sbatch scripts/slurm_train.sh llm latent --debug    # Stage 1: 29-head predictor
sbatch scripts/slurm_train.sh llm response --debug  # Stage 2: Response generator
sbatch scripts/slurm_train.sh llm joint --debug     # Stage 3: Joint fine-tuning
```

### Custom GPU / time

```bash
# Different partition
sbatch --partition=gpu-a30 scripts/slurm_train.sh slm small_lm --arch gpt

# More time for large models
sbatch --time=72:00:00 scripts/slurm_train.sh llm latent

# Multiple GPUs
sbatch --gpus-per-node=2 scripts/slurm_train.sh slm small_lm --arch moe
```

### Available GPU partitions

| Partition | GPU | VRAM | Nodes | Best for |
|-----------|-----|------|-------|----------|
| `gpu-a30` | A30 | 24GB | 15 | SLMs, encoders, Qwen3-1.7B |
| `gpu-l20` | L20 | 48GB | 6 | Gemma 4, Qwen3-4B |
| `gpu-rtx4090d` | RTX 4090D | 24GB | 2 | Fast SLM training |

---

## Evaluation

### Run evaluation

```bash
# SLM evaluation (encoders + dialogue + small LMs)
sbatch scripts/slurm_eval.sh slm

# With specific artifacts path
sbatch scripts/slurm_eval.sh slm --artifacts /path/to/artifacts

# Save to CSV
sbatch scripts/slurm_eval.sh slm --out-csv /scratch/$USER/artifacts/results.csv

# LLM evaluation
sbatch scripts/slurm_eval.sh llm latent
sbatch scripts/slurm_eval.sh llm all

# Both SLM + LLM
sbatch scripts/slurm_eval.sh all
```

### Train + auto-evaluate

```bash
sbatch scripts/slurm_train_eval.sh slm --arch gpt --epochs 20
sbatch scripts/slurm_train_eval.sh llm latent --debug
```

### Reading results

```bash
# Per-model summaries
find /scratch/$USER/npc/slm_training/artifacts -name "run_summary.json" | \
    xargs python3 -c "
import json, sys
for f in sys.argv[1:]:
    d = json.load(open(f))
    best = d.get('best', {})
    print(f'{d.get(\"run_id\",\"?\")[:40]:40s} {d.get(\"task\",\"?\"):30s} {best}')
"

# Evaluation bundle
cat /scratch/$USER/npc/slm_training/artifacts/evaluation/evaluation_summary.json

# MLflow experiments
ls /scratch/$USER/mlruns/
```

---

## Slurm / HPC Operations

### Key scripts

| Script | Purpose |
|--------|---------|
| `scripts/slurm_train.sh` | Single training job (SLM or LLM) |
| `scripts/slurm_array.sh` | Job arrays for multi-seed sweeps |
| `scripts/slurm_eval.sh` | Standalone evaluation |
| `scripts/slurm_train_eval.sh` | Train + auto-evaluate |
| `scripts/resume_training.sh` | Full pipeline with pre-flight check |
| `scripts/submit_test_jobs.sh` | Quick 1-epoch verification tests |
| `scripts/submit_missing.sh` | Auto-detect + resume incomplete runs |
| `scripts/env_setup_spack.sh` | One-time environment bootstrap |
| `scripts/gpu_test.sh` | Interactive GPU verification |
| `scripts/sync_to_scratch.sh` | Rsync code/data to scratch |

### Cluster conventions (HKUST HPC4)

```bash
# Use --gpus-per-node (NOT --gres=gpu)
# Use --account=xrimlab
# No --mem on GPU jobs (auto-allocated)
# Scratch at /scratch/$USER for fast I/O
# CUDA via: module load cuda/12.4.0
# Python via: module load miniconda3/24.3.0
```

### Monitor jobs

```bash
# Queue
squeue -u $USER
watch -n 5 'squeue -u $USER'

# Live logs
tail -f /scratch/$USER/logs/t_gpt_*.out
tail -f /scratch/$USER/logs/llm_latent_*.out

# Job history
sacct -u $USER --format="JobID,JobName,State,ExitCode,Elapsed" --starttime=2026-05-01

# Cancel jobs
scancel -u $USER                     # All jobs
scancel -u $USER -n t-gpt            # By name
scancel 123456                       # By job ID
```

### Export results for local download

```bash
# From your local machine:
scp -r skumyol@hpc4.ust.hk:/scratch/skumyol/npc/slm_training/artifacts ./artifacts/
scp -r skumyol@hpc4.ust.hk:/scratch/skumyol/logs ./slurm_logs/
scp -r skumyol@hpc4.ust.hk:/home/skumyol/llm_training/docs ./docs/
scp skumyol@hpc4.ust.hk:/home/skumyol/llm_training/docs/evaluation_paper.md ./
```

---

## Results & Paper

### Complete results table

| # | Track | Model | Best Metric | Epochs | Hardware |
|---|-------|-------|------------|--------|----------|
| 1 | A | MoE | val_ppl=42.07 | 20 | A30 24GB |
| 2 | A | PrefixGPT | val_ppl=44.54 | 20 | A30 24GB |
| 3 | A | GPT | val_ppl=45.32 | 20 | A30 24GB |
| 4 | A | Mamba-like | val_ppl=53.25 | 10 | A30 24GB |
| 5 | B | Personality encoder | val_f1=0.678 | 4 | A30 24GB |
| 6 | B | Affect encoder | val_ccc=0.559 | 13 | A30 24GB |
| 7 | C | ConditionalDialogue | val_ppl=2.90 | 2 | A30 24GB |
| 8 | C | TinyLlama 1.1B + LoRA | val_ppl=3.30 | 1 | A30 24GB |
| 9 | C | Gemma 4 E2B + QLoRA | val_ppl=16.24 | 1 | A30 24GB |
| 10 | D | Qwen3 latent (29-head) | f1=0.474 | 3 | A30 24GB |
| 11 | D | Qwen3 response | SFT loss | — | A30 24GB |
| 12 | D | Qwen3 joint | — | — | A30 24GB |

### Paper

Full technical paper: [`docs/evaluation_paper.md`](docs/evaluation_paper.md)

Covers: methodology, architecture stack, training configuration, results tables, discussion, limitations, and all code fixes applied.

---

## Code Fixes Applied

This evaluation required substantial debugging. Key fixes:

| Issue | File | Fix |
|-------|------|-----|
| `build_personality_cache` missing import | `smoke_test.py` | Use `encode_profiles` |
| Tokenizer returning NoneType | `smoke_test.py` | Use `distilbert-base-uncased` |
| `GradScaler` deprecation | `run_small_lm.py` | `torch.amp.GradScaler('cuda', ...)` |
| `get_sentence_embedding_dimension` | `memory_store.py` | `get_embedding_dimension()` fallback |
| Missing `faiss` dependency | venv | `pip install faiss-cpu` |
| Wrong working directory | `slurm_train.sh` | `cd slm_training/` before SLM training |
| `dtype` mismatch (float vs BFloat16) | `dialogue.py` | Cast prefix embeddings to model dtype |
| `token_type_ids` in encoder | `build_caches.py` | Filter from input dict |
| Missing personality cache | training pipeline | Built 414-NPC cache from training data |
| `datasets.py` name collision | project structure | Renamed to `dialogue_data.py` |
| `max_seq_len` truncation → NaN | `train_response.yaml` | Increased from 1024 → 2048 |
| `best_dir` not created | `train_latent/response/joint.py` | Added `mkdir(parents=True)` |
| Gemma 4 `ClippableLinear` | `run_gemma_unsloth.py` | `target_modules="all-linear"` |
| Gemma 4 no chat template | `run_gemma_unsloth.py` | Set Gemma chat template |
| Gemma 4 OOM on eval | `run_gemma_unsloth.py` | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| trl `tokenizer` → `processing_class` | `run_gemma_unsloth.py` | API migration |
| trl `dataset_text_field` → `formatting_func` | `run_gemma_unsloth.py` | API migration |
| `max_seq_length` → `max_length` | `run_gemma_unsloth.py` | API migration |

---

## Troubleshooting

### "FileNotFoundError: data/dialogue/train.txt"

The SLM training expects data in `slm_training/data/dialogue/`. Either generate mock data (see Data Preparation) or the `resume_training.sh` script auto-generates it.

### "ModuleNotFoundError: No module named 'faiss'"

```bash
source /scratch/$USER/venvs/slm_env/bin/activate
pip install faiss-cpu
```

### "CUDA out of memory" on Gemma 4

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Or use gpu-l20 (48GB) partition
sbatch --partition=gpu-l20 ...
```

### "NaN loss" in LLM response training

Caused by `max_seq_len` truncating inputs beyond 1024 tokens. Fixed in config by setting `max_seq_len: 2048`. If using custom data, verify input lengths.

### "GatedRepoError" for Gemma models

Gemma models require HuggingFace authentication:
1. Accept license at huggingface.co/google/gemma-4-E2B
2. Add `HF_TOKEN=hf_...` to `~/.env`
3. The training script auto-logins via the token

### Jobs stuck in PENDING

Check available nodes and account limits:
```bash
sinfo -p gpu-a30,gpu-l20    # Available nodes
sacctmgr show user $USER     # Account limits
```

### QOSMaxSubmitJobPerUserLimit

Submit jobs individually with a 1-2 second delay:
```bash
for arch in gpt mamba_like prefix_gpt moe; do
    sbatch scripts/slurm_train.sh slm small_lm --arch $arch --epochs 20
    sleep 2
done
```
