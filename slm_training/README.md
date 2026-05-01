# SLM Training from Scratch

Trains small language models (5–50M parameters) from scratch for NPC dialogue — no pre-trained LLM required. Includes personality/affect encoders, a conditional dialogue model, and a benchmark suite of 6 architectures.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    SLM Training Pipeline                         │
│                                                                  │
│  Track A: Encoders (parallel)                                    │
│  ┌─────────────────┐    ┌──────────────────┐                   │
│  │ Personality      │    │ Affect Encoder    │                   │
│  │ OCEAN → 5-vector│    │ VAD → 3-vector    │                   │
│  │ DistilBERT + MLP │    │ DistilBERT + MLP  │                   │
│  └────────┬────────┘    └────────┬─────────┘                   │
│           │                      │                               │
│           └──────┬───────────────┘                               │
│                  ▼                                               │
│  Track B: Language Models with Conditioning                     │
│  ┌───────────────────────────────────────────────────┐         │
│  │  6 architectures, 2 hardware profiles, Optuna HPO  │         │
│  │  OCEAN(5) + VAD(3) = cond_vec(8)                  │         │
│  │  → Prefix injection / token embedding              │         │
│  └───────────────────────────────────────────────────┘         │
│                                                                  │
│  Track C: Fine-Tuned LLM (optional)                              │
│  ┌───────────────────────────────────────────────────┐         │
│  │  Gemma 3/4 + Unsloth QLoRA                         │         │
│  │  TinyLlama 1.1B + LoRA + Prefix conditioning       │         │
│  └───────────────────────────────────────────────────┘         │
└──────────────────────────────────────────────────────────────────┘
```

---

## Track A: Personality & Affect Encoders

### Personality Encoder (OCEAN)

Predicts Big Five personality traits as continuous values.

| Architecture | Detail |
|-------------|--------|
| Backbone | `distilbert-base-uncased` (66M params) |
| Pooling | Mean + Max → concatenated → 2 × 768 = 1536 |
| Head | 3-layer MLP: 1536→768→384→5 |
| Output | 5 continuous values: O, C, E, A, N |
| Activation | GELU |
| Normalization | LayerNorm after pooling |

| Hyperparameter | Value |
|----------------|-------|
| Learning rate | 2×10⁻⁵ |
| LR schedule | Cosine |
| Epochs | 3 |
| Batch size | 16 |
| Max sequence length | 256 |
| Dropout | 0.1 |
| Freeze encoder epochs | 0 |
| Optimizer | AdamW |
| Loss | MSE (regression) |

### Affect Encoder (VAD)

Predicts Valence-Arousal-Dominance as continuous values with Concordance Correlation.

| Architecture | Detail |
|-------------|--------|
| Backbone | `distilbert-base-uncased` |
| Pooling | Mean pooling over non-padding tokens |
| Head | Single linear: 768 → 3 |
| Output | 3 continuous values: V, A, D |
| Sigmoid output | ✅ (values in [0,1]) |

| Hyperparameter | Value |
|----------------|-------|
| Learning rate | 2×10⁻⁵ |
| Epochs | 15 |
| Batch size | 16 |
| Dropout | 0.1 |
| CCC weight (λ) | 0.3 |
| Loss | (1-λ)×MSE + λ×(1-CCC) |
| Multi-sample dropout | 0 (off) |

---

## Track B: Small Language Models from Scratch

Six architectures for A/B comparison, two hardware profiles, Optuna HPO.

### Architecture Catalog

#### 1. GRU (`SmallGRULM`)
```
Embedding(256) → Dropout → GRU(3×512) → Dropout → Linear(V)
```
| Parameter | `m1_small` | `rtx4070_small` |
|-----------|-----------|-----------------|
| embed_dim | 256 | 512 |
| hidden_size | 512 | 1024 |
| num_layers | 3 | 3 |
| dropout | 0.3 | 0.3 |
| tie_weights | ✅ | ✅ |
| params | 42.9M | 94.5M |

#### 2. AWD-LSTM (`AWDLSTMLM`)
```
Embedding(256) → LockedDrop → LSTM(3×512, DropConnect) → LockedDrop → Linear(V)
```
| Parameter | `m1_small` | `rtx4070_small` |
|-----------|-----------|-----------------|
| embed_dim | 256 | 400 |
| hidden_size | 512 | 1150 |
| num_layers | 2 | 3 |
| dropout (output) | 0.4 | 0.4 |
| dropouth (variational) | 0.25 | 0.25 |
| dropouti (embedding) | 0.4 | 0.65 |
| wdrop (DropConnect) | 0.5 | 0.5 |
| tie_weights | ✅ | ✅ |
| params | 42.3M | 106.2M |

**DropConnect:** Applies dropout to LSTM hidden-to-hidden weight matrices (re-sampled every forward pass).

#### 3. GPT (`TinyGPTLM`)
```
Embed(256) + PosEmbed → 4× [ LN → CausalAttn(4 heads) → LN → FFN(4×expand→GELU) ] → LN → Linear(V)
```
| Parameter | `m1_small` | `rtx4070_small` |
|-----------|-----------|-----------------|
| n_embd | 256 | 512 |
| n_head | 4 | 8 |
| n_layer | 4 | 8 |
| FFN expansion | 4× | 4× |
| Activation | GELU | GELU |
| dropout | 0.1 | 0.1 |
| max_seq_len | 256 | 512 |
| tie_weights | ✅ | ✅ |
| params | 16.1M | 51.2M |

#### 4. Prefix GPT (`PrefixTinyGPTLM`)
```
cond_vec(8) → MLP → prefix(8×E)    # Conditioned soft-prefix
Token_Emb + Prefix_Emb + PosEmb → GPT Blocks → Linear(V)
```
| Parameter | `m1_small` | `rtx4070_small` |
|-----------|-----------|-----------------|
| prefix_length | 8 | 8 |
| cond_dim (OCEAN+VAD) | 8 | 8 |
| Prefix projection | Linear→Tanh→Linear | Linear→Tanh→Linear |
| Same GPT backbone as above | ✅ | ✅ |
| params | 16.6M | 53.3M |

#### 5. Mixture-of-Experts (`TinyMoELM`)
```
GPT backbone with sparse MoE FFN per layer
  Router: Linear(E → num_experts), top-k gating
  Experts: 4× Linear→GELU→Linear (per expert)
  Aux loss: load-balancing (frac × route)
```
| Parameter | `m1_small` | `rtx4070_small` |
|-----------|-----------|-----------------|
| num_experts | 4 | 8 |
| top_k | 2 | 2 |
| FFN expansion | 4× | 4× |
| Activation | GELU | GELU |
| Aux loss weight | 0.01 | 0.01 |
| params | 22.4M | 168.8M |

#### 6. Mamba-like SSM (`MambaLikeLM`)
```
Embed(256) → 6× [ LN → SelectiveSSM(d_state=16, d_conv=4) → Dropout ] → LN → Linear(V)
```
**SelectiveSSM:**
- Input projection: E → 2·E·expand (split x, z)
- 1D depthwise conv (kernel=4)
- SSM parameters: B, C (input-dependent), A (learned diagonal, input-independent)
- Discretization: dt = softplus(linear(log_dt))
- Scan: cumulative product (O(L) vectorized)
- Output: y = (h · C)·sum + D·x, gated by SiLU(z)

| Parameter | `m1_small` | `rtx4070_small` |
|-----------|-----------|-----------------|
| n_embd | 256 | 512 |
| n_layer | 6 | 12 |
| d_state (SSM state dim) | 16 | 16 |
| d_conv (local conv kernel) | 4 | 4 |
| expand (inner expansion) | 2 | 2 |
| dropout | 0.1 | 0.1 |
| max_seq_len | 256 | 512 |
| ~params | ~8M | ~50M |

### Hardware Profiles

| Profile | Target | GPUs | Memory |
|---------|--------|------|--------|
| `m1_small` | Apple Silicon MPS | M1–M4 | 2–8 GB unified |
| `rtx4070_small` | NVIDIA consumer | RTX 3070–4090 | 8–24 GB VRAM |

### Optuna HPO Search Space

| Hyperparameter | Range/Choices | Description |
|----------------|---------------|-------------|
| `lr` | [1×10⁻⁴, 5×10⁻³] log-uniform | Learning rate |
| `weight_decay` | [0.01, 0.5] log-uniform | L2 regularization |
| `batch_size` | {8, 16, 32} categorical | Per-device batch |
| `grad_accum` | {1, 2, 4} categorical | Gradient accumulation |
| `dropout` | [0.0, 0.5] uniform | Dropout probability |
| `embed_dim` | {128, 256, 512} categorical | Embedding dimension |
| `n_layer` | {2, 3, 4} (small) / {4, 6, 8} (large) | Number of layers |
| `seq_len` | {128, 256, 512} categorical | Sequence length |

### HPO Pipeline
```
20 trials × 5 epochs → Select best hyperparams → 3 seeds × 30 epochs final training
```

Metrics tracked: PPL, BLEU, Distinct-1/2, token diversity ratio.

---

## Track C: Fine-Tuned LLMs (Optional)

### Gemma 4 + Unsloth

Fine-tunes Google Gemma 4 MoE models using Unsloth + QLoRA.

| Parameter | Value |
|-----------|-------|
| Base model | `unsloth/gemma-3-4b-it` |
| Quantization | 4-bit QLoRA |
| LoRA r | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0.0 |
| Learning rate | 2×10⁻⁴ |
| Epochs | 3 |
| Max sequence length | 2048 |
| Batch size (effective) | 8 (1 × 8 grad_accum) |

### TinyLlama + LoRA + Prefix

Conditional dialogue model with personality-driven soft-prefix.

| Parameter | Value |
|-----------|-------|
| Base model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| LoRA r | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Prefix length | 8 tokens |
| Conditioning dim | 8 (OCEAN=5 + VAD=3) |
| Learning rate | 2×10⁻⁴ |
| Epochs | 3 |
| Max source length | 768 |
| Max target length | 192 |
| Batch size (effective) | 16 (2 × 8 grad_accum) |

---

## Training Orchestration

### Full SLM Pipeline
```bash
cd slm_training
bash run_full_slm_training.sh [arch]
```
Phases: Optuna HPO (20×5) → Final training (3 seeds × 30) → Evaluation (PPL/BLEU/Distinct)

### Individual Components
```bash
# Track A: Encoders (parallel-safe)
bash train_personality_encoder.sh
bash train_affect_encoder.sh

# Track B: Small LMs from scratch
bash train_small_lms.sh [arch]

# Track C: Fine-tuned LLMs
bash finetune_dialogue_lm.sh     # TinyLlama + LoRA + Prefix
bash train_all.sh --with-gemma   # Includes Gemma 4

# Smoke test (quick validation)
bash smoke_test.sh               # Full quick test (distilbert + distilgpt2)
bash smoke_test_external.sh      # With external 107M token corpus

# Full training (all tracks, auto-hardware detection)
bash train_all.sh --run-id my_experiment
```

### Hardware Auto-Detection (`train_all.sh`)
| GPU | Batch Size | Stage 1 | Notes |
|-----|-----------|---------|-------|
| CUDA ≥ 24 GB | 32 / 4 | Parallel | Full speed |
| CUDA 16-24 GB | 16 / 2 | Parallel | |
| CUDA 8-16 GB | 8 / 1 | Parallel (tight) | |
| CUDA < 8 GB | 4 / 1 | Sequential | AMP recommended |
| MPS (Apple) | 16 / 1 | Parallel | |
| CPU | 8 / 1 | Sequential | Very slow |

---

## MLflow Tracking

All SLM training and evaluation logs to the **shared project root `mlruns/`** alongside LLM fine-tuning experiments.

```bash
mlflow ui --backend-store-uri ../mlruns
# → http://localhost:5000
```

**Tracked experiments:** `personality_encoder`, `affect_encoder`, `small_lm`, `dialogue_model`, `slm_eval`

Every run logs: hyperparameters, step-level metrics (loss, lr, grad_norm), epoch-level metrics (PPL, CCC, R², BLEU), and artifacts (`run_summary.json`, `run_summary.md`, CSVs).

MLflow gracefully degrades if not installed — all scripts work without it.

---

## Export & Evaluation

```bash
cd slm_training

# Export trained models to inference-ready format
python scripts/export_small_lm_models.py --arch awdlstm --seed 42

# Evaluate all seeds
python scripts/eval_small_lms.py --arch all --seeds 42 43 44

# Aggregate SLM evaluation artifacts
python -m src.eval.run_eval --artifacts artifacts/

# Generate comprehensive training report
python scripts/comprehensive_training_report.py --phase all --n-seeds 3

# Upload checkpoints to HuggingFace Hub
python scripts/upload_to_hf.py
```

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| PPL | Perplexity on held-out validation text |
| BLEU-4 | 4-gram overlap with reference dialogue |
| Distinct-1/2 | Token type diversity ratio |
| CCC (Affect) | Concordance correlation coefficient |
| R² (Personality) | Coefficient of determination |
| Summary bundle | `evaluation/evaluation_summary.json` + `.md` |

---

## File Structure

```
slm_training/
├── configs/                        # YAML configs
│   ├── personality.yaml            # Personality encoder
│   ├── affect.yaml                 # Affect encoder
│   ├── dialogue.yaml               # TinyLlama + LoRA + Prefix
│   ├── dialogue_gemma_unsloth.yaml # Gemma 4 + Unsloth
│   └── small_lm.yaml               # From-scratch LM training
├── src/
│   ├── models/                     # Model architectures
│   │   ├── personality.py          # DistilBertRegressor (OCEAN)
│   │   ├── affect.py               # DistilBertRegressor (VAD)
│   │   └── dialogue.py             # ConditionalSoftPrefix + DialogueModel
│   ├── train/                      # Training runners
│   │   ├── run_personality.py      # OCEAN encoder training
│   │   ├── run_affect.py           # VAD encoder training
│   │   ├── run_dialogue.py         # TinyLlama conditional training
│   │   ├── run_gemma_unsloth.py    # Gemma 4 Unsloth training
│   │   ├── run_small_lm.py         # From-scratch LM training
│   │   ├── small_lm_architectures.py  # 6 LM architectures
│   │   └── mlflow_tracker.py       # MLflow integration
│   ├── data/                       # Data loaders and preprocessing
│   ├── infer/                      # Inference and chat
│   ├── eval/                       # Evaluation metrics
│   └── api/                        # FastAPI server
├── scripts/                        # HPO, full training, eval
│   ├── optuna_small_lm.py          # Optuna hyperparameter optimization
│   ├── train_final_small_lms.py    # Multi-seed final training
│   ├── eval_small_lms.py           # Small LM evaluation
│   ├── comprehensive_training_report.py  # Full report generation
│   ├── sequential_training_orchestrator.py  # Sequential training loop
│   └── download_external_datasets.py  # External dataset download
├── tests/                          # Unit tests
├── train_all.sh                    # Full orchestrator (all tracks)
├── run_full_slm_training.sh        # HPO + final training pipeline
├── smoke_test.sh                   # Quick smoke test
└── smoke_test.py                   # Smoke test implementation
```
