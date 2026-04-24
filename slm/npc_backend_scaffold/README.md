# NPC Backend — Research Training Stack

Personality-conditioned, affect-aware NPC dialogue generation.
This repository contains **three independent but interacting research tracks**:

| Track | Goal | Models |
|-------|------|--------|
| **A — Encoders** | Predict NPC personality (OCEAN) and emotional state (VAD) from text | Fine-tuned DistilBERT |
| **B — Small LMs** | Train compact dialogue models from scratch; compare architectures | GRU, AWD-LSTM, TinyGPT, PrefixGPT, MoE, Mamba-like |
| **C — LLM Fine-tuning** | Fine-tune a production-grade dialogue LM with LoRA + soft-prefix conditioning | TinyLlama-1.1B, Gemma-3 |

> **If you are new here:** Start with `bash train_personality_encoder.sh` and
> `bash train_affect_encoder.sh` (Track A), then run `bash train_small_lms.sh` (Track B).
> Track C (LLM fine-tuning) requires a GPU with ≥16 GB VRAM.

---

## System Architecture

```
  NPC profile text
       │
       ▼
  ┌─────────────────────┐
  │  Personality Encoder │──► p_vec  (5-dim OCEAN)   ──► FAISS personality cache
  │  DistilBERT + head   │    [O, C, E, A, N] ∈ ℝ⁵
  └─────────────────────┘

  Conversation context
       │
       ▼
  ┌─────────────────────┐
  │  Affect Encoder      │──► a_vec  (3-dim VAD)      ──► live inference
  │  DistilBERT + head   │    [valence, arousal, dom] ∈ ℝ³
  └─────────────────────┘

  p_vec + a_vec
       │
       ▼
  ┌────────────────────────────────────────────────┐
  │  Soft-Prefix MLP                               │
  │  Concatenates [p_vec; a_vec] → prefix tokens  │
  └────────────────────────────────────────────────┘
                          │
       ┌──────────────────┴────────────────────┐
       │    Dialogue LM  (Track C)              │
       │    TinyLlama-1.1B  +  LoRA  +  Prefix │──► NPC response
       │    ─── OR ───                          │
       │    Track B SLM (from scratch)          │
       └────────────────────────────────────────┘
```

---

## Track A — Personality & Affect Encoders

### What they do
Both encoders fine-tune **DistilBERT-base-uncased** (66M parameters) with a lightweight
regression/classification head on top of the `[CLS]` token embedding.

### Personality Encoder (OCEAN)
Predicts the Big Five personality dimensions (Openness, Conscientiousness, Extraversion,
Agreeableness, Neuroticism) as a 5-way multi-label classification from free-text NPC profiles.
Training uses **focal loss** to handle class imbalance in the generated profiles.

- **Architecture:** `DistilBERT → [CLS] → Dropout → Linear(768 → 5) → Sigmoid`
- **Loss:** Focal loss (γ tuned via Optuna)
- **Metric:** Macro F1 per OCEAN dimension
- **Optimiser:** AdamW with differential LR (encoder 10× lower than head), cosine warm-up
- **Hyperparameter search:** `scripts/hyperparam_search.py` (Optuna, 30 trials)
- **Train:** `bash train_personality_encoder.sh`

### Affect Encoder (VAD)
Predicts continuous valence, arousal, and dominance scores from conversation context.
Uses **Concordance Correlation Coefficient (CCC)** loss to optimise rank+scale agreement
simultaneously, critical for affect tasks where per-dimension variance differs greatly.

- **Architecture:** `DistilBERT → [CLS] → Dropout → Linear(768 → 3) → Tanh (scaled to [0,1])`
- **Loss:** CCC loss + MSE regulariser (α tuned via Optuna)
- **Metric:** Mean CCC across V/A/D dimensions
- **Train:** `bash train_affect_encoder.sh`

---

## Track B — Small Language Models (from scratch)

### Why from scratch?
We benchmark six architectures trained from scratch on the same NPC dialogue corpus (~545K tokens,
~2,183 dialogues). This gives a controlled comparison of inductive biases: sequential memory
(RNNs), self-attention (Transformers), sparse routing (MoE), and state-space models (Mamba).
All models use the same GPT-2 BPE tokeniser (tiktoken, vocab=50,257) for fair perplexity comparison.

### Architecture Descriptions

#### 1. GRU-LM (`SmallGRULM`)
A stacked **Gated Recurrent Unit** language model. GRUs (Cho et al., 2014) use reset and update
gates to control information flow across time, providing a computationally cheaper alternative to
LSTMs while retaining comparable performance on short sequences. Our implementation uses 2–3 layers
with tied input/output embeddings.

- **Key hyperparameter:** `seq_len` is critically short (64–128) — BPTT gradients vanish rapidly
  in deep stacks; longer contexts harm rather than help.
- **Parameters (default):** ~4–8M

#### 2. AWD-LSTM (`AWDLSTMLM`)
**ASGD Weight-Dropped LSTM** (Merity et al., 2018). Extends the vanilla LSTM with three
regularisation techniques proven to prevent overfitting on small corpora:
- **DropConnect** on hidden-to-hidden weight matrices (`wdrop`) — drops entire weight entries,
  not just activations, forcing robust hidden state usage
- **Variational (locked) dropout** — the same dropout mask is applied at every time step,
  preserving temporal structure
- **Embedding dropout** — drops entire word vectors with probability `dropouti`

This is the standard strong baseline for small-data recurrent LMs.

- **Parameters (default):** ~8–20M (width-dependent)

#### 3. TinyGPT-LM (`TinyGPTLM`)
A small **decoder-only causal Transformer** in the GPT style (Radford et al., 2018). Uses
multi-head causal self-attention with a causal mask, position embeddings, and pre-norm layer
normalisation. Our implementation targets 4–6 layers with `n_embd=128–256`, providing a compact
but expressive model well-suited to dialogue-length sequences.

- **Key advantage:** Parallelisable training (no sequential dependency); global context window
- **Parameters (default):** ~4–10M

#### 4. PrefixGPT-LM (`PrefixTinyGPTLM`)
**TinyGPT + conditioning prefix.** Prepends `prefix_length` learnable soft-token embeddings
derived from the personality+affect conditioning vector `c ∈ ℝ⁸` via a small MLP. This is
the minimal compatible interface with the production Track C model — the architecture can be
swapped in at inference time with zero API changes.

- **Key difference from TinyGPT:** Accepts `cond_vec` input; outputs are personality-conditioned
- **Research question:** Does even a tiny (8-dim) conditioning signal improve NPC dialogue consistency?

#### 5. TinyMoE-LM (`TinyMoELM`)
A **Mixture-of-Experts Transformer** with sparse routing (Shazeer et al., 2017; Fedus et al., 2022).
The FFN block in each Transformer layer is replaced by `num_experts` parallel feed-forward networks,
with a learned router selecting the top-K experts per token. A **load-balancing auxiliary loss**
(weight 0.01) prevents expert collapse. MoE models can achieve higher effective capacity with the
same per-token compute, but are sensitive to `top_k` and `num_experts` choices.

- **Parameters (active per token):** ~3–5M (sparse; total ~8–20M across experts)

#### 6. Mamba-like SSM (`MambaLikeLM`)
A **selective state-space model** inspired by Mamba (Gu & Dao, 2023). Replaces self-attention
with a linear-recurrent SSM layer where the state transition matrices (A, B, C) are
**input-dependent** (selected from the input token), allowing the model to selectively remember
or forget information. Our pure-PyTorch implementation uses a sequential scan (no CUDA kernels).

- **Critical constraint:** The Python sequential scan is O(seq_len) — use `seq_len ≤ 64`.
  With seq_len=32 this model matches GPT-level perplexity at lower parameter count.
- **Parameters (default):** ~5–15M

### Hyperparameter Optimisation
Each architecture has a dedicated search space in `scripts/optuna_small_lm.py`. Key axes differ
by architecture: RNNs need shorter `seq_len` and lower `lr`; Transformers tolerate larger models.

```bash
# Run HPO for one architecture (20 trials × 5 epochs each)
bash train_small_lms.sh --hpo-only --arch gru

# Run full pipeline: HPO → retrain best × 3 seeds × 30 epochs
bash train_small_lms.sh
```

### Current Results (5 epochs, Optuna-found params)

| Architecture | Val PPL (mean ± std) | Params | seq_len |
|---|---|---|---|
| PrefixGPT | 46.6 ± 0.4 | ~5M | 128 |
| GPT | 81.8 ± 0.3 | ~2M | 256 |
| MoE | 140.4 ± 6.8 | ~8M | 256 |
| GRU | 299.6 ± 32.0 | ~4M | 64 |
| AWD-LSTM | 296.2 ± 70.2 | ~12M | 128 |
| Mamba-like | ~53 (1 trial) | ~4M | 32 |

*Full 30-epoch × 3-seed results in progress.*

---

## Track C — Large LM Fine-tuning

### TinyLlama-1.1B + LoRA + Soft-Prefix
Fine-tunes **TinyLlama-1.1B** (Zhang et al., 2024) on the NPC dialogue corpus using
**Low-Rank Adaptation** (LoRA; Hu et al., 2022). LoRA freezes the base model and injects
trainable rank-decomposition matrices `W = W₀ + BA` (rank r=16) into the attention projections,
reducing trainable parameters by ~1000× vs full fine-tuning. The soft-prefix from Track A
is prepended to the prompt, injecting personality conditioning without modifying architecture.

```bash
bash finetune_dialogue_lm.sh --model tinyllama
```

### Gemma-3 + Unsloth
Fine-tunes **Gemma-3** with **Unsloth** (2–4× faster training via custom CUDA kernels and
Flash Attention). Uses 4-bit NF4 quantisation for memory efficiency on consumer GPUs.

```bash
bash finetune_dialogue_lm.sh --model gemma
```

---

## Quick Start

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Verify everything works (2–3 min)
bash smoke_test.sh

# 3. Train encoders (Track A, ~2 hrs each on RTX 4070)
bash train_personality_encoder.sh
bash train_affect_encoder.sh

# 4. Train small LMs (Track B, ~6 hrs HPO + ~4 hrs final)
bash train_small_lms.sh

# 5. Fine-tune dialogue LM (Track C, ~4 hrs on RTX 4070)
bash finetune_dialogue_lm.sh --model tinyllama

# 6. Evaluate all models
bash evaluate.sh
```

---

## Training Scripts Reference

| Script | Track | What it does |
|--------|-------|--------------|
| `train_personality_encoder.sh` | A | Optuna HPO → fine-tune DistilBERT for OCEAN personality |
| `train_affect_encoder.sh` | A | Optuna HPO → fine-tune DistilBERT for VAD affect |
| `train_small_lms.sh` | B | Optuna HPO for all 6 architectures → retrain × 3 seeds |
| `finetune_dialogue_lm.sh` | C | Fine-tune TinyLlama or Gemma with LoRA + soft-prefix |
| `evaluate.sh` | A+B+C | Evaluate all trained models; print comparison table |
| `smoke_test.sh` | all | End-to-end pipeline test with tiny data (~3 min) |

---

## Logs and Artifacts

All runs write to `artifacts/<model>/<run_id>/`:

```
artifacts/
├── personality_encoder/
│   └── <run_id>/
│       ├── run.log               # timestamped training log
│       ├── step_metrics.csv      # loss, lr, grad_norm per step
│       ├── epoch_metrics.csv     # val F1 / MSE per epoch
│       └── run_summary.json      # hyperparams + best results
├── affect_encoder/
│   └── <run_id>/  (same structure)
├── small_lm/
│   └── <run_id>/  (same structure; val_ppl instead of val_mse)
├── dialogue/
│   └── <run_id>/  (same structure)
└── optuna/
    ├── personality_best.json     # best Optuna params for personality
    ├── affect_best.json          # best Optuna params for affect
    └── small_lm_<arch>_best.json # best Optuna params per SLM arch
```

MLflow tracks all runs. View with:
```bash
mlflow ui --backend-store-uri ./mlruns
# open http://localhost:5000
```

---

## File Layout

```
slm/npc_backend_scaffold/
│
├── Train scripts (run these)
│   ├── train_personality_encoder.sh   Track A: personality encoder HPO + training
│   ├── train_affect_encoder.sh        Track A: affect encoder HPO + training
│   ├── train_small_lms.sh             Track B: SLM HPO + multi-seed training
│   ├── finetune_dialogue_lm.sh        Track C: TinyLlama or Gemma fine-tuning
│   ├── evaluate.sh                    Evaluate all models, print results table
│   └── smoke_test.sh                  Quick end-to-end sanity check
│
├── configs/
│   ├── personality.yaml               Personality encoder hyperparams
│   ├── affect.yaml                    Affect encoder hyperparams
│   ├── dialogue.yaml                  Dialogue LM (TinyLlama) hyperparams
│   └── dialogue_gemma_unsloth.yaml    Gemma fine-tuning hyperparams
│
├── src/
│   ├── train/
│   │   ├── run_personality.py         Personality encoder training loop
│   │   ├── run_affect.py              Affect encoder training loop
│   │   ├── run_small_lm.py            Small LM training loop (all 6 archs)
│   │   ├── run_dialogue.py            TinyLlama LoRA fine-tuning
│   │   ├── run_gemma_unsloth.py       Gemma Unsloth fine-tuning
│   │   └── small_lm_architectures.py  GRU/AWD-LSTM/GPT/PrefixGPT/MoE/Mamba-like
│   ├── models/
│   │   ├── personality.py             DistilBertRegressor (OCEAN head)
│   │   ├── affect.py                  DistilBertRegressor (VAD head)
│   │   └── dialogue.py                ConditionalDialogueModel (LoRA + prefix)
│   └── data/
│       ├── datasets.py                Dataset loaders
│       ├── build_caches.py            Build FAISS personality cache
│       ├── prepare_dialogue_data.py   Convert public datasets
│       └── convert_generated_data.py  Convert upper-level generated data
│
├── scripts/
│   ├── optuna_small_lm.py             Per-arch Optuna HPO for Track B
│   ├── train_final_small_lms.py       Multi-seed final training from Optuna bests
│   ├── eval_small_lms.py              Evaluate SLMs: PPL, Distinct-1/2, samples
│   └── hyperparam_search.py           Optuna HPO for Track A encoders
│
└── data/
    ├── dialogue/
    │   ├── train.txt / val.txt        Plain text for SLM training (Track B)
    │   └── train.jsonl / val.jsonl    Structured turns for Track C
    └── affect/
        └── train.csv / val.csv        VAD labels for affect encoder (Track A)
```

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Gemma fine-tuning (Track C), also install Unsloth:
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

---

## References

- Cho et al. (2014). *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.* EMNLP.
- Merity et al. (2018). *Regularizing and Optimizing LSTM Language Models.* ICLR.
- Radford et al. (2018). *Improving Language Understanding by Generative Pre-Training.* OpenAI.
- Shazeer et al. (2017). *Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer.* ICLR.
- Fedus et al. (2022). *Switch Transformers: Scaling to Trillion Parameter Models.* JMLR.
- Gu & Dao (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv.
- Hu et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR.
- Zhang et al. (2024). *TinyLlama: An Open-Source Small Language Model.* arXiv.
