# Model Architecture Reference

_All models, tracks, conditioning mechanisms, and how they connect._

---

## Architecture Stack Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        NPC DIALOGUE SYSTEM                          │
│                                                                     │
│  Player:"Where is the vault?"                                       │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  TRACK B: CONDITIONING ENCODERS (DistilBERT 66M)            │   │
│  │  ┌──────────────────┐  ┌──────────────────┐                 │   │
│  │  │Personality Encoder│  │  Affect Encoder  │                 │   │
│  │  │  dialogue → OCEAN │  │dialogue → VAD(3) │                 │   │
│  │  │     (5 floats)    │  │   (3 floats)     │                 │   │
│  │  └────────┬─────────┘  └────────┬─────────┘                 │   │
│  │           │   concat(5+3) = 8D   │                           │   │
│  │           └──────────┬───────────┘                           │   │
│  └──────────────────────┼───────────────────────────────────────┘   │
│                         │ cond_vec (8D)                              │
│                         ▼                                            │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  TRACK D: LATENT SOCIAL-STATE PREDICTOR                      │   │
│  │  ┌──────────────────────────────────────────────────────┐   │   │
│  │  │  Backbone (Qwen3/Gemma4/GPT-SLM/Mamba-SLM/MoE-SLM)   │   │   │
│  │  │  dialogue text → pooled hidden state                 │   │   │
│  │  └──────────────────────┬───────────────────────────────┘   │   │
│  │                         │                                    │   │
│  │          ┌──────────────┼──────────────┐                     │   │
│  │          ▼              ▼              ▼                     │   │
│  │   ┌──────────┐  ┌──────────┐  ┌──────────────┐              │   │
│  │   │ C_t head │  │ R_t head │  │ D_t head ... │  ×28 heads  │   │
│  │   │(context) │  │(stance)  │  │ (decision)   │              │   │
│  │   └──────────┘  └──────────┘  └──────────────┘              │   │
│  │                                                              │   │
│  │   Optional: SocialJEPAHead → predicts future Z_{t+1}        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                         │ predicted Z_t (28 labels)                 │
│                         ▼                                            │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  TRACK C: RESPONSE GENERATOR                                 │   │
│  │  cond_vec(8D) → prefix_proj → prefix tokens → Backbone      │   │
│  │  ┌──────────────────────────────────────────────────────┐   │   │
│  │  │  Backbone (TinyLlama 1.1B / Qwen3 1.7B / Gemma4)    │   │   │
│  │  │  [prefix tokens] + dialogue → NPC response           │   │   │
│  │  └──────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                         │                                            │
│                         ▼                                            │
│  NPC:"The vault lies beneath the eastern tower, but I'd             │
│        need the captain's seal to show you."                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Track A: From-Scratch Small Language Models (SLMs)

**Purpose:** Language modeling baselines. Trained from scratch on 16,905 lines of NPC dialogue text. Evaluated on next-token perplexity.

**All models share:** `vocab_size=50257` (GPT-2 tokenizer), trained with AdamW + cosine schedule.

### GPT (16.1M params, PPL 45.32)

```
Input tokens → TokenEmbed + PosEmbed → [GPTBlock ×4] → LayerNorm → Linear → logits
                                        ┌──────────────┐
                   GPTBlock:            │ MultiHeadAttn │
                     h → LN → Attn ──┬──│     (4 heads) │
                     h → LN → FFN  ──┼──│  FFN: 256→1024│
                                     │  │       →256    │
                                     └──┴──────────────┘
```

Standard transformer decoder. 4 layers, 4 heads, d_model=256. Uses causal masking and weight tying.

### PrefixGPT (16.6M params, PPL 44.54)

```
cond_vec(8D) → Linear(8→256)→Tanh→Linear(256→8×256) → 8 prefix tokens
                                                              │
Input tokens → TokenEmbed + PosEmbed ←── concat ─────────────┘
                    │
                    ▼
              [GPTBlock ×4] → LayerNorm → Linear → logits
```

Same as GPT, but prepends 8 learned soft-prefix tokens projected from the 8D OCEAN+VAD conditioning vector. Conditioning helps marginally (-1.7% PPL).

### MoE (22.4M params, PPL 42.07) — Best Architecture

```
Input tokens → TokenEmbed + PosEmbed → [MoEBlock ×4] → LayerNorm → Linear → logits
                                        ┌──────────────────────────────┐
                   MoEBlock:            │ MultiHeadAttn                │
                     h → LN → Attn  ──┬─│ FFN replaced by:             │
                     h → LN → MoE  ──┼─│ ┌──────────────────────────┐  │
                                     │  │ │ Router: 256→4 experts    │  │
                                     │  │ │ Softmax → top-2 experts  │  │
                                     │  │ │ Expert_i: 256→512→256    │  │
                                     │  │ │ Output: weighted sum      │  │
                                     │  │ └──────────────────────────┘  │
                                     │  │ + aux_loss for load balancing │
                                     └──┴──────────────────────────────┘
```

Mixture-of-Experts: 4 experts, top-2 routing. 39% more params than GPT but 7.2% better PPL. Auxiliary load-balancing loss prevents expert collapse.

### Mamba-like (15.4M params, PPL 53.25)

```
Input tokens → TokenEmbed → [MambaBlock ×6] → LayerNorm → Linear → logits
                             ┌──────────────────────────────────────┐
        MambaBlock:          │ Input → LayerNorm                   │
          h → LN → SSM  ──┬──│   ├── Conv1d(4) → SiLU              │
          h → LN → FFN  ──┼──│   ├── Selective Scan (SSM)          │
                          │  │   │   d_state=16: hidden state       │
                          │  │   │   A,B,C,D: input-dependent       │
                          │  │   │   discretised by Δ               │
                          │  │   └── Output gate                    │
                          │  │ FFN: 256→512→256 (expand=2)          │
                          └──┴──────────────────────────────────────┘
```

Pure PyTorch SSM (no CUDA acceleration). 6 layers, d_state=16. Converges fastest but plateaus higher than transformers. Research baseline for architecture comparison.

---

## Track B: Conditioning Encoders (DistilBERT 66M)

**Purpose:** Encode dialogue context into OCEAN personality (5D) and VAD affect (3D) vectors used by Track C generators.

```
┌──────────────────────────────────────────────┐
│           Personality Encoder                 │
│  dialogue text → DistilBERT → pooled →       │
│  Linear(768→256)→ReLU→Dropout→               │
│  Linear(256→5) → OCEAN vector                │
│  (openness, conscientiousness, extraversion,  │
│   agreeableness, neuroticism)                 │
│  Loss: MSE                                    │
│  F1: 0.678                                    │
└──────────────────────────────────────────────┘

┌──────────────────────────────────────────────┐
│              Affect Encoder                   │
│  dialogue text → DistilBERT → pooled →       │
│  Linear(768→256)→ReLU→Dropout→               │
│  Linear(256→3) → VAD vector                  │
│  (valence, arousal, dominance)                │
│  Loss: MSE                                    │
│  F1: 0.559 (3-class discretised)              │
└──────────────────────────────────────────────┘

  concat(OCEAN, VAD) = 8D cond_vec → feeds Track C generators
```

**Important finding:** Placebo ablations proved the specific OCEAN/VAD values are NOT used by the generator. The PPL gain is from having any 8D conditioning vector, not from its content.

---

## Track C: Dialogue Response Generators

**Purpose:** Generate NPC dialogue responses conditioned on dialogue context and (optionally) OCEAN+VAD prefix.

### TinyLlama 1.1B + Conditional Prefix (PPL 2.90)

```
cond_vec(8D) → Linear(8→768)→Tanh→Linear(768→8×768) → 8 prefix tokens
                                                                │
dialogue context → tokenizer → token_ids ←─── concat ───────────┘
                                     │
                                     ▼
                          TinyLlama 1.1B (22 layers, 2048 dim)
                          + LoRA (r=16, α=32)
                                     │
                                     ▼
                             NPC response tokens
```

The primary conditioned dialogue model. 8D OCEAN+VAD vector projected to 8 soft-prefix tokens prepended to dialogue context. LoRA fine-tuning on 2,183 dialogue turns.

**Placebo finding:** Shuffling OCEAN, randomizing VAD, or both → same PPL 2.88-2.91. The 12.3% gain (3.30→2.90) is from prefix capacity, not semantic content.

### Qwen3-1.7B + QLoRA (PPL 1.04)

```
dialogue context (with NPC profile) → Qwen3-1.7B (35 layers, 1536 dim)
                                      + QLoRA (r=32, α=64, 4-bit nf4)
                                               │
                                               ▼
                                       NPC response tokens
```

Pretrained transformer with QLoRA fine-tuning. Used as response generator in joint training with latent predictor. Best PPL (1.04) but largest model.

### Gemma-4-E2B + QLoRA (PPL 16.24, 1 epoch)

```
NPC profile + dialogue context → Gemma-4-E2B (35 layers, 1536 dim)
                                 MoE: 16B total, 2B active
                                 + QLoRA (r=16, α=16, 4-bit nf4)
                                          │
                                          ▼
                                  NPC response tokens
```

Exploratory larger baseline. Mixture-of-Experts with 2B active parameters per forward pass. Currently undertrained (1 epoch only).

---

## Track D: Latent Social-State Predictors

**Purpose:** Predict the 29-dimension social state Z_t from dialogue context. Uses multi-head classification on pooled backbone hidden states.

### Z_t Schema (29 heads across 6 groups)

```
┌──────────────────────────────────────────────────────────────────┐
│                     Z_t = {C, A, M, R, N, D}                     │
│                                                                  │
│  C (Context):     3 heads                                        │
│    dialogue_act, tone, risk_type                                 │
│                                                                  │
│  A (Affect):      2 heads                                        │
│    valence, arousal                                              │
│                                                                  │
│  M (Mental Model): 3 heads                                       │
│    player_intent, player_knowledge, player_credibility           │
│                                                                  │
│  R (Relational Stance): 12 heads                                 │
│    affection_level/δ, respect_level/δ, dominance_level/δ,        │
│    familiarity_level/δ, trust_level/δ, obligation_level/δ        │
│                                                                  │
│  N (Norms):       4 heads                                        │
│    duty_pressure, secrecy_pressure, face_pressure, value_conflict│
│                                                                  │
│  D (Decision Policy): 5 heads                                    │
│    response_policy, reveal_decision, repair_strategy,            │
│    threat, control                                               │
└──────────────────────────────────────────────────────────────────┘
```

### Shared Architecture

```
dialogue text → Backbone → pooled hidden state → 28 ClassificationHeads
                    │         (last token or            │
                    │          mean pooling)     ┌──────┼──────┐
                    │                           ▼      ▼      ▼
                    │                      C_t head  R_t head  D_t head
                    │                      (n_class) (n_class) (n_class)
                    │
          Optional: SocialJEPAHead
          pooled → Predictor(h) → predict Z_{t+1} in embedding space
                   cosine_loss(pred, target_embedding)
```

All backbones use the same 28 heads, same loss weights (λ_R=λ_D=2.0), same checkpoint selection metric (`val/response_policy_f1`).

### Backbone Variants

| Backbone | Params | Type | Hidden Dim | Accuracy | κ |
|----------|--------|------|------------|----------|-----|
| Qwen3-1.7B | 1.7B | Pretrained + QLoRA | 1536 | **0.686** | **0.441** |
| GPT-SLM | 17.9M | From-scratch | 256 | 0.627 | 0.370 |
| MoE-SLM | ~25M | From-scratch MoE | 256 | 0.578 | — |
| Gemma-4-E2B | 16B/2B | Pretrained MoE + QLoRA | 1536 | 0.539 | — |
| Mamba-SLM | 15M | From-scratch SSM | 256 | 0.474 | 0.148 |

### JEPA Auxiliary Objective

```
                    ┌─────────────────────────────────┐
                    │        SocialJEPAHead            │
                    │                                  │
  pooled h_t ──────→│  HorizonPredictor(h_t) → pred    │
                    │                                  │
  future Z_{t+1} ──→│  SocialStateEmbedding(Z) → target│
                    │       (stop-gradient)            │
                    │                                  │
                    │  Loss = cosine(pred, target)     │
                    │       + β·variance_reg(pred)     │
                    └─────────────────────────────────┘
```

Shuffled-future placebo: same label marginals, broken temporal alignment. JEPA detects temporal structure (4.7× higher loss on shuffled futures) but does not improve downstream classification accuracy.

---

## How Tracks Connect

```
TRACK B (Encoders)              TRACK D (Latent Predictor)
┌──────────────────┐           ┌──────────────────────────┐
│ DistilBERT →     │           │ Backbone + 28 heads → Z_t│
│ OCEAN(5)+VAD(3)  │           │ (Qwen/Gemma/GPT/Mamba)   │
└────────┬─────────┘           └────────────┬─────────────┘
         │ 8D cond_vec                      │ predicted Z_t
         │                                  │ (28 labels)
         └──────────┬───────────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │   TRACK C (Generator) │
         │   cond_vec → prefix   │
         │   TinyLlama/Qwen/     │
         │   Gemma → response    │
         └──────────────────────┘
```

**Current state:** Track B→Track C connection is PLACEBO (OCEAN/VAD values unused). Track D predicts Z_t but Track C doesn't consume it yet. Bridge experiment shows Z_t predicts generation difficulty (reveal_decision: 1.02σ effect on PPL), justifying the connection but not yet implemented end-to-end.

---

## Training Configurations

| Model | Optimizer | LR | Schedule | Epochs | Batch | Selection Metric |
|-------|-----------|-----|----------|--------|-------|-----------------|
| Track A SLMs | AdamW | 3e-4 | cosine | 10-20 | 4×4 | val_ppl |
| Track B Encoders | AdamW | 2e-4 | cosine+linear | 15 | 4×8 | val_loss |
| Track C TinyLlama | AdamW | 2e-4 | cosine+linear | 3 | 2×8 | val_ppl |
| Track C Qwen | AdamW | 5e-5 | cosine | 3 | 1×8 | val_loss |
| Track D Latent | AdamW | 2e-4 | cosine | 5 | 1×32 | val/response_policy_f1 |
| Track D SLM Latent | AdamW | 3e-4 | cosine | 10 | 4×4 | val_response_policy_f1 |

All use seed 42 unless multi-seed experiment. All on single NVIDIA L20 (48GB).
