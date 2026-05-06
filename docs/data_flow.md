# System Data Flow Diagrams

## Track A: From-Scratch Small Language Models

### GPT, MoE, Mamba-like (unconditioned)

```
┌─────────────────────────────────────────────────────────┐
│ INPUT                                                   │
│                                                         │
│  data/dialogue/train.txt (16,905 lines)                 │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Player approaches the guard and asks about the   │   │
│  │ siege. NPC: The guard responds in a formal       │   │
│  │ manner, deflecting the question about troop      │   │
│  │ movements.                                       │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  Tokenized with tiktoken (GPT-2 encoding, vocab=256)    │
│  Sequence length: 256 tokens                            │
│  Batch size: 32                                         │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ MODEL                                                   │
│                                                         │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐            │
│  │ Embedding │ → │ 6× Transf │ → │  LM Head │            │
│  │   256-dim │   │  /Mamba   │   │  → vocab │            │
│  └──────────┘   │  /MoE     │   └──────────┘            │
│                 └──────────┘                             │
│  Parameters: 15-16M                                     │
│  Training: next-token prediction (cross-entropy)        │
│  Epochs: 20                                             │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ OUTPUT                                                  │
│                                                         │
│  val_ppl (lower = better):                              │
│    MoE:         42.07  ← best                           │
│    PrefixGPT:   44.54                                   │
│    GPT:         45.32                                   │
│    Mamba-like:  53.25                                   │
│                                                         │
│  Checkpoint: artifacts/small_lm/<run_id>/best_model.pt  │
└─────────────────────────────────────────────────────────┘
```

### PrefixGPT (conditioned)

```
┌─────────────────────────────────────────────────────────┐
│ INPUT                                                   │
│                                                         │
│  Same dialogue text AS ABOVE                            │
│                +                                        │
│  OCEAN vector (5-dim):  [0.35, 0.25, -0.12, 0.08, 0.15]│
│  VAD vector (3-dim):    [0.42, 0.18, 0.55]             │
│                                                         │
│  Concatenated → 8-dim condition vector                  │
│  Prepended as prefix tokens to input                    │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ MODEL                                                   │
│                                                         │
│  ┌──────────┐   ┌──────────────┐   ┌──────────┐        │
│  │ [prefix] │   │              │   │          │        │
│  │  +embed  │ → │  6× GPT      │ → │ LM Head  │        │
│  │          │   │  transformer  │   │  → vocab │        │
│  └──────────┘   └──────────────┘   └──────────┘        │
│                                                         │
│  Prefix tokens are learned embeddings projected         │
│  from the 8-dim OCEAN+VAD vector                        │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ OUTPUT                                                  │
│                                                         │
│  val_ppl: 44.54                                         │
│  (small improvement over GPT's 45.32 — prefix helps)    │
└─────────────────────────────────────────────────────────┘
```

---

## Track B: Conditioning Encoders

### Personality Encoder (OCEAN)

```
┌─────────────────────────────────────────────────────────┐
│ INPUT                                                   │
│                                                         │
│  data/npc_profiles.csv (414 NPCs)                       │
│  ┌──────────────────────────────────────────────────┐   │
│  │ npc_id,profile_text                              │   │
│  │ guard_001,"A formal guard who values duty,       │   │
│  │           secrecy, and protecting the citadel."   │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  Tokenized with distilbert-base-uncased                  │
│  Max length: 512 tokens                                 │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ MODEL                                                   │
│                                                         │
│  ┌──────────────┐   ┌──────────┐   ┌──────────────┐    │
│  │ DistilBERT   │ → │  Pooler  │ → │ Linear(768,5)│    │
│  │ (66M params) │   │          │   │ → [O,C,E,A,N] │    │
│  └──────────────┘   └──────────┘   └──────────────┘    │
│                                                         │
│  Loss: MSE on 5 OCEAN dimensions                        │
│  Epochs: 15 (best at 4)                                 │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ OUTPUT                                                  │
│                                                         │
│  val_f1: 0.678    val_mse: 0.248                        │
│                                                         │
│  Example output for "guard_001":                        │
│    O: 0.35   C: 0.25   E: -0.12   A: 0.08   N: 0.15   │
│    (open)   (conscientious) (extrovert) (agreeable)     │
│                                                         │
│  → Saved to personality_cache.jsonl (414 entries)       │
│    Used by Track C dialogue model                       │
└─────────────────────────────────────────────────────────┘
```

### Affect Encoder (VAD)

```
┌─────────────────────────────────────────────────────────┐
│ INPUT                                                   │
│                                                         │
│  data/affect/train.csv (500 synthetic samples)          │
│  ┌──────────────────────────────────────────────────┐   │
│  │ text,valence,arousal,dominance                    │   │
│  │ "Player asks about the siege...",0.42,0.18,0.55  │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  Tokenized with distilbert-base-uncased                  │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ MODEL                                                   │
│                                                         │
│  ┌──────────────┐   ┌──────────┐   ┌──────────────┐    │
│  │ DistilBERT   │ → │  Pooler  │ → │ Linear(768,3)│    │
│  │ (66M params) │   │          │   │ → [V, A, D]   │    │
│  └──────────────┘   └──────────┘   └──────────────┘    │
│                                                         │
│  Loss: MSE + CCC on 3 VAD dimensions                    │
│  Epochs: 15 (best at 13)                                │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ OUTPUT                                                  │
│                                                         │
│  val_ccc: 0.559    val_mse: 0.005                       │
│                                                         │
│  Example output for a tense exchange:                   │
│    V: 0.42   A: 0.78   D: 0.30                         │
│    (negative valence, high arousal = agitated)          │
│                                                         │
│  → Used live by Track C dialogue model at inference     │
└─────────────────────────────────────────────────────────┘
```

---

## Track C: Response Generation

### ConditionalDialogue (OCEAN + VAD conditioned)

```
┌─────────────────────────────────────────────────────────┐
│ INPUT                                                   │
│                                                         │
│  Conversation history:                                  │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Player: "You there! What do you know about the   │   │
│  │          spy in the citadel?"                     │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  NPC profile → OCEAN vector (from cache):               │
│    guard_001 → [0.35, 0.25, -0.12, 0.08, 0.15]        │
│                                                         │
│  Dialogue context → VAD vector (live from encoder):     │
│    [0.42, 0.78, 0.30]                                   │
│                                                         │
│  Concatenated → 8-dim condition vector                  │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ MODEL                                                   │
│                                                         │
│  ┌──────────────┐                                       │
│  │ 8-dim cond   │ → SoftPrefix → prefix embeddings      │
│  │ [OCEAN+VAD]  │                                       │
│  └──────────────┘                                       │
│         +                                               │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────┐    │
│  │ TinyLlama    │ → │ 22× Transf   │ → │ LM Head  │    │
│  │ 1.1B params  │   │  + LoRA      │   │          │    │
│  └──────────────┘   └──────────────┘   └──────────┘    │
│                                                         │
│  Prefix prepended to input embeddings                   │
│  LoRA: r=8, alpha=16, target_modules=all-linear         │
│  Epochs: 5 (best at 2)                                  │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ OUTPUT                                                  │
│                                                         │
│  val_ppl: 2.90                                          │
│                                                         │
│  Generated response:                                    │
│  "The citadel's security is not a matter for casual     │
│   discussion. State your business or move along."       │
│                                                         │
│  Checkpoint: artifacts/dialogue_model/<run>/best_model  │
└─────────────────────────────────────────────────────────┘
```

### TinyLlama SFT (baseline, no conditioning)

```
┌─────────────────────────────────────────────────────────┐
│ INPUT                                                   │
│                                                         │
│  Same conversation history                              │
│  NO personality cache, NO affect encoder                │
│                                                         │
│  Just: "[dialogue context] → NPC response"              │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ MODEL                                                   │
│                                                         │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────┐    │
│  │ TinyLlama    │ → │ 22× Transf   │ → │ LM Head  │    │
│  │ 1.1B params  │   │  + LoRA      │   │          │    │
│  └──────────────┘   └──────────────┘   └──────────┘    │
│                                                         │
│  Standard supervised fine-tuning (no prefix)            │
│  Epochs: 3 (best at 1)                                  │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ OUTPUT                                                  │
│                                                         │
│  val_ppl: 3.30                                          │
│                                                         │
│  → 12.1% WORSE than ConditionalDialogue                 │
│  → Proves conditioning helps                            │
└─────────────────────────────────────────────────────────┘
```

### Gemma 4 E2B + QLoRA

```
┌─────────────────────────────────────────────────────────┐
│ INPUT                                                   │
│                                                         │
│  data/dialogue/from_gen_train.jsonl                     │
│  ┌──────────────────────────────────────────────────┐   │
│  │ {"npc_id": "guard_225",                          │   │
│  │  "npc_profile": "A methodical apothecary who     │   │
│  │   values knowledge, profit, discretion...",       │   │
│  │  "dialogue_context": [{"speaker":"player",        │   │
│  │    "text":"Do you have anything for headaches?"}],│   │
│  │  "target_response": "I might have a tincture..."} │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  Formatted via Gemma chat template:                     │
│  "<start_of_turn>user\n[NPC profile]\n\n[query]..."     │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ MODEL                                                   │
│                                                         │
│  ┌────────────────────────────────────────────────┐     │
│  │ Gemma 4 E2B (16B total, 2B active MoE)         │     │
│  │ 4-bit QLoRA + LoRA (r=16, alpha=16)            │     │
│  │ trainable: 37.9M / 5.1B params (0.74%)         │     │
│  │ target_modules: "all-linear"                    │     │
│  └────────────────────────────────────────────────┘     │
│                                                         │
│  Trained with SFTTrainer (trl)                          │
│  Requires: PYTORCH_CUDA_ALLOC_CONF=expandable_segments  │
│  Epochs: 1                                              │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ OUTPUT                                                  │
│                                                         │
│  val_ppl: 16.24 (1 epoch only — still converging)       │
│  Train loss: 0.99 → 0.58 (dropping)                    │
│  Token accuracy: 72% → 83% (climbing)                  │
│                                                         │
│  Checkpoint: artifacts/gemma4_e2b/<run>/best_model       │
└─────────────────────────────────────────────────────────┘
```

---

## Track D: Structured LLM Pipeline (Qwen3)

### Stage 1: Latent State Predictor

```
┌─────────────────────────────────────────────────────────┐
│ INPUT                                                   │
│                                                         │
│  data/splits/train.jsonl (generated turns with labels)  │
│  ┌──────────────────────────────────────────────────┐   │
│  │ {                                                │   │
│  │   "context": "<scene>Setting: temple_courtyard   │   │
│  │    NPC Role: priest\nGoals: protect_flock...</scene>│  │
│  │    \n\nPlayer: Have you caught the spy yet?",    │   │
│  │   "labels": {                                    │   │
│  │     "dialogue_act": 3,       // "question"       │   │
│  │     "tone": 1,               // "formal"         │   │
│  │     "valence": 2,            // "negative"       │   │
│  │     "trust_level": 0,        // "very_low"       │   │
│  │     "secrecy_pressure": 2,   // "high"           │   │
│  │     "response_policy": 1,    // "deflect"        │   │
│  │     ... (29 heads total)                         │   │
│  │   }                                              │   │
│  │ }                                                │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  Tokenized: max 1024 tokens                             │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ MODEL                                                   │
│                                                         │
│  ┌──────────────┐   ┌──────────────┐                    │
│  │ Qwen3-1.7B   │ → │ 28× Transf   │                    │
│  │ + LoRA       │   │  (debug)     │                    │
│  └──────────────┘   └──────┬───────┘                    │
│                            │                            │
│              ┌─────────────┼─────────────┐              │
│              ▼             ▼             ▼              │
│          ┌──────┐     ┌──────┐      ┌──────┐           │
│          │Head 1│     │Head 2│ ...  │Head29│           │
│          │dialog│     │ tone │      │repair│           │
│          │_act  │     │      │      │      │           │
│          └──┬───┘     └──┬───┘      └──┬───┘           │
│             │            │             │                │
│             ▼            ▼             ▼                │
│      "question"    "formal"      "none_needed"          │
│                                                         │
│  29 parallel classification heads                       │
│  Loss: sum of cross-entropy across all heads            │
│  Epochs: 5 (best at 3)                                  │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ OUTPUT                                                  │
│                                                         │
│  Per-epoch metrics (best at epoch 3):                   │
│    response_policy_f1: 0.474                            │
│    mean_accuracy:      0.704                            │
│    trust_delta_f1:     0.471                            │
│                                                         │
│  Checkpoint: checkpoints/latent_predictor_best          │
│                                                         │
│  Feeds into → Stage 2 response generator                │
└─────────────────────────────────────────────────────────┘
```

### Stage 2: Response Generator (SFT)

```
┌─────────────────────────────────────────────────────────┐
│ INPUT                                                   │
│                                                         │
│  data/splits/train_sft.jsonl                            │
│  ┌──────────────────────────────────────────────────┐   │
│  │ {                                                │   │
│  │   "input": "<scene>...</scene>\n\nPlayer: ...",  │   │
│  │   "target": "You come at a troubling time..."    │   │
│  │ }                                                │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  Tokenized as: input + "\n" + target                    │
│  Labels: target only (input tokens masked with -100)    │
│  Max sequence length: 2048 tokens                       │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ MODEL                                                   │
│                                                         │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────┐    │
│  │ Qwen3-1.7B   │ → │ 28× Transf   │ → │ LM Head  │    │
│  │ + LoRA       │   │  (debug)     │   │          │    │
│  └──────────────┘   └──────────────┘   └──────────┘    │
│                                                         │
│  Standard SFT: next-token prediction on target          │
│  Epochs: 3                                              │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ OUTPUT                                                  │
│                                                         │
│  val_loss: 0.037 (training), 0.044 (validation)         │
│                                                         │
│  Checkpoint: checkpoints/response_generator_best        │
│                                                         │
│  Feeds into → Stage 3 joint training                    │
└─────────────────────────────────────────────────────────┘
```

### Stage 3: Joint Model

```
┌─────────────────────────────────────────────────────────┐
│ INPUT                                                   │
│                                                         │
│  BOTH data sources COMBINED:                            │
│  ┌──────────────────────────────────────────────────┐   │
│  │ SFT data (like Stage 2)  +  Labeled data (Stg 1) │   │
│  │                                                  │   │
│  │ Each turn has:                                   │   │
│  │   • context (dialogue history + scene)            │   │
│  │   • target (NPC response)                        │   │
│  │   • labels (29 social state dimensions)          │   │
│  └──────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ MODEL                                                   │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │              Qwen3-1.7B + LoRA                   │   │
│  │                                                  │   │
│  │     ┌─────────────┐                              │   │
│  │     │   Shared    │                              │   │
│  │     │  Backbone   │                              │   │
│  │     └──────┬──────┘                              │   │
│  │            │                                      │   │
│  │   ┌────────┴────────┐                            │   │
│  │   ▼                 ▼                            │   │
│  │ ┌──────────┐  ┌──────────┐                       │   │
│  │ │ 29-Head  │  │   LM     │                       │   │
│  │ │Predictor │  │   Head   │                       │   │
│  │ │(Stage 1) │  │ (Stage 2)│                       │   │
│  │ └────┬─────┘  └────┬─────┘                       │   │
│  │      │             │                             │   │
│  │      ▼             ▼                             │   │
│  │  Cross-entropy  Cross-entropy                    │   │
│  │  (29 heads)     (next token)                     │   │
│  │      │             │                             │   │
│  │      └──────┬──────┘                             │   │
│  │             ▼                                    │   │
│  │     L_joint = L_heads + L_lm                     │   │
│  └──────────────────────────────────────────────────┘   │
│                                                         │
│  Trains both objectives simultaneously                  │
│  Gradients from both heads flow through shared backbone │
│  Epochs: 3                                              │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│ OUTPUT                                                  │
│                                                         │
│  Epoch 3: train=4.30, val=6.47                          │
│                                                         │
│  Checkpoint: checkpoints/joint_model_best               │
│                                                         │
│  The joint model can both:                              │
│    1. Predict social state from dialogue context        │
│    2. Generate socially-conditioned NPC responses       │
└─────────────────────────────────────────────────────────┘
```

---

## Full System Integration

```
                        ┌──────────────────┐
                        │   NPC Profiles   │
                        │   (414 entries)  │
                        └────────┬─────────┘
                                 │
                                 ▼
                        ┌──────────────────┐
                        │   Personality    │
                        │   Encoder (B)    │
                        │   → OCEAN [5]    │
                        └────────┬─────────┘
                                 │
                    ┌────────────┴────────────┐
                    │  Personality Cache      │
                    │  (precomputed, .jsonl)  │
                    └────────────┬────────────┘
                                 │
┌────────────────────┐           │           ┌────────────────────┐
│  Dialogue History  │           │           │  Dialogue History  │
│  (conversation)    │           │           │  + Scene Context   │
└────────┬───────────┘           │           └────────┬───────────┘
         │                       │                    │
         ▼                       │                    ▼
┌────────────────────┐           │           ┌────────────────────┐
│  Affect Encoder    │           │           │  Qwen3 Latent      │
│  → VAD [3]         │           │           │  Predictor (D1)    │
└────────┬───────────┘           │           │  → 29-dim state    │
         │                       │           └────────┬───────────┘
         │                       │                    │
         └───────────┬───────────┘                    │
                     │                                │
                     ▼                                ▼
         ┌──────────────────────┐        ┌──────────────────────┐
         │  ConditionalDialogue │        │  Qwen3 Response Gen  │
         │  (Track C)           │        │  (Track D2)          │
         │  [OCEAN+VAD] → text  │        │  [state+ctx] → text  │
         └──────────┬───────────┘        └──────────┬───────────┘
                    │                                │
                    ▼                                ▼
         ┌──────────────────────┐        ┌──────────────────────┐
         │  NPC Response        │        │  NPC Response        │
         │  "The citadel's      │        │  "You come at a      │
         │   security is not    │        │   troubling time..."  │
         │   a matter for       │        │                      │
         │   casual discussion" │        │                      │
         └──────────────────────┘        └──────────────────────┘
```
