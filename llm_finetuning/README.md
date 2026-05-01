# LLM Fine-Tuning: Latent State Predictor

Fine-tunes pre-trained LLMs (Qwen3 family) into a **structured social-state dialogue model** with 29 classification heads and multi-stage training.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     LatentStatePredictor                         │
│                                                                  │
│  Input: dialogue context text                                    │
│     ↓                                                            │
│  ┌──────────────────────────────────────┐                       │
│  │  Qwen3 Backbone (0.6B–4B) + LoRA     │                       │
│  │  - 4-bit QLoRA quantization          │                       │
│  │  - Pooling: last / mean / attention   │                       │
│  │  - hidden_size = 896–2560            │                       │
│  └──────────────────────────────────────┘                       │
│     ↓                                                            │
│  ┌──────────────────────────────────────┐                       │
│  │  29 Classification Heads             │                       │
│  │  Linear(256)→GELU→Dropout→Linear     │                       │
│  └──────────────────────────────────────┘                       │
│     ↓                                                            │
│  Output: 29 label predictions per turn                          │
└──────────────────────────────────────────────────────────────────┘
```

### Backbone Models

| Model | Params | Hidden | VRAM (4bit) | Use |
|-------|--------|--------|-------------|-----|
| `Qwen/Qwen3-0.6B` | 0.6B | 896 | ~2 GB | Debug/fast iteration |
| `Qwen/Qwen3-1.7B` | 1.7B | 2048 | ~4 GB | Debug/fast iteration |
| `Qwen/Qwen3-4B` | 4B | 2560 | ~6 GB | **Production** |

### Quantization

| Setting | BitsAndBytes Config |
|---------|---------------------|
| `4bit` (default) | `nf4`, double quantization, compute_dtype=bfloat16 |
| `8bit` | 8-bit linear quantization |
| none | Full precision (16GB+ VRAM) |

### LoRA Configuration

| Parameter | Stage 1 (Latent) | Stage 2 (Response) | Stage 3 (Joint) |
|-----------|-----------------|--------------------|--------------------|
| **r** | 16 | 32 | 32 |
| **alpha** | 32 | 64 | 64 |
| **dropout** | 0.05 | 0.05 | 0.05 |
| **target_modules** | q,k,v,o,gate,up,down | q,k,v,o,gate,up,down | q,k,v,o,gate,up,down |

### Pooling Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `last` | Last non-padding token hidden state | **Default** — best for autoregressive models |
| `mean` | Mean pooling over all non-padding tokens | Better for bidirectional context |
| `attention` | Learnable attention-weighted pooling | Most expressive, slower |

---

## Latent State Schema (29 Targets)

### C_t — Contextual Analysis (3 heads)
| Field | Classes | Multi-label | Description |
|-------|---------|-------------|-------------|
| `dialogue_act` | 10 | ✅ Yes | ask, accuse, threaten, flatter, apologize, negotiate, joke, confess, probe, command |
| `tone` | 6 | ❌ No | warm, neutral, confrontational, sarcastic, fearful, evasive |
| `risk_type` | 5 | ❌ No | none, secret-risk, face-risk, status-risk, conflict-risk |

### A_t — Affective Appraisal (4 heads)
| Field | Classes | Description |
|-------|---------|-------------|
| `valence` | 3 | negative, neutral, positive |
| `arousal` | 3 | low, medium, high |
| `threat` | 3 | low, medium, high |
| `control` | 3 | low, medium, high |

### M_t — Player Mental Model (3 heads)
| Field | Classes | Description |
|-------|---------|-------------|
| `player_intent` | 9 | seek-info, trap, bond, manipulate, test, persuade, intimidate, probe, negotiate |
| `player_knowledge` | 4 | unaware, partial, informed, knows-secret |
| `player_credibility` | 3 | low, medium, high |

### R_t — Relational Stance (12 heads — 6 dims × 2)
| Dimension | Level (5 classes) | Delta (5 classes) |
|-----------|-------------------|-------------------|
| `affection` | VL, L, N, H, VH | --, -, 0, +, ++ |
| `respect` | VL, L, N, H, VH | --, -, 0, +, ++ |
| `dominance` | VL, L, N, H, VH | --, -, 0, +, ++ |
| `familiarity` | VL, L, N, H, VH | --, -, 0, +, ++ |
| `trust` | VL, L, N, H, VH | --, -, 0, +, ++ |
| `obligation` | VL, L, N, H, VH | --, -, 0, +, ++ |

### N_t — Norm/Value Constraints (4 heads)
| Field | Classes | Description |
|-------|---------|-------------|
| `duty_pressure` | 3 | low, medium, high |
| `secrecy_pressure` | 3 | low, medium, high |
| `face_pressure` | 3 | low, medium, high |
| `value_conflict` | 3 | none, mild, strong |

### D_t — Response Policy (3 heads)
| Field | Classes | Description |
|-------|---------|-------------|
| `response_policy` | 10 | answer, partial, withhold, deflect, challenge, soothe, test, threaten, negotiate, clarify |
| `reveal_decision` | 4 | none, hint, partial, full |
| `repair_strategy` | 5 | none, soften, apologize, clarify, redirect |

---

## Training Pipeline

### Stage 1: Latent State Predictor
```bash
./scripts/pipeline.sh train latent
```

**Objective:** Predict 29 social-state labels from dialogue context.

| Hyperparameter | Value |
|----------------|-------|
| Learning rate (backbone) | 2×10⁻⁴ |
| Learning rate (heads) | 4×10⁻⁴ (2× backbone) |
| LR schedule | Cosine + 5% warmup |
| Epochs | 5 |
| Max sequence length | 512 |
| Batch size (effective) | 32 (1 × 32 grad_accum) |
| Label smoothing | 0.1 |
| Weighted sampler | ✅ (oversampling minority classes) |
| Gradient checkpointing | ✅ |
| Optimizer | AdamW (β₁=0.9, β₂=0.999) |
| Weight decay | 0.01 |
| Max grad norm | 1.0 |

**Loss weights (λ per component group):**

| Group | λ | Rationale |
|-------|---|-----------|
| C (context) | 1.0 | Baseline |
| A (affect) | 1.0 | Core signal |
| M (mental model) | 1.5 | ToM is harder |
| R (stance) | 2.0 | **Most important** for NPC control |
| N (norms) | 1.0 | Auxiliary |
| D (policy) | 2.0 | **Most important** for behavior |

**Class weights:** Inverse-frequency per head, clamped to [0.2, 5.0].

### Stage 2: Response Generator (SFT)
```bash
./scripts/pipeline.sh train response
```

**Objective:** Generate NPC dialogue conditioned on gold latent state.

| Hyperparameter | Value |
|----------------|-------|
| Learning rate | 1×10⁻⁴ |
| LR schedule | Cosine |
| Epochs | 3 |
| Max sequence length | 1024 |
| Batch size (effective) | 32 (1 × 32 grad_accum) |
| Conditioning mode | Gold labels from teacher LLM |
| Gradient checkpointing | ✅ |

### Stage 3: Joint Fine-Tuning
```bash
./scripts/pipeline.sh train joint
```

**Objective:** Train latent prediction + response generation jointly with consistency loss.

| Hyperparameter | Value |
|----------------|-------|
| Learning rate | 5×10⁻⁵ |
| Epochs | 3 |
| Max sequence length | 1024 |
| Batch size (effective) | 8 (1 × 8 grad_accum) |
| Initialization | Stage 1 backbone + Stage 2 adapter |

**Joint loss:**

$$L = L_{heads} + \lambda_Y \cdot L_{lm} + \lambda_{consistency} \cdot L_{consistency}$$

| Loss term | λ | Description |
|-----------|---|-------------|
| L_heads | 1.0 (per-group) | 29-head classification loss |
| L_lm | 1.0 | Causal LM cross-entropy |
| L_consistency | 0.5 | Penalizes high-secrecy + full-reveal |


## Data Pipeline

```
┌──────────────────┐     ┌───────────────┐     ┌─────────────────┐
│  ScenarioBank     │────▶│ EpisodePlanner │────▶│ TurnGenerator    │
│  35 templates     │     │ NPC profile    │     │ Teacher LLM: 10  │
│  7 scenario types │     │ Story arc      │     │ calls per turn   │
└──────────────────┘     └───────────────┘     └────────┬────────┘
                                                        │
    ┌───────────────────────────────────────────────────┘
    ▼
┌──────────────────┐     ┌───────────────┐     ┌─────────────────┐
│ Validator         │────▶│ Packager       │────▶│ Splitter         │
│ - Schema checks   │     │ - Context str  │     │ 80/10/10 split   │
│ - Secret leakage  │     │ - Latent state │     │ Stratified by     │
│ - Min/max tokens  │     │ - SFT format   │     │ scenario type    │
└──────────────────┘     └───────────────┘     └─────────────────┘
                                                        │
    ┌───────────────────────────────────────────────────┘
    ▼
┌──────────────────────────────────────────────────────┐
│  Counterfactual Augmenter                             │
│  - Trust: high ↔ low                                  │
│  - Secrecy: high ↔ low                                │
│  - Deception: sincere ↔ deceptive                     │
│  - Disclosure: withhold ↔ reveal                      │
└──────────────────────────────────────────────────────┘
```

### Scenario Types
| Type | Template Count | Description |
|------|---------------|-------------|
| `secret_extraction` | 5 | Player tries to extract secrets |
| `apology_repair` | 5 | NPC must apologize for past actions |
| `alliance_negotiation` | 5 | Player proposes an alliance |
| `rumor_confrontation` | 5 | Player confronts NPC about rumors |
| `threat_escalation` | 5 | Conflict threatens to turn violent |
| `trust_building` | 5 | Building trust with suspicious NPC |
| `deception_detection` | 5 | NPC tries to detect player lies |

### Counterfactual Dimensions
| Variable | Flip | Tests |
|----------|------|-------|
| `trust_level` | high ↔ low | State-sensitive policy |
| `secrecy_pressure` | high ↔ low | Secret-keeping behavior |
| `player_intent` | bond ↔ manipulate | Social perception |
| `reveal_decision` | none ↔ full | Disclosure control |
| `value_conflict` | none ↔ strong | Norm compliance |

---

## MLflow Tracking

Both LLM fine-tuning and SLM training log to a **single shared `mlruns/`** at the project root. All experiments, metrics, parameters, and artifacts are visible in one dashboard.

```bash
mlflow ui --backend-store-uri mlruns
# → http://localhost:5000
```

### Tracked experiments

| Experiment | System | What's logged |
|-----------|--------|---------------|
| `social_state_data_generation` | LLM | Data manifest, episode counts, hashes |
| `latent_state_prediction` | LLM | Train/val loss, per-head F1, accuracy, class weights (Stage 1 + 3) |
| `response_generation` | LLM | LM loss, ROUGE-L, secret leakage (Stage 2) |
| `routing_and_policy_eval` | LLM | All eval metrics, confusion matrices, sample generations |
| `personality_encoder` | SLM | MSE loss, R² per OCEAN trait, params |
| `affect_encoder` | SLM | CCC, MSE, MAE, R² per VAD dimension |
| `small_lm` | SLM | PPL, loss, lr, grad_norm per step/epoch |
| `dialogue_model` | SLM | Train loss, val PPL, LoRA params (TinyLlama + Gemma) |
| `slm_eval` | SLM | PPL, BLEU-1/2, Distinct-1/2 per architecture |

> **Note:** MLflow 2.20+ deprecated the filesystem backend. To future-proof, add to `.env`:
> ```
> MLFLOW_TRACKING_URI=sqlite:///mlflow.db
> ```

## Evaluation

```bash
./scripts/pipeline.sh eval all
```

### Metrics

| Metric | Target | Measures |
|--------|--------|----------|
| `response_policy_f1` | ≥ 0.75 | Classification accuracy of NPC response type |
| `stance_delta_accuracy` | ≥ 0.70 | Correctly predicting relationship changes |
| `secret_leakage_rate` | ≤ 0.05 | NPCs revealing secrets when told not to |
| `contradiction_rate` | ≤ 0.08 | Responses contradicting earlier statements |
| `rouge_l` | maximize | Lexical similarity to gold responses |
| `routing_precision` | maximize | Router correctly identifies slow-path turns |
| `router_fpr` | ≤ 0.15 | False positive rate for routing |

### Slow-Path Routing Triggers
| Condition | Trigger |
|-----------|---------|
| `value_conflict = strong` | Slow path |
| `response_policy ∈ {threaten, negotiate}` | Slow path |
| `secrecy_pressure = high AND reveal ≠ none` | Slow path |

---

## Usage

```bash
# Quick test (no API key)
cd llm_finetuning
PYTHONPATH=. python run_data_gen.py --config configs/data_gen.yaml --dry-run --n-episodes 20

# Full training via root pipeline
./scripts/pipeline.sh data-gen               # Generate data
./scripts/pipeline.sh train latent --debug    # Test stage 1
./scripts/pipeline.sh train all               # Full 3-stage training
./scripts/pipeline.sh eval all                # Evaluate

# Interactive chat
cd llm_finetuning
PYTHONPATH=. python src/inference/interactive.py \
    --checkpoint ../checkpoints/joint_model_best/ \
    --base_model Qwen/Qwen3-4B \
    --scenario ../data/scenario_bank/secret_extraction.yaml \
    --npc guard_captain
```

### Interactive Inference

```bash
python src/inference/interactive.py \
    --checkpoint checkpoints/joint_model_best \
    --base_model Qwen/Qwen3-4B \
    --scenario data/world_contexts/oakhaven_siege.yaml \
    --npc commander_vance \
    --quantization 4bit
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `--checkpoint` | Path to trained joint model (contains `backbone/` and `heads.pt`) |
| `--base_model` | HuggingFace model used during training (default: `Qwen/Qwen3-4B`) |
| `--scenario` | Path to scenario YAML defining world and NPCs |
| `--npc` | NPC ID to converse with (must exist in scenario) |
| `--quantization` | `4bit`, `8bit`, or `none` (default: `4bit`) |

**How it works:**

1. **Context Construction** — Builds prompt from scenario, NPC profile, and conversation history.
2. **Latent Prediction** — Classification heads predict the NPC's internal state from current context.
3. **Response Generation** — Predicted latent state is appended to context; fine-tuned backbone generates the response.
4. **State Update** — Conversation history and NPC stance are updated for the next turn.

### Hardware Requirements

| Task | VRAM | RAM | Notes |
|------|------|-----|-------|
| Data generation (API) | — | 4 GB | Uses remote LLM API |
| Data generation (local) | 16 GB | 8 GB | Gemma 4B teacher by default; Qwen3-0.6B test config available |
| Stage 1 training | 8 GB | 16 GB | 0.6B model + LoRA |
| Stage 2/3 training | 12 GB | 24 GB | 4B model + QLoRA |
| Inference | 8 GB | 8 GB | 4bit quantized |

### File Structure

```
llm_finetuning/
├── configs/                   # YAML configs
│   ├── data_gen.yaml          # Data generation settings
│   ├── data_gen_api.yaml      # API-based teacher config
│   ├── train_latent.yaml      # Stage 1: latent predictor
│   ├── train_response.yaml    # Stage 2: response SFT
│   ├── train_joint.yaml       # Stage 3: joint fine-tuning
│   └── eval.yaml              # Evaluation thresholds
├── prompts/                   # Teacher LLM prompt templates
│   ├── label_C.txt            # C_t labeling
│   ├── label_A_M.txt          # A_t + M_t labeling
│   ├── label_R_N_D.txt        # R_t + N_t + D_t labeling
│   └── response_generation.txt
├── src/                       # Source code
│   ├── data_gen/              # Episode generation pipeline (7 modules)
│   ├── training/              # Models, datasets, training loops (6 modules)
│   ├── eval/                  # Evaluation metrics (3 modules)
│   ├── packaging/             # Data packaging and splitting (2 modules)
│   ├── inference/             # Interactive chat
│   └── mlflow_utils.py        # MLflow tracking helpers
├── scripts/                   # Utility scripts
│   ├── check_gpu.py           # GPU availability check
│   ├── analyze_data_quality.py
│   ├── clean_labels.py        # Fix corrupted teacher outputs
│   └── visualize_dialogues.py # Human-readable dialogue transcripts
├── tests/                     # Unit tests
├── run_data_gen.py            # Entry: data generation
├── run_train.py               # Entry: training
└── run_eval.py                # Entry: evaluation
```
