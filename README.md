# NPC Dialogue with Structured Social State

**[📖 For Everyone](#what-is-this) | [🔧 For AI Developers](#technical-architecture) | [🚀 Quick Start](#quick-start)**

---

## What Is This?

This project builds **smarter AI characters for games and simulations**. Instead of NPCs that repeat the same lines, we create characters that:

- **Remember relationships** — they track if they trust you, respect you, or are angry at you
- **Keep secrets** — they won't reveal sensitive information to strangers
- **React emotionally** — their dialogue reflects mood, stress, and social pressure
- **Stay consistent** — they don't randomly flip between friendly and hostile

### The Core Idea: AI That "Reads the Room"

Current chatbots just predict the next word. Our system adds a **"social brain"** that thinks about the conversation first:

```
Player: "Have you caught the spy yet?"
        ↓
[Social Brain Thinks]
- Trust level: Low (I don't know this person)
- Secret: Yes (spy location is hidden)
- Secrecy pressure: High
        ↓
NPC: "Orders from above. Move along, citizen."
```

See [`docs/education_intro.md`](docs/education_intro.md) for a beginner-friendly explanation.

---

## Quick Start

### Prerequisites
- Python 3.10+
- CUDA GPU (12GB+ VRAM recommended) or CPU fallback

### Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run a Test (No API Key Required)
```bash
# Generate 20 mock dialogue episodes (uses fake LLM responses)
python run_data_gen.py --config configs/data_gen.yaml --dry-run --n-episodes 20

# Run the smaller training scaffold smoke test
cd slm/npc_backend_scaffold
bash smoke_test.sh
```

### Full Pipeline (Requires API Key)
```bash
# 1. Generate training data with a teacher LLM
export AZURE_API_KEY=your_key_here
./scripts/run_data_gen.sh configs/data_gen_api.yaml

# 2. Train the latent state predictor
./scripts/run_train.sh latent

# 3. Train the response generator
./scripts/run_train.sh response

# 4. Evaluate
./scripts/run_eval.sh all configs/eval.yaml
```

---

## Technical Architecture

### Overview

This repository contains **two complementary systems**:

1. **`/slm/npc_backend_scaffold/`** — Modular personality + affect encoders with soft-prefix conditioning
2. **`/src/`** — Full latent state predictor with 29 classification heads and multi-stage training

### System 1: Scaffold (Lightweight, Modular)

For researchers wanting clean, reusable components — includes a **Small-LM benchmark suite** (GRU, AWD-LSTM, GPT, PrefixGPT, MoE, Mamba-like) for A/B comparison against fine-tuned LLMs:

```
┌─────────────────────────────────────────────────────────────┐
│  NPC Profile → Personality Encoder → OCEAN vector (cached)   │
│                (DistilBERT regression)                        │
├─────────────────────────────────────────────────────────────┤
│  Conversation → Affect Encoder → VAD vector (live)           │
│                (DistilBERT regression)                        │
├─────────────────────────────────────────────────────────────┤
│  OCEAN + VAD → Soft Prefix → Causal LM + LoRA → Response   │
│                (TinyLlama default)                          │
└─────────────────────────────────────────────────────────────┘
```

**Key Files:**
- `slm/npc_backend_scaffold/src/models/personality.py` — OCEAN trait encoder
- `slm/npc_backend_scaffold/src/models/affect.py` — VAD emotion encoder  
- `slm/npc_backend_scaffold/src/models/dialogue.py` — Conditional dialogue model
- `slm/npc_backend_scaffold/src/train/small_lm_architectures.py` — 6 from-scratch LM baselines
- `slm/npc_backend_scaffold/train_all.sh` — Full training orchestration

### System 2: Full Pipeline (Complete, Research-Grade)

For the complete structured latent state approach:

**Data Generation Pipeline:**
```
ScenarioBank (35 templates)
    ↓
StateInit → EpisodePlanner → TurnGenerator (10-step teacher LLM)
    ↓
CounterfactualAugmenter → Validator → Packager → Splitter
```

**Training Pipeline:**
```
Stage 1: Latent State Predictor (Qwen3-0.6B + LoRA)
         ↓
Stage 2: Response Generator (Qwen3-4B + QLoRA)  
         ↓
Stage 3: Joint Fine-tuning
```

**The Latent State Schema (Z_t):**

| Component | Fields | Classes |
|-----------|--------|---------|
| **C_t** | dialogue_act, tone, risk_type | 10 + 6 + 5 |
| **A_t** | valence, arousal, threat, control | 3 each |
| **M_t** | player_intent, knowledge, credibility | 9 + 4 + 3 |
| **R_t** | 6 stance dims × (level + delta) | 5 + 5 each |
| **N_t** | duty/secrecy/face/value pressure | 3 each |
| **D_t** | response_policy, reveal_decision, repair | 10 + 4 + 5 |

**Total: 29 classification targets per turn**

---

## Project Structure

```
llm_training/
├── configs/              # YAML configs for all pipeline stages
│   ├── data_gen.yaml     # Data generation settings
│   ├── train_latent.yaml # Stage 1 training config
│   ├── train_response.yaml
│   └── eval.yaml         # Evaluation thresholds
│
├── data/
│   ├── scenario_bank/    # 7 scenario types × 5 templates = 35 YAMLs
│   ├── world_contexts/   # Setting descriptions (medieval siege, etc.)
│   └── splits/           # Train/val/test JSONL files
│
├── prompts/              # Teacher LLM prompt templates
│   ├── label_C.txt       # C_t (dialogue act, tone, risk)
│   ├── label_A_M.txt     # A_t + M_t (affect + player model)
│   ├── label_R_N_D.txt   # R_t + N_t + D_t (stance + norms + policy)
│   └── response_generation.txt
│
├── src/
│   ├── data_gen/         # Episode generation pipeline
│   │   ├── scenario_bank.py
│   │   ├── episode_planner.py
│   │   ├── turn_generator.py
│   │   ├── labeler.py    # Structured label extraction
│   │   └── validator.py
│   │
│   ├── training/         # Model training
│   │   ├── model.py      # LatentStatePredictor (29 heads)
│   │   ├── train_latent.py
│   │   ├── train_response.py
│   │   └── train_joint.py
│   │
│   └── eval/             # Evaluation metrics
│       ├── eval_latent.py
│       └── eval_response.py
│
├── slm/npc_backend_scaffold/  # Modular scaffold system
│   ├── src/models/       # personality.py, affect.py, dialogue.py
│   ├── src/train/        # Training scripts for each component
│   ├── train_all.sh      # Full training orchestration
│   └── smoke_test.sh     # Quick verification
│
├── run_data_gen.py       # Entry point: data generation
├── run_train.py          # Entry point: training
├── run_eval.py           # Entry point: evaluation
└── scripts/              # Shell wrappers
```

---

## Usage Examples

### Data Generation
```bash
# Dry run with mock LLM (no API key)
./scripts/run_data_gen.sh configs/data_gen.yaml --dry-run --n-episodes 50

# Full generation with Azure OpenAI
export AZURE_API_KEY=xxx
export AZURE_ENDPOINT=https://...openai.azure.com/
./scripts/run_data_gen.sh configs/data_gen_api.yaml
```

### Training
```bash
# Stage 1: Train latent predictor
./scripts/run_train.sh latent --debug  # Uses small model, fast

# Stage 2: Train response generator  
./scripts/run_train.sh response

# Stage 3: Joint fine-tuning
./scripts/run_train.sh joint

# Or all stages
./scripts/run_train.sh all
```

### Evaluation
```bash
# Run all evaluation stages
./scripts/run_eval.sh all configs/eval.yaml

# Key metrics and targets:
# - response_policy_f1 >= 0.75
# - stance_delta_accuracy >= 0.70
# - secret_leakage_rate <= 0.05
```

### Interactive Inference
```bash
# Chat with a trained NPC
python src/inference/interactive.py --checkpoint checkpoints/latent_predictor_best/
```

### Scaffold Training (Modular System)
```bash
cd slm/npc_backend_scaffold

# Quick smoke test
bash smoke_test.sh

# Full training with hardware auto-detection
./train_all.sh --run-id my_experiment

# Individual components
python -m src.train.run_personality --config configs/personality.yaml
python -m src.train.run_affect --config configs/affect.yaml
python -m src.train.run_dialogue --config configs/dialogue.yaml
```

---

## Documentation

| Document | Audience | Content |
|----------|----------|---------|
| [`docs/education_intro.md`](docs/education_intro.md) | General public | Non-technical introduction to social state AI |
| [`project_overview.md`](project_overview.md) | Researchers | System design and rationale |
| [`SUMMARY.md`](SUMMARY.md) | Developers | Detailed repository map and current status |
| [`schema.md`](schema.md) | ML Engineers | Full latent state schema specification |
| [`slm/npc_backend_scaffold/README.md`](slm/npc_backend_scaffold/README.md) | Engineers | Modular system docs |

---

## Evaluation Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| response_policy_f1 | ≥ 0.75 | Classification accuracy of NPC response type |
| stance_delta_accuracy | ≥ 0.70 | Correctly predicting relationship changes |
| secret_leakage_rate | ≤ 0.05 | NPCs shouldn't reveal secrets when pressured |
| contradiction_rate | ≤ 0.08 | Responses shouldn't contradict earlier statements |
| valence_ccc | ≥ 0.70 | Concordance correlation for emotional valence |
| router_fpr | ≤ 0.15 | False positive rate for selective routing |

---

## Development Status

| Component | Status |
|-----------|--------|
| Scenario bank (35 templates) | ✅ Complete |
| Data generation pipeline | ✅ Complete (449 episodes generated) |
| Stage 1 training | ✅ Implemented |
| Stage 2 training | ✅ Implemented |
| Stage 3 joint training | ✅ Implemented |
| Evaluation pipeline | ✅ Complete |
| MLflow tracking | ✅ Integrated |
| Interactive inference | ✅ Available |

---

## Hardware Requirements

| Task | Minimum | Recommended |
|------|---------|-------------|
| Data generation | CPU | GPU (for local teacher LLM) |
| Stage 1 training | 8GB VRAM | 12GB VRAM |
| Stage 2/3 training | 12GB VRAM | 16GB+ VRAM |
| Scaffold training | 4GB VRAM | 8GB+ VRAM |

---

## License

Research and educational use. See LICENSE file for details.

---

## Citation

If you use this work in research, please cite:

```bibtex
@software{npc_social_state_2024,
  title = {Structured Latent-State NPC Dialogue System},
  year = {2024},
  url = {https://github.com/yourorg/llm_training}
}
```