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
cd llm_finetuning
PYTHONPATH=. python run_data_gen.py --config configs/data_gen.yaml --dry-run --n-episodes 20

# Run the SLM scaffold smoke test
cd slm_training
bash smoke_test.sh
```

### Full Pipeline (Requires API Key)
```bash
# All operations via unified pipeline script
./scripts/pipeline.sh data-gen                         # Generate training data
./scripts/pipeline.sh train latent                     # Train latent state predictor
./scripts/pipeline.sh train response                   # Train response generator
./scripts/pipeline.sh train joint                      # Joint fine-tuning
./scripts/pipeline.sh eval all                         # Full evaluation
./scripts/pipeline.sh full                             # All of the above
```

---

## Technical Architecture

### Overview

This repository contains **two complementary systems**:

1. **`llm_finetuning/`** — Full latent state predictor with 29 classification heads and multi-stage fine-tuning of pre-trained LLMs (Qwen3)
2. **`slm_training/`** — Small language models trained from scratch with personality + affect encoders and soft-prefix conditioning

### System 1: LLM Fine-Tuning (`llm_finetuning/`)

Complete structured latent state approach:

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
Stage 1: Latent State Predictor (Qwen3-4B + QLoRA; Qwen3-0.6B CPU/debug config available)
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

### System 2: SLM Training from Scratch (`slm_training/`)

Modular, lightweight system for researchers wanting clean, reusable components — includes a **Small-LM benchmark suite** (GRU, AWD-LSTM, GPT, PrefixGPT, MoE, Mamba-like):

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

---

## Project Structure

```
llm_training/
├── data/                   # Training data and scenarios
│   ├── scenario_bank/      # 7 scenario types × 5 templates
│   ├── world_contexts/     # Setting descriptions
│   ├── splits/             # Train/val/test JSONL files
│   └── ...
│
├── llm_finetuning/         # LLM fine-tuning pipeline
│   ├── configs/            # YAML configs for all stages
│   ├── prompts/            # Teacher LLM prompt templates
│   ├── src/
│   │   ├── data_gen/       # Episode generation (scenario_bank, labeler, etc.)
│   │   ├── training/       # Models, datasets, training loops
│   │   ├── eval/           # Evaluation metrics
│   │   ├── packaging/      # Data packaging and splitting
│   │   └── inference/      # Interactive chat
│   ├── scripts/            # Analysis utilities (data quality, cleaning)
│   ├── tests/              # Unit tests
│   ├── run_data_gen.py     # Entry: data generation
│   ├── run_train.py        # Entry: training
│   └── run_eval.py         # Entry: evaluation
│
├── slm_training/           # SLM training from scratch
│   ├── configs/            # Training configs
│   ├── src/
│   │   ├── models/         # Personality, affect, dialogue models
│   │   ├── train/          # Training loops + small LM architectures
│   │   ├── data/           # Data loading and preparation
│   │   ├── infer/          # Inference and chat
│   │   ├── eval/           # Evaluation
│   │   └── api/            # FastAPI server
│   ├── scripts/            # HPO, final training, eval scripts
│   ├── tests/              # Tests
│   ├── train_all.sh        # Full training orchestrator
│   └── smoke_test.sh       # Quick verification
│
├── scripts/                # Shared orchestration scripts
│   └── pipeline.sh         # Unified pipeline (data-gen, train, eval, slm)
├── checkpoints/            # Trained model checkpoints
├── docs/                   # Documentation
├── logs/                   # Log files
└── eval_results/           # Evaluation outputs
```

---

## Usage Examples

### LLM Fine-Tuning

```bash
# Data generation
./scripts/pipeline.sh data-gen --dry-run --n-episodes 50
# With real API:
export AZURE_API_KEY=xxx
./scripts/pipeline.sh data-gen

# Training stages
./scripts/pipeline.sh train latent --debug     # Fast debug mode
./scripts/pipeline.sh train response
./scripts/pipeline.sh train joint
./scripts/pipeline.sh train all                # All 3 stages

# Evaluation
./scripts/pipeline.sh eval all
```

### SLM Training from Scratch

```bash
cd slm_training

# Quick smoke test
bash smoke_test.sh

# Full training with hardware auto-detection
bash train_all.sh --run-id my_experiment

# Train individual components
python -m src.train.run_personality --config configs/personality.yaml
python -m src.train.run_affect --config configs/affect.yaml
python -m src.train.run_dialogue --config configs/dialogue.yaml

# Or via root pipeline
./scripts/pipeline.sh slm-train all
```

### Interactive Inference
```bash
cd llm_finetuning
PYTHONPATH=. python src/inference/interactive.py \
    --checkpoint ../checkpoints/joint_model_best/ \
    --base_model Qwen/Qwen3-4B \
    --scenario data/scenario_bank/secret_extraction.yaml \
    --npc guard_captain
```

### Utility Scripts
```bash
cd llm_finetuning

# Check GPU availability
python scripts/check_gpu.py

# Analyze data quality
python scripts/analyze_data_quality.py

# Clean corrupted labels
python scripts/clean_labels.py --dry-run

# Visualize generated dialogues
python scripts/visualize_dialogues.py --all --output dialogues.txt

# Run tests
PYTHONPATH=. python -m pytest tests/ -v
```

---

## Documentation

| Document | Audience | Content |
|----------|----------|---------|
| [`docs/education_intro.md`](docs/education_intro.md) | General public | Non-technical introduction |
| [`docs/project_overview.md`](docs/project_overview.md) | Researchers | System design and rationale |
| [`docs/SUMMARY.md`](docs/SUMMARY.md) | Developers | Detailed repository map |
| [`docs/schema.md`](docs/schema.md) | ML Engineers | Latent state schema spec |
| [`slm_training/README.md`](slm_training/README.md) | Engineers | SLM system docs |

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

## Hardware Requirements

| Task | Minimum | Recommended |
|------|---------|-------------|
| Data generation | CPU | GPU (for local teacher LLM) |
| LLM Stage 1 training | 8GB VRAM | 12GB VRAM |
| LLM Stage 2/3 training | 12GB VRAM | 16GB+ VRAM |
| SLM training | 4GB VRAM | 8GB+ VRAM |

---

## License

Research and educational use. See LICENSE file for details.
