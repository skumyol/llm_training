Overview
========

This project builds **smarter AI characters for games and simulations** through two complementary approaches:

1. **LLM Fine-Tuning** — Fine-tunes pre-trained LLMs (Qwen3) into structured latent-state dialogue models
2. **SLM Training** — Trains small language models from scratch (5–50M params) with personality conditioning

The Core Idea
-------------

Instead of monolithic next-token prediction, our system adds a **"social brain"** that explicitly models:

- 🔄 **Relationship stance** — trust, respect, dominance, familiarity, affection, obligation
- 🎭 **Affective state** — valence, arousal, threat, control
- 🧠 **Player mental model** — what the NPC thinks the player knows/wants
- 🔐 **Norm compliance** — secrecy, duty, face-saving, value conflicts
- 📋 **Response policy** — whether to deflect, answer, challenge, negotiate, etc.

.. code-block:: text

   Player: "Have you caught the spy yet?"
           ↓
   [Social Brain Thinks]
   - Trust level: Low (I don't know this person)
   - Secret: Yes (spy location is hidden)
   - Secrecy_pressure: High
   - Response_policy: deflect
           ↓
   NPC: "Orders from above. Move along, citizen."

The Latent State Schema
------------------------

Every turn in the conversation is annotated with 29 structured labels:

.. list-table:: Latent State Components
   :header-rows: 1

   * - Component
     - Heads
     - Description
   * - C_t (Context)
     - dialogue_act, tone, risk_type
     - Player utterance analysis
   * - A_t (Affect)
     - valence, arousal, threat, control
     - NPC emotional appraisal
   * - M_t (Mental)
     - player_intent, player_knowledge, player_credibility
     - Theory of Mind modeling
   * - R_t (Stance)
     - 6 dims × (level + delta)
     - Relationship tracking
   * - N_t (Norms)
     - duty, secrecy, face, value_conflict
     - Social constraint pressure
   * - D_t (Decision)
     - response_policy, reveal_decision, repair_strategy
     - Behavioral policy

Quick Start
-----------

.. code-block:: bash

   # Install
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

   # LLM Fine-Tuning (quick test)
   cd llm_finetuning
   PYTHONPATH=. python run_data_gen.py --config configs/data_gen.yaml --dry-run --n-episodes 20

   # SLM Training (smoke test)
   cd slm_training
   bash smoke_test.sh

   # Full pipeline via orchestrator
   ./scripts/pipeline.sh data-gen
   ./scripts/pipeline.sh train all
   ./scripts/pipeline.sh eval all

Repository Structure
--------------------

.. code-block:: text

   llm_training/
   ├── llm_finetuning/      # LLM fine-tuning pipeline (Qwen3 + LoRA)
   │   ├── src/             # Data gen, training, eval, inference
   │   ├── configs/         # YAML configs for all stages
   │   └── tests/           # Unit tests
   ├── slm_training/        # SLM training from scratch
   │   ├── src/             # Models, training runners, data
   │   ├── scripts/         # HPO, final training, eval
   │   └── tests/           # Unit tests
   ├── data/                # Shared datasets
   ├── checkpoints/         # Trained model weights
   ├── scripts/             # Orchestration scripts
   └── docs/                # Documentation
