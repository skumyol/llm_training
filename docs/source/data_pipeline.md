Data Pipeline
=============

End-to-end data generation pipeline for creating structured social-state training data.

Pipeline Overview
------------------

.. code-block:: text

   ┌────────────────────────────────────────────────────────────────┐
   │                     Data Generation Pipeline                    │
   │                                                                │
   │  ScenarioBank (7 families / 35 templates) → samples scenarios  │
   │  StateInit                    → builds NPC profile             │
   │  EpisodePlanner               → generates story arc            │
   │  TurnGenerator                → teacher LLM generates turns    │
   │       ├── label_C()           → C_t (context analysis)         │
   │       ├── label_A_M()         → A_t + M_t (affect + ToM)       │
   │       ├── label_R_N_D()       → R_t + N_t + D_t (stance+policy)│
   │       └── generate_response() → NPC utterance                  │
   │  Validator                    → schema + leakage checks        │
   │  CounterfactualAugmenter      → flips variables for robustness│
   │  Packager                     → formats for training            │
   │  Splitter                     → 80/10/10 stratified split      │
   └────────────────────────────────────────────────────────────────┘

Scenario Types
---------------

.. list-table::
   :header-rows: 1

   * - Scenario
     - Templates
     - Stakes
     - Key Social Dynamics
   * - ``secret_extraction``
     - 5
     - medium
     - Trust, secrecy, disclosure
   * - ``apology_repair``
     - 5
     - low
     - Face-saving, guilt, forgiveness
   * - ``alliance_negotiation``
     - 5
     - high
     - Trust, reciprocity, deception
   * - ``rumor_confrontation``
     - 5
     - medium
     - Reputation, credibility, anger
   * - ``threat_escalation``
     - 5
     - high
     - Fear, dominance, de-escalation
   * - ``trust_building``
     - 5
     - low
     - Warmth, self-disclosure, bonding
   * - ``deception_detection``
     - 5
     - medium
     - Theory of mind, lie detection

Per-Turn Teacher LLM Flow
---------------------------

Each turn requires up to 10 API calls to the teacher LLM:

.. code-block:: text

   1. label_C()       → dialogue_act, tone, risk_type
   2. label_A_M()     → valence, arousal, threat, control, player_intent, knowledge, credibility
   3. label_R_N_D()   → 12 stance fields + 4 norm fields + 3 policy fields
   4. generate_response() → NPC utterance text

Teacher Models:

- **Azure-compatible API** (`gpt-5.4-mini` in `data_gen_api.yaml`)
- **Local Gemma** (`google/gemma-4-4b-it` in `data_gen.yaml`)
- **OpenRouter-compatible providers** (commented examples in `data_gen.yaml`)
- **Qwen3-0.6B** (lightweight, for testing) (`data_gen_qwen3_small.yaml`)

Counterfactual Augmentation
-----------------------------

For each validated episode, generate counterfactual variants by flipping one dimension:

.. list-table::
   :header-rows: 1

   * - Flip Variable
     - From
     - To
     - Tests
   * - ``trust``
     - high
     - low
     - State-sensitive withholding
   * - ``secrecy_pressure``
     - low
     - high
     - Secret-keeping under pressure
   * - ``player_intent``
     - bond
     - manipulate
     - Threat detection
   * - ``reveal_decision``
     - none
     - full
     - Disclosure policy adherence
   * - ``value_conflict``
     - none
     - strong
     - Norm compliance

Counterfactual episodes are re-labeled by the teacher LLM with the flipped variable.

Data Packaging
--------------

Each validated turn produces 3 synchronized JSONL files when generation has been run:

.. list-table::
   :header-rows: 1

   * - Artifact
     - Format
     - Contains
     - Use
   * - ``full_trace.jsonl``
     - Raw turn records
     - All social state fields
     - Debugging, routing eval
   * - ``head_supervision.jsonl``
     - context + labels
     - Label tensors
     - **Stage 1: Latent predictor**
   * - ``sft.jsonl``
     - input + target
     - Prompt + response text
     - **Stage 2: Response SFT**

All three files have exactly the same number of records in the same order (alignment guaranteed by Packager). Generated JSONL artifacts are not checked into the current repository snapshot; regenerate them before training.

Train/Val/Test Split
---------------------

Stratified by scenario type, 80/10/10 split at episode level (all turns from one episode go to the same split).

.. code-block:: text

   data/splits/
   ├── train_heads.jsonl
   ├── train_sft.jsonl
   ├── train_trace.jsonl
   ├── val_heads.jsonl
   ├── val_sft.jsonl
   ├── val_trace.jsonl
   ├── test_heads.jsonl
   ├── test_sft.jsonl
   └── test_trace.jsonl

Data Generation Usage
----------------------

.. code-block:: bash

   # Full generation via pipeline
   ./scripts/pipeline.sh data-gen

   # Direct invocation (LLM finetuning dir)
   cd llm_finetuning
   PYTHONPATH=. python run_data_gen.py --config configs/data_gen.yaml --n-episodes 500

   # Dry run (no API key, mock responses)
   PYTHONPATH=. python run_data_gen.py --config configs/data_gen.yaml --dry-run --n-episodes 20

   # Specific stages
   PYTHONPATH=. python run_data_gen.py --stage generate    # Generate only
   PYTHONPATH=. python run_data_gen.py --stage validate    # Validate only
   PYTHONPATH=. python run_data_gen.py --stage package     # Package only
   PYTHONPATH=. python run_data_gen.py --stage split       # Split only
