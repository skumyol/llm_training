SLM Training from Scratch
===========================

Trains small language models (5–50M parameters) for NPC dialogue without pre-trained LLMs.

The repo also keeps a checkpoint registry for the existing baseline runs under `slm/npc_backend_scaffold/runs/`, plus registry-based evaluation and social-state probe scripts for the PrefixGPT baseline.

Architecture Overview
---------------------

.. code-block:: text

   ┌──────────────────────────────────────────────────────────────┐
   │                  SLM Training Pipeline                       │
   │                                                              │
   │  Track A: Encoders                                           │
   │  ┌───────────────┐    ┌───────────────┐                    │
   │  │ Personality    │    │ Affect Encoder │                    │
   │  │ OCEAN→5-vector│    │ VAD→3-vector   │                    │
   │  │ DistilBERT+MLP │    │ DistilBERT+MLP │                    │
   │  └───────┬───────┘    └───────┬───────┘                    │
   │          └────────┬───────────┘                              │
   │                   ▼                                          │
   │  Track B: Small LMs from Scratch + Conditioning              │
   │  ┌─────────────────────────────────────────────────────┐   │
   │  │  6 architectures × 2 hardware profiles × Optuna HPO   │   │
   │  │  cond_vec(8) → prefix injection / token embedding     │   │
   │  │  condition_mode: ocean_vad | social_state | zero      │   │
   │  └─────────────────────────────────────────────────────┘   │
   │                                                              │
   │  Track C: Fine-Tuned LLMs (optional)                         │
   │  ┌─────────────────────────────────────────────────────┐   │
   │  │  Gemma 3/4 + Unsloth QLoRA                            │   │
   │  │  TinyLlama 1.1B + LoRA + Prefix conditioning          │   │
   │  └─────────────────────────────────────────────────────┘   │
   └──────────────────────────────────────────────────────────────┘

Track A: Personality & Affect Encoders
--------------------------------------

Personality Encoder (OCEAN)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Component
     - Detail
   * - Backbone
     - ``distilbert-base-uncased`` (66M)
   * - Pooling
     - Mean ⊕ Max → 2×768 = 1536
   * - Head
     - 1536 → 768 → 384 → 5
   * - Activation
     - GELU
   * - Output
     - 5 continuous: O, C, E, A, N
   * - Loss
     - MSE

.. list-table:: Personality Hyperparameters
   :header-rows: 1

   * - Parameter
     - Value
   * - Learning rate
     - 2×10⁻⁵
   * - Epochs
     - 3
   * - Batch size
     - 16
   * - Max seq length
     - 256
   * - Dropout
     - 0.1

Affect Encoder (VAD)
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Component
     - Detail
   * - Backbone
     - ``distilbert-base-uncased``
   * - Pooling
     - Mean over non-padding tokens
   * - Head
     - 768 → 3 (single linear)
   * - Output
     - 3 continuous: V, A, D (sigmoid)
   * - Loss
     - (1-λ)·MSE + λ·(1-CCC)

.. list-table:: Affect Hyperparameters
   :header-rows: 1

   * - Parameter
     - Value
   * - Learning rate
     - 2×10⁻⁵
   * - Epochs
     - 15
   * - Batch size
     - 16
   * - CCC weight (λ)
     - 0.3
   * - Multi-sample dropout
     - 0 (off)

Track B: Small Language Models
--------------------------------

Six architectures for comparison, two hardware profiles, Optuna HPO with 20 trials.

Hardware Profiles
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Profile
     - Target Hardware
     - Memory
     - ~Total Params
   * - ``m1_small``
     - Apple Silicon MPS
     - 2–8 GB
     - 5–15M
   * - ``rtx4070_small``
     - NVIDIA RTX
     - 8–24 GB
     - 20–60M

Training Parameters
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Parameter
     - HPO Phase
     - Final Phase
   * - Trials
     - 20
     - —
   * - Epochs
     - 5
     - 30
   * - Seeds
     - 1 (best config log)
     - 3 (42, 43, 44)
   * - Batch size
     - 16
     - 16
   * - Grad accum
     - 4
     - 4
   * - Optimizer
     - AdamW
     - AdamW
   * - LR schedule
     - Cosine
     - Cosine
   * - Weight decay
     - from HPO
     - from HPO
   * - Gradient clipping
     - 1.0
     - 1.0

Causal LM loss (all architectures):

.. math::

   L_{LM} = CE(logits.reshape(B·T, V), targets.reshape(B·T))

where :math:`V` = vocab_size and :math:`ignore\_index = -100` for padding.

For MoE, auxiliary load-balancing loss is added:

.. math::

   L = CE + 0.01 \cdot \frac{1}{L} \sum_{l=1}^L aux\_loss_l

Track C: Fine-Tuned LLMs
--------------------------

Gemma 4 + Unsloth
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Parameter
     - Value
   * - Base model
     - ``unsloth/gemma-3-4b-it``
   * - Quantization
     - 4-bit QLoRA
   * - LoRA r
     - 16
   * - LoRA alpha
     - 16
   * - LoRA dropout
     - 0.0
   * - Learning rate
     - 2×10⁻⁴
   * - Epochs
     - 3
   * - Max seq length
     - 2048
   * - Effective batch
     - 8

TinyLlama + LoRA + Prefix
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Parameter
     - Value
   * - Base model
     - ``TinyLlama/TinyLlama-1.1B-Chat-v1.0``
   * - LoRA r
     - 16
   * - LoRA alpha
     - 32
   * - LoRA dropout
     - 0.05
   * - Prefix length
     - 8 tokens
   * - Cond dim
     - 8 (OCEAN=5 + VAD=3)
   * - Learning rate
     - 2×10⁻⁴
   * - Epochs
     - 3
   * - Max source len
     - 768
   * - Max target len
     - 192
   * - Effective batch
     - 16

``social_state`` is wired as a text-derived proxy mode in the scratch runner, so the conditioning interface is already in place for richer labels later.

External baselines and tooling
------------------------------

- Baseline checkpoint paths used by the registry evaluator when the external artifact bundle or local `slm/` workspace is present:
  - `slm/npc_backend_scaffold/runs/gpt/gpt_best.pt`
  - `slm/npc_backend_scaffold/runs/prefix_gpt/prefix_gpt_best.pt`
- Registry file:
  - `slm_training/trained_models.yaml`
- Registry-based evaluation:
  - `slm_training/scripts/eval_registered_small_lms.py`
- Social-state probe:
  - `slm_training/scripts/probe_social_state.py`

The probe expects the labeled `train_heads.jsonl` / `val_heads.jsonl` splits produced by the latent-state pipeline, not the plain-text SLM corpus.

Running Training
-----------------

.. code-block:: bash

   # Full SLM pipeline (HPO + final training + eval)
   cd slm_training
   bash run_full_slm_training.sh [arch]

   # Individual components
   bash train_personality_encoder.sh
   bash train_affect_encoder.sh
   bash train_small_lms.sh [arch]
   bash finetune_dialogue_lm.sh

   # Full orchestration with auto hardware detection
   bash train_all.sh --run-id my_experiment --with-gemma

   # Smoke tests
   bash smoke_test.sh               # Quick (distilbert + distilgpt2)
   bash smoke_test_external.sh      # Full external corpus test
