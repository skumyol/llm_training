Configuration Reference
=======================

Complete guide to all YAML configuration files.

LLM Fine-Tuning Configs
------------------------

Located in ``llm_finetuning/configs/``.

data_gen.yaml
~~~~~~~~~~~~~

Data generation pipeline configuration.

.. list-table::
   :header-rows: 1

   * - Field
     - Type
     - Description
   * - output_dir
     - str
     - Where to save generated episodes
   * - n_episodes
     - int
     - Number of episodes to generate
   * - scenario_dir
     - str
     - Path to scenario YAML files
   * - teacher_model
     - str
     - Model for labeling (for example, google/gemma-4-4b-it locally or gpt-5.4-mini through the API config)
   * - use_api
     - bool
     - Use OpenAI API vs local model
   * - temperature
     - float
     - Sampling temperature for generation
   * - max_tokens
     - int
     - Max tokens per API call
   * - counterfactual.enabled
     - bool
     - Enable counterfactual augmentation
   * - counterfactual.flip_ratio
     - float
     - Fraction of episodes to augment

train_latent.yaml
~~~~~~~~~~~~~~~~~

Stage 1: Latent State Predictor training.

.. list-table::
   :header-rows: 1

   * - Field
     - Type
     - Description
   * - base_model
     - str
     - Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B
   * - lora.r
     - int
     - LoRA rank (16, 32, 64)
   * - lora.alpha
     - int
     - LoRA scaling factor (32, 64)
   * - lora.dropout
     - float
     - LoRA dropout rate
   * - learning_rate
     - float
     - Backbone LR (2e-4)
   * - head_learning_rate
     - float
     - Classification head LR (4e-4)
   * - epochs
     - int
     - Training epochs
   * - batch_size
     - int
     - Per-device batch size
   * - gradient_accumulation_steps
     - int
     - Grad accum for effective batch
   * - max_seq_length
     - int
     - Max sequence length
   * - quantization
     - str
     - 4bit, 8bit, or none
   * - loss_weights
     - dict
     - Component weights (C=1.0, A=1.0, M=1.5, R=2.0, N=1.0, D=2.0)

train_response.yaml
~~~~~~~~~~~~~~~~~~~

Stage 2: Response Generator SFT.

.. list-table::
   :header-rows: 1

   * - Field
     - Type
     - Description
   * - base_model
     - str
     - Same as stage 1 or larger model
   * - lora.r
     - int
     - Typically higher (32) for generation
   * - lora.alpha
     - int
     - 64 for stage 2
   * - learning_rate
     - float
     - Lower LR (1e-4)
   * - epochs
     - int
     - Usually 3
   * - conditioning_mode
     - str
     - "gold" or "predicted" latent state
   * - max_seq_length
     - int
     - Context window for generation

train_joint.yaml
~~~~~~~~~~~~~~~~

Stage 3: Joint fine-tuning.

.. list-table::
   :header-rows: 1

   * - Field
     - Type
     - Description
   * - stage1_checkpoint
     - str
     - Path to stage 1 heads
   * - stage2_checkpoint
     - str
     - Path to stage 2 LoRA adapter
   * - learning_rate
     - float
     - Very low (5e-5)
   * - epochs
     - int
     - Usually 3
   * - lambda_heads
     - float
     - Weight for latent loss (1.0)
   * - lambda_lm
     - float
     - Weight for language loss (1.0)
   * - lambda_consistency
     - float
     - Weight for logical consistency (0.5)
   * - batch_size
     - int
     - Smaller batch (8) for stability

eval.yaml
~~~~~~~~~

Evaluation configuration.

.. list-table::
   :header-rows: 1

   * - Field
     - Type
     - Description
   * - checkpoint
     - str
     - Model to evaluate
   * - test_data
     - str
     - Test split path
   * - metrics
     - list
     - Metrics to compute
   * - thresholds.response_policy_f1
     - float
     - Minimum F1 (0.75)
   * - thresholds.stance_delta_accuracy
     - float
     - Min accuracy (0.70)
   * - thresholds.secret_leakage_rate
     - float
     - Max leakage (0.05)
   * - thresholds.contradiction_rate
     - float
     - Max contradiction (0.08)

SLM Training Configs
--------------------

Located in ``slm_training/configs/``.

personality.yaml
~~~~~~~~~~~~~~~~

DistilBERT encoder for OCEAN personality traits.

.. list-table::
   :header-rows: 1

   * - Field
     - Type
     - Description
   * - model.backbone
     - str
     - distilbert-base-uncased
   * - model.pooling
     - str
     - mean_max (concatenated)
   * - model.head_dims
     - list
     - [1536, 768, 384, 5]
   * - training.learning_rate
     - float
     - 2e-5
   * - training.epochs
     - int
     - 3
   * - training.batch_size
     - int
     - 16
   * - training.dropout
     - float
     - 0.3

affect.yaml
~~~~~~~~~~~

DistilBERT encoder for VAD affect.

.. list-table::
   :header-rows: 1

   * - Field
     - Type
     - Description
   * - model.pooling
     - str
     - mean (not concatenated)
   * - model.head_dims
     - list
     - [768, 3] (single linear)
   * - training.ccc_weight
     - float
     - λ for CCC loss (0.3)
   * - training.epochs
     - int
     - 15 (longer training)
   * - training.output_activation
     - str
     - sigmoid (outputs in [0,1])

dialogue.yaml
~~~~~~~~~~~~~

TinyLlama + LoRA for conditional dialogue.

.. list-table::
   :header-rows: 1

   * - Field
     - Type
     - Description
   * - base_model
     - str
     - TinyLlama/TinyLlama-1.1B-Chat-v1.0
   * - lora.r
     - int
     - 16
   * - lora.alpha
     - int
     - 32
   * - prefix.length
     - int
     - 8 tokens
   * - prefix.cond_dim
     - int
     - 8 (OCEAN=5 + VAD=3)
   * - training.max_source_length
     - int
     - 768
   * - training.max_target_length
     - int
     - 192

dialogue_gemma_unsloth.yaml
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Gemma 3/4 with Unsloth QLoRA.

.. list-table::
   :header-rows: 1

   * - Field
     - Type
     - Description
   * - base_model
     - str
     - unsloth/gemma-3-4b-it
   * - quantization
     - str
     - 4-bit QLoRA
   * - lora.r
     - int
     - 16
   * - lora.alpha
     - int
     - 16
   * - max_seq_length
     - int
     - 2048

small_lm.yaml
~~~~~~~~~~~~~

Small language models from scratch.

.. list-table::
   :header-rows: 1

   * - Field
     - Type
     - Description
   * - architectures
     - list
     - [gru, awdlstm, gpt, prefix_gpt, moe, mamba_like]
   * - hpo.trials
     - int
     - 20 Optuna trials
   * - hpo.epochs
     - int
     - 5 per trial
   * - training.epochs
     - int
     - 30 final training
   * - training.seeds
     - list
     - [42, 43, 44]
   * - hardware_profile
     - str
     - m1_small or rtx4070_small

Scenario Bank
-------------

Located in ``data/scenario_bank/``.

Format
~~~~~~

Each YAML file defines a scenario type with 5 variant templates::

   scenario_type: "secret_extraction"
   templates:
     - name: "guard_at_gate"
       description: "Player tries to extract gate password from guard"
       stakes: "medium"
       npc_roles:
         - role: "guard_captain"
           traits: ["loyal", "suspicious", "duty_bound"]
           secrets: ["gate_password"]
       player_goals: ["learn_password", "avoid_detection"]
       success_conditions: ["npc_reveals_password"]
       failure_conditions: ["npc_raises_alarm"]

Common scenario types:

- ``secret_extraction`` — Information elicitation under secrecy pressure
- ``apology_repair`` — Face-saving and forgiveness dynamics
- ``alliance_negotiation`` — Trust building vs. exploitation detection
- ``rumor_confrontation`` — Reputation management and credibility
- ``threat_escalation`` — De-escalation and dominance dynamics
- ``trust_building`` — Warmth, self-disclosure, bonding
- ``deception_detection`` — Theory of mind and lie detection

World Contexts
--------------

Located in ``data/world_contexts/``.

Defines persistent world state (W component)::

   world_id: "oakhaven_siege"
   setting: "medieval_fantasy"
   factions:
     - name: "City Guard"
       values: ["order", "loyalty", "protection"]
     - name: "Thieves Guild"
       values: ["freedom", "secrecy", "profit"]
   locations:
     - name: "City Gate"
       controlled_by: "City Guard"
       secrets: ["gate_password", "escape_tunnel"]
   npcs:
     - id: "commander_vance"
       faction: "City Guard"
       role: "gate_commander"
       ocean: [0.7, 0.8, 0.6, 0.5, 0.3]  # O-C-E-A-N
       secrets_known: ["gate_password"]

Notes
-----

- All paths in configs are relative to repository root
- CPU variants (train_*_cpu.yaml) reduce batch size and disable CUDA-specific features
- Counterfactual augmentation generates 5x variants per episode
- Hardware profiles auto-configure batch sizes and model sizes
