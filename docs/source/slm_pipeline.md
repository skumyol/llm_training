SLM Training Pipeline
========================

Sequential and parallel training orchestration with hardware auto-detection.

Optuna HPO Pipeline
--------------------

.. code-block:: text

   ┌───────────────────────────────────────────────────────┐
   │  Phase 1: Optuna HPO (20 trials × 5 epochs per arch)  │
   │                                                       │
   │  For each architecture:                               │
   │    1. Define search space (lr, wd, bs, dropout, ...)  │
   │    2. Train 5-epoch trials, log PPL to Optuna         │
   │    3. Save best config to artifacts/optuna/           │
   │    4. (Optional) Dump best params for inspection      │
   └───────────────────────────────────────────────────────┘
                           ↓
   ┌───────────────────────────────────────────────────────┐
   │  Phase 2: Final Multi-Seed Training                    │
   │                                                       │
   │  For each arch × seed (42, 43, 44):                   │
   │    1. Load best config from Optuna                    │
   │    2. Train 30 epochs                                 │
   │    3. Save best checkpoint: artifacts/small_lm/       │
   │    4. Log PPL curve to MLflow                         │
   └───────────────────────────────────────────────────────┘
                           ↓
   ┌───────────────────────────────────────────────────────┐
   │  Phase 3: Evaluation                                  │
   │                                                       │
   │  For each arch × seed:                                │
   │    • Perplexity on val set                            │
   │    • BLEU-4 against reference text                    │
   │    • Distinct-1 / Distinct-2 (token diversity)        │
   │    • Save to eval_results.csv                         │
   └───────────────────────────────────────────────────────┘

Search Space
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Parameter
     - Type
     - Range
   * - ``lr``
     - log-uniform
     - [1×10⁻⁴, 5×10⁻³]
   * - ``weight_decay``
     - log-uniform
     - [0.01, 0.5]
   * - ``batch_size``
     - categorical
     - {8, 16, 32}
   * - ``grad_accum``
     - categorical
     - {1, 2, 4}
   * - ``dropout``
     - uniform
     - [0.0, 0.5]
   * - ``embed_dim``
     - categorical
     - {128, 256, 512}
   * - ``n_layer``
     - categorical
     - {2, 3, 4} small / {4, 6, 8} large
   * - ``seq_len``
     - categorical
     - {128, 256, 512}

Sequential Orchestrator
-------------------------

For single-GPU setups, trains architectures one at a time, auto-skipping completed runs.

.. code-block:: bash

   cd slm_training
   nohup ../.venv/bin/python scripts/sequential_training_orchestrator.py \
       > /tmp/sequential_training.log 2>&1 &

Auto-resumes if interrupted:

.. code-block:: python

   ARCHS = ["gpt", "prefix_gpt", "moe", "mamba_like"]
   SEEDS = [42, 43, 44]
   EPOCHS = 20

   for arch in ARCHS:
       for seed in SEEDS:
           if check_completed(arch, seed):
               print(f"Skipping {arch} s{seed} — already done")
               continue
           train_arch_seed(arch, seed, EPOCHS)

Hardware Auto-Detection
-------------------------

The ``train_all.sh`` orchestrator auto-detects GPU and configures accordingly:

.. list-table::
   :header-rows: 1

   * - GPU VRAM
     - Stage 1 Strategy
     - Batch Size
     - Grad Accum
   * - ≥ 24 GB (A100, 4090)
     - Parallel encoders
     - 32
     - 4
   * - 16–24 GB (3090, 4080)
     - Parallel encoders
     - 16
     - 2
   * - 8–16 GB (3070)
     - Parallel (tight)
     - 8
     - 1
   * - < 8 GB
     - Sequential only
     - 4
     - 1
   * - Apple MPS
     - Parallel encoders
     - 16
     - 1
   * - CPU
     - Sequential
     - 8
     - 1

Running the Pipeline
---------------------

.. code-block:: bash

   # Full pipeline via root orchestrator
   ./scripts/pipeline.sh slm-train all

   # Or directly in slm_training/
   cd slm_training

   # Full HPO + multi-seed training + eval
   bash run_full_slm_training.sh [arch]

   # Full orchestrator (all tracks, auto HW)
   bash train_all.sh --run-id my_experiment --with-gemma

   # Single architecture, manual
   bash train_small_lms.sh gpt     # HPO + train one arch
   bash train_personality_encoder.sh
   bash train_affect_encoder.sh
   bash finetune_dialogue_lm.sh    # TinyLlama + LoRA
