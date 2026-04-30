MLflow Tracking
===============

Both LLM fine-tuning and SLM training share a **single MLflow tracking server** at the project root. This gives you a unified dashboard for all experiments.

Quick Start
-----------

.. code-block:: bash

   mlflow ui --backend-store-uri mlruns
   # → http://localhost:5000

Shared Tracking URI
--------------------

.. code-block:: text

   mlruns/                    ← project root (shared)
   ├── 0/  Default
   ├── 1/  social_state_data_generation   (LLM data gen)
   ├── 2/  latent_state_prediction        (LLM Stage 1 + 3)
   ├── 3/  response_generation            (LLM Stage 2)
   ├── 4/  routing_and_policy_eval        (LLM eval)
   ├── 5/  personality_encoder            (SLM)
   ├── 6/  affect_encoder                 (SLM)
   ├── 7/  small_lm                       (SLM)
   ├── 8/  dialogue_model                (SLM)
   └── 9/  slm_eval                       (SLM eval)

LLM Fine-Tuning: What's Tracked
--------------------------------

Data Generation
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Parameter
     - Value
   * - Experiment
     - ``social_state_data_generation``
   * - Artifacts
     - ``data_manifest.json`` (episode counts, hashes, scenario distribution)
   * - Metrics
     - n_turns, n_episodes, counterfactual_count

Stage 1: Latent State Predictor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Parameter
     - Value
   * - Experiment
     - ``latent_state_prediction``
   * - Params logged
     - model_name, lora_r, lora_alpha, lr, head_lr, batch_size, epochs, pooling, quantization, loss_weights
   * - Step metrics
     - ``train/loss`` (every 20 steps)
   * - Epoch metrics
     - ``val/loss``, ``val/response_policy_f1``, ``val/mean_accuracy``, ``val/mean_f1``, ``val/trust_delta_f1``, ``val/reveal_decision_f1``
   * - Artifacts
     - Best model checkpoint directory

Stage 2: Response Generator (SFT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Parameter
     - Value
   * - Experiment
     - ``response_generation``
   * - Params logged
     - model_name, conditioning_mode, lora_r, lora_alpha, lr, epochs
   * - Step metrics
     - ``train/lm_loss`` (every 20 steps)
   * - Epoch metrics
     - ``val/lm_loss``, ``val/best_lm_loss``
   * - Artifacts
     - Best model adapter weights

Stage 3: Joint Fine-Tuning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Parameter
     - Value
   * - Experiment
     - ``latent_state_prediction`` (same as Stage 1)
   * - Step metrics
     - ``train/joint_loss``, ``train/lm_loss``
   * - Epoch metrics
     - ``val/joint_loss``

LLM Evaluation
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Parameter
     - Value
   * - Experiment
     - ``routing_and_policy_eval``
   * - Metrics
     - response_policy_f1, stance_delta_accuracy, secret_leakage_rate, contradiction_rate, rouge_l, routing_precision/recall/f1, false_positive_rate
   * - Artifacts
     - ``latent_eval_metrics.json``, ``response_eval_metrics.json``, ``routing_eval_metrics.json``, confusion matrices (PNG), sample generations

SLM Training: What's Tracked
------------------------------

Personality Encoder
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Parameter
     - Value
   * - Experiment
     - ``personality_encoder``
   * - Params
     - model_name, lr, dropout, batch_size, epochs, target_columns
   * - Step metrics
     - ``train_loss``, ``lr``
   * - Epoch metrics
     - ``val_mse``, ``val_r2``, per-trait MSE and R² (openness, conscientiousness, extraversion, agreeableness, neuroticism)
   * - Artifacts
     - ``run_summary.json``, ``epoch_metrics.csv``

Affect Encoder
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Parameter
     - Value
   * - Experiment
     - ``affect_encoder``
   * - Params
     - model_name, lr, encoder_lr, dropout, ccc_weight, loss_type, epochs
   * - Step metrics
     - ``train_loss``, ``lr``
   * - Epoch metrics
     - ``val_ccc``, ``val_mse``, ``val_r2``, per-dim CCC/MSE/R² (valence, arousal, dominance)
   * - Final metrics
     - ``best_val_ccc``, ``best_val_mse``, ``best_val_r2``, ``best_epoch``
   * - Artifacts
     - ``run_summary.json``, ``epoch_metrics.csv``

Small Language Models
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Parameter
     - Value
   * - Experiment
     - ``small_lm``
   * - Params
     - arch, hardware_profile, lr, weight_decay, batch_size, grad_accum, seq_len, embed_dim, n_layer, dropout
   * - Step metrics
     - ``train_loss``, ``train_ppl``, ``lr``, ``grad_norm``
   * - Epoch metrics
     - ``val_loss``, ``val_ppl``
   * - Artifacts
     - ``run_summary.json``, ``epoch_metrics.csv``

Dialogue Model (TinyLlama + Gemma)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Parameter
     - Value
   * - Experiment
     - ``dialogue_model``
   * - Params
     - base_model_name, lora_r, lora_alpha, prefix_length, lr, epochs, batch_size, grad_accum
   * - Step metrics
     - ``train_loss``, ``lr``, ``grad_norm``
   * - Epoch metrics
     - ``val_loss``, ``val_ppl``
   * - Final metrics
     - ``best_val_loss``, ``best_val_ppl``
   * - Artifacts
     - ``run_summary.json``, ``epoch_metrics.csv``

SLM Evaluation
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Parameter
     - Value
   * - Experiment
     - ``slm_eval``
   * - Metrics
     - Per-architecture: ``train_best_ppl``, ``test_ppl``, ``bleu_1``, ``bleu_2``, ``distinct_1``, ``distinct_2``, ``params_m``
   * - Artifacts
     - Eval CSV with full comparison table

Implementation
--------------

LLM Side
~~~~~~~~

.. code-block:: python

   # llm_finetuning/src/mlflow_utils.py
   import mlflow

   def setup_mlflow(tracking_uri="mlruns"):
       mlflow.set_tracking_uri(tracking_uri)

   # Usage in training scripts:
   with mlflow.start_run(run_name="latent_train"):
       mlflow.log_params(cfg)
       mlflow.log_metric("train/loss", loss, step=global_step)
       mlflow.log_artifact("checkpoints/best_model")

SLM Side
~~~~~~~~

.. code-block:: python

   # slm_training/src/train/mlflow_tracker.py
   class MLflowTracker:
       """Graceful degradation if mlflow not installed."""

       def log_metrics(self, metrics, step=None): ...
       def log_params(self, params): ...
       def log_artifact(self, path): ...
       def start_run(self, run_name, tags): ...
       def end_run(self): ...

   # Usage:
   tracker = MLflowTracker(experiment="affect_encoder")
   tracker.start_run(run_name="run_42", tags={"seed": "42"})
   tracker.log_params(cfg)
   tracker.log_metrics({"val_ccc": 0.68}, step=epoch)
   tracker.log_artifact("run_summary.json")
   tracker.end_run()

Graceful Degradation
~~~~~~~~~~~~~~~~~~~~~

If MLflow is not installed, ``MLflowTracker`` silently becomes a no-op — all scripts work without it:

.. code-block:: python

   try:
       import mlflow
       HAS_MLFLOW = True
   except ImportError:
       mlflow = None
       HAS_MLFLOW = False

   if not self.enabled:
       return  # silent no-op

Future-Proofing
---------------

MLflow 2.20+ deprecated the filesystem backend. To migrate:

.. code-block:: bash

   # Add to .env:
   echo 'MLFLOW_TRACKING_URI=sqlite:///mlflow.db' >> .env

   # Or set per-run:
   export MLFLOW_TRACKING_URI=sqlite:///mlflow.db

This keeps a single-file SQLite database instead of nested directories.
