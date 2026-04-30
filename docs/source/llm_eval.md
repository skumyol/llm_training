LLM Evaluation
===============

Metrics and thresholds for evaluating the trained latent state predictor and response generator.

Latent State Evaluation
------------------------

.. list-table::
   :header-rows: 1

   * - Metric
     - Target
     - Description
   * - ``response_policy_f1``
     - ≥ 0.75
     - Macro F1 on response policy classification
   * - ``stance_delta_accuracy``
     - ≥ 0.70
     - Accuracy predicting relationship change directions
   * - ``secret_leakage_rate``
     - ≤ 0.05
     - Fraction of turns where NPC reveals secrets
   * - ``mean_accuracy``
     - —
     - Average accuracy across all heads
   * - ``mean_f1``
     - —
     - Average macro F1 across all heads
   * - ``trust_delta_f1``
     - —
     - F1 specifically for trust level changes
   * - ``reveal_decision_f1``
     - —
     - F1 for disclosure decisions

Secret Leakage Detection
~~~~~~~~~~~~~~~~~~~~~~~~

Secrets are defined per NPC profile with ``leakage_keywords``. A leakage event occurs when:

1. The NPC's ``reveal_decision`` is ``none`` (should not disclose)
2. The NPC's response contains any leakage keywords

.. code-block:: python

   SECRECY_KEYWORDS = [
       "chalice", "vault location", "patrol schedule", "ledger",
       "affair", "supply theft", "succession", "smuggling",
       "contraband", "heresy", "poison", "bribe", "corruption",
   ]

Response Generation Evaluation
--------------------------------

.. list-table::
   :header-rows: 1

   * - Metric
     - Target
     - Description
   * - ``rouge_l``
     - maximize
     - Longest common subsequence overlap with gold
   * - ``secret_leakage_rate``
     - ≤ 0.05
     - Same as above, post-generation
   * - ``contradiction_rate``
     - ≤ 0.08
     - Responses contradicting earlier statements

Contradiction Detection
~~~~~~~~~~~~~~~~~~~~~~~

Heuristic pattern matching for contradictions:

.. code-block:: python

   CONTRADICTION_PATTERNS = [
       ("i know nothing", "i saw"),
       ("i was not there", "i watched"),
       ("there is no secret", "the secret"),
       ("i never met", "i have known"),
   ]

Routing Evaluation
-------------------

The selective routing module decides when to use the "slow path" (reflective generation). Evaluated as a binary classifier:

.. list-table::
   :header-rows: 1

   * - Metric
     - Target
     - Description
   * - ``routing_precision``
     - maximize
     - Ratio of correct slow-path invocations
   * - ``routing_recall``
     - maximize
     - Fraction of gold slow-path turns caught
   * - ``routing_f1``
     - maximize
     - Harmonic mean of precision/recall
   * - ``false_positive_rate``
     - ≤ 0.15
     - Unnecessary slow-path invocations
   * - ``slow_path_rate``
     - —
     - Percentage of turns routed to slow path

Slow-Path Triggers
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Condition
     - Action
   * - ``value_conflict == "strong"``
     - Slow path
   * - ``response_policy ∈ {threaten, negotiate}``
     - Slow path
   * - ``secrecy_pressure == "high" AND reveal ≠ "none"``
     - Slow path
   * - All other cases
     - Fast path

Confusion Matrices
-------------------

Evaluation generates per-head confusion matrices saved as PNG images, e.g.:

.. code-block:: text

   eval_results/
   ├── cm_response_policy.png      # 10×10 matrix for response policy
   ├── cm_trust_delta.png          # 5×5 matrix for trust deltas
   ├── cm_reveal_decision.png      # 4×4 matrix for disclosure
   └── ...                         # 26 more confusion matrices

Running Evaluation
-------------------

.. code-block:: bash

   # All evaluation stages
   ./scripts/pipeline.sh eval all

   # Individual stages
   cd llm_finetuning
   PYTHONPATH=. python run_eval.py --stage latent   --config configs/eval.yaml
   PYTHONPATH=. python run_eval.py --stage response --config configs/eval.yaml
   PYTHONPATH=. python run_eval.py --stage routing  --config configs/eval.yaml

   # Results saved to eval_results/
