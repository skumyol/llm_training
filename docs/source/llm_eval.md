LLM Evaluation
===============

The canonical evaluation contract lives in ``eval_plan.md``.

Use that page for the exact artifact schema, metric names, and script-to-artifact mapping.

Quick reference:

.. list-table::
   :header-rows: 1

   * - Metric
     - Target
     - Written by
   * - ``response_policy_f1``
     - ≥ 0.75
     - ``src/eval/eval_latent.py`` and ``src/training/train_latent.py``
   * - ``stance_delta_accuracy``
     - ≥ 0.70
     - ``src/eval/eval_latent.py``
   * - ``secret_leakage_rate``
     - ≤ 0.05
     - ``src/eval/eval_latent.py`` and ``src/eval/eval_response.py``
   * - ``contradiction_rate``
     - ≤ 0.08
     - ``src/eval/eval_response.py``
   * - ``routing_f1``
     - maximize
     - ``src/eval/eval_routing.py``
   * - ``false_positive_rate``
     - ≤ 0.15
     - ``src/eval/eval_routing.py``

Running Evaluation
------------------

.. code-block:: bash

   ./scripts/pipeline.sh eval all
   cd llm_finetuning
   PYTHONPATH=. python run_eval.py --stage latent --config configs/eval.yaml
   PYTHONPATH=. python run_eval.py --stage response --config configs/eval.yaml
   PYTHONPATH=. python run_eval.py --stage routing --config configs/eval.yaml
