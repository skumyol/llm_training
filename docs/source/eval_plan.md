# Evaluation Plan

This project uses one evaluation contract across training, offline evaluation, MLflow, and paper tables.

The goal is not to accumulate separate metrics in separate formats. The goal is to emit the same structured artifact family everywhere:

- JSON for machine parsing
- Markdown for paper drafting and review
- MLflow metrics for experiment tracking
- PNG confusion matrices for latent-head diagnostics

## Canonical Records

Offline evaluation scripts consume JSONL rows with these canonical fields:

- `episode_id`
- `turn_idx`
- `scenario_family`
- `history`
- `player_utterance`
- `gold_latent`
- `pred_latent`
- `gold_response`
- `pred_response`

The latent-state entries mirror the 29-head schema from the paper:

- `dialogue_act`
- `tone`
- `risk_type`
- `valence`
- `arousal`
- `threat`
- `control`
- `player_intent`
- `player_knowledge`
- `player_credibility`
- `affection_level`, `affection_delta`
- `respect_level`, `respect_delta`
- `dominance_level`, `dominance_delta`
- `familiarity_level`, `familiarity_delta`
- `trust_level`, `trust_delta`
- `obligation_level`, `obligation_delta`
- `duty_pressure`
- `secrecy_pressure`
- `face_pressure`
- `value_conflict`
- `response_policy`
- `reveal_decision`
- `repair_strategy`

## Canonical Metric Names

### Latent evaluation

Written by `llm_finetuning/src/eval/eval_latent.py` and by `llm_finetuning/src/training/train_latent.py`.

Artifact files:

- `latent_eval_metrics.json`
- `latent_eval_report.md`
- `cm_<field>.png` for each evaluated head
- `evaluation_summary.json` when the top-level runner is used

Summary metrics:

- `mean_accuracy`
- `mean_balanced_accuracy`
- `mean_cohen_kappa`
- `mean_macro_f1`
- `mean_weighted_f1`
- `mean_mcc`
- `response_policy_f1`
- `response_policy_accuracy`
- `reveal_decision_f1`
- `reveal_decision_accuracy`
- `stance_delta_f1`
- `stance_delta_accuracy`
- `trust_delta_f1`
- `trust_delta_accuracy`
- `secret_leakage_rate`

Field-level metrics:

- `accuracy`
- `balanced_accuracy`
- `cohen_kappa`
- `macro_f1`
- `weighted_f1`
- `mcc`
- `support`
- `dialogue_act.micro_f1`
- `dialogue_act.macro_f1`
- `dialogue_act.weighted_f1`
- `dialogue_act.subset_accuracy`
- `dialogue_act.hamming_loss`

Group-level metrics:

- `C.mean_accuracy`, `C.mean_macro_f1`, `C.mean_weighted_f1`
- `A.mean_accuracy`, `A.mean_macro_f1`, `A.mean_weighted_f1`
- `M.mean_accuracy`, `M.mean_macro_f1`, `M.mean_weighted_f1`
- `R.mean_accuracy`, `R.mean_macro_f1`, `R.mean_weighted_f1`
- `N.mean_accuracy`, `N.mean_macro_f1`, `N.mean_weighted_f1`
- `D.mean_accuracy`, `D.mean_macro_f1`, `D.mean_weighted_f1`

Each group also reports `mean_balanced_accuracy`, `mean_cohen_kappa`, and `mean_mcc`.
For the paper, prefer macro-F1, balanced accuracy, Cohen's kappa, and MCC over raw accuracy whenever a field has skewed class marginals.

### Response evaluation

Written by `llm_finetuning/src/eval/eval_response.py`.

Artifact files:

- `response_eval_metrics.json`
- `response_eval_report.md`
- `sample_generations.json`

Canonical metrics:

- `rouge_l`
- `rouge_l_ci_low`
- `rouge_l_ci_high`
- `bleu_1`
- `bleu_2`
- `bleu_4`
- `bertscore_precision`
- `bertscore_recall`
- `bertscore_f1`
- `mauve_score` when optional MAUVE evaluation is enabled
- `sentiment_valence_accuracy`
- `sentiment_valence_support`
- `secret_leakage_rate`
- `secret_leakage_gated_count`
- `secret_leakage_gated_total`
- `secret_leakage_rate_ungated`
- `secret_leakage_ungated_count`
- `secret_leakage_by_reveal_decision`
- `contradiction_rate`
- `distinct_1`
- `distinct_2`
- `repeated_3gram_rate`
- `degenerate_repetition_rate`
- `prompt_artifact_rate`
- `avg_len`
- `avg_ref_len`
- `length_ratio`
- `n_evaluated`

ROUGE and BLEU are diagnostic overlap metrics, not evidence of good role-play quality.
BERTScore and optional MAUVE add semantic and distributional views, while sentiment/valence alignment checks whether generated affect matches the intended `A_t.valence`.
For EMNLP reporting, pair all automatic metrics with degeneration metrics, constraint-violation rates, and human preference judgments.

Additional diagnostic artifacts:

- `topic_coverage.json` from `llm_finetuning/scripts/analyze_topic_coverage.py`
- `human_eval/human_eval_items.jsonl`, `human_eval/human_eval_sheet.csv`, and `human_eval/human_eval_answer_key.json` from `llm_finetuning/scripts/build_human_eval_packet.py`

### Routing evaluation

Written by `llm_finetuning/src/eval/eval_routing.py`.

Artifact files:

- `routing_eval_metrics.json`
- `routing_eval_report.md`
- `predicted_zt.jsonl` from latent evaluation when `routing_mode: predicted`

Canonical metrics:

- `routing_precision`
- `routing_recall`
- `routing_f1`
- `false_positive_rate`
- `slow_path_rate`
- `n_evaluated`
- `n_trace_records`
- `missing_predictions`
- `prediction_coverage`

Routing supports two modes:

- `routing_mode: gold` is the deterministic sanity check over gold `D_t`/`N_t`; F1 near 1.0 is expected and should not be reported as routing generalization.
- `routing_mode: predicted` consumes `predicted_zt_file` from `eval_latent.py` and routes on predicted `value_conflict`, `secrecy_pressure`, `response_policy`, and `reveal_decision`. `routing_missing_predictions: error` is the publication-safe default because it prevents silent fallback to gold labels.

## Training Artifacts

Training scripts write summary bundles so the paper can cite run outputs directly.

### Latent training

Written by `llm_finetuning/src/training/train_latent.py`.

Artifact files:

- `checkpoint_dir/metrics/epoch_###_latent.json`
- `checkpoint_dir/metrics/epoch_###_latent.md`
- `checkpoint_dir/metrics/latent_training_summary.json`
- `checkpoint_dir/metrics/latent_training_summary.md`

### Response training

Written by `llm_finetuning/src/training/train_response.py`.

Artifact files:

- `checkpoint_dir/metrics/response_training_summary.json`
- `checkpoint_dir/metrics/response_training_summary.md`

### Joint training

Written by `llm_finetuning/src/training/train_joint.py`.

Artifact files:

- `checkpoint_dir/metrics/joint_training_summary.json`
- `checkpoint_dir/metrics/joint_training_summary.md`

## MLflow Mapping

The shared helper `llm_finetuning/src/metrics_report.py` flattens nested metrics before logging them to MLflow.

Canonical MLflow prefixes:

- `val/...` for validation metrics during training
- `eval/...` for offline evaluation
- `response/...` for response-train summaries
- `joint/...` for joint-train summaries

## Paper Reporting Rule

The paper should prefer these numbers in order:

1. A metric bundle JSON or Markdown artifact generated by the pipeline.
2. The corresponding MLflow run metric.
3. A manually transcribed number only if the artifact is missing.

That keeps the paper tied to the actual experiment outputs instead of a hand-maintained secondary source.

## EMNLP-Grade Evaluation Upgrade

The next evaluation batch should be framed around validity, not metric quantity.
The current strongest story is that structured social state is an auditable intermediate representation; therefore the evaluation must test state validity, causal usefulness, and generated-behaviour consistency.

### What can be improved without new training

- Re-run offline latent evaluation so `latent_eval_metrics.json` contains true Cohen's kappa, balanced accuracy, and MCC instead of any post-hoc estimated agreement. The rerun also writes `predicted_zt.jsonl` for predicted-state routing.
- Re-run response evaluation with `length_aware_max_tokens: true` so `response_eval_metrics.json` includes BLEU, bootstrap ROUGE-L confidence intervals, repetition rate, degeneration rate, length ratio, prompt-artifact rate, and both gated/ungated leakage denominators.
- Build a 100-200 item blinded human-evaluation packet from `sample_generations.json`, stratified by scenario family and `reveal_decision`.
- Reconcile aggregate `secret_leakage_rate` with per-sample `secret_leak` flags before reporting leakage. If the fixed result is zero, report it as "0 detected by keyword/rule audit" with a Wilson upper confidence bound, not as a safety guarantee.
- Re-label gold-mode routing F1 as a deterministic policy sanity check. For the next batch, report predicted-mode routing with `routing_missing_predictions: error` or include `prediction_coverage` if skipping missing predictions.

### Next cluster runs

- Multi-seed all headline comparisons: Qwen latent, joint vs separate, no-consistency, and Track A GPT/MoE. Use seeds 42/43/44 at minimum.
- Add a parameter-matched dense GPT baseline near the MoE parameter count before making architectural claims about MoE.
- Run an intervention/counterfactual generation evaluation: hold context fixed, change one `Z_t` field, and measure whether the response changes in the expected direction.
- Evaluate predicted-`Z_t` routing and predicted-`Z_t` conditioning separately from gold-`Z_t` modes. This is the causal bridge reviewers will expect.
- Add an out-of-domain split by scenario family or NPC role to test whether the schema generalizes rather than memorizing the synthetic generator.

### Human evaluation protocol

Use a within-item, blinded pairwise design whenever possible.
Annotators see the scenario, dialogue history, hidden social-state rubric, and two anonymized model responses.
Collect:

- Role consistency: does the NPC stay in character?
- Social-state consistency: does the response match duty, secrecy, face, stance, and reveal policy?
- Helpfulness/coherence: does the answer respond naturally to the player?
- Safety/constraint violation: does it reveal forbidden information or contradict the specified state?
- Preference: which response would be better in an interactive narrative?

Report Krippendorff's alpha or Fleiss' kappa for categorical labels, bootstrap confidence intervals for preferences, and a short qualitative error taxonomy.

### Paper framing

For EMNLP, the paper should not sell this as a new SOTA dialogue generator.
The defensible claim is narrower and stronger: structured social-state supervision exposes which social variables are recoverable from dialogue, identifies which conditioning signals are placebo, and provides an auditable control interface for NPC dialogue.
