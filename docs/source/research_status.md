# Research Status and Evidence Ledger

This page is the repo-local checkpoint for paper claims. It separates results that are backed by checked-in artifacts from claims that need a rerun, stronger evaluation, or external artifact sync before submission.

## Canonical Sources

Use these files before copying numbers into a paper or README:

| Source | What it supports |
|---|---|
| `eval_results/latent_eval_metrics.json` | Flat per-head latent accuracies, `response_policy_f1`, keyword leakage flag |
| `eval_results/predicted_zt.jsonl` | Per-record predicted routing fields emitted by the updated latent evaluator; required for predicted-state routing |
| `eval_results/response_eval_metrics.json` | Current checked-in response ROUGE-L, keyword leakage, contradiction rate, sample count |
| `eval_results/routing_eval_metrics.json` | Rule-router sanity check on validation states |
| `eval_results/sample_generations.json` | 100 generated samples with gold response, generated response, ROUGE-L, and keyword leakage flag |
| `paper/main.tex` | Current manuscript draft |
| `paper/results_reassessment.md` | Publication-oriented interpretation and claim guardrails |
| `docs/source/model_registry.md` | Generated registry from code/config, not experiment-result truth |
| `docs/source/eval_plan.md` | Evaluation contract and upgrade plan |

The generated packaged JSONL splits, SLM training text, model checkpoints, and run-summary artifacts are not present in this checkout. Claims that depend on those artifacts should either cite the external artifact location or be framed as previously reported runs, not as fully reproducible from the checked-in files alone.

## Current Checked-In Evaluation Artifacts

| Area | Checked-in artifact | Current value | Interpretation |
|---|---|---:|---|
| Latent heads | `latent_eval_metrics.json` | Mean accuracy across 28 single-label heads: 0.702 | Useful signal, but raw accuracy is sensitive to class imbalance |
| Latent policy | `latent_eval_metrics.json` | `response_policy_f1`: 0.512 | Better than chance, below deployment threshold |
| Response quality | `response_eval_metrics.json` | ROUGE-L: 0.098 over 683 examples | Diagnostic overlap only; not evidence of good role-play quality |
| Response samples | `sample_generations.json` | 100 samples, mean sample ROUGE-L about 0.115 | Shows verbosity and repetition qualitatively |
| Safety checks | response metrics and samples | Inconsistent: aggregate response metric reports 0.0 leakage, but `sample_generations.json` has 10/100 samples flagged with `secret_leak=true` | Treat leakage reporting as unresolved until the evaluator is rerun |
| Routing | `routing_eval_metrics.json` | F1: 1.000, slow path: 0.537 | Deterministic rule sanity check over validation states |

## Publishable Claims With Current Evidence

- The 29-head latent predictor learns a meaningful portion of the synthetic social-state labels: the checked-in flat metrics average about 0.70 accuracy across the 28 single-label heads.
- Recoverability varies by head. Risk type, face pressure, arousal, and duty pressure are strong; relational deltas such as familiarity and dominance are weaker.
- The response generator artifacts show weak lexical overlap and obvious verbosity. The paper should frame response generation as an auditable prototype, not a state-of-the-art dialogue model.
- The rule router should be described as a deterministic policy sanity check until it is evaluated on predicted `Z_t` or independently annotated routing labels.

## Claims To Avoid

- Do not claim zero leakage. The checked-in aggregate metric and sample-level flags disagree.
- Do not call the normalized above-chance agreement values true Cohen's kappa unless the run has saved predictions, labels, and confusion matrices.
- Do not claim that OCEAN/VAD semantics caused the response perplexity gain. Placebo controls show similar PPL for real, shuffled, and random conditioning vectors.
- Do not claim joint training improves generation unless the joint checkpoint is evaluated with the same response metrics as the separate response model.
- Do not claim all checkpoints and artifacts are released from this checkout; the checked-in repository does not include model checkpoints or generated split data.

## Evaluation Work Needed Before Submission

- Re-run latent evaluation from saved predictions so balanced accuracy, true Cohen's kappa, MCC, and macro-F1 are computed from confusion matrices.
- Re-run response evaluation with length-aware decoding so the JSON artifact includes BLEU, ROUGE-L confidence intervals, distinct-n, repeated n-gram rate, length ratio, and prompt-artifact rate.
- Reconcile `secret_leakage_rate` with per-sample `secret_leak` flags by reporting both gated leakage (`reveal_decision=none`) and ungated leakage across all samples, with exact denominators and Wilson upper bound.
- Evaluate routing on predicted social state using `predicted_zt.jsonl` and `routing_missing_predictions: error`, not only gold or constructed labels.
- Re-run packaging before secret-masked SFT if the current SFT splits predate the per-record `secret_strings` field.
- Build DPO/KTO preference pairs from refreshed `sample_generations.json`; the updated samples include the full prompt, gold response, generated response, ROUGE-L, leak flag, and reveal decision.
- Add a blinded human or LLM-assisted preference packet focused on role consistency, social-state consistency, naturalness, and constraint violations.
- Add counterfactual `Z_t` intervention tests: hold context fixed, change one state field, and measure whether the generated response changes in the expected direction.
