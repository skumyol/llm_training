# Results Reassessment for Publication

This document is a source-driven reassessment of the current uncommitted results. It separates publishable claims from claims that need a rerun or human evaluation before submission.

## Executive Takeaways

| Question | Updated assessment | Evidence |
|---|---|---|
| Best from-scratch architecture | MoE remains best among Track A SLMs, with val_ppl=42.07 versus GPT val_ppl=45.32. | `slm_training/artifacts/small_lm/*/run_summary.json` |
| Best response model | ConditionalDialogue remains the best validation-perplexity response model: val_ppl=2.90. | `slm_training/artifacts/dialogue_model/.../run_summary.json` |
| Conditioning effect | OCEAN+VAD soft-prefix gives a 12.3% relative PPL reduction over unconditioned TinyLlama LoRA (3.30 -> 2.90). | paired Track C runs |
| Fixed Gemma result | The publishable Gemma baseline should be `google/gemma-2-2b-it` with val_ppl=6.38, not the earlier Gemma-4-E2B exploratory run with val_ppl=16.24. | two Gemma run summaries |
| Latent state | Independent flat eval reports mean accuracy=0.702, estimated kappa=0.611; per-epoch training logs report response_policy_f1 max=0.448 at epoch 2. | `eval_results` + checkpoint metrics |
| Response generation | Checked-in response metrics are weak: `response_eval_metrics.json` reports ROUGE-L=0.098 over 683 examples; the 100 checked-in samples average roughly ROUGE-L=0.115 and show verbose repetition qualitatively. The evaluator now computes BLEU, repetition, length ratio, and confidence intervals, but those fields must be regenerated on the remote checkpoints before submission. | `response_eval_metrics.json` + `sample_generations.json` |
| Safety checks | Leakage reporting is inconsistent in checked-in artifacts: aggregate response metrics report 0.0, but `sample_generations.json` contains 10/100 `secret_leak=true` flags. The updated evaluator separates gated leakage (`reveal_decision=none`) from ungated leakage; treat leakage as unresolved until that rerun completes. | response eval artifacts |
| Routing | Routing F1=1.000 is a gold-state sanity check. The updated routing evaluator can consume `predicted_zt.jsonl` from latent eval and refuses silent gold fallback by default. | `eval_routing.py` |

## What Changed Relative to the Previous Report

1. **Gemma baseline corrected.** The strongest local Gemma run is `google/gemma-2-2b-it` (`gemma4_20260503_102545`) with best val_loss=1.854, val_ppl=6.38, train_size=500, val_size=115. The later `google/gemma-4-E2B` run is useful as an engineering feasibility result but is not the best scientific baseline.
2. **Qwen scale corrected.** The saved adapters identify `Qwen/Qwen3-1.7B` as the base model. Paper text should not call the current checkpoints 0.6B models.
3. **Joint model status corrected.** The joint checkpoint exists and has a training summary (best_val_joint_loss=6.468), but it has not been independently evaluated for generation quality or per-head latent metrics. Treat it as trained-but-not-fully-evaluated.
4. **Latent F1 discrepancy documented.** `eval_results/latent_eval_metrics.json` reports response_policy_f1=0.512, while checkpoint epoch logs peak at 0.448. Until a clean rerun from the selected checkpoint is performed, cite mean accuracy/kappa as stable and report response-policy F1 as approximately 0.45-0.51.

## Publishable Claims

- **Architecture benchmark:** Sparse MoE is the best 15-22M from-scratch SLM in this benchmark. The exact relative improvement over GPT is 7.2% PPL reduction.
- **Conditioning result:** Explicit social conditioning improves response PPL from 3.30 to 2.90; this is the cleanest RQ3 result because it compares related LoRA response models on the same validation split.
- **Latent interpretability:** Mean latent accuracy around 0.70 across 28-29 heads supports the claim that structured social state is learnable, with relational deltas remaining the hardest group.
- **Generation limitation:** The Qwen response model is not yet a high-quality dialogue model by checked-in lexical metrics. It should be framed as an auditable structured-generation prototype, not SOTA open-dialogue generation.

## Claims to Avoid or Qualify

- Do not claim routing generalization from F1=1.0. It is a deterministic rule sanity check over gold labels.
- Do not claim zero leakage. The checked-in aggregate metric and per-sample flags disagree.
- Do not call estimated kappa a true Cohen's kappa unless predictions and class marginals are available. The current comprehensive script estimates kappa from accuracy and a uniform chance baseline.
- Do not use Gemma-4-E2B as the main Gemma baseline unless it is rerun longer and resolves the high training-loss anomaly.
- Do not say the joint model has improved response generation until it is evaluated with the same response metrics.

## Figure Inventory

| Figure file | Purpose |
|---|---|
| `paper/figures/architecture_stack.pdf` | Overall four-track benchmark structure |
| `paper/figures/data_flow.pdf` | Scenario-to-split data generation pipeline |
| `paper/figures/structured_llm_pipeline.pdf` | Qwen3 latent, router, response, joint model flow |
| `paper/figures/best_model_diagram.pdf` | Best response model architecture |
| `paper/figures/slm_ppl_comparison.pdf` | Track A perplexity comparison |
| `paper/figures/response_ppl_comparison.pdf` | Track C response-model comparison including fixed Gemma |
| `paper/figures/latent_group_scores.pdf` | Latent group accuracy/kappa |
| `paper/figures/latent_head_heatmap.pdf` | Per-head latent metric heatmap |
| `paper/figures/qwen_latent_training.pdf` | Qwen latent training dynamics |

## Recommended Paper Framing

The most defensible story is: **structured social state is useful as an auditable bottleneck and as a conditioning signal, but current automatic generation quality still needs human evaluation and decoding improvements.** This framing lets the paper publish strong system and benchmark contributions without overstating response quality.

## Checked-In Artifact Gaps

- Generated split JSONL files and trained checkpoints are not present in this checkout.
- Checked-in `response_eval_metrics.json` predates the evaluator upgrade and does not yet include BLEU, confidence intervals, distinct-n, repeated n-gram rates, length ratio, gated/ungated leakage denominators, or prompt-artifact rate.
- `sample_generations.json` has 10/100 samples flagged for `secret_leak=true`, while `response_eval_metrics.json` reports `secret_leakage_rate=0.0`; this must be reconciled before publication.
- True Cohen's kappa, balanced accuracy, MCC, and macro-F1 require saved predictions and labels; the current paper table uses normalized above-chance estimates for historical comparison.
- Routing is only publishable as a deterministic rule over validation states until the remote rerun reports predicted-state routing from `predicted_zt.jsonl`.

## Next Remote Batch Must Produce

- Refreshed `response_eval_metrics.json` with length-aware decoding, BLEU, ROUGE-L CI, repetition, length ratio, prompt-artifact rate, and gated/ungated leakage counts.
- Refreshed `sample_generations.json` with full prompt, reveal decision, leak flag, gold response, generated response, and ROUGE-L for DPO/KTO pair construction.
- `predicted_zt.jsonl` from `eval_latent.py`, aligned by `(episode_id, turn_idx)`.
- `routing_eval_metrics.json` in `routing_mode: predicted` with `prediction_coverage=1.0` or an explicit missing-prediction policy.
- Repackaged SFT splits containing per-record `secret_strings` before running secret-masked SFT.
