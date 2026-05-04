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
| Response generation | Automatic overlap metrics are weak: ROUGE-L=0.120, BLEU-4=0.009, with high 3-gram repetition=0.667. | `sample_generations.json` |
| Safety checks | Secret leakage and contradiction heuristics both report 0.0, but these are keyword/rule checks and should be stated as automatic checks rather than exhaustive safety proof. | response eval artifacts |
| Routing | Routing F1=1.000 is a sanity check only because gold and predicted labels use the same deterministic rule over `Z_t`. | `eval_routing.py` |

## What Changed Relative to the Previous Report

1. **Gemma baseline corrected.** The strongest local Gemma run is `google/gemma-2-2b-it` (`gemma4_20260503_102545`) with best val_loss=1.854, val_ppl=6.38, train_size=500, val_size=115. The later `google/gemma-4-E2B` run is useful as an engineering feasibility result but is not the best scientific baseline.
2. **Qwen scale corrected.** The saved adapters identify `Qwen/Qwen3-1.7B` as the base model. Paper text should not call the current checkpoints 0.6B models.
3. **Joint model status corrected.** The joint checkpoint exists and has a training summary (best_val_joint_loss=6.468), but it has not been independently evaluated for generation quality or per-head latent metrics. Treat it as trained-but-not-fully-evaluated.
4. **Latent F1 discrepancy documented.** `eval_results/latent_eval_metrics.json` reports response_policy_f1=0.512, while checkpoint epoch logs peak at 0.448. Until a clean rerun from the selected checkpoint is performed, cite mean accuracy/kappa as stable and report response-policy F1 as approximately 0.45-0.51.

## Publishable Claims

- **Architecture benchmark:** Sparse MoE is the best 15-22M from-scratch SLM in this benchmark. The exact relative improvement over GPT is 7.2% PPL reduction.
- **Conditioning result:** Explicit social conditioning improves response PPL from 3.30 to 2.90; this is the cleanest RQ3 result because it compares related LoRA response models on the same validation split.
- **Latent interpretability:** Mean latent accuracy around 0.70 across 28-29 heads supports the claim that structured social state is learnable, with relational deltas remaining the hardest group.
- **Generation limitation:** The Qwen response model is not yet a high-quality dialogue model by lexical metrics: it is diverse but verbose and repetitive. This should be framed as an auditable structured-generation prototype, not SOTA open-dialogue generation.

## Claims to Avoid or Qualify

- Do not claim routing generalization from F1=1.0. It is a deterministic rule sanity check over gold labels.
- Do not claim zero leakage as a formal guarantee. Say zero keyword-detected leakage in the evaluated sample.
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
