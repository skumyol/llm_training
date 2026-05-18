# Experiment Results Report

This report reconciles the local checkout with the fuller experiment artifacts present on the HKUST scratch checkout at `/scratch/$USER/npc`.

## Artifact Coverage

Local checkout currently has only compact evaluation artifacts:

- `eval_results/latent_eval_metrics.json`
- `eval_results/response_eval_metrics.json`
- `eval_results/routing_eval_metrics.json`
- `eval_results/sample_generations.json`

The cluster checkout also contains the fuller result set:

- `eval_results/comprehensive_results.json`
- `eval_results/paper_tables.md`
- `eval_results/per_head_metrics.csv`
- `eval_results/eval_latent_predictor_best.csv`
- `eval_results/eval_latent_predictor_jepa_shuffled_best.csv`
- `eval_results/eval_slm_latent_gpt_best.csv`
- `eval_results/eval_slm_latent_mamba_best.csv`
- `eval_results/ablation_joint_vs_separate.json`
- `eval_results/gold_zt_bridge.json`
- trained checkpoints and per-epoch metric summaries under `checkpoints/*/metrics/`

Therefore, the paper can report the complete result matrix only if it cites or syncs the cluster artifacts. The local checkout alone is not sufficient to reproduce all result tables.

## Result Matrix

| Track | Result | Current assessment |
|---|---|---|
| A: From-scratch SLMs | MoE has best validation PPL: 42.07 vs GPT 45.32, PrefixGPT 44.54, Mamba-like 53.25 | Reflected in the paper, but should be framed as "best among tested models" because MoE has 39% more parameters than GPT |
| B: Encoders | Personality F1 0.678; affect CCC 0.559 | Reflected in the paper and docs |
| C: Response PPL | ConditionalDialogue 2.90; TinyLlama SFT 3.30; Gemma-2-2B-it 6.38; Gemma-4-E2B exploratory 16.24 | Reflected after the Gemma correction |
| C: Placebo conditioning | Real, shuffled, and random OCEAN/VAD conditioning converge around PPL 2.88-2.91 | Reflected; the causal claim is prefix capacity, not OCEAN/VAD semantics |
| C: Response quality | ROUGE-L 0.1199 in comprehensive results; BLEU-4 0.0094; 3-gram repetition 0.6667; length ratio 2.334 | Partly reflected. The paper now marks these as weak automatic diagnostics and says they need regeneration/sync |
| C: Leakage | Comprehensive table says 0.0 leakage, but sample flags show 10/100 `secret_leak=true` locally | Not safe to report as a positive result. Treat as unresolved until evaluator outputs are reconciled |
| D: Latent flat metrics | Mean accuracy 0.702, estimated kappa 0.611 over 28 single-label heads | Reflected, with caveat that kappa is estimated |
| D: Rigorous latent eval | Qwen latent rigorous CSV: mean accuracy 0.6857, true mean kappa 0.4407, mean macro-F1 0.4844 | Needs to replace or sit beside estimated-kappa tables in the final paper |
| D: SLM latent comparison | GPT-SLM mean accuracy 0.6265, kappa 0.3699, F1 0.4284; Mamba mean accuracy 0.4736, kappa 0.1478, F1 0.3134 | Not fully reflected in the current paper |
| D: JEPA | Qwen JEPA training summary improves best response-policy F1, but rigorous shuffled CSV has similar mean accuracy to baseline; Gemma JEPA remains weak | Current framing as null/uncertain is appropriate |
| D: Joint vs separate | Separate latent mean accuracy 0.6825 vs joint 0.6741; response PPL both 1.04; consistency violations mixed | Reflected as no clear joint-generation improvement |
| Routing | F1 1.0, slow-path rate 0.5373 | Reflected as deterministic sanity check only |
| Gold `Z_t` bridge | `reveal_decision` has strongest difficulty effect, effect size sigma 1.017; repair strategy 0.586; response policy 0.570 | Should be reported as evidence that `Z_t` fields predict generation difficulty, not as causal control |

## Main Quality Concerns

1. **Two latent metric families disagree in magnitude.** The comprehensive result reports estimated kappa 0.611 from accuracy and uniform chance. The rigorous CSV reports true mean kappa 0.4407 and macro-F1 0.4844. The final paper should prefer the rigorous CSV.
2. **Leakage metrics are inconsistent.** The aggregate response metric says 0.0, but per-sample flags show 10/100. This blocks any "no leakage" claim.
3. **Response quality is weak.** ROUGE/BLEU are low, repetition is high, and generated responses are more than twice reference length. This supports an auditability paper, not a high-quality dialogue-generation paper.
4. **MoE is not parameter-matched.** MoE wins Track A but has more parameters. A dense 22M GPT baseline is needed before claiming an architectural MoE advantage.
5. **Routing is circular.** The router is evaluated against the same deterministic rule used for gold labels. It must be evaluated on predicted `Z_t` or independent route labels.

## Next Batch

The next batch should prioritize evaluation validity over new model training. The code changes below have been staged in the local checkout.

### Response generation fixes (no retrain)

1. **Re-run response eval with length-aware decoding** — caps `max_new_tokens` at `length_multiplier * median(reference tokens)`. This directly attacks the length_ratio=2.334 finding.
2. **Dual leakage reporting + semantic metrics** — eval now reports both gated (reveal=none only) and ungated (all turns) leakage, per-reveal_decision breakdown, BERTScore, optional MAUVE, and sentiment/valence alignment. This resolves the 0.0 aggregate vs 10/100 per-sample discrepancy and adds a semantic quality lens beyond ROUGE/BLEU.

```bash
# Local
cd llm_finetuning
PYTHONPATH=. python run_eval.py --stage response --config configs/eval.yaml
# eval.yaml already has length_aware_max_tokens: true and semantic_eval.enabled: true
```

### Response training with secret-span masking + optional DPO

3. **Re-train response model with secret-token masking** — SFT labels set to -100 on tokens overlapping secret strings. Re-run packaging first if the current SFT splits predate the `secret_strings` field. Trains the model to *not* reproduce literal secret text.

```bash
# train_response.yaml now has mask_secret_spans: true
sbatch scripts/slurm_train.sh llm response
```

4. **Build DPO preference pairs** from the eval run and do one short DPO pass (1-2 epochs) to push away from verbose/repetitive/leaking generations.

```bash
# After response eval produces eval_results/sample_generations.json:
python llm_finetuning/scripts/build_dpo_pairs.py  \
    --input eval_results/sample_generations.json  \
    --output data/splits/dpo_pairs.jsonl          \
    --max-length-ratio 1.5 --max-repeated-3gram 0.30

# Then run a short DPO/KTO fine-tune on those pairs (add a DPO trainer script).
```

### Latent head improvement

5. **Focal loss** — `train_latent.yaml` now supports `focal_gamma`. Set to ~1.5-2.0 to up-weight hard examples (especially relational delta heads in group R, the weakest group per kappa 0.44).

```bash
# train_latent.yaml now has focal_gamma: 1.5
sbatch scripts/slurm_train.sh llm latent
```

6. **Rigorous latent re-eval** — re-run `eval_latent` from the selected best checkpoint (not just the final epoch). It now saves `eval_results/predicted_zt.jsonl` per-record, needed for step 7.

```bash
sbatch scripts/slurm_eval.sh llm latent
# Checkpoints best model: checkpoints/latent_predictor_best
```

### Routing on predicted Z_t

7. **Predicted-Z_t routing eval** — set `routing_mode: predicted` in `eval.yaml`. This evaluates the router against predicted `value_conflict`, `response_policy`, `reveal_decision`, and `secrecy_pressure` from the latent head, not gold labels. This is the real generalization test (gold mode F1=1.0 is only a sanity check).

```bash
# eval.yaml: routing_mode: predicted, predicted_zt_file: eval_results/predicted_zt.jsonl
# eval.yaml: routing_missing_predictions: error  # do not silently fall back to gold
sbatch scripts/slurm_eval.sh llm routing
```

### Parameter-matched Track A baseline

8. **Dense GPT at ~22M params** to make the MoE claim defensible.

```bash
sbatch scripts/slurm_train.sh slm small_lm --arch gpt --seed 42 --epochs 20
# (requires run_small_lm to expose size args; if not, manual torchrun)
```

### Human evaluation packet

9. **Build 100-200 item blinded packet** from sampled generations for human quality rating (fluency, relevance, consistency, safety). Use the new `sample_generations.json` which includes `reveal_decision` per sample.

```bash
python llm_finetuning/scripts/analyze_topic_coverage.py \
    --input eval_results/sample_generations.json \
    --output eval_results/topic_coverage.json

python llm_finetuning/scripts/build_human_eval_packet.py \
    --input eval_results/sample_generations.json \
    --output-dir eval_results/human_eval \
    --n-items 200
```

The browser-readable execution plan for today's remote run is `docs/remote_experiment_plan.html`.

### Ordering rule

Do not launch broad retraining until the response evaluator inconsistency is fixed (step 1-2). Once the new eval produces consistent leakage numbers, steps 3-5 can run in parallel.

| Step | What | Needs retrain? | Expected impact |
|---|---|---|---|
| 1-2 | Length-aware eval + dual leakage + semantic metrics | No | Fixes credibility of eval numbers |
| 3 | Secret-masked SFT | Yes (response) | Cuts leakage at source |
| 4 | DPO on bad generations | Yes (response) | Lifts ROUGE-L, cuts repetition |
| 5 | Focal loss latent | Yes (latent) | Lifts kappa on weak heads |
| 6 | Rigorous latent re-eval | No | Clean kappa/F1 for paper tables |
| 7 | Predicted-Z_t routing | No | First real routing F1 |
| 8 | Dense GPT baseline | Yes (SLM) | Defensible Track A claim |
| 9 | Human eval packet | No | Quality evidence beyond auto metrics |
