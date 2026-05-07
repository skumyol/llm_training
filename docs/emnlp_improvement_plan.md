# EMNLP Improvement Plan

This plan treats the current work as a structured-state NLP paper, not as a game demo.
The central scientific claim should be:

> Structured social-state supervision provides an auditable intermediate representation for NPC dialogue; it reveals which social variables are recoverable from dialogue and which conditioning signals causally affect generation.

Do not frame the paper as a state-of-the-art open-domain dialogue generation paper unless human evaluation and stronger generation baselines support that claim.

## Immediate Paper Reframe

Use three core research questions:

1. **Recoverability:** Which social-state dimensions can be predicted from dialogue context?
2. **Causal utility:** Does using the social state change generation in the intended direction?
3. **System validity:** Does a structured pipeline reduce constraint violations or improve human preference relative to unstructured baselines?

Current results already support recoverability.
Current results weakly support causal utility because OCEAN/VAD prefix conditioning is a placebo.
System validity needs better evaluation before submission.

## Next Evaluation Batch

Run these before any new model training:

1. Re-run latent evaluation with true Cohen's kappa, balanced accuracy, MCC, macro-F1, and per-class support.
2. Re-run response evaluation with BLEU, ROUGE-L bootstrap confidence intervals, repeated 3-gram rate, prompt-artifact rate, secret-leakage Wilson upper bound, and contradiction rate.
3. Stratify every metric by scenario family, `reveal_decision`, `response_policy`, and secrecy pressure.
4. Build a blinded human-evaluation packet of 100-200 examples, balanced across high-risk and low-risk turns.
5. Report deterministic routing as a sanity check only unless independently annotated routing labels exist.

## Next Training Batch

Prioritize experiments that close reviewer objections:

1. **Parameter-matched Track A:** train a dense GPT near the MoE parameter count. Without this, only claim "best among tested models."
2. **Gold vs predicted `Z_t`:** compare generation conditioned on gold state, predicted state, shuffled state, and no state.
3. **Counterfactual intervention:** hold dialogue context fixed and change one state variable such as `reveal_decision` or `secrecy_pressure`; measure whether the generated response changes accordingly.
4. **Out-of-domain split:** hold out scenario families or NPC roles to test schema generalization.
5. **Multi-seed headline runs:** at least seeds 42, 43, 44 for all claims used in the abstract.

Run order:

1. Re-run offline evaluation with the upgraded metrics.
2. Build `gold_Z_t`, `predicted_Z_t`, `shuffled_Z_t`, and `none` response-generation eval sets.
3. Run response generation for those four settings on the same validation split.
4. Run counterfactual intervention tests for `reveal_decision`, `secrecy_pressure`, and `response_policy`.
5. Run held-out scenario-family evaluation.
6. Run the parameter-matched dense GPT baseline.
7. Multi-seed only the comparisons that affect the abstract.
8. Build the human-evaluation packet from the final systems.

## Architecture Improvement Roadmap

The current architecture story is too broad: Track A language-model PPL, Track B OCEAN/VAD encoders, Track C response SFT, and Track D latent heads are all useful, but reviewers will ask which architectural choices actually improve structured social-state control.
The next architecture batch should therefore test architecture changes against `Z_t` prediction, causal control, and response consistency, not only perplexity.

### Low-risk next changes

1. **Pooling ablation for latent heads.**
   Current decoder-only latent heads use the last token as the sequence representation.
   Add `pooling_strategy: last_token | avg_pool | attention_pool`.
   Average pooling or learned attention pooling may improve relational and decision heads because those depend on the whole dialogue context, not only the final token.

2. **Ordinal heads for relational levels and deltas.**
   The `R_t` fields are ordinal, but currently behave like flat classification.
   Replace or compare flat softmax with ordinal regression, cumulative-link classification, or Earth Mover's Distance loss.
   This is academically cleaner because mistaking `VL` for `L` should be penalized less than mistaking `VL` for `VH`.

3. **Group-specific adapters or LoRA blocks.**
   Instead of one shared adapter for all 29 heads, test small group-specific adapters for `C/A/M/R/N/D`.
   This can reduce negative transfer between affect, relational stance, and decision policy.

4. **Hierarchical latent predictor.**
   Predict coarse groups first, then dependent fields:
   `N_t` secrecy and duty should inform `D_t` reveal and policy; `A_t` threat should inform repair strategy.
   Implement this as a lightweight dependency layer over head logits before loss computation.

5. **Class-balanced and focal losses.**
   Several fields are likely skewed.
   Add inverse-frequency weighting or focal loss per head, then select by macro-F1/MCC rather than accuracy.

### Response-model architecture changes

1. **Structured control tokens instead of soft OCEAN/VAD prefixes.**
   The placebo result shows OCEAN/VAD prefix values are not semantically used.
   Replace them with explicit serialized `Z_t` control tokens and compare gold, predicted, shuffled, and random states.

2. **Two-stage generate-then-verify decoding.**
   Generate candidate responses, predict their implied `Z_t`, and rerank by consistency with target `Z_t`.
   This may improve constraint adherence without retraining.

3. **Constrained decoding for reveal decisions.**
   For high-secrecy or `reveal_decision=none`, apply a lexical/semantic blocklist from scenario secrets during decoding or post-generation reranking.
   Report this as an inference-time safety layer, not as a learned guarantee.

4. **State-delta conditioning.**
   Condition the response not only on current state but on intended transition `Z_t -> Z_{t+1}`.
   This better matches interactive dialogue, where the NPC response should move trust, threat, or obligation.

### More ambitious research directions

1. **Graph-structured social state.**
   Treat `Z_t` fields as nodes in a dependency graph and use a small GNN or transformer over fields before classification/generation.
   This directly models constraints such as secrecy affecting reveal policy.

2. **Latent-state world model.**
   Train a transition model `p(Z_{t+1} | Z_t, player_utterance, npc_response)`.
   This would make the system more academically grounded as dialogue state tracking plus controllable generation.

3. **Mixture-of-experts by social function.**
   Route examples to experts by social group or scenario type: secrecy, repair, negotiation, threat, bonding.
   This is stronger than generic MoE if routing corresponds to interpretable social functions.

4. **Multi-objective training.**
   Jointly optimize response likelihood, latent prediction, constraint loss, and contrastive counterfactual loss.
   The key ablation is whether each objective improves causal-control metrics.

5. **Preference optimization after SFT.**
   Use human or high-quality judge preferences on role consistency and social-state consistency to run DPO/IPO.
   This should come after the control experiments, not before.

## Stronger Evaluation Design

Automatic metrics should be organized into validity categories:

- **Predictive validity:** macro-F1, balanced accuracy, true Cohen's kappa, MCC.
- **Calibration:** expected calibration error and reliability diagrams for latent heads.
- **Constraint validity:** secrecy/reveal, hostile/affection, threat/repair, and policy/reveal violation counts.
- **Causal control:** intervention success rate under controlled changes to `Z_t`.
- **Generation quality:** human preference, role consistency, social-state consistency, coherence, and safety violation labels.
- **Degeneration:** repeated n-gram rate, prompt artifact rate, average length, and stop-token compliance.

Lexical overlap metrics are acceptable as diagnostics, but should not be the main evidence.

## Human Evaluation Rubric

Use pairwise blinded comparison when possible.
Each item should show annotators:

- Scenario and dialogue history.
- Hidden target social-state rubric translated into natural language.
- Two anonymized responses from different systems.

Collect 1-5 Likert scores for:

- Naturalness.
- Role consistency.
- Social-state consistency.
- Player relevance.
- Constraint safety.

Also collect a forced preference and a binary violation flag.
Report bootstrap confidence intervals and inter-annotator agreement.

## Paper Changes

Move the paper away from broad architecture benchmarking and toward a sharper representation-learning contribution:

- Put the 29-dimensional schema and consistency constraints earlier.
- Make the placebo conditioning result a strength: it shows why naive personality/affect conditioning is insufficient.
- Treat JEPA and consistency-loss null results honestly as negative findings.
- Move Track A SLM architecture results to a secondary experiment unless the parameter-matched baseline is complete.
- Add an "Evaluation Limitations" paragraph that explicitly says keyword leakage is not a formal safety proof.

## Submission Bar

The paper is EMNLP-plausible if it has:

- True agreement metrics, not estimated agreement.
- Confidence intervals or multi-seed variance on headline results.
- A causal intervention experiment for `Z_t`.
- Human evaluation or a very carefully justified LLM-as-judge protocol.
- Honest negative results for placebo conditioning, JEPA, and consistency loss.

Without those, the safer venue framing is a workshop paper on controllable NPC dialogue systems.
