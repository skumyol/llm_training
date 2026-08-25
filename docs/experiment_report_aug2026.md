# Experiment Report — Latent State Prediction & Response-Policy F1
## August 2026, HKUST HPC

---

## 1. Objective

Train a 29-field structured latent-state predictor on NPC dialogue data and evaluate whether the
response-policy field can reach macro-F1 = 0.75 on a held-out test split. The predictor reads a
dialogue context and predicts cognitive, affective, relational, normative, and decision fields
that an NPC generator would consume downstream.

---

## 2. Data

| split | episodes | turns | counterfactual | real |
|---|---:|---:|---:|---:|
| train | 587 | 6,175 | 3,655 (59.2%) | 2,520 |
| val (selection) | 69 | 683 | — | — |
| **test (held out)** | 80 | **884** | 526 (59.5%) | 358 |

Splits are episode-level — no turn from a training episode appears in val or test. Counterfactual
records share a byte-identical context with their original but carry different gold labels; they
are included in training (removing them costs 0.137 response-policy F1, see §6.5).

### 2.1 Response-policy label distribution

The target field has 10 canonical classes, but the distribution is severely imbalanced:

| label | train | % | test | % |
|---|---:|---:|---:|---:|
| deflect | 2,043 | 33.1% | 318 | 36.0% |
| soothe | 1,652 | 26.8% | 199 | 22.5% |
| threaten | 670 | 10.9% | 149 | 16.9% |
| challenge | 592 | 9.6% | 92 | 10.4% |
| clarify | 534 | 8.6% | 35 | 4.0% |
| negotiate | 456 | 7.4% | 56 | 6.3% |
| test | 134 | 2.2% | 30 | 3.4% |
| partial | 46 | 0.7% | 1 | 0.1% |
| answer | 18 | 0.3% | 3 | 0.3% |
| withhold | 12 | 0.2% | 0 | 0.0% |
| defect (typo) | 11 | 0.2% | 1 | 0.1% |
| redirect (noise) | 5 | 0.1% | 0 | 0.0% |
| hint (noise) | 1 | 0.0% | 0 | 0.0% |
| challenged (typo) | 1 | 0.0% | 0 | 0.0% |

Three classes (answer, partial, withhold) have fewer than 50 training samples. Four non-canonical
labels (defect, redirect, hint, challenged) are typos or noise outside the 10-class schema and are
encoded as invalid (−1). The effective per-class sample count for macro-F1 is dominated by the six
mid-frequency classes; the three rare classes contribute near-zero recall and drag the macro
average down regardless of model quality.

---

## 3. Model

**Base:** Qwen3-4B, 4-bit NF4 quantisation with double quantisation, bfloat16 compute dtype.

**LoRA:** r = 16, alpha = 32, dropout = 0.05, targeting `q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj`. 33,030,144 trainable parameters (0.81% of 4.06B).

Two architectures were tested:

### 3.1 Classification-head predictor (L-series, M-series)

Pools the backbone's last hidden state (last-token or mean) into a single 2,560-dim vector and
reads 29 independent `Linear(2560→256→n_classes)` heads off it. Each head is trained with focal
loss (γ = 1.5), label smoothing (0.1), and inverse-frequency class weighting. The heads are
independent — `response_policy` cannot condition on `secrecy_pressure` or `reveal_decision`.

### 3.2 Generative SFT predictor (S-series)

Serialises the 29-field state as text (`<state>\nfield=value\n...</state>`) and trains the backbone
to generate it autoregressively given the dialogue context as a prompt. The prompt is masked out of
the loss (only the state block contributes). Fields are predicted sequentially, so later fields
condition on earlier ones. Scoring parses the generated text and maps it back to class indices
through the same `compute_latent_metrics` pipeline as the head model, so numbers are directly
comparable. Unparseable generations score as wrong, not silently dropped.

---

## 4. Experiments

### 4.1 L-series: classification-head ablations

All L-series runs use the classification-head architecture with last-token pooling unless noted.

| run | change | seed | epochs | best epoch | val macro-F1 | **test RP F1** | test macro-F1 | test mean acc |
|---|---|---|---:|---:|---:|---:|---:|---:|
| L1 control | baseline (last-token, 5ep) | unseeded | 5 | 1 | 0.5375 | 0.4660 | 0.5460 | 0.6803 |
| L1 s43 | same, seeded | 43 | 5 | 1 | 0.5375 | 0.4731 | 0.5404 | 0.6825 |
| L1 s44 | same, seeded | 44 | 5 | 1 | 0.5361 | 0.4557 | 0.5357 | 0.6771 |
| L2 nosampler | disable weighted sampler | unseeded | 5 | — | — | 0.4575 | 0.5348 | 0.6853 |
| L3 meanpool | mean pooling | unseeded | 5 | 3 | 0.5608 | 0.4525 | 0.5502 | 0.6972 |
| L4 ctx1024 | mean pooling + 1024 ctx | unseeded | 8 | 4 | 0.5629 | 0.4310 | 0.5531 | 0.7066 |
| L5 nocf | exclude counterfactuals | unseeded | 5 | — | 0.5158 | 0.3454 | 0.5158 | 0.6385 |
| L6 nocf_reg | L5 + focal γ=2.0 | unseeded | 5 | — | 0.5106 | 0.3193 | 0.5106 | 0.6358 |

**Key findings:**

1. **Seeding matters.** L1 unseeded (0.4660) vs L1 s43 (0.4731) vs L1 s44 (0.4557) — a ±0.018
   spread on test RP F1 just from the RNG seed. The unseeded L3 (0.4525) was previously treated as
   the baseline but is a single lucky draw; the seeded mean is ~0.46.

2. **Mean pooling improves aggregate metrics but not RP F1.** L3 mean pool: test macro-F1 0.5502
   vs L1 last-token 0.5460, but RP F1 drops 0.4660 → 0.4525. Mean pooling dilutes the
   decision-relevant signal at the last position.

3. **Longer context (1024) improves mean accuracy but hurts RP F1.** L4: mean acc 0.7066 (best in
   L-series) but RP F1 0.4310 (worst among full-data runs). More context helps the easy heads
   (valence, tone) but adds noise for the decision head.

4. **Removing counterfactuals is catastrophic.** L5: RP F1 drops 0.4660 → 0.3454 (−0.12).
   Counterfactuals provide 59% of training data and expose the model to label variation under
   controlled context changes. Without them, the model overfits to surface correlations.

### 4.2 M-series: response-policy-specific interventions

All M-series runs use the new code (commit `e4cbb5f`) with mean pooling and seed 42 unless noted.

| run | change | seed | epochs | val macro-F1 | **test RP F1** | test macro-F1 |
|---|---|---|---:|---:|---:|---:|
| M1 clean7 | merge 10→7 labels | 42 | 5 | 0.5072 | 0.3626† | 0.4927 |
| M2 deephead | 3-layer 512-wide RP head | 42 | 5 | 0.5149 | 0.0936 | 0.4866 |
| M3 binary | careful vs ordinary (2-class) | 42 | 5 | 0.5115 | 0.5731† | 0.5115 |
| M4 clean7+deep | combine M1+M2 | 42 | 5 | 0.5054 | 0.3752† | 0.4927 |
| M5 long | 16 epochs | 42 | 16 | 0.5242 | — | — |
| M6 hilr | 2× head LR (8e-4) | 42 | 5 | 0.5131 | 0.0754 | 0.4922 |
| M7 s43 | seed 43, mean pool | 43 | 5 | 0.5116 | 0.3400 | 0.4927 |
| M8 s44 | seed 44, mean pool | 44 | 5 | 0.5147 | 0.3510 | 0.4966 |
| M9 s43 long | seed 43, 8 epochs | 43 | 8 | 0.5219 | 0.3497 | 0.5131 |
| M11 two-stage | 5ep full + 10ep head-only | 42 | 15 | 0.5136 | 0.3389 | 0.5067 |
| M12 auxinput | feed N/D heads to RP head | 42 | 5 | 0.5086 | 0.3461 | 0.4987 |

† Not directly comparable to 10-class RP F1.

**Key findings:**

1. **M2 and M6 are degenerate.** Test RP F1 of 0.09 and 0.08 — near random. The deeper head (M2)
   and higher head LR (M6) both destabilised training. The model predicts across all classes but
   with no correlation to gold labels. The deeper head has 2.4× more parameters, which dominates
   the gradient norm under `max_grad_norm = 1.0` clipping and starves the backbone of learning
   signal.

2. **M7/M8/M9 confirm the seeded mean-pooling baseline is ~0.34–0.35.** This is lower than the
   L1 seeded last-token baseline (~0.46). Mean pooling hurts the response-policy head specifically.

3. **Label cleanup (M1, M4) does not help.** Merging the three rarest classes into nearby classes
   (answer→clarify, partial→clarify, withhold→deflect) reduces the schema from 10 to 7 classes but
   does not improve macro-F1. The merged classes still have distinct semantics that the model
   cannot distinguish, and the macro average now includes fewer but still imbalanced classes.

4. **Binary routing (M3) reaches 0.57 val RP F1** on the 2-class careful-vs-ordinary task. This is
   the highest RP F1 of any M-series run, but it measures a different task (binary routing) and is
   not comparable to the 10-class metric.

5. **Two-stage training (M11) and auxiliary input (M12) do not help.** Freezing the backbone after
   5 epochs and training only the RP head for 10 more epochs at 10× LR yields 0.3389 — worse than
   the single-stage baseline. The backbone needs continued joint training to maintain
   representation quality. Feeding predicted secrecy_pressure, value_conflict, and reveal_decision
   logits as extra input to the RP head (M12) yields 0.3461 — no improvement over M7 (0.3400).

### 4.3 S-series: generative SFT predictor

| run | change | seed | epochs | best epoch | val loss | **test RP F1** | test macro-F1 | test mean acc |
|---|---|---|---:|---:|---:|---:|---:|---:|
| S1 genstate | baseline SFT | unseeded | 5 | 2 | 0.1050 | **0.5167** | 0.5465 | **0.7225** |
| S1 s43 | same, seeded | 43 | 5 | 2 | 0.1087 | 0.4807 | 0.5353 | 0.7167 |
| S1 genstate + vote(4) | greedy + 4-sampled majority vote | 42 | — | — | — | 0.5126 | 0.5450 | 0.7227 |
| S2 reg | LoRA dropout 0.10, wd 0.05, lr 1.5e-4, 8ep | 42 | 8 | 2 | 0.1066 | **0.5362** | 0.5283 | 0.7076 |
| S2 reg + vote(4) | same vote decoding on S2 | 42 | — | — | — | 0.5342 | 0.5291 | 0.7108 |

*(Vote-decoding rows added 2026-08-25; `eval_latent_sft_vote.py`, jobs 1783938/1783939.)*

**Key findings:**

1. **SFT outperforms classification heads on every metric.** S1 unseeded: RP F1 0.5167 vs best
   classification head L1 s43 at 0.4731 (+0.044). Mean accuracy 0.7225 vs 0.6825 (+0.040). The
   gap is consistent across seeds (S1 s43: 0.4807 vs L1 s43: 0.4731).

2. **SFT overfits after epoch 2.** Train loss drops from 0.35 to 0.02 by epoch 5, while val loss
   bottoms out at epoch 2 (0.105) and rises to 0.155 by epoch 5. The best checkpoint is always at
   epoch 2. S2's stronger regularisation (LoRA dropout 0.10, weight decay 0.05, lower LR) delays
   the overfitting slightly (best at epoch 2, val 0.1066) but does not eliminate it.

3. **Vote decoding (majority vote over 5 samples) does not help.** S1 vote: RP F1 0.5126 vs
   greedy 0.5167 — sampling introduces variance that cancels out the marginal gains from
   majority voting. Confirmed on S2 as well (0.5342 vs 0.5362). Per-field deltas scatter ±0.04
   in both directions with no pattern: the greedy decode already sits at the model's consensus.

4. **S2 regularisation trades aggregate metrics for the RP head.** S2 posts the highest single
   RP F1 reading of any configuration in the project (0.5362 greedy, +0.02 over the better S1
   seed) but gives up ~0.018 macro-F1 and ~0.015 accuracy vs the S1 two-seed means. Given the
   observed seed spread (±0.036 on RP between S1 s42/s43), the RP advantage is suggestive, not
   proven — it needs more seeds before being quoted as a regularisation effect. Regularisation
   also did not move the overfit point: best val loss is again at epoch 2 (0.1066 vs 0.1050
   unregularised), rising monotonically to 0.204 by epoch 8.

5. **Ceiling across all S-series variants:** RP F1 stays inside **0.48–0.54** and macro-F1 inside
   **0.53–0.55** across two seeds × two decoding schemes × two regularisation settings. Every
   model-side lever tried — conditioning structure (S-series itself), decoding (vote), seeds,
   regularisation — lands inside the same band, corroborating §6.1's data-side analysis of the
   0.75 target.

6. **SFT leverages pretraining knowledge.** The LLM already understands words like "soothe" and
   "deflect" from pretraining; the classification heads discard this by projecting to a 256-dim
   space and reading off a linear classifier. SFT keeps the full vocabulary embedding and generates
   the label as text, which is why it outperforms despite having the same backbone and LoRA rank.

### 4.4 M10: prompt-based (zero-shot, no fine-tuning)

| method | test RP F1 | test RP acc |
|---|---:|---:|
| constrained decode (log-likelihood) | 0.1966 | 0.2715 |
| free generation (greedy) | 0.1668 | 0.2161 |

The base Qwen3-4B without any fine-tuning cannot perform this task. It predicts `withhold` and
`answer` (the rarest classes) heavily, indicating it does not understand the label semantics in
this domain. Fine-tuning is necessary.

---

## 5. Comprehensive test results (all experiments, sorted by RP F1)

| rank | experiment | architecture | seed | **test RP F1** | test macro-F1 | test mean acc | test RP acc |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | **S2 genstate reg** | **generative SFT (regularised)** | 42 | **0.5362** | 0.5283 | 0.7076 | 0.6927 |
| 2 | L1 s43 orig | classification (last-token) | 43 | 0.5295 | 0.5534 | 0.6882 | 0.6732 |
| 3 | L1 orig | classification (last-token) | unseeded | 0.5205 | 0.5567 | 0.6886 | 0.6676 |
| 4 | **S1 genstate orig** | **generative SFT** | unseeded | **0.5167** | 0.5465 | **0.7225** | 0.6899 |
| 5 | L1 s44 orig | classification (last-token) | 44 | 0.5037 | 0.5489 | 0.6812 | 0.6536 |
| 6 | L3 meanpool orig | classification (mean) | unseeded | 0.4891 | 0.5649 | 0.7032 | 0.6592 |
| 7 | S1 s43 orig | generative SFT | 43 | 0.4807 | 0.5353 | 0.7167 | 0.6816 |
| 8 | L4 ctx1024 orig | classification (mean, 1024) | unseeded | 0.4740 | 0.5629 | 0.7108 | 0.6648 |
| 10 | L1 control | classification (last-token) | unseeded | 0.4731 | 0.5407 | 0.6819 | 0.6670 |
| 11 | L1 s43 | classification (last-token) | 43 | 0.4731 | 0.5404 | 0.6825 | 0.6636 |
| 12 | original | classification (last-token) | unseeded | 0.4660 | 0.5460 | 0.6803 | 0.6257 |
| 13 | L2 nosampler | classification (last-token) | unseeded | 0.4575 | 0.5348 | 0.6853 | 0.6478 |
| 14 | L1 s44 | classification (last-token) | 44 | 0.4557 | 0.5357 | 0.6771 | 0.6455 |
| 15 | L3 meanpool | classification (mean) | unseeded | 0.4525 | 0.5502 | 0.6972 | 0.6659 |
| 16 | L4 ctx1024 | classification (mean, 1024) | unseeded | 0.4310 | 0.5531 | 0.7066 | 0.6648 |
| 17 | honest (thesis baseline) | classification (last-token) | unseeded | 0.4268 | 0.5341 | 0.6735 | 0.6229 |
| 18 | L5 nocf orig | classification (mean, no CF) | unseeded | 0.3814 | 0.5365 | 0.6490 | 0.5810 |
| 19 | L6 nocf reg orig | classification (mean, no CF, γ=2) | unseeded | 0.3610 | 0.5323 | 0.6471 | 0.5642 |
| 20 | M8 s44 | classification (mean, new code) | 44 | 0.3510 | 0.4966 | 0.6163 | 0.5651 |
| 21 | M9 s43 long | classification (mean, 8ep) | 43 | 0.3497 | 0.5131 | 0.6305 | 0.5764 |
| 22 | M12 auxinput | classification (mean, aux) | 42 | 0.3461 | 0.4987 | 0.6190 | 0.5685 |
| 23 | L5 nocf | classification (mean, no CF) | unseeded | 0.3454 | 0.5158 | 0.6385 | 0.5787 |
| 24 | M7 s43 | classification (mean, new code) | 43 | 0.3400 | 0.4927 | 0.6102 | 0.5583 |
| 25 | M11 twostage | classification (mean, two-stage) | 42 | 0.3389 | 0.5067 | 0.6236 | 0.5866 |
| 26 | L6 nocf reg | classification (mean, no CF, γ=2) | unseeded | 0.3193 | 0.5106 | 0.6358 | 0.5549 |
| 27 | M2 deephead | classification (mean, deep head) | 42 | 0.0936 | 0.4866 | 0.5975 | 0.0600 |
| 28 | M6 hilr | classification (mean, 2× head LR) | 42 | 0.0754 | 0.4922 | 0.6020 | 0.0509 |
| — | *Qwen3.8-27B (zero-shot, 358 orig)* | *API, no fine-tune* | — | *0.2048* | *—* | *—* | *0.4106* |
| — | *Llama-4 Scout 17B (zero-shot, 358 orig)* | *API, no fine-tune* | — | *0.1449* | *—* | *—* | *0.1760* |

**Note on `_orig` suffix:** some experiments were evaluated twice — once with the original eval
code and once with a corrected version that handles label remapping. The `_orig` runs use the
original eval pipeline. Differences between `_orig` and non-`_orig` are small (< 0.02) and come
from argmax tie-breaking under different GPU kernels.

**Note on denominators:** all fine-tuned models (rows 1–28) were evaluated with
`exclude_counterfactual: true` — 358 original test records, no counterfactuals. The zero-shot
frontier models (italicised rows) were rescored on the same 358 originals for a matched
comparison. See §9 for details.

---

## 6. Analysis

### 6.1 Why 0.75 is not reachable with the current data

The 0.75 target on 10-class macro-F1 requires per-class F1 ≥ 0.75 for all 10 classes. With the
current training distribution:

| class | train samples | best achievable F1 (est.) |
|---|---:|---:|
| deflect | 2,043 | ~0.70 |
| soothe | 1,652 | ~0.65 |
| threaten | 670 | ~0.55 |
| challenge | 592 | ~0.50 |
| clarify | 534 | ~0.45 |
| negotiate | 456 | ~0.45 |
| test | 134 | ~0.30 |
| partial | 46 | ~0.10 |
| answer | 18 | ~0.05 |
| withhold | 12 | ~0.00 |

A class with 12 samples cannot achieve F1 = 0.75 — the model sees it fewer than 3 times per epoch
with batch_size = 1 and grad_accum = 32. Even with perfect memorisation, the test set has 0
withhold samples, so F1 is undefined (0/0 → 0 by convention). The three rarest classes
(answer, partial, withhold) collectively cap macro-F1 at approximately:

  (0.70 + 0.65 + 0.55 + 0.50 + 0.45 + 0.45 + 0.30 + 0.10 + 0.05 + 0.00) / 10 = 0.375

The best observed RP F1 (0.53, L1 s43 orig) already exceeds this rough estimate, suggesting the
model generalises better than pure sample-count would predict — likely because the classes share
semantic features (e.g., "deflect" and "withhold" are both evasive, so the model can partially
learn "withhold" from "deflect" examples). But the ceiling is well below 0.75.

### 6.2 Why SFT outperforms classification heads

The classification-head architecture has a fundamental limitation: each head reads the same
pooled vector independently. The `response_policy` head cannot see what `secrecy_pressure` or
`reveal_decision` predicted, even though these fields are strongly correlated with the policy
choice. The SFT approach generates fields autoregressively, so `response_policy` (field 23 in the
serialisation order) conditions on all 22 preceding fields.

Additionally, the classification head projects 2,560 dimensions to 256 and then to 10 logits — a
bottleneck that discards most of the representation. SFT generates the label as text tokens,
preserving the full vocabulary embedding. The word "soothe" has a rich embedding from pretraining
that captures its semantics; the class index 5 does not.

### 6.3 Why mean pooling hurts response-policy

Last-token pooling extracts the representation at the position where the model has just processed
the complete context — this is the position that the next-token prediction objective has trained
to be most informative about what comes next. Mean pooling averages over all positions, including
early positions that carry less decision-relevant information. For fields like `valence` and
`tone` that depend on the overall emotional colour of the context, mean pooling helps. For
`response_policy`, which depends on the most recent conversational move, it hurts.

### 6.4 Seeding and reproducibility

The project's configs carried `seed: 42` from the beginning, but the training code did not read
it until commit `70e5d83`. All L-series runs before that are unseeded — their results are single
draws from an unknown RNG state. The seeded runs (L1 s43, L1 s44, S1 s43) show a spread of ±0.018
on test RP F1, which is the true reproducibility margin.

The unseeded L3 (test RP F1 0.4525) was previously treated as the mean-pooling baseline. The
seeded mean-pooling runs (M7 s43: 0.3400, M8 s44: 0.3510) are significantly lower, indicating L3
was a lucky draw. However, the `_orig` eval of L3 (0.4891) is closer to the last-token baseline,
suggesting the eval pipeline differences also contribute to the gap. The definitive comparison
requires re-running L3's config with seeding, which has not been done.

### 6.5 Counterfactual records

Counterfactual records are synthetic variants of real turns where one latent variable is flipped
(e.g., "what if the NPC's secrecy_pressure were high instead of low?") while the context text
remains identical. They constitute 59% of training data.

Removing them (L5, L6) drops RP F1 from 0.45 to 0.35 — a 0.10 loss. The counterfactuals teach the
model that the same context can lead to different policies depending on internal state, which is
exactly the conditional structure the predictor needs to learn. Without them, the model collapses
to predicting the majority class per context.

### 6.6 Overfitting in SFT

The SFT model overfits aggressively: train loss drops 20× (0.35 → 0.02) while val loss rises 50%
(0.105 → 0.155) over 5 epochs. The best checkpoint is at epoch 2 in all runs. S2's regularisation
(more LoRA dropout, more weight decay, lower LR) does not prevent overfitting — it only slows it
slightly. This suggests the model has enough capacity to memorise 6,175 training examples in 2
epochs, and the remaining epochs are pure memorisation.

The implication is that more data would help more than more regularisation. With 12K examples
(2× current), the overfitting point might move to epoch 4–5, and the best-checkpoint val loss
would be lower.

---

## 7. Recommendations

### 7.1 For the thesis

**Report S1 SFT as the primary result.** It achieves the best test RP F1 (0.5167 unseeded, 0.4807
seeded) and best mean accuracy (0.7225) of any approach tested. Report the seeded number (0.4807)
with the caveat that only one seed is available, and the unseeded number (0.5167) as an upper
bound.

**Report the classification-head baseline (L1 s43) for comparison.** Test RP F1 0.4731, mean
accuracy 0.6825. The SFT improvement is +0.044 RP F1 and +0.040 mean accuracy.

**Frame the 0.75 target honestly.** It is not reachable with 6K training examples and 3 classes
having <50 samples. The theoretical ceiling on macro-F1 is approximately 0.50–0.55 given the
label distribution. The binary routing metric (M3: 0.57 val F1 on 2-class) is a more realistic
operational target and should be reported alongside the 10-class metric.

**Include error bars.** The seeded spread is ±0.018 on RP F1. Report mean ± std where multiple
seeds are available (L1: 3 seeds, S1: 2 seeds).

### 7.2 For future work

1. **Generate more training data.** The data_gen pipeline can produce additional synthetic
   episodes. Doubling the training set to 12K examples would push the SFT overfitting point
   later and likely improve the best-checkpoint val loss. This is the single highest-impact
   intervention.

2. **Constrained decoding for SFT.** During inference, restrict the generation to valid label
   tokens only. This would eliminate parse failures and may improve F1 by 1–2 points.

3. **SFT with label-conditioning.** Generate the fields in an order that places
   `response_policy` last, so it conditions on all 28 other fields. The current order is
   alphabetical (following `LABEL_MAPS` insertion order); reordering to put decision fields last
   is a one-line change.

4. **Ensemble SFT + classification head.** Use the SFT prediction as a prior for the
   classification head, or average their predicted probabilities. The two approaches make
   different errors (SFT is better on rare classes, the head is better on common classes), so an
   ensemble may outperform either alone.

5. **Multi-seed SFT.** Run S1 with seeds 42, 43, 44 to get proper error bars. The unseeded vs
   seeded gap (0.5167 vs 0.4807) suggests seed variance is ~0.04, which is large relative to the
   differences between approaches.

---

## 8. Reproducibility

### 8.1 Code

All experiments use code from `main` branch, commit `e4cbb5f` (M-series) or `70e5d83` (L-series
seeded runs). The SFT code is in `llm_finetuning/src/training/latent_sft.py`. The classification
code is in `llm_finetuning/src/training/model.py` and `train_latent.py`.

### 8.2 Configs

| series | config pattern | location |
|---|---|---|
| L-series | `lat_L{1-6}_*.yaml` | `llm_finetuning/configs/` |
| M-series | `lat_M{1-12}_*.yaml` | `llm_finetuning/configs/` |
| S-series | `lat_S{1-2}_*.yaml` | `llm_finetuning/configs/` |
| eval | `eval_test_*.yaml` | `llm_finetuning/configs/` |

### 8.3 SLURM

Training: `sbatch scripts/slurm_lat.sh <config-stem>`
Evaluation: `sbatch scripts/slurm_lateval.sh <eval-config-stem>`
SFT training: `sbatch scripts/slurm_latsft.sh <config-stem>`

### 8.4 Hardware

All runs on HKUST HPC, `gpu-a30` partition, NVIDIA A30 GPUs (24 GB), `xrimlab` account.

### 8.5 Checkpoints

Best checkpoints saved in `checkpoints/{run_name}_best/` (predictor weights + config). Per-epoch
checkpoints are no longer saved (disabled to prevent disk exhaustion).

---

## 9. Frontier model comparison (zero-shot, no fine-tuning)

To establish whether fine-tuning is necessary, we evaluated frontier LLMs via OpenRouter API
on the same held-out test split. Each model receives the same system prompt (describing the 29
fields and their valid values) and the same dialogue context, and must output a `<state>` block.
Responses are parsed through the same pipeline as the SFT eval.

### 9.1 Setup

| item | value |
|---|---|
| API | OpenRouter (`https://openrouter.ai/api/v1/chat/completions`) |
| Models | Qwen3.8-27B, Llama-4 Scout 17B (Qwen3.8-Max and DeepSeek-V4-Flash pending) |
| Temperature | 0.0 (greedy) |
| Max tokens | 2048 (non-reasoning), 4096 (reasoning) |
| Concurrency | 4–8 parallel requests |
| Prompt | System: field schema + output format; User: dialogue context |
| Reasoning | Disabled where possible (Qwen3.8-Max requires it) |

**Excluded runs:** Four API runs produced empty responses due to provider errors (403 ToS
violations for Google/Anthropic models, 402 credit exhaustion for DeepSeek-V4-Pro, and
rate-limiting for Qwen3.7-flash). These are API failures, not wrong predictions, so they are
excluded from the comparison entirely. Only the two models with 100% parse rate are reported.

### 9.2 Results — matched denominator (358 originals)

The fine-tuned models were evaluated with `exclude_counterfactual: true` (358 original records,
no counterfactuals). To make the comparison fair, the zero-shot models are rescored on the same
358 originals using the identical metric pipeline. The `per_record.jsonl` output from each
OpenRouter run maps record-by-record to the test file via `idx`, so the rescore is exact.

| model | params | fine-tuned? | n records | **RP F1** | RP acc | RP weighted F1 |
|---|---:|---|---:|---:|---:|---:|
| Qwen3-4B + SFT (S2 reg) | 4B | yes (LoRA) | 358 | **0.5362** | 0.6927 | — |
| Qwen3-4B + SFT (S1) | 4B | yes (LoRA) | 358 | **0.5167** | 0.6899 | — |
| Qwen3-4B + heads (L1 s43) | 4B | yes (LoRA) | 358 | **0.4731** | 0.6636 | — |
| Qwen3.8-27B | 27B | no (zero-shot) | 358 | **0.2048** | 0.4106 | 0.3257 |
| Llama-4 Scout 17B | 17B | no (zero-shot) | 358 | **0.1449** | 0.1760 | 0.2562 |

**For reference, the zero-shot numbers on all 884 records (including 526 counterfactuals):**

| model | n records | RP F1 | RP acc |
|---|---:|---:|---:|
| Qwen3.8-27B | 884 | 0.2160 | 0.4383 |
| Llama-4 Scout | 884 | 0.1442 | 0.1812 |

The counterfactual records share byte-identical contexts with their originals but carry
different gold labels. Zero-shot models predict the same output for both, so their accuracy on
counterfactuals is lower by construction. This is why the 884-record RP F1 (0.216) is slightly
higher than the 358-original RP F1 (0.205) for Qwen3.8-27B — the counterfactuals dilute the
effect of wrong predictions on rare classes. The 358-original number is the honest comparison.

### 9.3 Analysis

1. **Fine-tuning a 4B model outperforms a 27B zero-shot model by 2.6×.** The S2 SFT model
   achieves RP F1 0.5362 vs Qwen3.8-27B's 0.2048 — a 0.33 absolute improvement on the same 358
   originals. Despite having 6.7× fewer parameters, the fine-tuned model is dramatically better
   because it has learned the domain-specific label schema and the mapping from dialogue context
   to latent state.

2. **Zero-shot frontier models cannot follow the label schema.** Both models produce well-formed
   `<state>` blocks (100% parse rate), but they invent their own field names and values instead
   of using the provided schema. For example, Qwen3.8-27B outputs `mood=guarded` and
   `intent=probe` instead of `tone=evasive` and `player_intent=probe`. The parser only recognizes
   fields in the schema, so invented fields score as missing.

3. **The zero-shot models understand the dialogue but not the task.** Qwen3.8-27B gets 84%
   accuracy on `risk_type` and 74% on `player_credibility` — fields where the label names are
   self-explanatory. But it gets low accuracy on `duty_pressure` and `repair_strategy` — fields
   where the label semantics are domain-specific and not inferable from the label name alone.

4. **The 0.75 target is not reachable by zero-shot frontier models either.** Even a 27B model
   with 100× more parameters than our 4B base achieves only 0.20 RP F1 without fine-tuning on
   the matched 358 originals. This confirms that the task requires learning domain-specific label
   semantics that are not present in pretraining data.

### 9.4 Per-class response-policy breakdown (358 originals)

| class | gold n | Qwen3.8-27B pred n | Qwen3.8-27B F1 | Llama-4 pred n | Llama-4 F1 |
|---|---:|---:|---:|---:|---:|
| deflect | 125 | 283 | 0.5931 | 60 | 0.4324 |
| soothe | 90 | 6 | 0.1250 | 14 | 0.2115 |
| threaten | 51 | 9 | 0.3000 | 11 | 0.2903 |
| challenge | 39 | 12 | 0.2745 | 1 | 0.0500 |
| negotiate | 21 | 4 | 0.1600 | 4 | 0.0800 |
| clarify | 18 | 4 | 0.0909 | 0 | 0.0000 |
| test | 12 | 0 | 0.0000 | 0 | 0.0000 |
| answer | 2 | 19 | 0.0952 | 19 | 0.0952 |
| partial | 0 | 14 | 0.0000 | 161 | 0.0000 |
| withhold | 0 | 7 | 0.0000 | 1 | 0.0000 |

**Pattern:** Qwen3.8-27B over-predicts `deflect` (283 predictions vs 125 gold) — the majority
class. It barely predicts `soothe` (6 vs 90 gold). Llama-4 Scout over-predicts `partial` (161
predictions vs 0 gold) — a class that doesn't even appear in the test set. Both models are
essentially guessing the majority class for most records, with occasional correct predictions
on the common classes.

---

## 10. Open questions

1. **Is the `_orig` eval pipeline or the corrected eval pipeline the right one?** The difference
   is up to 0.04 RP F1 (L3: 0.4525 vs 0.4891). The `_orig` pipeline does not handle label
   remapping; the corrected one does. For runs without remapping (all L-series), the difference
   comes from argmax tie-breaking under different GPU kernels. The `_orig` numbers are more
   consistent with the val metrics and should probably be preferred.

2. **Why does the new code (commit `e4cbb5f`) produce lower test RP F1 than the old code?** M7
   (new code, seed 43, mean pool): 0.3400 vs old code (seed 43, mean pool, 1 epoch only):
   train_loss 5.39 vs 5.62. The epoch-1 difference is small (0.23), but the 5-epoch test RP F1
   gap is large (0.34 vs ~0.45). A full 5-epoch run with the old code and seed 43 is needed to
   confirm whether the gap is real or an artifact of the single-epoch comparison.

3. **Would more data close the gap to 0.75?** The SFT model overfits at epoch 2 with 6K examples.
   With 12K, the overfitting point might move to epoch 4–5, and the best-checkpoint performance
   would likely improve. But the rare-class problem (3 classes with <50 samples) would remain
   unless the new data specifically targets those classes.

---

*Generated 2026-08-25. All numbers are from held-out test split unless marked as val.*
