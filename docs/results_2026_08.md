# Experimental Results — August 2026

Two independent result sets, both produced on HKUST HPC (`gpu-a30`, `xrimlab`):

1. **Small LM (from-scratch NPC dialogue LM)** — perplexity reduced 42.18 → 18.11 on held-out test.
2. **29-head latent state predictor (Qwen3-4B + LoRA)** — first evaluation on a genuinely held-out test split.

Both sections report numbers on data that was **never used for checkpoint selection**. Where an
earlier reported number differs, the discrepancy is decomposed rather than silently replaced.

---

## 1. Small LM — dialogue language modelling

### 1.1 Setup

| item | value |
|---|---|
| architecture | `TinyGPTLM`, 6 layers, `n_embd` 512, 8 heads, 44.77 M params, weights tied |
| tokenizer | GPT-2 BPE via tiktoken, vocab 50257 (identical across all runs) |
| context | 256 tokens |
| in-domain train | `data/dialogue/train.txt` — 545,223 tokens (2,184 blocks) |
| selection set | `data/dialogue/val_sel.txt` — 12,455 tokens (46 blocks, 8 openers) |
| **test set** | `data/dialogue/test.txt` — 15,744 tokens (69 blocks, 17 openers) |
| external corpus | `data/external/merged_dialogue.txt` — 107,338,428 tokens (402 MB) |
| seeds | 42 / 43 / 44 (3 per config) |

The val/test split is produced by `slm_training/scripts/make_val_test_split.py`, which groups blocks
by their opening PLAYER line before splitting. The corpus expands each dialogue into growing-prefix
blocks that share an opener, so a naive per-block split would place a block and its own prefix on
opposite sides. The script asserts that openers and NPC lines are disjoint across the two halves;
neither half shares an NPC line with `train.txt`.

### 1.2 Results

Perplexity, mean ± population standard deviation over 3 seeds. **`test_ppl` is the number to quote**;
`val_ppl` selected the checkpoint and is therefore optimistically biased.

| config | change from baseline | val_ppl | **test_ppl** | test_bpc | best epoch |
|---|---|---:|---:|---:|---:|
| `slm_D_finetune` | **pretrain on 107 M external tokens → fine-tune** | 14.65 ± 0.03 | **18.11 ± 0.01** | 1.036 | 10–12 |
| `slm_F_reg` | dropout 0.35, weight decay 0.3 | 29.97 ± 0.41 | 39.39 ± 0.45 | 1.314 | 5–6 |
| `slm_G_reg2` | dropout 0.45, weight decay 0.5 | 29.79 ± 0.89 | 39.58 ± 1.12 | 1.316 | 7–8 |
| `slm_H_stride` | F + stride 64 (4× windows) | 29.81 ± 0.28 | 39.84 ± 0.99 | 1.318 | 3 |
| `slm_B_improved` | warmup+cosine, stride 128, dropout 0.2 | 31.49 ± 0.49 | 40.98 ± 0.89 | 1.328 | 4 |
| `slm_A_baseline` | previous best recipe (control) | 33.17 ± 0.53 | 42.18 ± 0.80 | 1.339 | 12 |
| `slm_E_small` | 16 M params (`n_embd` 256, 4 layers) | 32.20 ± 0.70 | 43.41 ± 0.46 | 1.349 | 7–9 |
| `slm_C_pretrain` | external corpus only, zero-shot in-domain | 50.57 | 51.83 | 1.412 | 5 |

Relative to the `slm_A_baseline` control, on test perplexity:

- `slm_D_finetune` **−57.1 %**
- `slm_F_reg` −6.6 %, `slm_G_reg2` −6.2 %, `slm_H_stride` −5.5 %, `slm_B_improved` −2.9 %
- `slm_E_small` +2.9 % (worse)

`test_bpc` (bits per character) is reported alongside perplexity because it is tokenizer-invariant:
unlike perplexity it remains comparable if the vocabulary is ever changed.

### 1.3 Interpretation

**The from-scratch ceiling is a data limit, not a capacity or recipe limit.** Three independent
observations support this:

1. Recipe and regularisation improvements saturate. `slm_F_reg`, `slm_G_reg2` and `slm_H_stride`
   (39.39 / 39.58 / 39.84) are statistically indistinguishable given seed spreads of ±0.45–1.12.
   Pushing dropout from 0.35 to 0.45, or quadrupling window overlap, buys nothing further.
2. Reducing capacity makes results *worse* (`slm_E_small`, 43.41), ruling out over-parameterisation
   as the binding constraint.
3. Every from-scratch configuration peaks at **epoch 3–6** of 20–30 and then overfits, whereas the
   pretrained model keeps improving through **epoch 10–12**. Pretraining regularises more
   effectively than any dropout setting tested.

The binding constraint is 545 K in-domain tokens against 44.77 M parameters. Pretraining on 107 M
external tokens breaks it, yielding a **54 % improvement over the best from-scratch run** (39.39 →
18.11) and 57 % over the control.

Notably, the pretrained model *before* fine-tuning is **worse** in-domain than any from-scratch model
(`slm_C_pretrain`, 51.83), because the external corpus is anime/roleplay dialogue while the target is
medieval-fantasy NPC dialogue. Its value is entirely as an initialisation. In-domain validation
perplexity nonetheless improved monotonically across pretraining epochs (65.17 → 58.60 → 53.78 →
51.67 → 50.57), indicating transferable dialogue structure rather than source-domain memorisation.

### 1.4 Threats to validity

- **`slm_D_finetune`'s ±0.01 is not a full error bar.** All three seeds fine-tune from the *same*
  `slm_C_pretrain_s42` checkpoint, so the spread measures fine-tuning variance only; pretraining is
  effectively n = 1. Independent error bars require 2–3 separate pretraining runs.
- **The test set is small** (15,744 tokens, 69 blocks). Differences below roughly 1 perplexity point
  are not meaningful. The 24-point D-vs-control gap is far outside that margin; the F/G/H ordering
  is not.
- **Contamination was checked, not assumed.** All 67 unique test NPC lines of ≥ 40 characters were
  searched against the full 402 MB external corpus: 0 matches. 0 of 215 test NPC lines appear in
  `train.txt`.
- The numbers above were produced by the code as it stood on the cluster at the time of the runs.
  The merged `main` uses the `warmup_cosine` scheduler, which decays to `min_lr_ratio` 0.1 rather
  than to `eta_min/lr` ≈ 0.003. A re-run should land close but is not guaranteed bit-identical.

### 1.5 Code changes underlying the improvement

| change | file | effect |
|---|---|---|
| GPT-2 residual init scaling `0.02/√(2·n_layer)` | `small_lm_architectures.py` | early gradient norms 300–500 → 43–78; previously the clipper (max_norm 1.0) renormalised away nearly all early signal |
| fused `scaled_dot_product_attention` | `small_lm_architectures.py` | throughput; avoids materialising the (B,H,T,T) matrix |
| full-val evaluation (`max_batches` 200 → 0) | `run_small_lm.py` | validation previously scored only the first 200 batches — always the same prefix, since val is unshuffled — and that truncated number drove checkpoint selection |
| token-weighted validation loss | `run_small_lm.py` | a short final batch can no longer skew the mean |
| `train_stride` (overlapping windows) | `run_small_lm.py` | more training examples per epoch from the same corpus |
| warmup + single cosine decay | `run_small_lm.py` | `cosine_warm_restarts` (T_0 = 5, T_mult = 2) restarts at epochs 5/15/35, so a 20-epoch run ended mid-cycle at high LR and was never annealed |
| held-out test evaluation of the best checkpoint | `run_small_lm.py` | previously no test split existed at all |
| `device` selected before the embedding extractor | `run_small_lm.py` | fixed an `UnboundLocalError` swallowed by a bare `except`, which silently fell back to zero conditioning — every prior "semantic conditioning" A/B compared zero against zero |
| full RNG seeding | `run_small_lm.py` | reproducibility |

Reproduce with `bash scripts/push_slm_to_hpc.sh` (sync + submit) and
`python slm_training/scripts/aggregate_slm_push.py` (table). Raw per-run data:
`slm_training/artifacts/slm_push_results.json` (untracked — `artifacts/` is gitignored).

---

## 2. 29-head latent state predictor

### 2.1 Setup

Qwen3-4B, 4-bit NF4 quantisation, LoRA r = 16 on `q,v,k,o,gate,up,down`, 29 classification heads
(17 schema fields + 6 stance dimensions × {level, delta}), last-token pooling, `max_seq_len` 512.
Checkpoint `checkpoints/latent_predictor_best`.

| split | episodes | turns |
|---|---:|---:|
| train | 587 | 6,175 |
| val (selection) | 69 | 683 |
| **test (held out)** | 80 | **884** |

Splits are episode-level (`src/packaging/splitter.py`), so no turn from a training episode appears in
val or test. **The test split existed but had never been evaluated** — all previously reported latent
numbers are val numbers, and val also drove early stopping and best-checkpoint selection.

### 2.2 Results

| metric | val (as previously reported) | val (corrected metric) | **test (corrected metric)** |
|---|---:|---:|---:|
| mean accuracy | 0.6875 | 0.6880 | 0.6735 |
| mean macro-F1 | 0.5425 | 0.5409 | 0.5341 |
| mean balanced accuracy | 0.5489 | 0.5474 | 0.5516 |
| mean MCC | 0.4879 | 0.4890 | 0.4663 |
| **`response_policy` macro-F1** | 0.6210 | 0.6218 | **0.4268** |
| stance-delta accuracy | 0.5961 | 0.5946 | 0.5711 |

Against the project's own acceptance thresholds, **on test**:

| check | value | threshold | verdict |
|---|---:|---:|---|
| `response_policy_f1` | 0.4268 | ≥ 0.75 | **FAIL** |
| `stance_delta_accuracy` | 0.5711 | ≥ 0.70 | **FAIL** |
| `secret_leakage_rate` | 0.0000 | ≤ 0.05 | PASS |

### 2.3 Interpretation

**The aggregate system generalises; the decision head does not.** Mean macro-F1 moves only
0.5409 → 0.5341 from val to test and balanced accuracy actually rises slightly — across 80 unseen
episodes that is a stable result. But `response_policy`, the head the routing layer depends on,
drops **31 % relative** (0.6218 → 0.4268; accuracy 0.716 → 0.623). Both splits contain 9 of its 10
classes in gold, so this is not a class-mix artefact.

Per-head macro-F1 on test, extremes:

| weakest | | strongest | |
|---|---:|---|---:|
| `tone` | 0.376 | `valence` | 0.802 |
| `familiarity_delta` | 0.388 | `duty_pressure` | 0.777 |
| `risk_type` | 0.407 | `face_pressure` | 0.731 |
| `dominance_level` | 0.410 | `affection_level` | 0.699 |
| `dominance_delta` | 0.415 | | |
| `response_policy` | 0.427 | | |

`risk_type` is diagnostic: **86.9 % accuracy but 0.407 macro-F1 and 0.398 balanced accuracy** — a
head predicting the majority class and little else. The pattern is consistent across the weak heads:
balanced-class heads work, rare-class and stance-*delta* heads collapse toward the majority class.
This occurs despite the training pipeline applying three imbalance corrections simultaneously
(inverse-frequency class weights, a `WeightedRandomSampler`, and focal loss with γ = 1.5). The
sampler assigns each record the maximum class weight across all 28 single-label heads, so it
reshapes the marginal distribution of every other head as a side effect — a plausible contributor
rather than a remedy.

### 2.4 Correction to the macro-F1 metric

`compute_latent_metrics` previously called `f1_score(..., average="macro")` without `labels=`,
letting scikit-learn infer the label set from the values present. Two variants are now reported:

- **`macro_f1`** — averaged over classes with gold support in the split. The standard choice and the
  one quoted above: a class that never occurs cannot be scored, and counting it as 0 would penalise
  the model for the split's composition. (This still differs from the scikit-learn default, which
  averages over gold ∪ pred and so also counts classes the model hallucinates but that never occur.)
- **`macro_f1_schema`** — averaged over the head's full label schema with absent classes scored 0.
  Pessimistic, but the only variant comparable across splits and ablations, since its denominator
  does not shift with the class mix. For `response_policy`: 0.560 val, 0.384 test.

The correction's effect on previously reported figures is **negligible** (0.6210 → 0.6218 for
`response_policy` on val). The val → test change is what matters.

### 2.5 Caveats carried forward

- These are **latent-head numbers only**. `eval_response` conditions the generator on gold Z_t, so
  response-quality and leakage figures measure an oracle-conditioned upper bound, not the end-to-end
  pipeline. `routing_mode: predicted` in the eval config is read only by `eval_routing`.
- The head-ablation script defaults its training file to `cfg["data"]["test_heads_file"]` when
  `train_heads_file` is absent, and no invocation in `scripts/experiments.sh` or
  `scripts/slurm_experiments.sh` passes it — so previously reported ablation rows were trained on
  their own evaluation split.
- Temperature/isotonic calibration is documented as being fit on `train_heads.jsonl`, i.e. data the
  model has already fitted, which under-corrects confidence.
- Single seed: the latent predictor has not been run with multiple seeds, so no error bars.

Reproduce: `python llm_finetuning/run_eval.py --stage latent --config llm_finetuning/configs/eval_test.yaml`
(writes `eval_results/test_honest/`). The val-with-corrected-metric comparison uses
`configs/eval_val_fixedmetric.yaml`.

---

## 3. The evaluation set contains counterfactual duplicates

Every packaged split carries **up to four records per turn**: one original plus counterfactual
variants. Across the 335 duplicated groups in the test split, **100% share an identical `context`
string** and **91% carry differing labels**. The same input therefore appears several times with
conflicting gold labels, which mechanically caps what any deterministic classifier can score.

| split | records | unique turns | counterfactual records |
|---|---:|---:|---:|
| train | 6,175 | 2,571 | — |
| val | 683 | 274 | — |
| test | 884 | **363** | **526 (60%)** |

`packager.py` tags and counts `counterfactual` deliberately, but **no evaluation code filters on
it** — not `eval_latent.py`, not `eval_response.py`, not `dataset.py`. Every latent metric reported
anywhere in this project, including §2 above, was computed over a set that is 60% augmentations.

The per-field conflict rate tracks the weak-head pattern:

| field | groups with conflicting labels | test macro-$F_1$ |
|---|---:|---:|
| `valence` | 0% | 0.802 |
| `reveal_decision` | 10% | 0.659 |
| `response_policy` | 15% | 0.427 |
| `secrecy_pressure` | 22% | 0.491 |
| `trust_delta` | 23% | weak |

Re-evaluating the same checkpoint on originals only (`test_heads_original.jsonl`, 358 records,
358 unique turns) raises `response_policy` macro-$F_1$ from **0.4268 to 0.4660** (+9%).

This is a second, independent cause alongside class imbalance. `risk_type` has 0% conflict yet still
collapses (0.869 accuracy, 0.407 macro-$F_1$), so imbalance explains that one; conflicting
counterfactual labels explain the stance-delta and decision fields.

**Recommendation:** report the 358-turn original-only figure as the primary number, and treat
counterfactual records as a training augmentation, not evaluation data.

---

## 4. Configuration sweep — what actually helps

Four configurations, each isolating one variable, all selecting on `val/mean_macro_f1` rather than
the noisier `val/response_policy_f1`. Test = 884 records.

| run | change | val macro-$F_1$ | test macro-$F_1$ | test acc | test `response_policy` | test stance-delta |
|---|---|---:|---:|---:|---:|---:|
| L1_control (s42) | current recipe | 0.5277 | 0.5407 | 0.6819 | 0.4731 | 0.5930 |
| L1_control (s43) | " | 0.5193 | 0.5404 | 0.6825 | 0.4731 | 0.6043 |
| L1_control (s44) | " | 0.5297 | 0.5357 | 0.6771 | 0.4557 | 0.5853 |
| L2_nosampler | weighted sampler removed | 0.5376 | 0.5348 | 0.6853 | 0.4575 | 0.5892 |
| L3_meanpool | + mean pooling | 0.5550 | 0.5502 | 0.6972 | 0.4525 | 0.6126 |
| **L4_ctx1024** | + 1024 context, 8 epochs | **0.5590** | **0.5531** | **0.7066** | 0.4310 | **0.6283** |

### 4.1 The sampler hypothesis is rejected

Removing the `WeightedRandomSampler` (L2) did **not** improve the weak heads: test macro-$F_1$ falls
slightly (0.5348 vs 0.5407) and `response_policy` falls (0.4575 vs 0.4731). The earlier conjecture
that the sampler distorts other heads' marginals by oversampling on the per-record rarest field is
**not supported**. The limitation is not a sampler artefact.

### 4.2 Pooling and context help the aggregate, not the decision head

Mean pooling (L3) and longer context (L4) improve mean macro-$F_1$ by 1.8% and 2.3%, accuracy by
2.2% and 3.6%, and stance-delta accuracy by 3.3% and 6.0%. None improves `response_policy`; L4 is in
fact the worst on it (0.4310). Across all six runs `response_policy` stays in **0.43--0.47** on test.

### 4.3 The chapter's 0.621 is a selection artefact, not a generalisation gap

This is the most consequential result of the sweep. Three seeds of the identical control
configuration:

| | seed 42 | seed 43 | seed 44 | sd |
|---|---:|---:|---:|---:|
| **val** `response_policy` | 0.5519 | 0.3988 | 0.3989 | **0.072** |
| **test** `response_policy` | 0.4731 | 0.4731 | 0.4557 | **0.008** |

Validation `response_policy` has a seed spread nine times larger than test. The original checkpoint
was selected with `metric_for_best_model: val/response_policy_f1`, i.e. the best epoch *for that
specific noisy metric on the selection split*, which is the textbook way to obtain an optimistic
point estimate. Selecting instead on `val/mean_macro_f1` yields a checkpoint whose test
`response_policy` is **better** (0.4731 vs the original checkpoint's 0.4268).

So the earlier framing — "validation 0.622 collapses to 0.427 on test, a 31% generalisation gap" —
is wrong. The model was never at 0.62 in any unbiased sense. Its unbiased performance is ~0.45--0.47
on both splits, and the validation figure was inflated by maximising the reported metric over epochs.

---

## 5. Head ablations are unusable in both existing forms

| experiment | heads | original matrix (val, train-on-eval) | corrected (test, retrained) | precision | recall | slow-path rate |
|---|---:|---:|---:|---:|---:|---:|
| exp_a_routing_only | 4 | 0.698 | 0.6662 | 0.4994 | 1.0000 | 0.9966 |
| exp_b_plus_affect | 7 | 0.686 | 0.6559 | 0.4965 | 0.9659 | 0.9683 |
| exp_c_plus_relational | 6 | 0.666 | 0.6636 | 0.4972 | 0.9977 | 0.9989 |
| exp_d_full_29head | 28 | 0.604 | 0.6590 | 0.4977 | 0.9750 | 0.9751 |

The **original** matrix was trained on its own evaluation split (`run_head_ablation.py` defaulted
`train_file` to `cfg["data"]["test_heads_file"]`; no caller passed `--train-heads-file`). That
default now raises.

The **corrected** re-run is degenerate. Every configuration routes ~97--100% of turns to the careful
path with recall ≈ 1.0 and precision ≈ 0.5, which is exactly the always-careful baseline
($F_1 = 2 \times 0.5 / 1.5 = 0.667$), not a learned router. The ablation protocol — freshly
initialised LoRA and heads, 3 epochs at lr $2\times10^{-5}$, against the main recipe's
$2\times10^{-4}$/$4\times10^{-4}$ over 5 epochs — is far too weak to train a usable model.

**Neither version supports the claim that more fields make routing worse.** The experiment needs a
stronger protocol (warm-start from the full checkpoint, or the main recipe's learning rate and
schedule) before it can be quoted in either direction.

---

## 6. Predictor--human agreement (the third leg of the triangle)

Previously the study reported human--teacher, human--human and AI-validator--teacher agreement, but
never predictor--human. It is now computed. All 78 annotated episodes fall in the **test** split, so
the predictor never trained on them, and all 356 audit turns join to the packaged records at 100%.

Cohen's $\kappa$, 150 turns shared by both annotators, eight annotated fields:

| field | pred--teacher | pred--Human A | pred--Human B | human--human | human--teacher | AI--teacher |
|---|---:|---:|---:|---:|---:|---:|
| valence | 0.498 | 0.230 | 0.040 | −0.06 | 0.12 | 0.32 |
| arousal | 0.364 | 0.071 | 0.004 | 0.07 | −0.01 | 0.04 |
| secrecy_pressure | 0.174 | −0.013 | −0.034 | −0.01 | −0.01 | 0.15 |
| reveal_decision | 0.197 | 0.075 | −0.013 | 0.03 | −0.01 | 0.15 |
| response_policy | 0.410 | 0.313 | 0.012 | 0.00 | 0.14 | 0.40 |
| repair_strategy | 0.310 | 0.176 | −0.040 | 0.04 | 0.04 | 0.14 |
| trust_level | 0.299 | 0.038 | −0.047 | 0.05 | 0.04 | 0.02 |
| familiarity_level | 0.192 | 0.014 | 0.018 | −0.02 | −0.02 | 0.02 |
| **mean** | **0.305** | **0.113** | **−0.008** | **0.013** | **0.036** | **0.155** |

Three readings, in decreasing order of confidence:

1. **The predictor reproduces the teacher far better than humans do** (0.305 vs 0.036) and better
   than the zero-shot AI validator (0.155). This is expected and is *imitation*, not validation: the
   predictor was trained on teacher labels, so this measures how well it learned its supervision.
2. **Predictor--human agreement is near chance** (0.053 averaged over the two annotators), which is
   approximately where human--human agreement already sits (0.013). The predictor is not measurably
   closer to human judgement than two humans are to each other.
3. **The comparison has no usable ceiling.** Because inter-annotator reliability is at chance, there
   is no stable "human judgement" for any model to agree with, and predictor--human $\kappa$ cannot
   be read as a validity measure in either direction. The honest conclusion is that **this annotation
   study cannot validate the predictor**, and computing the third leg confirms that rather than
   resolving it.

The one field with signal across every rater pair is `response_policy` (pred--teacher 0.410,
pred--Human A 0.313, AI--teacher 0.40), suggesting it is the most objectively determined of the
eight. Note also the sharp disagreement between annotators: pred--Human A averages 0.113 while
pred--Human B averages −0.008, consistent with `audit_interpretation.md`'s observation that the two
used very different label distributions.

**Excluded:** seven `audit_synthetic_*.jsonl` files carry `"synthetic": true` in their metadata and
are not human annotations. Only `654cfad…` (Human A) and `67c87fc…` (Human B) are used, matching the
pairing in `audit_interpretation.md`.

Reproduce:
```
python paper/predict_audit_turns.py --config llm_finetuning/configs/eval_test.yaml \
    --audit paper/audit_input_clean.jsonl --heads-file data/splits/test_heads.jsonl \
    --out paper/audit_results/audit_predictor.jsonl
python paper/compute_audit_agreement.py --a paper/audit_results/audit_predictor.jsonl \
    --b paper/audit_results/audit_654cfad67f990b0393b85132.jsonl \
    --teacher paper/audit_input_clean.jsonl --output paper/audit_results/agreement_pred_vs_humanA.json
```
Note `compute_audit_agreement.py` labels its columns `ht_*` and `hh_*` for the human study; with the
predictor passed as `--a`, `ht_*` is predictor--teacher and `hh_*` is predictor--annotator.

---

## 3. Routing evaluation (predicted Z_t)

Routing uses the predicted latent state to decide fast-path vs slow-path for each turn. This is the
end-to-end routing layer, not oracle-conditioned.

| metric | value |
|---|---:|
| routing F1 | 0.7345 |
| routing precision | 0.7155 |
| routing recall | 0.7545 |
| false positive rate | 0.2973 |
| slow-path rate | 0.5249 |
| prediction coverage | 1.0000 |
| n evaluated | 884 |

Source: `eval_results/routing_eval_report.json` (test split, predicted mode).

---

## 4. Response generation evaluation (oracle Z_t)

The response generator is conditioned on **gold** Z_t (oracle), so these numbers are an upper bound
on end-to-end response quality.

| metric | value |
|---|---:|
| BLEU-1 | 0.2617 |
| BLEU-2 | 0.1231 |
| BLEU-4 | 0.0388 |
| ROUGE-L | 0.1387 (CI: 0.1342–0.1431) |
| distinct-1 | 0.1677 |
| distinct-2 | 0.6030 |
| length ratio (gen/ref) | 0.8640 |
| avg gen length | 30.04 tokens |
| avg ref length | 34.77 tokens |
| contradiction rate | 0.0000 |
| degenerate repetition rate | 0.0000 |
| repeated 3-gram rate | 0.0002 |
| prompt artifact rate | 0.0000 |
| **secret leakage rate (gated)** | **0.0000** |
| secret leakage rate (ungated) | 0.0667 |
| sentiment valence accuracy | 0.4016 |

The gated leakage rate is 0 because the reveal_decision head gates output. The ungated rate (0.067)
shows what would happen without the gating mechanism — 59 of 884 turns would leak a secret.

Source: `eval_results/response_eval_report.json` (test split, oracle-conditioned).

---

## 5. Decision-card A/B test (baseline vs structured card)

A paired comparison on 683 val turns: baseline (no decision card) vs treatment (structured decision
card with policy/reveal guidance). Bootstrap CI with episode-level resampling, 95% confidence.

### 5.1 Per-system metrics

| metric | baseline | treatment (card) |
|---|---:|---:|
| ROUGE-L | 0.1263 | 0.0797 |
| BLEU-1 | 0.2189 | 0.1603 |
| BLEU-2 | 0.0439 | 0.0191 |
| distinct-2 | 0.5662 | 0.2077 |
| avg gen length | 30.2 | 25.2 |
| secret leakage rate | 0.0000 | 0.0000 |
| contradiction rate | 0.0000 | 0.0000 |
| **policy consistency** | **0.7349** | **0.8497** |
| exact disclosure match | 0.3382 | 0.3104 |
| over-disclosure rate | 0.0029 | 0.0015 |
| under-disclosure rate | 0.6589 | 0.6881 |

### 5.2 Bootstrap significance tests

| metric | delta (treatment − baseline) | 95% CI | p-value | significant? |
|---|---:|---|---:|---|
| ROUGE-L | −0.0557 | [−0.0691, −0.0423] | 0.012 | **yes** |
| Policy consistency | +0.0499 | [+0.0057, +0.1013] | 0.044 | **yes** |
| Secret leakage | 0.0000 | [0.0, 0.0] | 1.000 | no |
| Contradiction rate | 0.0000 | [0.0, 0.0] | 1.000 | no |
| Over-disclosure | −0.0013 | [−0.0084, +0.0043] | 0.693 | no |
| Under-disclosure | +0.0288 | [−0.0085, +0.0697] | 0.257 | no |

**Interpretation:** The decision card trades fluency (ROUGE-L ↓0.056, BLEU-1 ↓0.059, distinct-2
↓0.36) for policy consistency (+0.050, a 6.8% relative improvement). Both systems achieve zero
gated leakage. The card makes the NPC more policy-compliant but less lexically diverse and less
similar to the reference.

Source: `eval_results/decision_card_ab_report.json`, `eval_results/bootstrap_significance.json`.

---

## 6. Masking ablations (routing head importance)

Each ablation masks one head's prediction (replaced with majority class or random) and re-evaluates
routing. Baseline routing F1 = 0.6721 on val (n=683).

| ablated head | mode | routing F1 | Δ F1 | unsafe fast-path rate |
|---|---|---:|---:|---:|
| (baseline, none) | — | 0.6721 | — | 0.1742 |
| response_policy | majority | 0.5936 | −0.0785 | 0.2401 |
| response_policy | random | 0.6133 | −0.0588 | — |

 is the single most important head for routing: masking it drops F1 by 0.08
(11.7% relative) and increases unsafe fast-path rate from 17.4% to 24.0%.

Source: `eval_results/masking_ablations.json`.

---

## 7. Relational memory evaluation

Compares per-turn relational-state prediction with and without an explicit relational memory module
(accumulating stance vectors across the episode).

| dimension | baseline acc | memory acc | delta | n |
|---|---:|---:|---:|---:|
| trust_level | 0.7169 | 0.7169 | 0.0000 | 862 |
| respect_level | 0.6713 | 0.6678 | −0.0035 | 867 |
| affection_level | 0.7759 | 0.7724 | −0.0035 | 870 |
| familiarity_level | 0.6322 | 0.6334 | +0.0012 | 851 |
| dominance_level | 0.6406 | 0.6338 | −0.0068 | 871 |
| obligation_level | 0.5919 | 0.5930 | +0.0011 | 860 |
| **macro avg** | **0.6717** | **0.6698** | **−0.0019** | — |

**The relational memory module provides no measurable benefit.** The overall delta is −0.002
(worse). The Qwen3-4B backbone already encodes episode context via its context window; an explicit
memory module is redundant at this scale.

Source: `eval_results/relational_memory_eval.json`.

---

## 8. Joint vs separate model comparison

Compares a jointly trained latent+response model against the separate pipeline (latent predictor +
response generator trained independently).

| metric | separate | joint |
|---|---:|---:|
| latent mean accuracy | 0.6825 | 0.6741 |
| latent mean kappa | 0.4375 | 0.4135 |
| response perplexity | 1.04 | 1.04 |
| high-secrecy + full-reveal violations | 1 | 0 |
| hostile + affection-high violations | 1 | 4 |
| total consistency violations / 652 | 2 | 4 |

The separate model has slightly higher latent accuracy (0.683 vs 0.674) and kappa (0.438 vs 0.414).
The joint model eliminates high-secrecy/full-reveal violations but introduces more
hostile/affection-high violations. Response perplexity is identical (1.04).

Source: `eval_results/ablation_joint_vs_separate.json`.

---

## 9. Ablation curve (head subset routing)

Routing F1 as heads are progressively added to the routing decision.

| experiment | n heads | routing F1 | precision | recall | slow-path rate |
|---|---:|---:|---:|---:|---:|
| exp_a_routing_only | 4 | 0.6985 | 0.5374 | 0.9973 | 0.9971 |
| exp_c_plus_relational | 6 | 0.6660 | 0.5451 | 0.8556 | 0.8433 |
| exp_b_plus_affect | 7 | 0.6861 | 0.5332 | 0.9619 | — |

The 4-head routing-only configuration (response_policy, reveal_decision, secrecy_pressure,
value_conflict) achieves the highest F1 (0.6985) with near-perfect recall but very high slow-path
rate (99.7%). Adding relational and affective heads increases precision but reduces recall.

Source: `eval_results/ablation_curve.json`.

---

## 10. Currently running latent ablation experiments (2026-08-21)

Four SLURM jobs launched on HKUST HPC (gpu-a30 partition) at 19:56 HKT, 16-hour time limit:

| job ID | name | GPU | pooling | sampler | ctx len | total steps | status |
|---|---|---|---|---|---|---:|---|
| 1776582 | L1_control | gpu01 | last | weighted | 512 | 965 | RUNNING |
| 1776583 | L2_nosampler | gpu01 | last | none | 512 | 965 | RUNNING |
| 1776584 | L3_meanpool | gpu07 | mean | weighted | 512 | 1544 | RUNNING |
| 1776585 | L4_ctx1024 | gpu07 | last | weighted | 1024 | 965 | RUNNING |

**Common setup:** Qwen3-4B, 4-bit NF4, LoRA r=16 on q/v/k/o/gate/up/down, 33.03 M trainable
params (0.81%), focal loss γ=1.5, label smoothing 0.1, cosine LR (heads 4e-4, backbone 2e-2),
best model by val/mean_macro_f1.

**Ablation variables:**
- L1 → L2: removes the WeightedRandomSampler (tests whether class-imbalance correction helps or
  distorts the marginal distribution of other heads).
- L1 → L3: replaces last-token pooling with mean pooling (tests whether averaging hidden states
  across the sequence improves multi-head prediction).
- L1 → L4: doubles context length to 1024 (tests whether longer dialogue history improves
  relational and decision heads).

Results will be written to `checkpoints/L{1-4}_*/` and evaluated with
`eval_test.yaml` on the held-out test split. As of 20:23 HKT (27 min runtime), all four jobs are
in Epoch 1 at ~2.3 it/s (L1/L2) and ~2.4 it/s (L3/L4).
