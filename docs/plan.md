# Research Improvement Plan

_Last updated: 2026-05-05. Single source of truth for the remaining work needed to defend RQ\* (SLMs ≈ pretrained backbones on structured social-state understanding) and ship a coherent paper + docs + code._

---

## 0. Headline reframing (decided)

The paper's RQ1 ("which from-scratch SLM wins on PPL?") cannot defend the hypothesis "small models match larger ones on emotion/dialogue understanding" because PPL 42 vs PPL 3 is a pretraining gap, not a capability gap. We replace it with:

> **RQ\*:** Trained on identical 29-head Z\_t supervision (with optional Social-State JEPA), does a 15–22M from-scratch SLM achieve comparable per-head Cohen's κ and policy F1 to a 1.7B / 2B pretrained backbone on emotion and social-state understanding?

RQ\* is the new headline. RQ1–RQ3 become supporting. Compute budget: full matrix.

---

## 1. State of the world

### 1.1 Already done (verified in repo, 2026-05-05)

| Item | Evidence |
|---|---|
| SLM latent trainer | `slm_training/src/train/train_latent_slm.py` |
| SLM latent configs (GPT, Mamba × base, JEPA) | `slm_training/configs/latent_{gpt,mamba}_{base,jepa}.yaml` |
| Shuffled-future-label dataset mode | `shuffle_future_labels` plumbed in `llm_finetuning/src/training/dataset.py` (6 refs) |
| Rigorous eval script with Cohen's κ + bootstrap | `eval_results/evaluate_latent_rigorous.py` (25 matches for `bootstrap`/`cohen_kappa`/`paired`) |
| JEPA module | `llm_finetuning/src/training/jepa.py` |
| Joint-vs-separate ablation | `eval_results/ablation_joint_vs_separate.{py,json}` |
| OCEAN/VAD placebo runs (single seed) | jobs 887458/887459/887461 complete |

### 1.2 Currently running (4 slots)

| Job | Ablation | Config | ETA |
|---|---|---|---|
| 89621 2 | Consistency λ=0.5 baseline | `train_joint.yaml` | ~3h |
| 89621 3 | Consistency λ=0.0 retry | `train_joint_no_consist.yaml` | ~3h |
| 89621 4 | JEPA Qwen | `train_latent_jepa.yaml` | ~1.5h |
| 89621 5 | JEPA Qwen shuffled (placebo) | `train_latent_qwen_jepa_shuffled.yaml` | ~1.5h |

### 1.3 Queued (hit QOS limit)

`latent_gpt_base.yaml`, `latent_gpt_jepa.yaml`, `latent_mamba_base.yaml`, `latent_mamba_jepa.yaml`.

---

## 2. Remaining work (the actual gap)

### 2.1 Critical path while jobs run (no GPU)

These four items are the highest ROI and are independent — do in parallel.

#### A. Gold-Z\_t conditioning experiment ⭐ paper-changing
**Why:** This is the bridge experiment. Currently Track C (response gen) and Track D (latent prediction) are disconnected — the paper cannot claim "better latents → better generation." A single oracle run closes the loop.

**What:** Add an evaluation mode that feeds **gold** Z\_t into the response generator and compares response PPL / ROUGE-L / policy-consistency against (a) no-conditioning, (b) predicted-Z\_t.

**Files:**
- New `eval_results/eval_gold_zt_conditioning.py` — load Qwen3 response model, three conditioning modes (none / predicted / gold).
- Extend `llm_finetuning/src/eval/eval_response.py` to accept `--zt-source {none,predicted,gold}`.
- Result table → `eval_results/gold_zt_conditioning.json` and a row in `paper/main.tex` Track C.

**Acceptance:** Three numbers on the same val split. If gold Z\_t > predicted > none, the structured-state thesis holds. If gold ≈ predicted ≈ none, we honestly report a null.

#### B. Decoding fix ⭐ Table 5 polish, 30 min
**Why:** Track C generations show 66.7% 3-gram repetition and 2.3× length ratio. This makes the response-quality table look broken. Models are fine; decoding config is wrong.

**What:** Add `repetition_penalty=1.15`, `no_repeat_ngram_size=3`, `top_p=0.92`, `max_new_tokens=128` to:
- `llm_finetuning/src/eval/eval_response.py:95`
- `llm_finetuning/src/inference/interactive.py:212-213`

Re-emit `eval_results/sample_generations.json` and Table 5 (`tab:response_quality`).

**Acceptance:** 3-gram repetition < 0.30, length ratio < 1.4.

#### C. Paired-bootstrap audit ⭐ unblocks all CI claims
**Why:** Eval script has bootstrap, but only one "paired" reference (a docstring comment). Independent bootstrap on differences gives CIs that are too wide and too lenient — JEPA-vs-baseline significance claims become unreliable.

**What:** Audit `evaluate_latent_rigorous.py`. The paired-bootstrap function must, for each resample, draw a single index set and apply it to both models' predictions. ~20 LOC.

**Acceptance:** Paired CI on (model A − model B) is strictly tighter than independent CI on the same data when predictions are correlated.

#### D. Correctness fixes from `docs/TODOS.md`
- **TODO-2** (10 lines): `(episode_id, turn_idx)` alignment check at top of `JointDataset.__init__`. Prevents silent training corruption.
- **TODO-1**: counterfactual flip temporal-corruption fix in `src/data_gen/counterfactual.py`. Apply before any 5000-episode regeneration.
- **TODO-4**: semantic secret-leakage validator in `src/data_gen/validator.py` + `secrets[*].keywords` schema field.
- **TODO-3** (optional): `pooling_strategy: {last_token, avg_pool}` config; one paired ablation on best Qwen run.

### 2.2 Configs to add for "full matrix" (10 min each)

The previously chosen "full matrix" budget requires three configs that don't exist yet:

- `slm_training/configs/latent_mamba_jepa_shuffled.yaml` — SLM placebo
- `slm_training/configs/latent_moe_base.yaml` + `latent_moe_jepa.yaml` — param-matched comparison
- `slm_training/configs/latent_prefix_gpt_base.yaml` + `latent_prefix_gpt_jepa.yaml` — optional

If skipping, the SLM scope becomes GPT vs Mamba (transformer vs SSM), which is still defensible.

### 2.3 Gemma path (currently broken)

- `train_latent_gemma_base.yaml` + `_jepa.yaml` missing. QLoRA target modules differ from Qwen.
- Gemma-4-E2B-it response PPL 6.38 vs TinyLlama 3.30 is suspicious — likely chat-template mismatch in SFT formatting. Audit `prompts/` first, then re-run.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` should live in the SLURM wrapper, not in code.

### 2.4 Multi-seed placebo array

OCEAN/VAD placebo result is single-seed. Need seeds 42/43/44 × {real, shuffled, random, both, none} = 15 short jobs via `scripts/slurm_array.sh`. Reviewers will reject 2.88 vs 2.91 differences without CIs.

### 2.5 Param-matched GPT-22M (Ablation 1)

Add `--num-layers` / `--hidden-dim` to `run_small_lm.py` CLI. Train GPT-8L-512d (~21M) to match MoE 22.4M. Until this exists, paper says "MoE performs best among models tested, but uses 39% more parameters" — not "MoE is architecturally superior."

---

## 3. Experimental matrix (what to run once slots free)

Single selection metric: `val/response_policy_f1`. Never select on `jepa_loss`.

| Tier | Run | Backbone | Status |
|---|---|---|---|
| ✓ Done | `Q-base` | Qwen3-1.7B | baseline complete |
| 🔄 Running | `Q-jepa` | Qwen3-1.7B | job 89621 4 |
| 🔄 Running | `Q-jepa-shuf` | Qwen3-1.7B | job 89621 5 (placebo confirmed plumbed) |
| 🔄 Running | `Q-consist05` | Qwen3-1.7B | job 89621 2 |
| 🔄 Running | `Q-consist00` | Qwen3-1.7B | job 89621 3 |
| Queued | `SG-base` / `SG-jepa` | GPT-SLM 16M | configs ready |
| Queued | `SM-base` / `SM-jepa` | Mamba-SLM 15M | configs ready |
| TODO | `SM-jepa-shuf` | Mamba-SLM | needs §2.2 config |
| TODO | `SX-base` / `SX-jepa` | MoE-SLM 22M | needs §2.2 configs |
| TODO | `G-base` / `G-jepa` | Gemma-4-E2B | needs §2.3 |
| TODO | `Param-matched GPT-22M` | from-scratch | needs CLI flag |
| TODO | `placebo×3 seeds` | TinyLlama | 15 short jobs (§2.4) |
| TODO | `Q-pool-avg` | Qwen3-1.7B | TODO-3 ablation |

---

## 3.5 Better experiment protocol for the next batch

The next batch should be organized around reviewer-grade questions, not around "try another model."
Every run must answer one of these:

1. **Recoverability:** which parts of `Z_t` can be predicted from dialogue?
2. **Causal utility:** does `Z_t` change generation in the intended direction?
3. **System validity:** does the structured pipeline reduce violations or improve human preference?

### Required response-generation comparisons

Run all four settings on the same validation split:

| Setting | Purpose |
|---|---|
| `none` | lower-bound response baseline |
| `gold_Z_t` | oracle upper bound for structured state |
| `predicted_Z_t` | deployed pipeline condition |
| `shuffled_Z_t` | placebo/control for state semantics |

Report PPL, ROUGE-L, BLEU-4, repeated 3-gram rate, prompt-artifact rate, constraint violations, and secret-leakage Wilson upper bound.
If `gold_Z_t` is not better than `none`, the structured-state generation claim is weak and should be reported as a null.
If `gold_Z_t > predicted_Z_t > none > shuffled_Z_t`, the bridge claim is strong.

### Counterfactual control evaluation

Hold context fixed and change exactly one state field.
Start with:

- `reveal_decision`: `none`, `hint`, `partial`, `full`
- `secrecy_pressure`: `low`, `medium`, `high`
- `response_policy`: `answer`, `deflect`, `negotiate`, `challenge`, `soothe`

Measure intervention success with rule-based checks plus human/LLM-assisted judgments.
Examples:

- Higher `reveal_decision` should increase specific information content.
- Higher `secrecy_pressure` should reduce direct disclosure.
- `response_policy=soothe` should reduce threat/challenge language.

### Generalization split

Add a non-random held-out split:

- Hold out entire scenario families.
- Hold out NPC roles if enough data exists.
- Report in-domain vs held-out macro-F1, MCC, and violation rates.

This is necessary to show the schema generalizes beyond the synthetic generator's surface patterns.

### Multi-seed policy

Do not multi-seed everything.
Run seeds 42/43/44 for only the headline results:

- Qwen latent predictor.
- Gold/predicted/shuffled/no-`Z_t` generation.
- Joint vs separate.
- Parameter-matched GPT vs MoE.

Report mean ± std or paired bootstrap confidence intervals.

### Human evaluation packet

Build a 100-200 item blinded packet after the automatic eval stabilizes.
Balance it across scenario family, `reveal_decision`, `response_policy`, and secrecy pressure.
Collect naturalness, role consistency, social-state consistency, player relevance, constraint safety, and forced preference.

---

## 3.6 Architecture improvement roadmap

The architecture work should improve structured-state understanding and controllability, not just raw PPL.
Prioritize changes that are easy to ablate and easy to defend.

### Latent predictor improvements

| Idea | Why it matters | Acceptance |
|---|---|---|
| Pooling strategy: `last_token` vs `avg_pool` vs learned attention pooling | Decoder final token may be a weak sequence summary for classification | improves R/D macro-F1 or MCC without hurting mean metrics |
| Ordinal heads for stance levels/deltas | `R_t` labels are ordered, not nominal | better delta/level MAE and macro-F1 |
| Group-specific adapters | Reduces negative transfer across `C/A/M/R/N/D` | improves at least two groups with no large regression |
| Hierarchical dependency layer | Lets `N_t` influence `D_t`, and affect/stance influence repair | reduces consistency violations and improves D-head F1 |
| Class-balanced or focal loss | Handles skewed labels better than raw cross-entropy | improves macro-F1/MCC over accuracy-only gains |
| Calibration loss or temperature scaling | Makes confidence meaningful for routing and selective generation | lower ECE while preserving F1 |

### Response model improvements

| Idea | Why it matters | Acceptance |
|---|---|---|
| Serialized `Z_t` control tokens | Replaces placebo OCEAN/VAD prefix with auditable state | gold/predicted beat shuffled/no-state |
| Generate-then-verify reranking | Uses the latent predictor as a response consistency critic | fewer violations and lower repetition at same PPL range |
| Constrained decoding for secrets | Prevents obvious forbidden disclosure at inference time | lower leakage with Wilson CI reported |
| State-delta conditioning | Optimizes intended social transition, not only current state | better intervention success on trust/threat/obligation |
| Stop-token and artifact training cleanup | Current samples include tags and prompt reasoning | prompt-artifact rate below 5% |

### Higher-risk research directions

- **Graph-structured `Z_t`:** model fields as dependency graph nodes; useful if consistency rules are central to the paper.
- **Latent transition world model:** train `p(Z_{t+1} | Z_t, player, response)` for trajectory-level social dynamics.
- **Social-function MoE:** route by interpretable social function such as secrecy, repair, negotiation, threat, or bonding, not by generic token likelihood.
- **Contrastive counterfactual loss:** same context with altered `Z_t` should produce distinguishable hidden states/responses.
- **Preference optimization:** run DPO/IPO using human or carefully audited judge preferences for role and social-state consistency.

Do not add all of these at once.
The strongest near-term architecture package is: pooling ablation, ordinal `R_t` heads, serialized `Z_t` response conditioning, and generate-then-verify reranking.

---

## 4. Documentation alignment

- `docs/ablations.md` — replace "estimated κ via normalized above-chance agreement" with "Cohen's κ from confusion matrices" once §2.1.C lands; mark Ablation 3 as superseded.
- `docs/jepa_plan.md` — remove §8.5 "SLURM implementation gap" (T2/T3 already shipped); cross-link to `train_latent_slm.py`.
- `docs/TODOS.md` — close TODO-2/4 with PR refs once §2.1.D lands.
- `docs/schema.md` — add `secrets[*].keywords` field.
- `docs/data_flow.md` — re-render after counterfactual fix.
- `docs/project_overview.md` + `docs/SUMMARY.md` — add RQ\* and the SLM-vs-pretrained latent comparison.
- New `docs/results_matrix.md` — auto-written by eval script; paper tables source from this. Kills copy-paste drift.

---

## 5. Paper changes (`paper/main.tex`)

**Do not touch the paper until §2.1 produces real numbers.** All κ values, all Track C tables, and the new SLM-vs-pretrained table will shift.

When ready:

1. **Abstract** — add: *"Trained with the same 29-head supervision, a 15–22M from-scratch SLM achieves [X]% of Qwen3-1.7B's mean per-head accuracy on Z\_t, suggesting that structured social-state understanding at this annotation density is largely a question of supervision rather than of pretrained scale."*
2. **§Introduction** — add **RQ\*** as primary; demote RQ1–3 to supporting.
3. **§Method** — add JEPA subsection (text in `docs/jepa_plan.md` §7.3); add `lecun2022path`, `assran2023self`, `bardes2022vicreg` to `references.bib`.
4. **§Experimental Setup** — add SLM latent training paragraph: same 7,742 turns, same heads, same loss, same selection metric.
5. **§Results — new subsection "SLM vs Pretrained on Z\_t"** — main RQ\* table comparing per-group mean acc and Cohen's κ across {GPT-SLM, Mamba-SLM, MoE-SLM, Qwen3-1.7B, Gemma-4-E2B} with paired-bootstrap CIs.
6. **§Results — new subsection "Gold vs Predicted Z\_t conditioning"** — three-row table from §2.1.A.
7. **§Results — JEPA table** — populate from §2.1.C output; honor acceptance criteria in `docs/jepa_plan.md` §6.
8. **§Results — consistency ablation** — fill λ=0.0 vs λ=0.5 row from running jobs.
9. **§Track A** — once param-matched GPT-22M exists, replace 7.2% / 39% hedge with clean comparison.
10. **§Track C** — re-table after Gemma chat-template fix and decoding fix; add multi-seed placebo CIs.
11. **§Limitations** — drop "single seed" once §2.4 lands.
12. **§Appendix per-head table** — replace estimated κ with true Cohen's κ.

---

## 6. Sequencing

**Now (no GPU, parallel):**
1. §2.1.B Decoding fix (30 min)
2. §2.1.C Paired-bootstrap audit (~1h)
3. §2.1.D TODO-2 episode-alignment check (10 LOC)
4. §2.1.A Gold-Z\_t conditioning experiment (highest leverage on paper claim)
5. §2.1.D TODO-1, TODO-4 (correctness, before any 5000-ep regen)
6. §2.2 missing configs if pursuing full matrix
7. §2.3 Gemma chat-template audit

**As GPU slots free:**
8. Queued SLM jobs (already submitted)
9. Multi-seed placebo array (§2.4)
10. Param-matched GPT-22M (§2.5)
11. Gemma base + JEPA (§2.3)
12. Optional MoE/PrefixGPT latent runs (§2.2)

**Then paper:**
13. Run `eval_results/evaluate_latent_rigorous.py` across all checkpoints → `docs/results_matrix.md`.
14. Apply §5 edits in one pass; update §4 docs in the same PR.

---

## 7. Acceptance criteria for the paper

### RQ\* (new headline)
- **Strong:** SLM mean per-head κ ≥ 0.9 × Qwen3 κ on the test split, with paired-bootstrap CI excluding zero in favor of "comparable" (i.e., difference CI overlapping ±0.05).
- **Moderate:** SLM matches Qwen on 4+ of 6 social-state groups, lags on R\_t (relational deltas).
- **Null:** SLM substantially below pretrained — report honestly; reframe as "supervision alone is insufficient at 15–22M scale."

### Gold-Z\_t conditioning (bridge)
- **Strong:** gold > predicted > none on response PPL with non-overlapping CIs.
- **Moderate:** gold > none, predicted ≈ none — predictor needs work.
- **Null:** gold ≈ none — structured-state thesis weakened; report honestly.

### JEPA
- See `docs/jepa_plan.md` §6. Real-future > shuffled-future on `response_policy_f1`, `reveal_decision_f1`, and one delta F1, or no claim of temporal benefit.

### Multi-seed placebo
- Seeds 42/43/44 mean ± std on each of {real, shuffled, random, both, none}. If 2.88 vs 2.91 is within seed noise, paper says so explicitly.

---

## 8. Open risks

- **Gold-Z\_t result might be null.** If it is, RQ\* loses some force ("we can predict Z\_t with an SLM, but Z\_t doesn't help generation"). The honest paper still ships, framed as a negative result on structured conditioning.
- **Gemma chat-template fix might not move PPL.** If Gemma stays at PPL 6.38 after the fix, drop it from the headline table and keep as exploratory.
- **Param-matched GPT-22M might match MoE.** That eliminates the MoE architectural-win claim entirely. Fine — report honestly; the SLM-on-Z\_t result doesn't depend on it.
- **Counterfactual fix invalidates current data.** If TODO-1 changes the train distribution materially, all current latent results need a rerun. Apply *before* the next round of base-model experiments, not after.
