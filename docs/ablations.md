# Rigorous Ablation Studies

Each ablation tests a specific non-obvious claim in the paper.  
Unlike "train on less data" ablations (which reviewers expect and ignore),  
these provide genuine scientific signal.

---

## Ablation 1: Parameter-Matched MoE vs GPT ⚠️ REQUIRED FOR STRONG CLAIM
**Claim tested:** "MoE beats GPT by 7.2%, but MoE has 39% more params"

**Current status:** Not yet a true parameter-matched ablation. MoE at 22.4M beats GPT at 16.1M by 7.2%, but this is a capacity-confounded comparison.
The 39% parameter advantage is disclosed honestly in the paper, but a reviewer can still ask whether a 22M dense GPT closes the gap.

**Implementation:** Add `--num-layers` or `--hidden-dim` to `run_small_lm.py` CLI,
then train GPT-8L-512d (~21M params) to match MoE's 22.4M budget.

```bash
# After adding CLI arg:
sbatch scripts/slurm_train.sh slm small_lm --arch gpt --epochs 20 --num-layers 8
```

**Paper rule:** Until this run exists, say "MoE performs best among the models tested, but uses 39% more parameters" rather than "MoE is architecturally superior."

---

## Ablation 2: Conditioning Placebo Test ← HIGHEST IMPACT ✅ DONE, NEEDS MULTI-SEED CI
**Claim tested:** "OCEAN+VAD conditioning reduces PPL by 12.3%"

**Question:** Does the model use the actual OCEAN/VAD VALUES, or just benefits from having extra conditioning tokens regardless of content?

**Results (best val PPL, 3 epochs, TinyLlama-1.1B):**

| Condition | Best PPL | Job ID | Verdict |
|-----------|----------|--------|----------|
| Real OCEAN + Real VAD | 2.90 | 885827 | Baseline |
| No conditioning | 3.30 | (prior) | Lower bound |
| SHUFFLED OCEAN + Real VAD | 2.88 | 887458 | ⚠ NOT USED |
| Real OCEAN + RANDOM VAD | 2.91 | 887459 | ⚠ NOT USED |
| SHUFFLED OCEAN + RANDOM VAD | 2.90 | 887461 | ⚠ NOT USED |

**Key finding: The conditioning is a PLACEBO.** All 4 conditioned variants converge to the same PPL ~2.90 regardless of whether OCEAN/VAD values are real, shuffled, or random. The model achieves the 12.3% improvement from "no conditioning" (3.30 → 2.90) purely from having conditioning tokens present, NOT from their specific content.

**Implication:** The dialogue model learns to predict responses from the text context alone. OCEAN personality and VAD affect vectors provide no useful signal beyond a generic conditioning bias. This is strong evidence that the latent state predictors (which predict OCEAN/VAD from dialogue) are solving a different task than what the dialogue model needs.

**Academic consistency fix:** The paper has been rewritten to describe this as a prefix-capacity/placebo result rather than a semantic OCEAN/VAD result.

**Still needed on SLURM:** rerun the four conditioned variants plus no-conditioning for seeds 42/43/44 and report mean±std or bootstrap CI. Do not interpret 2.88 vs 2.91 without seed variance.

---

## Ablation 3: Class-Normalized Agreement ✅ ROBUSTNESS CHECK
**Claim tested:** "Static dimensions predictable (κ>0.78), deltas near chance (κ<0.50)"

**Result:** The gap WIDENS after normalizing to 3 classes (0.090 → 0.135 gap).
Deltas are genuinely harder — not a class-count artifact.
This is already computed and can go straight into the paper.

**Academic consistency fix:** Treat this as a robustness analysis, not a training ablation. The paper now labels the reported κ values as normalized above-chance estimates until true Cohen's κ is computed from confusion matrices.

---

## Ablation 4: Joint vs Separate Evaluation ← HIGH IMPACT ✅ DONE
**Claim tested:** "Joint model trains end-to-end" (but unevaluated)

**Question:** Does joint training actually improve anything over separate latent+response?

**Latent head results (Qwen3-1.7B, 28 heads, val set):**

| Metric | Separate | Joint | Δ |
|--------|----------|-------|---|
| Mean accuracy | 0.6825 | 0.6741 | -0.0084 |
| Mean κ | 0.4375 | 0.4135 | -0.0240 |
| Secrecy+Reveal violations | 1/652 | 0/652 | -1 |
| Hostile+Affection violations | 1/652 | 4/652 | +3 |

**Response PPL (SFT eval set):**

| Metric | Separate | Joint | Δ |
|--------|----------|-------|---|
| PPL | 1.04 | 1.04 | 0.00 |
| Avg loss | 0.0431 | 0.0431 | 0.0000 |
| Tokens | 1,094,942 | 1,094,942 | same |

**Key finding:** Joint training is close to separate training on mean metrics, but individual heads show non-trivial tradeoffs. The shared backbone achieves similar latent prediction accuracy (-0.008) and identical response generation PPL, but the paper should not claim strict equivalence until paired confidence intervals are computed.

Some heads show tradeoffs (secrecy_pressure +0.12 in joint, reveal_decision -0.10) but these average out. Consistency violations are minimal in both configurations (1-4 per 652 samples).

**Implementation:** Script at `eval_results/ablation_joint_vs_separate.py`. Results at `eval_results/ablation_joint_vs_separate.json`.

**Still needed on SLURM/eval:** paired bootstrap CIs on mean accuracy, macro-F1, response_policy_f1, reveal_decision_f1, secrecy_pressure_f1, and per-rule violation counts.

---

## Ablation 5: Consistency Loss Ablation 🔄 RUNNING (35% epoch 1/3, ~3h remaining)
**Claim tested:** "Consistency constraints prevent impossible states"

**Question:** Does the consistency loss λ_consist actually reduce violations, or are the constraints learned from data alone?

**Status:** Training joint model with λ_consist=0.0 (job 887542, config `train_joint_no_consist.yaml`). Baseline λ_consist=0.5 already trained. Will compare violation counts after training completes.

**Config:** `base_model: Qwen/Qwen3-1.7B`, `lambda_consistency: 0.0`, all other settings identical to `train_joint.yaml`.

**Academic consistency target:** Compare λ=0.0 vs λ=0.5 on both per-rule violation counts and head metrics. If compute allows, add λ∈{0.1,1.0}; otherwise report as a binary ablation.

---

## Ablation 6: Social-State JEPA Auxiliary Objective 🔄 RUNNING (job 887772)
**Claim tested:** "Future social-state prediction improves trajectory-level social dynamics more than token fluency."

**Question:** Does a JEPA-style latent prediction objective improve future social-state and trajectory metrics when added as an auxiliary loss?

**Method:** Add a horizon-specific predictor on the pooled dialogue representation:

```text
history H≤t → encoder → pooled h_t → p_k(h_t) ≈ sg(e(Z_{t+k}))
```

The target `e(Z_{t+k})` embeds focused future social labels:

```yaml
fields:
  - trust_delta
  - respect_delta
  - dominance_delta
  - secrecy_pressure
  - player_knowledge
  - response_policy
  - reveal_decision
```

Training loss:

```text
L = L_heads + λ_JEPA L_JEPA
```

where `L_JEPA` is normalized cosine distance with optional VICReg-style variance regularization.

**Current implementation:** Qwen latent training supports this behind `jepa.enabled` in `llm_finetuning/configs/train_latent.yaml`. Default is `false`, so all old runs remain reproducible. When enabled, the dataset attaches `future_{horizon}_{field}` labels using `(episode_id, turn_idx + horizon)` and masks missing horizons with `-1`.

**Recommended first run:**

```yaml
jepa:
  enabled: true
  horizons: [1]
  lambda_jepa: 0.05
  var_weight: 0.01
```

**Evaluation metrics:**

| Metric | Why |
|--------|-----|
| `trust_delta_f1` | Relationship trajectory |
| `respect_delta_f1` | Relationship trajectory |
| `dominance_delta_f1` | Power dynamics |
| `response_policy_f1` | Actionable policy prediction |
| `reveal_decision_f1` | Secret-disclosure control |
| `secret_leakage_rate` | Safety |
| `jepa_loss` | Future social-state representation prediction |

**Ablation table to fill:**

| Model | JEPA | Horizon | Policy F1 ↑ | Trust Δ F1 ↑ | Reveal F1 ↑ | Leakage ↓ | JEPA Loss ↓ |
|-------|-----:|--------:|------------:|-------------:|------------:|----------:|------------:|
| Qwen3 latent | no | — | | | | | — |
| Qwen3 latent + JEPA | yes | 1 | | | | | |
| Qwen3 latent + JEPA | yes | 1–3 | | | | | |
| Qwen3 latent + shuffled JEPA | yes | 1 | | | | | |
| Mamba-like latent + JEPA | yes | 1 | | | | | |
| Mamba-like latent + shuffled JEPA | yes | 1 | | | | | |

**Paper-safe claim:** JEPA is an auxiliary predictive representation objective, not the main architecture. Report it as improving future-state/trajectory metrics if the ablation confirms the effect.

**Academic consistency target:** Real-future JEPA must beat shuffled-future JEPA on held-out metrics before claiming temporal social-state prediction helps. `jepa_loss` is diagnostic only; main evidence must be response_policy/reveal/delta F1 and true Cohen's κ.

---

## Priority Order & Status

| # | Ablation | Status | Result |
|---|----------|--------|--------|
| 2 | Conditioning placebo | ✅ DONE | **PLACEBO** — all conditions converge to PPL ~2.90 |
| 4 | Joint vs separate eval | ✅ DONE | Joint ≈ Separate (Δacc -0.008, ΔPPL 0.00) |
| 6 | Social-State JEPA | 🔄 RUNNING | Qwen JEPA pilot job 887772; shuffled-future control still required |
| 5 | Consistency loss | 🔄 RUNNING | λ_consist=0.0 training (job 887542, ~3h remaining) |
| 1 | Param-matched GPT | ⚠️ TODO | Need dense GPT ~22M to remove MoE capacity confound |
| 3 | Class-norm agreement | ✅ ROBUSTNESS | Gap widens after normalization |

## SLURM Priority Queue

Use the HPC cluster for jobs that materially improve paper rigor:

1. **Qwen shuffled-future JEPA:** essential placebo for Ablation 6.
2. **Mamba-like latent + JEPA + shuffled JEPA:** cheap non-transformer replication.
3. **GPT SLM latent + JEPA:** canonical small transformer replication.
4. **Conditioning placebo multi-seed array:** seeds 42/43/44 for real, shuffled, random, shuffled+random, and no-conditioning.
5. **Param-matched GPT:** GPT ~22M against MoE 22.4M.
6. **Final eval job:** true Cohen's κ, bootstrap CIs, per-head CSV, paper tables.

**Files created:**
- `eval_results/ablation_joint_vs_separate.py` — Ablation 4 eval script
- `eval_results/ablation_joint_vs_separate.json` — Ablation 4 results
- `llm_finetuning/configs/train_joint_no_consist.yaml` — Ablation 5 config
- `llm_finetuning/src/training/jepa.py` — Social-State JEPA embedding, predictor, and loss
- `llm_finetuning/src/training/dataset.py` — future social-label support for JEPA horizons
- `llm_finetuning/src/training/train_latent.py` — optional JEPA auxiliary loss for Qwen latent training
- `llm_finetuning/configs/train_latent.yaml` — disabled-by-default JEPA config block
- `slm_training/artifacts/personality_cache_SHUFFLED.jsonl` — pre-existing shuffled cache
- `slm_training/src/train/run_dialogue.py` — added `--randomize-vad` flag
- `slm_training/src/train/train_dialogue.py` — VAD randomization support
- `slm_training/src/common/config.py` — added `randomize_vad` field

**SLURM jobs:**
| Job ID | Ablation | Status |
|--------|----------|--------|
| 887458 | Shuffled OCEAN + Real VAD | ✅ Complete (ppl=2.88) |
| 887459 | Real OCEAN + Random VAD | ✅ Complete (ppl=2.91) |
| 887461 | Shuffled OCEAN + Random VAD | ✅ Complete (ppl=2.90) |
| 887524 | Joint vs Separate (attempt 1) | ❌ Crashed (dialogue_act bug) |
| 887543 | Joint vs Separate (attempt 2) | ❌ Crashed (consistency bug) |
| 887549 | Joint vs Separate (attempt 3) | ❌ Crashed (tokenizer bug) |
| 887621 | Joint vs Separate (attempt 4) | ✅ Complete |
| 887460 | Consistency loss (attempt 1) | ❌ Crashed (wrong base model) |
| 887542 | Consistency loss (attempt 2) | 🔄 Running |
