# Social-State JEPA — Implementation and Paper Update Plan

_Last updated: 2026-05-04_

This document is the single source of truth for integrating **Social-State JEPA** into the training/evaluation pipeline and updating the paper accordingly. It consolidates the decisions made during the JEPA, OCEAN/VAD placebo, and ablation discussions.

---

## 0. Guiding Principles

- **JEPA is an auxiliary representation objective, not a new architecture.** It must always be compared as `backbone + heads` vs `backbone + heads + JEPA` with matched compute, data, and checkpoint selection.
- **Never sell JEPA as replacing the generator.** It improves the representation that downstream heads (and optionally the response generator) consume.
- **Always include a shuffled-future-label placebo** to prove JEPA uses temporal structure, not just extra capacity.
- **Respect the OCEAN/VAD placebo finding.** Do not rely on OCEAN/VAD values as the causal source of gains; report them as a placebo control.
- **Pretrained JEPA gains must be matched by the same selection metric** used for baselines, e.g. `val/response_policy_f1` or `val/macro_f1`. Do not pick checkpoints by `jepa_loss`.

---

## 1. Current State

### 1.1 What is already implemented

- `llm_finetuning/src/training/jepa.py`
  - `JEPA_FIELDS`
  - `SocialStateEmbedding`
  - `SocialJEPAHead` with per-horizon predictors
  - Cosine JEPA loss with optional variance regularization
  - Helper to pull future label IDs from a batch
- `llm_finetuning/src/training/dataset.py`
  - `HeadSupervisionDataset` optionally attaches `future_{horizon}_{field}` labels by `(episode_id, turn_idx + horizon)`
  - Missing horizons are masked with `-1`
- `llm_finetuning/src/training/train_latent.py`
  - Reads `jepa:` block from config
  - Instantiates `SocialJEPAHead` and adds `lambda_jepa * L_JEPA`
  - Includes JEPA params in optimizer and gradient clipping
  - Logs `train/jepa_loss` and `val/jepa_loss`
  - Saves `jepa_head.pt` with the best checkpoint
- `llm_finetuning/configs/train_latent.yaml`
  - Default `jepa.enabled: false` so existing runs are unchanged
- `llm_finetuning/configs/train_latent_jepa.yaml`
  - Enabled JEPA pilot config
- `docs/ablations.md`
  - Ablation 6 plan

### 1.2 What must change

- Add **shuffled-future-label** mode in the dataset.
- Generalize latent training beyond Qwen (Gemma and SLMs).
- Add SLM latent predictor training path with optional JEPA.
- Add final held-out test evaluation with bootstrap CIs.
- Update paper claims regarding OCEAN/VAD and add JEPA section.

---

## 2. Experimental Matrix

### 2.1 Core rule

For every backbone:

```
backbone + supervised 29-head
vs
backbone + supervised 29-head + JEPA
vs
backbone + supervised 29-head + shuffled-future JEPA
```

Optional additional controls:

```
backbone + supervised 29-head + consistency loss
backbone + supervised 29-head + next-token aux LM loss
```

### 2.2 Backbones

| ID | Backbone | Type | Tier |
|----|----------|------|------|
| Q  | Qwen3-1.7B | pretrained transformer + QLoRA | required |
| G  | Gemma-4-E2B | pretrained transformer + QLoRA | required |
| SG | GPT (from-scratch SLM) | transformer | required |
| SM | Mamba-like (from-scratch SLM) | state-space | required |
| SX | MoE (from-scratch SLM) | expert transformer | optional |
| SP | PrefixGPT (from-scratch SLM) | prefix-conditioned transformer | optional |

### 2.3 Run table

| Run | Backbone | Objective | Placebo | Required? |
|-----|----------|-----------|---------|-----------|
| Q-base | Qwen3-1.7B | heads only | — | yes |
| Q-jepa | Qwen3-1.7B | heads + JEPA | no | yes |
| Q-jepa-shuf | Qwen3-1.7B | heads + JEPA | shuffled future | yes |
| Q-consist | Qwen3-1.7B | heads + consistency | — | already running |
| G-base | Gemma-4-E2B | heads only | — | yes |
| G-jepa | Gemma-4-E2B | heads + JEPA | no | yes |
| G-jepa-shuf | Gemma-4-E2B | heads + JEPA | shuffled future | optional |
| SG-base | GPT SLM | heads only | — | yes |
| SG-jepa | GPT SLM | heads + JEPA | no | yes |
| SM-base | Mamba-like | heads only | — | yes |
| SM-jepa | Mamba-like | heads + JEPA | no | yes |
| SM-jepa-shuf | Mamba-like | heads + JEPA | shuffled future | yes |
| SX-base | MoE SLM | heads only | — | optional |
| SX-jepa | MoE SLM | heads + JEPA | no | optional |
| SP-base | PrefixGPT | heads only | — | optional |
| SP-jepa | PrefixGPT | heads + JEPA | no | optional |

Minimum viable set: Q-base, Q-jepa, Q-jepa-shuf, G-base, G-jepa, SG-base, SG-jepa, SM-base, SM-jepa, SM-jepa-shuf.

---

## 3. JEPA Configuration

### 3.1 Main JEPA config for all backbones

```yaml
jepa:
  enabled: true
  target: social_state
  fields:
    - trust_delta
    - respect_delta
    - dominance_delta
    - secrecy_pressure
    - player_knowledge
    - response_policy
    - reveal_decision
  horizons: [1]
  horizon_weights:
    1: 1.0
  emb_dim: 32
  target_dim: 128
  predictor_dim: 128
  dropout: 0.1
  lambda_jepa: 0.03
  var_weight: 0.01
  shuffle_future_labels: false
```

### 3.2 Placebo variant

Same config with:

```yaml
jepa:
  shuffle_future_labels: true
```

### 3.3 Ablation grid (for one backbone, likely Qwen or Mamba-like)

| Variant | horizons | lambda_jepa | emb_dim | var_weight |
|---------|----------|-------------|---------|------------|
| weak    | [1]      | 0.01        | 32      | 0.01       |
| default | [1]      | 0.03        | 32      | 0.01       |
| strong  | [1]      | 0.05        | 64      | 0.01       |
| multi-horizon | [1,2] | 0.03     | 32      | 0.01       |
| no-var  | [1]      | 0.03        | 32      | 0.0        |
| shuffled | [1]     | 0.03        | 32      | 0.01       |

### 3.4 Checkpoint selection

All runs use the same metric:

```yaml
metric_for_best_model: val/response_policy_f1
```

or alternatively `val/macro_f1`. Never `jepa_loss`.

---

## 4. Implementation Tasks

### 4.1 T1 — Dataset: shuffled-future-label mode

**File:** `llm_finetuning/src/training/dataset.py`

- Add constructor flag `shuffle_future_labels: bool = False`.
- In `_encode_future_labels`, if enabled:
  - For each horizon, replace the real `(episode_id, turn_idx+h)` lookup with labels from a randomly chosen record.
  - Preserve label marginals; break temporal alignment.
  - Use a deterministic `random.Random(seed)` stream per dataset instance.
- Plumb the flag through `train_latent.py` where dataset is constructed.

Acceptance:

- Same label marginals vs real JEPA mode (histogram within noise).
- Zero correlation between real `Z_{t+h}` and served `future_h_*` labels.

### 4.2 T2 — Backbone generalization for latent training

**File:** `llm_finetuning/src/training/train_latent.py` and `src/training/model.py` (if needed).

- Detect backbone family from `base_model`:
  - `qwen`, `gemma`, `tinyllama`, etc.
- Ensure `LatentStatePredictor`:
  - uses `AutoModelForCausalLM` with `output_hidden_states=True`.
  - pooling last hidden state works regardless of family.
- Add Gemma-specific QLoRA target modules if they differ from Qwen.
- Verify `predictor.hidden_size` resolves for Gemma (it will differ from Qwen).
- Add CLI: `python -m src.training.train_latent --config <yaml>`.

Acceptance:

- `configs/train_latent_gemma_base.yaml` and `configs/train_latent_gemma_jepa.yaml` train end-to-end without code changes.

### 4.3 T3 — SLM latent predictor training

**New file:** `slm_training/src/train/train_latent_slm.py`

- Loads SLM architecture from `slm_training/configs/latent_<arch>_*.yaml`:
  - `gpt`, `prefix_gpt`, `moe`, `mamba_like`.
- Wraps the SLM with a pooling layer and the 29 classification heads used in `llm_finetuning/src/training/model.py`.
- Reuses `HeadSupervisionDataset`.
- Imports `SocialJEPAHead` from `llm_finetuning.src.training.jepa`.
- Implements the same training loop contract:
  - progressive LR optional
  - grad accumulation
  - cosine schedule
  - MLflow logging
  - best-checkpoint selection by `val/response_policy_f1`
  - `jepa_head.pt` save alongside SLM checkpoint

Acceptance:

- `SG-base` and `SG-jepa` train to completion on GPT SLM.
- `SM-base` and `SM-jepa` train to completion on Mamba-like.
- Optional: MoE and PrefixGPT configs work.

### 4.4 T4 — Config files

Create:

```
llm_finetuning/configs/train_latent_qwen_base.yaml
llm_finetuning/configs/train_latent_qwen_jepa.yaml
llm_finetuning/configs/train_latent_qwen_jepa_shuffled.yaml
llm_finetuning/configs/train_latent_gemma_base.yaml
llm_finetuning/configs/train_latent_gemma_jepa.yaml
slm_training/configs/latent_gpt_base.yaml
slm_training/configs/latent_gpt_jepa.yaml
slm_training/configs/latent_mamba_base.yaml
slm_training/configs/latent_mamba_jepa.yaml
slm_training/configs/latent_mamba_jepa_shuffled.yaml
```

All configs must:

- Fix `seed: 42`.
- Match training budget per backbone-family.
- Use the same selection metric.
- Disable JEPA by default unless the file ends in `_jepa` or `_jepa_shuffled`.

### 4.5 T5 — Evaluation script

**New file:** `eval_results/evaluate_latent_jepa.py`

Functions:

- Load checkpoint (with optional `jepa_head.pt`).
- Load `test_heads.jsonl`.
- Predict heads; compute:
  - per-head accuracy, macro-F1, weighted-F1, Cohen's κ.
  - overall macro-F1, weighted-F1, mean accuracy.
  - consistency violation rate (reuse existing rule evaluator).
  - JEPA validation loss per horizon.
  - bootstrap 95% CIs on key metrics (N=1000 resamples).
- Write:
  - `eval_results/jepa/<run_id>.json`
  - `eval_results/jepa/<run_id>.csv` (per-head)
  - Aggregated `eval_results/jepa/jepa_comparison.md`

Acceptance:

- One command reproduces Table 1 in the paper.

### 4.6 T6 — Orchestration

Optional: SLURM/Bash wrappers.

```
scripts/run_jepa_matrix.sh
```

Submits:

- Q-base, Q-jepa, Q-jepa-shuf (only if Q-base and Q-jepa finish first)
- G-base, G-jepa
- SG-base, SG-jepa
- SM-base, SM-jepa, SM-jepa-shuf

Include:

- deterministic seeds
- `HF_TOKEN` for Gemma
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for Gemma

---

## 5. Metrics and Reporting

### 5.1 Metrics per run

- Training:
  - `train/loss`
  - `train/jepa_loss`
- Validation:
  - `val/loss`
  - `val/mean_accuracy`
  - `val/macro_f1`
  - `val/response_policy_f1`
  - `val/reveal_decision_f1`
  - `val/trust_delta_f1`
  - `val/respect_delta_f1`
  - `val/dominance_delta_f1`
  - `val/secrecy_pressure_f1`
  - `val/consistency_violation_rate`
  - `val/jepa_loss`
  - `val/jepa_h1_loss`
- Test (same as validation, held-out):
  - plus bootstrap CIs on macro-F1, response_policy_f1, reveal_decision_f1, trust_delta_f1.

### 5.2 Significance

- Bootstrap 95% CIs on macro-F1 and policy/reveal/delta F1.
- Paired bootstrap where applicable to compare JEPA vs baseline on the same test examples.
- Report mean ± CI in the paper.

---

## 6. Acceptance Criteria for the Paper

### 6.1 Strong positive

Real JEPA > baseline **and** real JEPA > shuffled JEPA on at least:

- `response_policy_f1`
- `reveal_decision_f1`
- one relationship delta (trust/respect/dominance)

Claim allowed:

> Social-State JEPA improves future-relevant social-state prediction beyond supervised heads and non-temporal controls.

### 6.2 Moderate

Real JEPA > baseline but ≈ shuffled JEPA on most metrics.

Claim allowed:

> JEPA acts as an effective regularizer for social-state heads; temporal structure does not provide additional benefit at this data scale.

### 6.3 Null / negative

Real JEPA ≈ baseline or worse.

Claim allowed:

> At this data scale, future-state auxiliary prediction did not improve over supervised social-state heads.

All outcomes are reported honestly.

---

## 7. Paper Update Plan

### 7.1 Files to edit

- `paper/main.tex`
- `paper/references.bib`

### 7.2 Abstract

Change:

```
Conditioning a 1.1B response model on just 8 numbers---a compressed OCEAN+VAD
personality vector---reduces perplexity by 12.3%...
```

to:

```
Soft-prefix conditioning reduces perplexity by 12.3% over an unconditioned
1.1B baseline, but placebo ablations show this gain is not attributable to
the semantic content of the current OCEAN/VAD vectors. We therefore evaluate
a Social-State JEPA auxiliary objective that predicts future structured
social states in embedding space and report its effect across transformer,
mixture-of-experts, and state-space backbones.
```

### 7.3 New subsection in Method

Add after current Track D description:

```latex
\subsection{Social-State JEPA}
\label{sec:jepa}

Let $h_t = f_\theta(H_{\leq t})$ be the pooled dialogue representation and
$e_\phi(Z_{t+k})$ an embedding of the future structured social state at
horizon $k$. A predictor $p_k$ maps $h_t$ into the target space. We minimise

\begin{equation}
\mathcal{L}_{\mathrm{JEPA}}
= \sum_{k \in \mathcal{K}} w_k
  \left(1 - \cos\!\left(p_k(h_t),\; \mathrm{sg}(e_\phi(Z_{t+k}))\right)\right)
+ \beta \mathcal{L}_{\mathrm{var}},
\end{equation}

where $\mathrm{sg}$ is stop-gradient and $\mathcal{L}_{\mathrm{var}}$ is a
VICReg-style variance term. The total loss is
$\mathcal{L} = \mathcal{L}_{\mathrm{heads}}
             + \lambda_{\mathrm{JEPA}} \mathcal{L}_{\mathrm{JEPA}}$.
We attach the same head to Qwen, Gemma, and SLM backbones using identical
data, optimizer, schedule, and checkpoint-selection metric, and we include a
shuffled-future-label control that preserves marginals but breaks temporal
alignment.
```

### 7.4 New subsection in Experiments

Add a JEPA benchmark subsection presenting the run table from §2.3 and the
ablation grid from §3.3.

### 7.5 New tables

- **Table: JEPA latent-state comparison** across Qwen, Gemma, GPT SLM,
  Mamba-like SLM, each with baseline and JEPA (and shuffled-JEPA for Qwen and
  one SLM).
- **Table: JEPA ablations** (lambda, horizons, emb_dim, var_weight) on one
  backbone.
- **Table: OCEAN/VAD placebo** (already supported by Ablation 2 data).

### 7.6 Softened claims

- Replace unconditional OCEAN/VAD claims with placebo-aware framing.
- Frame JEPA as auxiliary predictive representation objective, not a new
  architecture.
- Do not claim JEPA improves generation unless Track C/D experiments
  explicitly show it.

### 7.7 New references

Add to `paper/references.bib`:

```bibtex
@article{lecun2022path,
  title   = {A Path Towards Autonomous Machine Intelligence},
  author  = {LeCun, Yann},
  journal = {OpenReview},
  year    = {2022}
}

@inproceedings{assran2023self,
  title     = {Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture},
  author    = {Assran, Mahmoud and Duval, Quentin and Misra, Ishan and Bojanowski, Piotr and Vincent, Pascal and Rabbat, Michael and LeCun, Yann and Ballas, Nicolas},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2023}
}

@inproceedings{bardes2022vicreg,
  title     = {{VICReg}: Variance-Invariance-Covariance Regularization for Self-Supervised Learning},
  author    = {Bardes, Adrien and Ponce, Jean and LeCun, Yann},
  booktitle = {International Conference on Learning Representations},
  year      = {2022}
}
```

Cite in the JEPA subsection and in related work.

### 7.8 Related work addition

Add a short paragraph under the controlled-generation or representation-learning
subsection:

```
Joint-embedding predictive architectures (JEPA) have recently emerged as a
general framework for predicting abstract future representations rather than
raw inputs \cite{lecun2022path,assran2023self}. We adapt this principle to
structured dialogue-state prediction: instead of predicting pixels or future
tokens, our Social-State JEPA predicts an embedding of the future 29-dim
social state $Z_{t+k}$, regularised by a VICReg-style variance term
\cite{bardes2022vicreg} to prevent representational collapse.
```

---

## 8. Milestones

| Milestone | Deliverable | Depends on |
|-----------|-------------|------------|
| M1 | Shuffled-future dataset mode (T1) | — |
| M2 | Qwen `jepa_shuffled` run complete | M1, current Qwen JEPA run |
| M3 | Gemma training path works (T2) + configs (T4) | — |
| M4 | Gemma baseline + JEPA runs complete | M3 |
| M5 | SLM latent trainer (T3) + configs (T4) | — |
| M6 | GPT SLM and Mamba-like baseline + JEPA runs complete | M5 |
| M7 | Mamba-like shuffled JEPA run complete | M5 |
| M8 | Evaluation script (T5) + all JSON/CSV artifacts | M2, M4, M6, M7 |
| M9 | Paper Section update (§7) merged | M8 |
| M10 | Final polish: bootstrap CIs, figures, abstract rewrite | M9 |

Minimum publishable milestones: M1, M2, M4, M6, M8, M9.

---

## 8.5 SLURM Execution Plan

The project runs on an HPC cluster using SLURM. Existing wrappers:

- `scripts/slurm_train.sh`
- `scripts/slurm_eval.sh`
- `scripts/slurm_array.sh`
- `scripts/slurm_train_eval.sh`
- `scripts/mlflow_env.sh`

Cluster conventions:

- Use `--gpus-per-node`, not `--gres=gpu`.
- Default account: `xrimlab`.
- Recommended partitions:
  - `gpu-l20` for Gemma/Qwen and larger QLoRA jobs.
  - `gpu-a30` for Qwen if memory fits.
  - `gpu-rtx4090d` for fast single-GPU SLM/JEPA jobs if available.
- Gemma jobs should export:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Priority 1: Qwen shuffled-future JEPA placebo

Purpose: essential placebo for the current Qwen JEPA pilot.

```bash
sbatch \
  --job-name=qwen_jepa_shuf \
  --partition=gpu-l20 \
  --account=xrimlab \
  --gpus-per-node=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=8 \
  --time=24:00:00 \
  scripts/slurm_train.sh llm latent \
  --config llm_finetuning/configs/train_latent_qwen_jepa_shuffled.yaml
```

### Priority 2: Qwen tuned JEPA

Purpose: rerun JEPA with paper-default hyperparameters if the pilot used stronger settings.

```bash
sbatch \
  --job-name=qwen_jepa_tuned \
  --partition=gpu-l20 \
  --account=xrimlab \
  --gpus-per-node=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=8 \
  --time=24:00:00 \
  scripts/slurm_train.sh llm latent \
  --config llm_finetuning/configs/train_latent_qwen_jepa.yaml
```

### Priority 3: SLM JEPA matrix

Purpose: compare JEPA across small architectures.

Once `slm_training/src/train/train_latent_slm.py` exists, use:

```bash
sbatch \
  --job-name=slm_latent_gpt_jepa \
  --partition=gpu-a30 \
  --account=xrimlab \
  --gpus-per-node=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=8 \
  --time=12:00:00 \
  scripts/slurm_train.sh slm latent \
  --config slm_training/configs/latent_gpt_jepa.yaml
```

and:

```bash
sbatch \
  --job-name=slm_latent_mamba_jepa \
  --partition=gpu-a30 \
  --account=xrimlab \
  --gpus-per-node=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=8 \
  --time=12:00:00 \
  scripts/slurm_train.sh slm latent \
  --config slm_training/configs/latent_mamba_jepa.yaml
```

Required SLM controls:

```bash
sbatch scripts/slurm_train.sh slm latent --config slm_training/configs/latent_gpt_base.yaml
sbatch scripts/slurm_train.sh slm latent --config slm_training/configs/latent_mamba_base.yaml
sbatch scripts/slurm_train.sh slm latent --config slm_training/configs/latent_mamba_jepa_shuffled.yaml
```

### Priority 4: Conditioning placebo multi-seed array

Purpose: validate the OCEAN/VAD placebo result with seeds 42/43/44.

Use existing `scripts/slurm_array.sh` or submit individual dialogue jobs with:

```bash
sbatch scripts/slurm_train.sh slm dialogue --seed 42
sbatch scripts/slurm_train.sh slm dialogue --seed 43
sbatch scripts/slurm_train.sh slm dialogue --seed 44
```

Then repeat for:

- real OCEAN + real VAD
- shuffled OCEAN + real VAD
- real OCEAN + random VAD
- shuffled OCEAN + random VAD
- no-conditioning baseline

If no-conditioning is not currently supported by CLI/config, add an explicit `condition_mode: none` or `disable_conditioning: true` config before running.

### Priority 5: Final evaluation job

Purpose: true Cohen's κ, bootstrap CIs, per-head CSV, and paper tables.

```bash
sbatch \
  --job-name=jepa_final_eval \
  --partition=gpu-l20 \
  --account=xrimlab \
  --gpus-per-node=1 \
  --ntasks-per-node=1 \
  --cpus-per-task=4 \
  --time=04:00:00 \
  scripts/slurm_eval.sh llm latent \
  --config llm_finetuning/configs/eval_jepa.yaml
```

### SLURM implementation gap

`scripts/slurm_train.sh` currently supports:

- `llm latent|response|joint`
- `slm personality|affect|small_lm|dialogue`

It does **not** yet support:

```bash
scripts/slurm_train.sh slm latent --config ...
```

Required wrapper patch:

```bash
slm_latent)
  cd \"${REPO_DIR}/slm_training\"
  export PYTHONPATH=\"${REPO_DIR}/slm_training:${REPO_DIR}/llm_finetuning\"
  python -m src.train.train_latent_slm \"${EXTRA_ARGS[@]}\" \\
      2>&1 | tee \"${LOG_DIR}/${RUN_ID}.log\"
  ;;
```

Also update the valid-combos help text to include `slm: latent`.

---

## 8.6 Paper Consistency Checklist

Before final submission:

- [x] Replace OCEAN/VAD semantic-causality claims with prefix-placebo framing.
- [x] Mark reported κ values as estimated normalized-above-chance agreement unless true Cohen's κ is recomputed.
- [ ] Compute true Cohen's κ from confusion matrices in final eval.
- [ ] Add bootstrap 95% CIs for headline metrics.
- [ ] Add paired CIs for joint vs separate.
- [ ] Add shuffled-future JEPA result before making a temporal JEPA claim.
- [ ] Add multi-seed CIs for conditioning placebo results.
- [ ] Normalize or clearly caveat Track C PPL comparisons with different tokenizers, data subsets, and epoch counts.
- [ ] Verify all 2025 NPC-dialogue references in `paper/references.bib`; remove unverifiable citations.
- [ ] Evaluate routing on predicted `Z_t` or remove perfect-F1 routing as a result.
- [ ] Report leakage as `0/n` with a confidence interval, not as a guarantee.

---

## 9. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| JEPA helps `jepa_loss` but hurts `response_policy_f1` | lower `lambda_jepa` to 0.01–0.03, reduce `emb_dim` to 32, keep horizon 1 |
| Gemma QLoRA memory issues | use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, reduce `batch_size`, increase `grad_accum` |
| Missing future turns reduce effective batch | start with horizon 1, mask `-1`, require at least 50% valid per batch |
| Shuffled-JEPA leaks via correlated labels | sample across episodes, not within; assert zero `(episode, turn+h)` overlap |
| Overclaiming JEPA | stick to §6 acceptance criteria and §7.6 softened claims |
| Insufficient compute for full matrix | fall back to minimum viable set: Q-base, Q-jepa, Q-jepa-shuf, SG/SM baseline + JEPA |

---

## 10. Open Questions

- Do we want JEPA embeddings to also condition the response generator (Track C/D)? This is a separate experiment and should not be mixed with the representation ablation.
- Should `val/macro_f1` replace `val/response_policy_f1` as the primary selection metric? Decide once M2 and M4 results are in.
- Should `emb_dim` scale with label cardinality per field? Current implementation uses a single `emb_dim` for all fields.

---

## 11. Pointers

- Implementation:
  - `llm_finetuning/src/training/jepa.py`
  - `llm_finetuning/src/training/dataset.py`
  - `llm_finetuning/src/training/train_latent.py`
  - `llm_finetuning/configs/train_latent.yaml`
  - `llm_finetuning/configs/train_latent_jepa.yaml`
- Docs:
  - `docs/ablations.md` (Ablation 6)
  - `docs/jepa_plan.md` (this file)
- Paper:
  - `paper/main.tex`
  - `paper/references.bib`
