# Complete Results — All Tracks, All Backbones

_Generated 2026-05-07. Single source of truth for paper tables._

---

## Track D: Latent Social-State Prediction (RQ*)

**Metric:** Mean per-head accuracy / Cohen's κ / Macro-F1 on val set (683 turns).  
**Protocol:** Same 29-head schema, same loss weights, same checkpoint selection (`val/response_policy_f1`). Single seed (42).

| Backbone | Params | Type | Accuracy | κ | F1 | % of Qwen |
|----------|--------|------|----------|-----|-----|-----------|
| **Qwen3-1.7B** | 1.7B | Pretrained + QLoRA | **0.686** | **0.441** | 0.484 | 100% |
| **GPT-SLM** | 17.9M | From-scratch | 0.627 | 0.370 | 0.428 | 91.4% |
| **MoE-SLM** | ~25M | From-scratch MoE | 0.578 | — | — | 84.3% |
| **Gemma-4-E2B** | 16B/2B | Pretrained MoE + QLoRA | 0.539 | — | 0.198 | 78.6% |
| **Mamba-SLM** | 15M | From-scratch SSM | 0.474 | 0.148 | 0.313 | 69.0% |

---

## Track D: JEPA Auxiliary Objective

| Backbone | Base Acc | +JEPA Acc | Δ | JEPA-shuf Acc | Verdict |
|----------|----------|-----------|-----|---------------|---------|
| Qwen3-1.7B | 0.686 | 0.698 | +0.013 | 0.686 | Null |
| GPT-SLM | 0.627 | 0.623 | −0.004 | — | Null |
| Mamba-SLM | 0.474 | 0.495 | +0.022 | 0.366 | Null |
| Gemma-4-E2B | 0.539 | 0.546 | +0.007 | — | Null |

**Finding:** JEPA is null across all 4 backbones (Δ ≤ ±0.02, within noise). Shuffled-future placebo confirms temporal structure detected (4.7× higher jepa_loss) but doesn't improve downstream accuracy.

---

## Track A: From-Scratch SLM Language Modeling

**Metric:** Next-token PPL on 16,905-line NPC dialogue corpus. 20 epochs.

| Model | Params | PPL |
|-------|--------|-----|
| **MoE** | 22.4M | **42.07** |
| PrefixGPT | 16.6M | 44.54 |
| GPT | 16.1M | 45.32 |
| Mamba-like | 15.4M | 53.25 |

---

## Track C: Dialogue Response Generation

**Metric:** Response generation PPL on NPC dialogue val set.

| Model | Params | Conditioning | PPL | Epochs |
|-------|--------|-------------|-----|--------|
| **Qwen3-1.7B + QLoRA** | 1.7B | None (SFT) | **1.04** | 3 |
| **TinyLlama 1.1B + LoRA** | 1.1B | OCEAN+VAD prefix | **2.88±0.02** | 3 |
| TinyLlama 1.1B + LoRA | 1.1B | None | 3.30 | 3 |
| Gemma-4-E2B + QLoRA | 16B/2B | NPC profile SFT | 16.24 | 1 |

---

## Ablation 2: Conditioning Placebo Test

**3 seeds (42/43/44) on TinyLlama 1.1B. 3 epochs each.**

| Condition | Seed 42 | Seed 43 | Seed 44 | Mean ± std |
|-----------|---------|---------|---------|------------|
| Real OCEAN + Real VAD | 2.90 | 2.87 | 2.87 | **2.88 ± 0.02** |
| Shuffled OCEAN + Real VAD | 2.88 | — | — | 2.88 |
| Real OCEAN + Random VAD | 2.91 | — | — | 2.91 |
| Shuffled OCEAN + Random VAD | 2.90 | — | — | 2.90 |
| No conditioning | 3.30 | — | — | 3.30 |

**Finding:** All conditioned variants converge to PPL 2.88–2.91. The 12.3% gain is from prefix capacity, not OCEAN/VAD semantic content.

---

## Ablation 4: Joint vs Separate Training

| Metric | Separate | Joint | Δ |
|--------|----------|-------|---|
| Mean accuracy | 0.683 | 0.674 | −0.008 |
| Mean κ | 0.438 | 0.414 | −0.024 |
| Response PPL | 1.04 | 1.04 | 0.00 |

Per-head tradeoffs: secrecy_pressure +0.12, reveal_decision −0.10, respect_level −0.09, familiarity_level −0.09.

---

## Ablation 5: Consistency Loss

| λ | Train Loss | Val Loss | Violations |
|---|-----------|---------|------------|
| 0.5 | 4.31 | 6.48 | 0/652 |
| 0.0 | 4.31 | 6.47 | 0/652 |

**Finding:** No measurable effect.

---

## Ablation 6: JEPA (see Track D JEPA table above)

**Finding:** Null across all backbones. Shuffled-future placebo confirms temporal structure detection but no downstream benefit.

---

## Gold Z_t Bridge Experiment

| Field | Effect (σ) | Hardest → Easiest | Δ PPL |
|-------|-----------|-------------------|-------|
| **reveal_decision** | **1.02** | full (11.73) → partial (7.23) | +4.50 |
| repair_strategy | 0.59 | apologize (10.67) → redirect (8.08) | +2.59 |
| response_policy | 0.57 | negotiate (8.43) → clarify (7.34) | +1.09 |

**Finding:** Z_t fields systematically predict generation difficulty. Full-reveal turns are 62% harder than partial-reveal turns.

---

## Conditioning Encoders (Track B)

| Encoder | Backbone | Metric | Value |
|---------|----------|--------|-------|
| Personality (OCEAN) | DistilBERT 66M | F1 | 0.678 |
| Affect (VAD) | DistilBERT 66M | F1 | 0.559 |
