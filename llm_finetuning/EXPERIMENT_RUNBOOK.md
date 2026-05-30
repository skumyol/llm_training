# Experiment Runbook: From Social State to Decision State

## Compressing Inspectable Bottlenecks for Controllable NPC Dialogue

This runbook tests whether a 29-dimensional social-state schema can be compressed into a smaller, calibrated decision-relevant bottleneck without losing routing or disclosure-control performance.

**Research framing:** This is not a bag of engineering improvements. It is a study of **which social variables survive operational pressure**. The core question is:

> Which parts of a structured social state are actually decision-relevant for controllable NPC dialogue?

**Prerequisites:**
- Trained latent predictor checkpoint at `checkpoints/latent_predictor_best`
- Trained response generator checkpoint at `checkpoints/response_generator_best`
- Data splits: `data/splits/{train,val,test}_heads.jsonl`, `data/splits/{val,test}_sft.jsonl`, `data/splits/{val,test}_trace.jsonl`

---

## Explicit Hypotheses

| Hypothesis | Statement | Test |
|------------|-----------|------|
| **H1** | A compressed decision-state subset can match the routing performance of the full 29D social state. | Head ablation (Study 1) |
| **H2** | Low-agreement / low-utility heads add little or no operational value to routing. | MI ranking + ablation impact (Study 1) |
| **H3** | Confidence-aware selective routing reduces leakage risk at acceptable slow-path cost. | Calibration + selective router eval (Study 2) |
| **H4** | Decision-card prompting performs at least as well as full-state prompting while improving interpretability. | Response generation with/without decision cards (Study 3) |

---

## Study Structure

| Study | Question | Phases | Key Deliverable |
|-------|----------|--------|-----------------|
| **Study 1:** State Compression | Which heads are decision-relevant? | 1–3 | Ablation matrix, minimal sufficient head set |
| **Study 2:** System Hardening | Does a compressed calibrated state improve reliability? | 4–6 | ECE reduction, FP rate, leakage reduction |
| **Study 3:** Generation Control | Does decision-card prompting improve generation? | 5–6 | ROUGE-L, leakage, interpretability comparison |
| **Study 4:** Adversarial Stress | Is the system robust under manipulative pressure? | 7 | Adversarial leakage, slow-path recall |

---

## Important Caveat: Mutual Information is Exploratory

MI and NMI are used for **exploratory ranking only**, not as the sole pruning criterion. A head may have low individual MI but still be useful in combination with another head. The **real criterion is ablation impact on routing and leakage**. Always validate MI rankings with controlled ablation experiments.

---

## Important Methodological Note: Episode-Level Bootstrap

Because data contains multi-turn episodes, turns are **not independent**. All bootstrap confidence intervals must use **episode-level resampling**: resample episodes (not turns), then collect all turns from each resampled episode. Turn-level bootstrap yields optimistically narrow CIs and is **not valid** for this data.

---

## Phase 1: Head Utility Audit (CPU, ~5 min)

**Goal:** Rank every head by mutual information with routing decisions. Identify candidate noise heads.

**Note:** MI is exploratory. The real test is ablation impact in Phase 3.

```bash
cd llm_finetuning
PYTHONPATH=. python scripts/analyze_head_utility.py \
    --heads-file data/splits/val_heads.jsonl \
    --output-dir eval_results/head_utility \
    --min-support 10
```

**Outputs:**
- `eval_results/head_utility/head_utility_report.md` — ranked NMI table with redundancy analysis
- `eval_results/head_utility/head_utility.json` — raw scores

**Decision Gate 1:**
- If `response_policy`, `reveal_decision`, `value_conflict`, `secrecy_pressure` are top-4 by NMI → proceed to ablation.
- If any routing head has NMI < 0.05 → note as candidate for ablation, but do not drop without ablation test.
- If relational heads (trust_level, respect_level) have NMI > 0.10 → include in ablation experiments.

---

## Phase 2: Label Collapse (CPU, ~2 min)

**Goal:** Test whether collapsing ambiguous 5-class relational deltas to 3-class improves reliability. **Run this before final ablation** so you don't conclude relational heads are useless when the real problem is over-resolved labels.

```bash
# Collapse stance deltas and levels
PYTHONPATH=. python scripts/collapse_labels.py \
    --input data/splits/train_heads.jsonl \
    --output data/splits/train_heads_collapsed.jsonl \
    --collapse stance_deltas stance_levels

# Also collapse val set
PYTHONPATH=. python scripts/collapse_labels.py \
    --input data/splits/val_heads.jsonl \
    --output data/splits/val_heads_collapsed.jsonl \
    --collapse stance_deltas stance_levels
```

Then compute head utility on collapsed labels:
```bash
PYTHONPATH=. python scripts/analyze_head_utility.py \
    --heads-file data/splits/val_heads_collapsed.jsonl \
    --output-dir eval_results/head_utility_collapsed
```

**Decision Gate 2:**
- If collapsed-label NMI for relational heads improves by > 20% relative to fine-grained → **schema was over-resolved**.
- If no improvement → keep fine-grained labels; the heads are genuinely low-utility.

---

## Phase 3: Head Ablation (GPU, ~6 jobs × 30 min each)

**Goal:** Compare routing performance across head subsets. **Distinguish masking ablation from retraining ablation.**

### Two Types of Ablation

| Type | Command | What it answers |
|------|---------|-----------------|
| **Masking** | `--masking-mode` | Does the current router depend on this head? (remove at eval time) |
| **Retraining** | `--train` (default) | Can we build a smaller system from scratch without this head? |

Both are useful but answer different questions. For the paper, run both if possible.

### M0 — Baseline (all 29 heads)

Evaluate existing full model:
```bash
PYTHONPATH=. python scripts/run_head_ablation.py \
    --config configs/eval.yaml \
    --name m0_baseline \
    --test-trace-file data/splits/val_trace.jsonl
```

### M1mask — Masking: routing-only (6 heads)

Load full model, evaluate with only 6 heads:
```bash
PYTHONPATH=. python scripts/run_head_ablation.py \
    --config configs/eval.yaml \
    --heads response_policy reveal_decision value_conflict secrecy_pressure player_intent threat \
    --name m1_masking_routing_only \
    --masking-mode \
    --test-trace-file data/splits/val_trace.jsonl
```

### M1 — Retraining: routing-only (6 heads)

Train ablated predictor from scratch:
```bash
PYTHONPATH=. python scripts/run_head_ablation.py \
    --config configs/eval.yaml \
    --heads response_policy reveal_decision value_conflict secrecy_pressure player_intent threat \
    --name m1_retrain_routing_only \
    --train \
    --epochs 3 \
    --batch-size 8 \
    --lr 2e-5 \
    --test-trace-file data/splits/val_trace.jsonl
```

### M2 — Retraining: +affect (9 heads)

```bash
PYTHONPATH=. python scripts/run_head_ablation.py \
    --config configs/eval.yaml \
    --heads response_policy reveal_decision value_conflict secrecy_pressure player_intent threat valence arousal tone \
    --name m2_retrain_plus_affect \
    --train \
    --test-trace-file data/splits/val_trace.jsonl
```

### M3 — Retraining: +relational levels (12 heads)

```bash
PYTHONPATH=. python scripts/run_head_ablation.py \
    --config configs/eval.yaml \
    --heads response_policy reveal_decision value_conflict secrecy_pressure player_intent threat trust_level respect_level \
    --name m3_retrain_plus_relational \
    --train \
    --test-trace-file data/splits/val_trace.jsonl
```

### M4 — Retraining: +all relational (18 heads)

```bash
PYTHONPATH=. python scripts/run_head_ablation.py \
    --config configs/eval.yaml \
    --heads response_policy reveal_decision value_conflict secrecy_pressure player_intent threat trust_level respect_level affection_level familiarity_level dominance_level obligation_level \
    --name m4_retrain_all_relational \
    --train \
    --test-trace-file data/splits/val_trace.jsonl
```

### Aggregate results

```bash
PYTHONPATH=. python scripts/aggregate_ablation_results.py \
    --results-dir eval_results/ablation \
    --output eval_results/ablation_matrix.md \
    --n-bootstrap 1000
```

**Outputs:**
- `eval_results/ablation/m*/ablation_metrics.json` — per-experiment metrics
- `eval_results/ablation_matrix.md` — comparison table
- `eval_results/ablation_matrix.tex` — LaTeX table for paper

**Decision Gate 3:**
- If M1 (retraining, 6 heads) F1 is within 0.03 of M0 → **compression is viable**.
- If M2 F1 > M1 + 0.02 → affect heads add value; consider keeping.
- If M3 F1 ≈ M2 → relational levels add no operational value; drop them.
- If M4 F1 < M3 → too many heads hurts; overfitting confirmed.
- If M1mask F1 ≈ M1retrain → the original model already learned to ignore extra heads.
- If M1mask F1 << M1retrain → the full model uses extra heads at test time even if they are not necessary.

**Recommended commit point:** Pick the smallest head set with F1 ≥ M0 − 0.03.

---

## Phase 4: Calibration (GPU, ~10 min)

**Goal:** Apply temperature scaling per head. Reduce ECE.

```bash
PYTHONPATH=. python scripts/calibrate_head.py \
    --config configs/eval.yaml \
    --method temperature \
    --calib-heads-file data/splits/val_heads.jsonl \
    --output-dir calibrators/temperature \
    --batch-size 16
```

**Outputs:**
- `calibrators/temperature/calibration_summary.json` — ECE before/after per head
- `calibrators/temperature/apply_calibration.py` — auto-generated inference wrapper

**Decision Gate 4:**
- If ECE for `reveal_decision` and `response_policy` drops below 0.05 → enable selective router.
- If ECE stays high (> 0.15) → try isotonic method, or investigate label quality.

---

## Phase 5: Selective Router Evaluation (GPU, ~5 min)

**Goal:** Compare deterministic routing vs confidence-aware selective routing.

### 5a. Baseline (deterministic)

```yaml
# configs/eval.yaml
selective_router:
  enabled: false
routing_mode: predicted
```

```bash
PYTHONPATH=. python run_eval.py --stage routing --config configs/eval.yaml
```

### 5b. Selective router (confidence-aware)

Requires calibration from Phase 4. Edit `configs/eval.yaml`:
```yaml
selective_router:
  enabled: true
  thresholds:
    response_policy: 0.65
    reveal_decision: 0.70
    value_conflict: 0.75
    secrecy_pressure: 0.75
  calibration_dir: calibrators/temperature
```

```bash
PYTHONPATH=. python run_eval.py --stage routing --config configs/eval.yaml
```

### 5c. Threshold sweep (optional, for paper figure)

```bash
for tau in 0.50 0.60 0.70 0.80 0.90; do
  sed -i "s/reveal_decision: .*/reveal_decision: $tau/" configs/eval.yaml
  PYTHONPATH=. python run_eval.py --stage routing --config configs/eval.yaml
  mv eval_results/routing_eval_metrics.json eval_results/routing_tau_${tau}.json
done
```

**Decision Gate 5:**
- If selective router reduces FP rate by > 0.02 with < 0.05 increase in slow-path rate → **keep it**.
- If slow-path rate explodes (> 0.15 increase) → thresholds are too conservative; relax them.

---

## Phase 6: Decision Card Evaluation (GPU, ~15 min)

**Goal:** Compare full 29-head prompt vs compressed decision card for slow-path generation.

### 6a. Baseline (full state dump)

```yaml
# configs/eval.yaml
decision_card:
  enabled: false
```

Run latent eval first to produce predicted state file:
```bash
PYTHONPATH=. python run_eval.py --stage latent --config configs/eval.yaml
```

Then response eval:
```bash
PYTHONPATH=. python run_eval.py --stage response --config configs/eval.yaml
```

Save baseline:
```bash
cp eval_results/response_eval_metrics.json eval_results/response_baseline.json
cp eval_results/sample_generations.json eval_results/sample_generations_baseline.json
```

### 6b. Decision card

```yaml
# configs/eval.yaml
decision_card:
  enabled: true
```

```bash
PYTHONPATH=. python run_eval.py --stage response --config configs/eval.yaml
```

**Decision Gate 6:**
- If decision card leakage rate ≤ baseline AND ROUGE-L within 0.05 → **decision cards are better** (cleaner + safer).
- If leakage drops but ROUGE-L drops > 0.10 → cards are too restrictive; relax constraints.
- If leakage increases → cards are missing critical state; add back necessary heads.

---

## Phase 7: Leakage Validator (GPU, ~20 min)

**Goal:** Measure leakage reduction from validate-and-regenerate loop.

Requires `sample_generations.json` from Phase 6. Run validator:

```bash
PYTHONPATH=. python scripts/validate_and_regenerate.py \
    --config configs/eval.yaml \
    --input eval_results/sample_generations.json \
    --classifier-dir leakage_classifier/final \
    --max-retries 2 \
    --leak-threshold 0.5 \
    --output eval_results/validated_generations.json \
    --device auto
```

**Outputs:**
- `eval_results/validated_generations.json` — per-response validation + retries
- Console summary: accepted first try, accepted after retry, max retries exceeded, gated leakage rate

**Decision Gate 7:**
- If validator reduces gated leakage by > 50% with < 20% of responses needing retry → **keep it**.
- If > 30% need retry → leakage classifier is too sensitive; increase threshold or improve calibration.

---

## Phase 8: Adversarial Evaluation (GPU, ~20 min)

**Goal:** Stress-test the compressed system under manipulative player inputs.

### 8a. Generate adversarial test set

```bash
PYTHONPATH=. python scripts/generate_adversarial_set.py \
    --config configs/eval.yaml \
    --output data/adversarial/adversarial_test.jsonl \
    --n-episodes 50 \
    --attack-types threat bribery false_alliance authority_mimicry emotional_blackmail
```

### 8b. Run full adversarial pipeline

```bash
PYTHONPATH=. python run_eval.py --stage adversarial --config configs/eval.yaml
```

**Outputs:**
- `eval_results/adversarial_eval/adversarial_eval_metrics.json`
- Metrics: adversarial leakage rate, slow-path recall under pressure, false refusal rate

**Decision Gate 8:**
- If adversarial leakage < 0.05 AND slow-path recall > 0.80 → **system is robust**.
- If adversarial leakage > 0.10 → router is failing under pressure; strengthen selective thresholds or add more adversarial training data.

---

## Phase 9: Final Aggregation (CPU, ~1 min)

**Goal:** Collect all results into a single report for the paper.

```bash
PYTHONPATH=. python scripts/aggregate_ablation_results.py \
    --results-dir eval_results/ablation \
    --output eval_results/final_ablation_matrix.md

cat eval_results/head_utility/head_utility_report.md > eval_results/FINAL_REPORT.md
echo "" >> eval_results/FINAL_REPORT.md
cat eval_results/ablation_matrix.md >> eval_results/FINAL_REPORT.md
```

Then manually append:
- Selective router comparison (Phase 5)
- Decision card comparison (Phase 6)
- Validator summary (Phase 7)
- Adversarial results (Phase 8)

---

## Quick Reference: One-Command Full Pipeline

```bash
cd llm_finetuning

# Study 1: State Compression
PYTHONPATH=. python scripts/analyze_head_utility.py --heads-file data/splits/val_heads.jsonl --output-dir eval_results/head_utility
PYTHONPATH=. python scripts/collapse_labels.py --input data/splits/train_heads.jsonl --output data/splits/train_heads_collapsed.jsonl --collapse stance_deltas stance_levels
PYTHONPATH=. python scripts/run_head_ablation.py --config configs/eval.yaml --heads response_policy reveal_decision value_conflict secrecy_pressure player_intent threat --name m1_retrain_routing_only --train --test-trace-file data/splits/val_trace.jsonl
PYTHONPATH=. python scripts/run_head_ablation.py --config configs/eval.yaml --heads response_policy reveal_decision value_conflict secrecy_pressure player_intent threat trust_level respect_level --name m3_retrain_plus_relational --train --test-trace-file data/splits/val_trace.jsonl
PYTHONPATH=. python scripts/aggregate_ablation_results.py --results-dir eval_results/ablation --output eval_results/ablation_matrix.md

# Study 2: System Hardening
PYTHONPATH=. python scripts/calibrate_head.py --config configs/eval.yaml --method temperature --calib-heads-file data/splits/val_heads.jsonl --output-dir calibrators/temperature
# Edit configs/eval.yaml: selective_router.enabled = true
PYTHONPATH=. python run_eval.py --stage routing --config configs/eval.yaml

# Study 3: Generation Control
# Edit configs/eval.yaml: decision_card.enabled = true
PYTHONPATH=. python run_eval.py --stage all --config configs/eval.yaml
PYTHONPATH=. python scripts/validate_and_regenerate.py --config configs/eval.yaml --input eval_results/sample_generations.json --classifier-dir leakage_classifier/final --max-retries 2 --output eval_results/validated_generations.json

# Study 4: Adversarial Stress
PYTHONPATH=. python run_eval.py --stage adversarial --config configs/eval.yaml
```

---

## Hardware Requirements

| Phase | GPU | Time | Notes |
|-------|-----|------|-------|
| 1. Head Utility | No | 5 min | Pure statistics on JSONL |
| 2. Label Collapse | No | 2 min | Data rewriting |
| 3. Ablation | Yes | 30 min × 6 | Parallelize; both masking and retraining |
| 4. Calibration | Yes | 10 min | Forward pass over val set |
| 5. Selective Router | Yes | 5 min | Routing eval only |
| 6. Decision Card | Yes | 15 min | Response generation |
| 7. Validator | Yes | 20 min | Generation + classifier forward |
| 8. Adversarial | Yes | 20 min | Full pipeline on adversarial data |

**Total GPU time (serial):** ~2.5 hours  
**Total GPU time (parallel ablations):** ~1.5 hours  
**Total wall time:** ~3–4 hours

---

## Expected Final Deliverables

1. `eval_results/head_utility/head_utility_report.md` — MI ranking + redundancy analysis
2. `eval_results/head_utility_collapsed/` — MI ranking on collapsed labels
3. `eval_results/ablation_matrix.md` — routing F1 vs # heads (masking + retraining)
4. `eval_results/ablation_matrix.tex` — LaTeX table for paper
5. `eval_results/routing_eval_metrics.json` — selective router comparison
6. `eval_results/response_eval_metrics.json` — decision card comparison
7. `eval_results/validated_generations.json` — validator effectiveness
8. `eval_results/adversarial_eval/` — robustness under pressure
9. `calibrators/temperature/calibration_summary.json` — per-head ECE

These form the empirical core of the follow-up paper:
> "From Social State to Decision State: Compressing Inspectable Bottlenecks for Controllable NPC Dialogue"
