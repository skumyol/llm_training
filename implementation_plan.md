# Implementation Plan: Data Generation & Model Training

## Project Summary

Phase 2 of the NPC dialogue research project. Goal: train a compact (≤4B param),
edge-deployable model that explicitly predicts a structured social latent state per
turn and generates believable NPC dialogue conditioned on that state.

- **Primary backbone:** Qwen3-4B (QLoRA, 4-bit, 12GB GPU)
- **Debug/fast-iteration backbone:** Qwen3-1.7B
- **Framework:** HuggingFace Transformers + PEFT + bitsandbytes
- **Tracking:** MLflow (experiments, datasets, artifacts, traces)
- **Timeline:** 10–12 weeks

---

## Latent State Schema (Source of Truth)

Every component below is built around this schema.

### Per-turn state Z_t

| Symbol | Name                | Timescale | Fields |
|--------|---------------------|-----------|--------|
| `C_t`  | Interpretation      | Fast      | `dialogue_act`, `tone`, `risk_type` |
| `A_t`  | Affect/Appraisal    | Fast      | `valence`, `arousal`, `threat`, `control` |
| `M_t`  | Player Model (ToM)  | Fast/Med  | `player_intent`, `player_knowledge`, `player_credibility` |
| `R_t`  | Relational Stance   | Medium    | `affection`, `respect`, `dominance`, `familiarity`, `trust`, `obligation` (each: level + delta) |
| `N_t`  | Norm Constraints    | Medium    | `duty_pressure`, `secrecy_pressure`, `face_pressure`, `value_conflict` |
| `D_t`  | Response Policy     | Fast      | `response_policy`, `reveal_decision`, `repair_strategy` |

### Persistent state W (conditioned on, not predicted per turn)

```json
{
  "npc_id": "guard_01",
  "role": "castle_guard",
  "persona_style": ["formal", "guarded"],
  "core_goals": ["protect_vault", "avoid_scandal"],
  "values": ["duty", "loyalty", "self_preservation"],
  "secrets": [{"secret_id": "chalice_location", "severity": "high"}],
  "initial_stance": {
    "affection": "neutral", "respect": "low",
    "dominance": "medium", "familiarity": "low", "trust": "low"
  }
}
```

### Allowed label values

```python
LABEL_SCHEMA = {
    "dialogue_act": ["ask", "accuse", "threaten", "flatter", "apologize",
                     "negotiate", "joke", "confess", "probe", "command"],
    "tone":         ["warm", "neutral", "confrontational", "sarcastic", "fearful", "evasive"],
    "risk_type":    ["none", "secret-risk", "face-risk", "status-risk", "conflict-risk"],
    "valence":      ["negative", "neutral", "positive"],
    "arousal":      ["low", "medium", "high"],
    "threat":       ["low", "medium", "high"],
    "control":      ["low", "medium", "high"],
    "player_intent":      ["seek-info", "trap", "bond", "manipulate", "test", "persuade", "intimidate"],
    "player_knowledge":   ["unaware", "partial", "informed", "knows-secret"],
    "player_credibility": ["low", "medium", "high"],
    "stance_level": ["VL", "L", "N", "H", "VH"],
    "stance_delta": ["--", "-", "0", "+", "++"],
    "duty_pressure":    ["low", "medium", "high"],
    "secrecy_pressure": ["low", "medium", "high"],
    "face_pressure":    ["low", "medium", "high"],
    "value_conflict":   ["none", "mild", "strong"],
    "response_policy":  ["answer", "partial", "withhold", "deflect", "challenge",
                         "soothe", "test", "threaten", "negotiate"],
    "reveal_decision":  ["none", "hint", "partial", "full"],
    "repair_strategy":  ["none", "soften", "apologize", "clarify", "redirect"],
}
```

---

## Project Folder Structure

```
llm_training/
├── configs/
│   ├── schema_v1.json          # canonical label schema (auto-generated from LABEL_SCHEMA)
│   ├── data_gen.yaml           # teacher LLM, batch sizes, turn budget, filters
│   ├── train_latent.yaml       # QLoRA config for latent-state predictor
│   ├── train_response.yaml     # QLoRA config for response generator
│   └── eval.yaml               # eval thresholds, held-out scenario list
│
├── data/
│   ├── scenario_bank/          # YAML scenario templates
│   ├── npc_profiles/           # JSON NPC character definitions
│   ├── raw_episodes/           # unvalidated generator output (JSONL)
│   ├── validated_turns/        # per-turn records passing all validators (JSONL)
│   ├── counterfactuals/        # augmented counterfactual episodes (JSONL)
│   └── splits/
│       ├── train.jsonl
│       ├── val.jsonl
│       └── test.jsonl
│
├── prompts/
│   ├── scenario_instantiation.txt
│   ├── label_C.txt             # prompts for C_t labeling
│   ├── label_A_M.txt           # prompts for A_t + M_t labeling
│   ├── label_R_N_D.txt         # prompts for R_t + N_t + D_t labeling
│   └── response_generation.txt # conditioned NPC response prompt
│
├── src/
│   ├── data_gen/
│   │   ├── scenario_bank.py    # load/sample scenario templates
│   │   ├── state_init.py       # programmatic NPC profile builder
│   │   ├── episode_planner.py  # social arc planner (phase schedule + required shifts)
│   │   ├── turn_generator.py   # per-turn teacher pipeline (10-step sequence)
│   │   ├── labeler.py          # structured label extraction from teacher output
│   │   ├── counterfactual.py   # counterfactual episode augmenter
│   │   └── validator.py        # consistency checks and filter rules
│   │
│   ├── packaging/
│   │   ├── packager.py         # assemble validated turns → JSONL artifacts
│   │   └── splitter.py         # deterministic train/val/test split
│   │
│   ├── training/
│   │   ├── model.py            # backbone + classification heads definition
│   │   ├── dataset.py          # HuggingFace Dataset wrappers
│   │   ├── train_latent.py     # Stage 1: latent-state predictor training loop
│   │   ├── train_response.py   # Stage 2: response generator SFT
│   │   ├── train_joint.py      # Stage 3: joint multi-task fine-tuning
│   │   └── loss.py             # weighted multi-head loss
│   │
│   ├── eval/
│   │   ├── eval_latent.py      # head accuracy, F1, confusion matrices
│   │   ├── eval_response.py    # BLEU/ROUGE, secret-leakage, contradiction rate
│   │   └── eval_routing.py     # routing precision/recall, latency, memory
│   │
│   └── mlflow_utils.py         # logging helpers, dataset registration, run hierarchy
│
├── scripts/
│   ├── run_data_gen.sh
│   ├── run_train.sh
│   └── run_eval.sh
│
└── requirements.txt
```

---

# Part 1: Data Generation Pipeline

## Overview

The data generation pipeline runs in **5 sequential stages** via a teacher LLM
(GPT-4o or equivalent large model). Target: **20k–50k validated per-turn records**
for training, with counterfactuals adding 2–5× density on key scenarios.

```
scenario_bank → state_init → episode_planner → turn_generator → counterfactual_augmenter
                                                      ↓
                                               validator
                                                      ↓
                                              packager → splits
```

---

## Stage A: Scenario Bank (`src/data_gen/scenario_bank.py`)

Define template-driven scenario types as YAML. Template controls what the generator
can produce, preventing open-ended drift.

**Scenario template fields:**

```yaml
# data/scenario_bank/secret_extraction.yaml
- scenario_id: se_001
  scenario_type: secret_extraction
  setting: castle_vault
  stakes: high                        # low / medium / high
  turn_budget: 8
  required_events:
    - probe
    - pressure
    - deflect
  success_condition: no_full_secret_reveal
  allowed_reveal_ceiling: hint
  npc_role_pool: [castle_guard, steward, chamberlain]

- scenario_id: ap_001
  scenario_type: apology_repair
  setting: market_square
  stakes: medium
  turn_budget: 6
  required_events: [conflict, apology, stance_repair]
  success_condition: trust_delta_positive
```

**Priority scenario types to cover (minimum 5 templates each):**

| Type | Why |
|------|-----|
| `secret_extraction` | Forces secrecy pressure, reveal/withhold decisions |
| `apology_repair` | Tests repair strategy, stance recovery |
| `alliance_negotiation` | Tests obligation + face dimensions |
| `rumor_confrontation` | Tests player_knowledge handling, credibility |
| `threat_escalation` | Tests arousal + threat appraisal + policy shift |
| `trust_building` | Tests slow stance accumulation |
| `deception_detection` | Tests ToM / player_intent inference |

**Implementation tasks:**

- [ ] Write 35+ scenario YAML templates (5 per type minimum)
- [ ] `ScenarioBank.sample(n, scenario_type=None)` — uniform or weighted sampling
- [ ] Track scenario distribution in MLflow as dataset metadata

---

## Stage B: State Initialization (`src/data_gen/state_init.py`)

Programmatically generate `W` (persistent NPC state) from scenario + role constraints.
**Must be rule-constrained, not free-form LLM generation.**

```python
def build_npc_profile(role: str, scenario: dict, rng: random.Random) -> dict:
    """
    Returns a validated W dict.
    Secrets are drawn from a role-appropriate pool.
    Initial stance is set by relationship_prior[role][stakes].
    """
```

**NPC profile constraints:**
- Each role has a fixed `values` pool (e.g., guard → [duty, loyalty, self_preservation])
- Secret severity must match scenario stakes
- Initial stance levels must match `relationship_prior` table (e.g., strangers start at `familiarity=VL`)
- Persona style drawn from a per-role style vocabulary

**Implementation tasks:**

- [ ] Define `ROLE_PROFILES` dict covering 10+ NPC roles
- [ ] Define `RELATIONSHIP_PRIOR` table: role × stakes → initial stance vector
- [ ] `validate_npc_profile(profile)` — schema check + constraint check
- [ ] Serialize profiles to `data/npc_profiles/`

---

## Stage C: Episode Planner (`src/data_gen/episode_planner.py`)

Before any dialogue generation, produce a **hidden social arc** that controls what
must happen across turns. This prevents the generator from wandering.

```python
def plan_episode(scenario: dict, npc_profile: dict) -> dict:
    """
    Returns a phase schedule + required state shifts + reveal ceiling.
    """
```

**Arc structure:**

```json
{
  "phases": [
    {"turns": [1, 2], "phase": "approach"},
    {"turns": [3, 5], "phase": "probing"},
    {"turns": [6, 7], "phase": "pressure"},
    {"turns": [8, 8], "phase": "resolution"}
  ],
  "required_shifts": [
    {"turn": 4, "var": "trust", "delta": "-"},
    {"turn": 6, "var": "secrecy_pressure", "delta": "+"}
  ],
  "allowed_reveal": "hint_only",
  "target_outcome": "no_full_secret_reveal"
}
```

**Implementation tasks:**

- [ ] `plan_episode()` generates arc from scenario template + NPC profile
- [ ] Arc validator: verify required_shifts are reachable given initial stance
- [ ] Arc is injected into the turn generator as a constraint, not shown to player

---

## Stage D: Turn Generator (`src/data_gen/turn_generator.py`)

The core of data generation. At each turn, run this **10-step sequence** using a
teacher LLM with structured prompts:

```
Step 1:  Read scene + W + arc + prior turn state
Step 2:  Sample or generate player utterance x_t
Step 3:  Label C_t  (prompt: label_C.txt)
Step 4:  Infer A_t  (prompt: label_A_M.txt, section A)
Step 5:  Infer M_t  (prompt: label_A_M.txt, section M)
Step 6:  Update R_t (prompt: label_R_N_D.txt, section R)
Step 7:  Infer N_t  (prompt: label_R_N_D.txt, section N)
Step 8:  Choose D_t (prompt: label_R_N_D.txt, section D)
Step 9:  Generate NPC response y_t  (prompt: response_generation.txt)
Step 10: Validate + persist turn record
```

**Per-turn output record (full trace JSONL):**

```json
{
  "episode_id": "ep_0042",
  "turn_idx": 3,
  "scenario_type": "secret_extraction",
  "W": { "role": "castle_guard", "secrets": [...], "initial_stance": {...} },
  "arc_phase": "probing",
  "input": "I heard you stole the chalice. Why hide it?",
  "dialogue_history": [...],
  "C_t": { "dialogue_act": ["accuse", "probe"], "tone": "confrontational", "risk_type": "secret-risk" },
  "A_t": { "valence": "negative", "arousal": "medium", "threat": "high", "control": "medium" },
  "M_t": { "player_intent": "trap", "player_knowledge": "partial", "player_credibility": "medium" },
  "R_t": {
    "affection":   { "level": "L",  "delta": "-" },
    "respect":     { "level": "L",  "delta": "-" },
    "dominance":   { "level": "H",  "delta": "+" },
    "familiarity": { "level": "N",  "delta": "0" },
    "trust":       { "level": "L",  "delta": "-" },
    "obligation":  { "level": "L",  "delta": "0" }
  },
  "N_t": { "duty_pressure": "high", "secrecy_pressure": "high", "face_pressure": "high", "value_conflict": "mild" },
  "D_t": { "response_policy": "challenge", "reveal_decision": "none", "repair_strategy": "none" },
  "response": "You accuse me without proof. Watch your tone if you want answers."
}
```

**Prompt engineering notes:**
- Each labeled section uses structured output (JSON mode / constrained decoding)
- Feed prior R_t as context so stance updates are incremental, not reset each turn
- Arc phase and required_shifts are injected as system constraints, not user-visible
- Player utterances: 50% sampled from a player-move vocabulary per scenario type,
  50% generated by teacher to ensure diversity

**Implementation tasks:**

- [ ] `TurnGenerator.generate_episode(scenario, npc_profile, arc)` → list of turn records
- [ ] Implement retry logic for malformed JSON (max 3 retries per step)
- [ ] Implement arc-adherence enforcement: if required_shift not triggered by target turn,
  inject a forcing nudge in the next player utterance
- [ ] Persist raw output to `data/raw_episodes/` before validation

---

## Stage E: Counterfactual Augmenter (`src/data_gen/counterfactual.py`)

For every validated episode, generate **2–5 counterfactual variants** by holding
the episode arc constant and flipping exactly one variable.

**Target counterfactual dimensions:**

| Variable to flip | Low → High |
|-----------------|------------|
| `player_credibility` | low → high |
| `tone` | warm → confrontational |
| `secrecy_pressure` | low → high |
| `player_knowledge` | unaware → knows-secret |
| `value_conflict` | none → strong |

**Implementation:**

```python
def generate_counterfactual(episode: list[dict], flip_var: str, flip_to: str) -> list[dict]:
    """
    Re-runs the response generation step only (steps 8–9) with the specified
    variable overridden. All other labels are preserved.
    Returns a new episode list with modified D_t and response for each affected turn.
    """
```

- Only regenerate `D_t` and `response` when the flipped variable would change policy
- Label the counterfactual with `{"counterfactual": true, "flip_var": "...", "flip_to": "..."}`
- This is the **cheapest** high-value augmentation: reuses all expensive labeling steps

**Implementation tasks:**

- [ ] `CounterfactualAugmenter.augment(episode, n_variants=3)` — randomly sample flip dimensions
- [ ] Skip re-generation if predicted policy change is below threshold (avoid noise)

---

## Validation (`src/data_gen/validator.py`)

All records must pass these checks before entering `validated_turns/`:

```python
VALIDATION_RULES = [
    # Schema compliance
    "all label values in LABEL_SCHEMA allowed sets",
    "R_t delta is consistent with R_t level change vs prior turn",

    # Consistency checks
    "response_policy=answer implies reveal_decision!=none",
    "secrecy_pressure=high AND reveal_decision=full → flag as violation",
    "trust_level=VL AND response_policy=answer → flag as suspicious",

    # Arc adherence
    "required_shifts triggered by required turn",
    "allowed_reveal_ceiling not exceeded",

    # Coherence
    "response length in [10, 150] tokens",
    "no NPC response contains player's secret verbatim when reveal_decision=none",
]
```

**Implementation tasks:**

- [ ] `Validator.validate_turn(turn_record, prior_R_t, arc)` → (bool, list[str] errors)
- [ ] `Validator.validate_episode(episode)` → episode-level arc check
- [ ] Log rejection rate and rejection reason distribution to MLflow

---

## Dataset Packaging (`src/packaging/packager.py`)

After validation, produce **three aligned JSONL artifacts** from `validated_turns/`:

### Artifact 1: Full trace JSONL
One line per turn, all fields (used for analysis and debugging).

### Artifact 2: Head-supervision JSONL (for latent-state predictor training)

```json
{
  "context": "<scene>\nNPC: castle_guard\nGoals: ...\nSecrets: ...\nPrior stance: ...\n\n[dialogue history]\nPlayer: I heard you stole the chalice.",
  "labels": {
    "dialogue_act": ["accuse", "probe"],
    "tone": "confrontational",
    "valence": "negative",
    "threat": "high",
    "player_intent": "trap",
    "player_knowledge": "partial",
    "player_credibility": "medium",
    "trust_level": "L",
    "trust_delta": "-",
    "secrecy_pressure": "high",
    "response_policy": "challenge",
    "reveal_decision": "none"
  }
}
```

### Artifact 3: SFT JSONL (for response generator training)

```json
{
  "input": {
    "player_utterance": "I heard you stole the chalice. Why hide it?",
    "C_t": {"dialogue_act": ["accuse","probe"], "tone": "confrontational", "risk_type": "secret-risk"},
    "A_t": {"valence": "negative", "arousal": "medium", "threat": "high", "control": "medium"},
    "M_t": {"player_intent": "trap", "player_knowledge": "partial", "player_credibility": "medium"},
    "R_t": {"trust": {"level": "L", "delta": "-"}, "dominance": {"level": "H", "delta": "+"}},
    "N_t": {"secrecy_pressure": "high", "face_pressure": "high"},
    "D_t": {"response_policy": "challenge", "reveal_decision": "none", "repair_strategy": "none"}
  },
  "target": "You accuse me without proof. Watch your tone if you want answers."
}
```

**Splits:** Deterministic 80/10/10 by episode (not by turn, to prevent leakage).

**Target dataset sizes:**

| Split | Turns | Episodes |
|-------|-------|----------|
| Train | ~40k  | ~5k      |
| Val   | ~5k   | ~625     |
| Test  | ~5k   | ~625     |

**Implementation tasks:**

- [ ] `Packager.build_all(validated_turns_dir, output_dir)` → writes 3 artifacts
- [ ] `Splitter.split_by_episode(...)` — episode-level split, stratified by scenario_type
- [ ] Log dataset manifest (counts, split hashes, schema version) to MLflow

---

## Data Generation MLflow Experiment

**Experiment:** `social_state_data_generation`

Each run tracks:
- `generator_version`, `prompt_pack_version`, `schema_version`
- `n_episodes`, `n_turns`, `n_counterfactuals`
- `rejection_rate`, `rejection_reasons_csv`
- `scenario_type_distribution.json`
- `turn_records sample (50 examples)`

---

# Part 2: Model Training Pipeline

## Architecture: Shared Backbone + Classification Heads

```
Input: [context tokens] → Backbone (Qwen3-4B, QLoRA) → pooled representation
                                                              ↓
                          ┌────────────────┬────────────────┴──────────────────┐
                     C_t heads         A_t heads        M_t heads
                  dialogue_act          valence          player_intent
                     tone               arousal          player_knowledge
                    risk_type           threat           player_credibility
                                        control
                          └────────────────┬────────────────┐
                                      R_t heads         N_t + D_t heads
                                  (6 stances × 2)      duty/secrecy/face pressure
                                                        response_policy
                                                        reveal_decision
                                                        repair_strategy
                                              ↓
                                     LM Head (response generation)
```

Each classification head is a 2-layer MLP: `hidden_size → 256 → num_classes`.

---

## Training Stage 1: Latent-State Predictor

**Goal:** Train the model to predict all of C_t, A_t, M_t, R_t, N_t, D_t from dialogue context.

**Model:** `src/training/model.py` — `LatentStatePredictor`

```python
class LatentStatePredictor(nn.Module):
    def __init__(self, backbone, label_schema):
        # QLoRA backbone (frozen base, trainable LoRA adapters)
        # One ClassificationHead per label field
        # heads grouped by schema group (C, A, M, R, N, D)
```

**Input format (text, tokenized):**

```
<scene>
NPC Role: castle_guard
Goals: protect_vault, avoid_scandal
Secrets: chalice_location [high]
Persona: formal, guarded

<prior_stance>
affection=N  respect=L  dominance=M  familiarity=L  trust=L

<history>
[Turn 1] Player: May I see the vault records?
[Turn 1] NPC: No visitors. That is the rule.
[Turn 2] Player: I heard you stole the chalice. Why hide it?

<task>
Predict: dialogue_act, tone, risk_type, valence, arousal, threat, control,
         player_intent, player_knowledge, player_credibility,
         affection_level, affection_delta, ..., response_policy, reveal_decision, repair_strategy
```

**Loss function:**

```python
L_total = (
    λ_C * (L_dialogue_act + L_tone + L_risk) +
    λ_A * (L_valence + L_arousal + L_threat + L_control) +
    λ_M * (L_player_intent + L_player_knowledge + L_player_credibility) +
    λ_R * sum(L_stance_level[d] + L_stance_delta[d] for d in STANCE_DIMS) +
    λ_N * (L_duty + L_secrecy + L_face + L_value_conflict) +
    λ_D * (L_response_policy + L_reveal_decision + L_repair_strategy)
)
```

**Initial loss weights:** λ_C=1.0, λ_A=1.0, λ_M=1.5, λ_R=2.0, λ_N=1.0, λ_D=2.0
(Upweight R_t and D_t as they directly drive behavior; tune after first run.)

**Training config (`configs/train_latent.yaml`):**

```yaml
base_model: Qwen/Qwen3-4B
quantization: 4bit
lora_r: 16
lora_alpha: 32
lora_target_modules: [q_proj, v_proj, k_proj, o_proj]
lr: 2e-4
epochs: 5
max_seq_len: 1024
batch_size: 4
grad_accum: 8      # effective batch = 32
warmup_ratio: 0.05
weight_decay: 0.01
bf16: true
schema_version: v1
```

**Evaluation metrics (Stage 1):**
- Per-head accuracy and macro-F1
- Stance delta direction accuracy (most important: is the sign right?)
- Response policy top-1 accuracy
- Secret leakage rate: `reveal_decision=full` when `secrecy_pressure=high`
- Confusion matrices for all D_t fields

---

## Training Stage 2: Response Generator

**Goal:** Train the model to generate NPC responses conditioned on dialogue context
**and** the latent state.

**Two sub-stages:**

### 2a: Gold-state conditioning (upper bound)

Use gold C_t…D_t from dataset as input context. This establishes the ceiling.

**Input format:**

```
<scene>...</scene>
<prior_stance>...</prior_stance>
<history>...</history>

<latent_state>
C_t: dialogue_act=["accuse","probe"]  tone=confrontational  risk=secret-risk
A_t: valence=negative  arousal=medium  threat=high  control=medium
M_t: player_intent=trap  player_knowledge=partial  credibility=medium
R_t: trust=L(-) dominance=H(+) familiarity=N(0)
N_t: secrecy=high  face=high  duty=high  conflict=mild
D_t: policy=challenge  reveal=none  repair=none
</latent_state>

Generate NPC response:
```

**Target:** `"You accuse me without proof. Watch your tone if you want answers."`

**Training config (`configs/train_response.yaml`):**

```yaml
base_model: Qwen/Qwen3-4B
quantization: 4bit
lora_r: 32
lora_alpha: 64
lr: 1e-4
epochs: 3
max_seq_len: 2048
batch_size: 2
grad_accum: 16
label_smoothing: 0.05
```

### 2b: Predicted-state conditioning (deployment realism)

Same as 2a but replace gold latent state with Stage 1 model's predictions.
Measures the quality drop from prediction errors — this gap is a key paper result.

**Evaluation metrics (Stage 2):**
- ROUGE-L vs gold response
- BERTScore
- Rule-based behavioral checks:
  - `secrecy maintained`: response does not leak secret when `reveal_decision=none`
  - `tone match`: detected tone of response matches D_t policy (e.g., challenge → confrontational)
  - `contradiction rate`: response contradicts prior NPC turn on same topic

---

## Training Stage 3: Joint Multi-Task Fine-tuning

After Stages 1 and 2 converge separately, run joint fine-tuning to close the
gold-vs-predicted state gap.

```python
L_joint = (
    L_latent_heads +           # all classification heads
    λ_Y * L_response_NLL +     # LM generation loss
    λ_consistency * L_consistency  # penalize responses inconsistent with predicted D_t
)
```

**Consistency loss:** if `D_t.reveal_decision=none` but the generated response contains
a secret string → add penalty. Implement as a rule-based reward signal.

**Config:** Start from Stage 1 checkpoint + Stage 2 checkpoint as initialization.
Use λ_Y=1.0, λ_consistency=0.5.

---

## Training Stage 4: Selective Router (Phase 2b)

Train a lightweight binary classifier on top of Stage 1 outputs that decides when
to invoke a slow reflective path:

**Slow path triggers:**
- `value_conflict=strong`
- `secrecy_pressure=high` AND `player_knowledge=informed`
- `trust_delta` in `{--, -}` for 2+ consecutive turns (stance instability)
- `response_policy=threaten` or `negotiate` (high-stakes decisions)

**Implementation:** Single linear classifier on the concatenation of D_t and N_t
hidden states. No additional LLM call unless router fires.

---

## MLflow Tracking

### Experiment hierarchy

```
social_state_data_generation/
latent_state_prediction/
  └─ qwen3_4b_schema_v1_dataset_v1 (parent run)
      ├─ latent_heads_train (child)
      ├─ response_sft_gold (child)
      ├─ response_sft_predicted (child)
      └─ offline_eval (child)
response_generation/
routing_and_policy_eval/
```

### What to log per training run

```python
# Parameters
mlflow.log_params({
    "base_model": "Qwen/Qwen3-4B",
    "quantization": "4bit",
    "lora_r": 16,
    "schema_version": "v1",
    "generator_version": "v1",
    "loss_weights": "C=1.0,A=1.0,M=1.5,R=2.0,N=1.0,D=2.0",
    "train_turns": 40000,
    "max_seq_len": 1024,
})

# Metrics (per epoch)
mlflow.log_metrics({
    "val/response_policy_f1": ...,
    "val/stance_delta_accuracy": ...,
    "val/secret_leakage_rate": ...,
    "val/contradiction_rate": ...,
    "val/response_rouge_l": ...,
    "train/loss_total": ...,
    "system/gpu_memory_gb": ...,
})

# Artifacts
mlflow.log_artifact("confusion_matrix_response_policy.png")
mlflow.log_artifact("sample_generations_epoch3.json")
mlflow.log_dict(schema_spec, "schema_v1.json")
mlflow.log_dict(data_manifest, "data_manifest.json")
mlflow.log_input(train_dataset, context="training")
```

---

## Evaluation Plan

### Automated metrics

| Metric | Target | Stage |
|--------|--------|-------|
| Response policy F1 | ≥ 0.75 | Stage 1 |
| Stance delta accuracy | ≥ 0.70 | Stage 1 |
| Secret leakage rate | ≤ 0.05 | Stage 1+2 |
| Contradiction rate | ≤ 0.08 | Stage 2 |
| ROUGE-L (gold state) | baseline | Stage 2a |
| ROUGE-L (predicted state) | ≤ 5pt drop vs gold | Stage 2b |
| Router false positive rate | ≤ 0.15 | Stage 4 |

### Ablation study (Study A: which abstraction matters)

Train 6 variants, all else equal:
1. Response only (no latent state conditioning)
2. Response + D_t policy only
3. Response + R_t stance only
4. Response + A_t affect only
5. Response + C_t + A_t + M_t + R_t (no N_t)
6. Full model (all heads)

Report: human believability rating + automated metrics per variant.

---

## Week-by-Week Schedule

| Week | Work |
|------|------|
| 1 | Finalize `schema_v1.json`; write scenario templates (35+); implement `state_init.py` |
| 2 | Implement `episode_planner.py`, `turn_generator.py`; write all 5 prompt files |
| 3 | Run generator on 200 seed episodes; manual spot-check; implement `validator.py` |
| 4 | Scale to 1000 episodes; implement `counterfactual.py`; run packaging; log to MLflow |
| 5 | Scale to 5000 episodes; train Stage 1 (Qwen3-1.7B first for speed) |
| 6 | Evaluate Stage 1 heads; tune loss weights; switch to Qwen3-4B |
| 7 | Train Stage 2a (gold state); evaluate behavioral checks |
| 8 | Train Stage 2b (predicted state); measure gold-vs-predicted gap |
| 9 | Stage 3: joint fine-tuning; consistency loss experiments |
| 10 | Stage 4: router; edge benchmarks (latency, memory) |
| 11 | Ablation study (Study A); human evaluation package |
| 12 | Final model registration; paper writing support |

---

## Implementation Priorities (Start Here)

1. **`configs/schema_v1.json`** — export `LABEL_SCHEMA` to JSON; this is the contract all other code reads
2. **`data/scenario_bank/*.yaml`** — 35+ templates; gates all data generation
3. **`src/data_gen/state_init.py`** + **`validator.py`** — can be tested without a teacher LLM
4. **`src/data_gen/turn_generator.py`** — first integration test with 10 episodes
5. **`src/training/model.py`** — backbone + heads; test forward pass on dummy data
6. **`src/mlflow_utils.py`** — set up experiment hierarchy before any real runs

---

## Key Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Teacher LLM produces invalid JSON | Retry logic (3 attempts) + constrained decoding |
| Stance deltas inconsistent across turns | Arc planner enforces required shifts; validator catches violations |
| Secret leakage in training data | Validation rule: flag `reveal=full` when `secrecy=high` |
| 12GB VRAM limit with Qwen3-4B | 4-bit QLoRA + grad_accum=8–16 + seq_len=1024 |
| Gold-vs-predicted state gap too large | Counterfactual augmentation + Stage 3 joint loss |
| Scenario diversity too low | Weighted sampling from scenario bank to ensure all 7 types covered |
