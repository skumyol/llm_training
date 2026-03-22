# LLM Social State Training

Training pipeline for NPC dialogue models with structured latent social state prediction (Z_t).

## Project Structure

```
configs/          — YAML/JSON configs for each pipeline stage
data/
  scenario_bank/  — 35 scenario templates across 7 types
  splits/         — train/val/test JSONL artifacts (after generation)
  packaged/       — full_trace, head_supervision, sft artifacts
prompts/          — teacher LLM prompt templates
src/
  data_gen/       — scenario_bank, state_init, episode_planner,
                    turn_generator, labeler, counterfactual, validator
  packaging/      — packager, splitter
  training/       — model, dataset, loss, train_latent,
                    train_response, train_joint
  eval/           — eval_latent, eval_response, eval_routing
  mlflow_utils.py
run_data_gen.py   — data generation entry point
run_train.py      — training entry point
run_eval.py       — evaluation entry point
scripts/          — shell wrappers for each pipeline stage
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Generation

**Dry-run (no API key needed, mock LLM):**
```bash
./scripts/run_data_gen.sh configs/data_gen.yaml --dry-run 20
# or directly:
python run_data_gen.py --config configs/data_gen.yaml --dry-run --n-episodes 20
```

**Full run (requires OPENAI_API_KEY):**
```bash
export OPENAI_API_KEY=sk-...
./scripts/run_data_gen.sh configs/data_gen.yaml
```

**Stages** (default: `all`): `generate` → `validate` → `package` → `split`

```bash
python run_data_gen.py --config configs/data_gen.yaml --stage generate --dry-run --n-episodes 100
```

Output artifacts under `data/`:
| File | Description |
|------|-------------|
| `splits/train_trace.jsonl` | Full Z_t trace for training |
| `splits/train_heads.jsonl` | Flat label dict for head supervision |
| `splits/train_sft.jsonl`   | Context + latent + response for SFT |

## Training

```bash
# Stage 1 — Latent state predictor (Qwen3-4B + QLoRA)
./scripts/run_train.sh latent

# Stage 2 — Response generator SFT (Qwen3-1.7B + QLoRA)
./scripts/run_train.sh response

# Stage 3 — Joint fine-tuning
./scripts/run_train.sh joint

# All stages sequentially
./scripts/run_train.sh all

# Debug mode (uses smaller model from config)
./scripts/run_train.sh latent --debug
```

## Evaluation

```bash
./scripts/run_eval.sh all configs/eval.yaml
# individual stages:
./scripts/run_eval.sh latent
./scripts/run_eval.sh response
./scripts/run_eval.sh routing
```

Key thresholds:
- `response_policy_f1` >= 0.75
- `stance_delta_accuracy` >= 0.70
- `secret_leakage_rate` <= 0.05
- `contradiction_rate` <= 0.08
- `router_false_positive_rate` <= 0.15

## Latent State Schema (Z_t)

| Component | What it captures |
|-----------|-----------------|
| `W`       | Stable NPC profile (role, goals, secrets, values) |
| `C_t`     | Conversational interpretation (dialogue act, tone, risk) |
| `A_t`     | Affect / appraisal (valence, arousal, threat, control) |
| `M_t`     | Mental model of player (intent, knowledge, credibility) |
| `R_t`     | Relational stance (6 dims x level + delta) |
| `N_t`     | Normative constraints (secrecy, face, duty, conflict) |
| `D_t`     | Discourse policy (response_policy, reveal_decision, repair) |

## Scenario Types

`secret_extraction` · `apology_repair` · `alliance_negotiation` ·
`rumor_confrontation` · `threat_escalation` · `trust_building` · `deception_detection`

## MLflow

```bash
mlflow ui --backend-store-uri mlruns
```