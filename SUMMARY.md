# Project Summary: LLM Social State Training

> **Generated:** 2026-03-27 | **Status:** Active development — Stage 1 training in progress

---

## 1. Project Goal

Train a compact (0.6B–4B parameter), edge-deployable dialogue model that:
1. Predicts a **structured social latent state** (Z_t) at every conversational turn
2. Generates **believable NPC dialogue** conditioned on that state
3. Maintains **auditable behavior** (secret-keeping, stance consistency, norm compliance)

The project evolves from a prior multi-agent prompting system into a single shared-backbone model with explicit classification heads — shifting from orchestration to distillation.

---

## 2. Repository Structure

### Actual file tree (verified)

```
llm_training/
├── configs/                          # 10 YAML/JSON config files
│   ├── schema_v1.json                # Canonical label schema (29 classification targets)
│   ├── data_gen.yaml                 # Main data gen config (local Qwen3-8B teacher)
│   ├── data_gen_api.yaml             # API-based teachers (Azure, OpenRouter)
│   ├── data_gen_qwen3_small.yaml     # Lightweight Qwen3-0.6B teacher config
│   ├── train_latent.yaml             # Stage 1: latent predictor (Qwen3-0.6B + LoRA)
│   ├── train_response.yaml           # Stage 2: response generator (Qwen3-4B + QLoRA 4bit)
│   ├── train_response_cpu.yaml       # CPU fallback for Stage 2
│   ├── train_joint.yaml              # Stage 3: joint fine-tuning (Qwen3-4B + QLoRA 4bit)
│   ├── train_joint_cpu.yaml          # CPU fallback for Stage 3
│   └── eval.yaml                     # Evaluation thresholds and checkpoint paths
│
├── data/
│   ├── scenario_bank/                # 7 YAML files, 5 templates each = 35 scenarios
│   │   ├── secret_extraction.yaml
│   │   ├── apology_repair.yaml
│   │   ├── alliance_negotiation.yaml
│   │   ├── rumor_confrontation.yaml
│   │   ├── threat_escalation.yaml
│   │   ├── trust_building.yaml
│   │   └── deception_detection.yaml
│   ├── world_contexts/
│   │   └── oakhaven_siege.yaml       # Single world context (medieval siege setting)
│   ├── npc_profiles/                 # (empty — profiles generated at runtime by state_init.py)
│   ├── raw_episodes/                 # 449 episode JSONL files
│   ├── validated_turns/              # 429 validated episode files
│   ├── counterfactuals/              # 915 counterfactual variant files
│   ├── merged_validated/             # Intermediate merge directory
│   ├── packaged/                     # 3 aligned JSONL artifacts (2,734 records each)
│   │   ├── full_trace.jsonl
│   │   ├── head_supervision.jsonl
│   │   └── sft.jsonl
│   └── splits/                       # 80/10/10 episode-level split (9 files)
│       ├── train_heads.jsonl         # 2,110 records
│       ├── train_sft.jsonl           # 2,110 records
│       ├── train_trace.jsonl         # 2,110 records
│       ├── val_heads.jsonl           #   263 records
│       ├── val_sft.jsonl             #   263 records
│       ├── val_trace.jsonl           #   263 records
│       ├── test_heads.jsonl          #   361 records
│       ├── test_sft.jsonl            #   361 records
│       └── test_trace.jsonl          #   361 records
│
├── prompts/                          # Teacher LLM prompt templates
│   ├── scenario_instantiation.txt    # Scene opening generation
│   ├── label_C.txt                   # C_t: dialogue act, tone, risk
│   ├── label_A_M.txt                 # A_t + M_t: affect/appraisal + player model
│   ├── label_R_N_D.txt              # R_t + N_t + D_t: stance, norms, policy
│   ├── response_generation.txt       # NPC response conditioned on D_t
│   ├── player_generation.txt         # Synthetic player utterance generation
│   └── templates/                    # Model-specific response templates
│       ├── qwen_response_generation.txt
│       ├── openai_response_generation.txt
│       ├── gemini_response_generation.txt
│       └── mistral_response_generation.txt
│
├── src/
│   ├── data_gen/                     # Data generation pipeline (10 files)
│   │   ├── scenario_bank.py          # Load/sample scenario templates
│   │   ├── state_init.py             # Rule-based NPC profile builder (W)
│   │   ├── episode_planner.py        # Social arc planner (phases + required shifts)
│   │   ├── turn_generator.py         # Per-turn 10-step teacher pipeline
│   │   ├── teacher.py                # Teacher LLM client factory (local_hf, azure, openrouter)
│   │   ├── labeler.py                # Structured label extraction + response cleaning
│   │   ├── counterfactual.py         # Counterfactual episode augmenter
│   │   ├── validator.py              # Schema/consistency/arc validation
│   │   └── world_context.py          # World context loader
│   │
│   ├── packaging/                    # Post-generation packaging (3 files)
│   │   ├── packager.py               # Validated turns → 3 JSONL artifacts
│   │   └── splitter.py               # Episode-level stratified train/val/test split
│   │
│   ├── training/                     # Model training (7 files)
│   │   ├── model.py                  # LatentStatePredictor: backbone + 29 classification heads
│   │   ├── dataset.py                # HeadDataset, SFTDataset, JointDataset + collators
│   │   ├── loss.py                   # Weighted multi-head loss function
│   │   ├── train_latent.py           # Stage 1: latent predictor training loop
│   │   ├── train_response.py         # Stage 2: response generator SFT
│   │   └── train_joint.py            # Stage 3: joint multi-task fine-tuning
│   │
│   ├── eval/                         # Evaluation (4 files)
│   │   ├── eval_latent.py            # Per-head accuracy, F1, confusion matrices
│   │   ├── eval_response.py          # ROUGE-L, BERTScore, secret leakage, contradiction
│   │   └── eval_routing.py           # Router precision/recall, latency, memory
│   │
│   ├── inference/                    # Interactive inference (2 files)
│   │   ├── interactive.py            # CLI chat with trained NPC model
│   │   └── README.md                 # Usage instructions
│   │
│   └── mlflow_utils.py               # MLflow logging helpers
│
├── scripts/                          # Shell wrappers
│   ├── run_data_gen.sh
│   ├── run_train.sh
│   └── run_eval.sh
│
├── docs/                             # Documentation
│   ├── education_intro.md            # Non-technical intro (high school / freshman level)
│   └── technical_blog.md             # Technical architecture blog post
│
├── checkpoints/                      # Trained model checkpoints
│   ├── latent_predictor/
│   ├── latent_predictor_best/
│   ├── response_generator/
│   ├── response_generator_best/
│   ├── joint_model/
│   └── joint_model_best/
│
├── eval_results/                     # Evaluation output
│   ├── latent_eval_metrics.json
│   ├── response_eval_metrics.json
│   ├── sample_generations.json
│   └── cm_*.png                      # 24 confusion matrix plots (all 29 heads)
│
├── run_data_gen.py                   # Data generation entry point
├── run_train.py                      # Training entry point (--stage latent|response|joint)
├── run_eval.py                       # Evaluation entry point
├── requirements.txt                  # 39 Python dependencies
├── README.md                         # Setup + usage guide
└── .env                              # API keys (AZURE_API_KEY, AZURE_ENDPOINT, OPENROUTER_API_KEY)
```

### Files documented in README/plan but NOT in the repo

| Documented | Status |
|------------|--------|
| `data/npc_profiles/*.json` | **Empty dir** — profiles are generated at runtime, not persisted |
| `data/splits/test_secret_index.json` | **Missing** — referenced in `eval.yaml` but not generated |
| Stage 4 selective router training code | **Not implemented** — planned for Phase 2b |

### Files in the repo but NOT documented

| File | What it is |
|------|-----------|
| `src/inference/interactive.py` | Interactive CLI chat with trained model — has its own README |
| `src/data_gen/teacher.py` | Teacher LLM client factory (local_hf, azure, openrouter, dashscope) |
| `src/data_gen/world_context.py` | World context YAML loader |
| `docs/education_intro.md` | Non-technical educational intro |
| `docs/technical_blog.md` | Technical architecture blog draft |
| `configs/data_gen_api.yaml` | API-only teacher config (Azure/OpenRouter) |
| `configs/data_gen_qwen3_small.yaml` | Small model teacher config |
| `configs/train_*_cpu.yaml` | CPU fallback training configs |
| `check_gpu.py` | GPU availability diagnostic script |

---

## 3. Latent State Schema (Z_t)

Every turn is annotated with **29 classification targets** across 6 structured components:

| Symbol | Group | Fields | Classes |
|--------|-------|--------|---------|
| **C_t** | Interpretation | `dialogue_act` (multi-label), `tone`, `risk_type` | 10 + 6 + 5 |
| **A_t** | Affect/Appraisal | `valence`, `arousal`, `threat`, `control` | 3 each |
| **M_t** | Player Model (ToM) | `player_intent`, `player_knowledge`, `player_credibility` | 9 + 4 + 3 |
| **R_t** | Relational Stance | 6 dims × (level + delta) | 5 levels + 5 deltas each |
| **N_t** | Norm Constraints | `duty_pressure`, `secrecy_pressure`, `face_pressure`, `value_conflict` | 3 + 3 + 3 + 3 |
| **D_t** | Response Policy | `response_policy`, `reveal_decision`, `repair_strategy` | 10 + 4 + 5 |

**R_t dimensions:** affection, respect, dominance, familiarity, trust, obligation

The persistent state **W** (NPC role, goals, secrets, values, persona) is conditioned on but not predicted per turn.

---

## 4. Data Generation Pipeline

```
ScenarioBank → StateInit → EpisodePlanner → TurnGenerator → CounterfactualAugmenter
                                                   ↓
                                              Validator → Packager → Splitter
```

### 10-step turn generation sequence

| Step | Output | Prompt File |
|------|--------|-------------|
| 1 | Scene context assembled | — |
| 2 | Player utterance x_t | `player_generation.txt` |
| 3 | C_t (dialogue act, tone, risk) | `label_C.txt` |
| 4–5 | A_t + M_t (affect + player model) | `label_A_M.txt` |
| 6–8 | R_t + N_t + D_t (stance, norms, policy) | `label_R_N_D.txt` |
| 9 | NPC response y_t | `response_generation.txt` |
| 10 | Validate and persist | — |

### Teacher providers supported

| Provider | Client | Status |
|----------|--------|--------|
| `local_hf` | HuggingFace transformers (GPU) | Working — used for 449 episodes |
| `azure` | OpenAI SDK (`base_url` mode) | Blocked by content filters on siege scenarios |
| `openrouter` | OpenAI SDK (`base_url` mode) | Available but not actively used |
| `dashscope` | OpenAI SDK (`base_url` mode) | Configured but untested |

### Current data status

| Metric | Count |
|--------|-------|
| Raw episodes generated | 449 |
| Validated episodes | 429 (95.5% pass rate) |
| Counterfactual variant files | 915 |
| Total packaged records | 2,734 |
| Train split | 2,110 records |
| Val split | 263 records |
| Test split | 361 records |
| Scenario types covered | 7/7 |
| Templates per type | 5 |

---

## 5. Training Pipeline

### Three-stage architecture

```
Stage 1: Latent Predictor   →   Stage 2: Response Generator   →   Stage 3: Joint Fine-tuning
(predict Z_t from context)       (generate y_t given Z_t)          (both objectives together)
```

### Stage 1 — Latent State Predictor

| Parameter | Current Config |
|-----------|---------------|
| **Model** | Qwen/Qwen3-0.6B (LoRA, no quantization, float32) |
| **LoRA** | r=16, alpha=32, targets: q/k/v/o_proj + gate/up/down_proj |
| **Training** | 5 epochs, lr=2e-4, cosine schedule, batch=1, grad_accum=32 |
| **Data** | `train_heads.jsonl` (2,110 samples) |
| **Loss** | Weighted multi-head CE: λ_R=2.0, λ_D=2.0, λ_M=1.5, others=1.0 |
| **Best metric** | `val/response_policy_f1` |
| **Status** | **Training in progress** |

### Stage 2 — Response Generator (SFT)

| Parameter | Current Config |
|-----------|---------------|
| **Model** | Qwen/Qwen3-4B (QLoRA 4-bit, bfloat16) |
| **LoRA** | r=32, alpha=64 |
| **Training** | 3 epochs, lr=1e-4, label_smoothing=0.05 |
| **Data** | `train_sft.jsonl` (2,110 samples) |
| **Conditioning** | Gold latent state (then predicted state for deployment realism) |
| **Best metric** | `val/rouge_l` |
| **Status** | Not started — waiting for Stage 1 |

### Stage 3 — Joint Multi-Task Fine-tuning

| Parameter | Current Config |
|-----------|---------------|
| **Model** | Qwen/Qwen3-4B (QLoRA 4-bit) |
| **Init from** | Stage 1 best + Stage 2 best checkpoints |
| **Loss** | L_latent_heads + λ_Y·L_response_NLL + λ_consistency·L_consistency |
| **Status** | Not started — waiting for Stages 1 & 2 |

### Stage 4 — Selective Router (planned, not implemented)

Binary classifier on D_t/N_t hidden states to decide when to invoke a slow reflective path.

---

## 6. Evaluation

### Automated metrics and targets

| Metric | Target | Stage |
|--------|--------|-------|
| Response policy F1 | ≥ 0.75 | Stage 1 |
| Stance delta accuracy | ≥ 0.70 | Stage 1 |
| Secret leakage rate | ≤ 0.05 | Stage 1+2 |
| Contradiction rate | ≤ 0.08 | Stage 2 |
| ROUGE-L gold→predicted drop | ≤ 5pt | Stage 2b |
| Router false positive rate | ≤ 0.15 | Stage 4 |

### Existing evaluation artifacts

- `eval_results/latent_eval_metrics.json` — per-head accuracy and F1
- `eval_results/response_eval_metrics.json` — ROUGE-L, secret leakage, contradiction
- `eval_results/sample_generations.json` — 100 sample NPC responses
- `eval_results/cm_*.png` — 24 confusion matrix plots for all classification heads

---

## 7. Known Issues and TODOs

### From TODOS.md (engineering review)

| ID | Issue | Severity | Status |
|----|-------|----------|--------|
| TODO-1 | Counterfactual flip applies a single midpoint value to ALL turns, erasing arc dynamics | High | Open |
| TODO-2 | JointDataset zips by index with no episode_id alignment check | Medium | Open |
| TODO-3 | Ablate avg-pool vs last-token pooling for classification heads | Low | Open |
| TODO-4 | Secret-leakage validator uses verbatim matching only, misses paraphrases | Medium | Open |

### Issues discovered during current session

| Issue | Detail | Status |
|-------|--------|--------|
| Azure content filters | Siege/war scenarios trigger "self_harm" flags, blocking all Azure generation | Unresolved — Azure unusable for this world context |
| `max_tokens` vs `max_completion_tokens` | Azure gpt-5.4-mini requires `max_completion_tokens` | Fixed in `labeler.py` |
| `base_url` env var resolution | Azure `AZURE_ENDPOINT` not resolved from env | Fixed in `teacher.py` |
| YAML syntax error | Non-comment line in `data_gen_api.yaml` | Fixed |
| Training speed | Stage 1 on Qwen3-0.6B running at ~1.67 it/s (batch=1, grad_accum=32) | Under investigation |

---

## 8. Documentation Cross-Reference

| Document | Purpose | Accuracy vs Repo |
|----------|---------|-----------------|
| `README.md` | Setup, usage, schema overview | **Mostly accurate.** Training section says "Qwen3-4B + QLoRA" for Stage 1, but actual config uses Qwen3-0.6B. |
| `implementation_plan.md` | Full implementation blueprint (821 lines) | **Good reference.** Some values differ from actual configs (e.g., plan says batch_size=4, config has batch_size=1). |
| `schema.md` | Theoretical latent-state schema design | **Accurate.** All 6 components (C, A, M, R, N, D) are implemented as designed. |
| `plan.md` | Phase 2 research plan and paper positioning | **Strategic document.** 12-week roadmap, hypothesis list, paper structure. Still relevant. |
| `project_overview.md` | Technical system overview | **Most accurate and concise.** Matches actual architecture closely. |
| `impl.md` | Initial MLflow + stack recommendations | **Background context.** Informed early design decisions. |
| `progress.md` | Pipeline progress tracker | **Stale.** References PID 1000233 (old), 500-episode target (surpassed). Needs update. |
| `TODOS.md` | Engineering review action items | **Current.** All 4 TODOs still open. |
| `docs/education_intro.md` | Non-technical intro for students | **Supplementary.** Not directly related to implementation. |
| `docs/technical_blog.md` | Technical blog draft | **Draft.** References Qwen3-8B as backbone (actual training uses 0.6B/4B). |

### README discrepancies

| README Says | Actual |
|-------------|--------|
| Stage 1: Qwen3-4B + QLoRA | Config: Qwen3-0.6B + LoRA (no quantization) |
| Stage 2: Qwen3-1.7B + QLoRA | Config: Qwen3-4B + QLoRA 4-bit |
| `OPENAI_API_KEY` for full run | Actually uses `AZURE_API_KEY`/`OPENROUTER_API_KEY` from `.env` |
| 35 scenario templates | Confirmed: 7 types × 5 = 35 |

---

## 9. Dependency Stack

| Category | Packages |
|----------|----------|
| **Core ML** | torch ≥2.2, transformers ≥4.47, peft ≥0.14, bitsandbytes ≥0.43, accelerate ≥0.27, trl ≥0.8.6 |
| **Evaluation** | evaluate, rouge-score, bert-score, scikit-learn, scipy |
| **Tracking** | mlflow ≥2.13 |
| **Data** | pyyaml, jsonlines, tqdm, pandas, tabulate |
| **LLM API** | openai ≥1.30, tiktoken, tenacity |
| **Utilities** | click, python-dotenv, rich, psutil |

---

## 10. Current Status Summary

| Component | Status |
|-----------|--------|
| Scenario bank (35 templates) | Complete |
| World context (Oakhaven Siege) | Complete |
| Data generation pipeline | Complete (all 7 stages functional) |
| Data generated | 449 episodes, 2,734 packaged records |
| Stage 1 training (latent predictor) | **In progress** (Qwen3-0.6B) |
| Stage 2 training (response generator) | Not started |
| Stage 3 training (joint fine-tuning) | Not started |
| Stage 4 (selective router) | Not implemented |
| Evaluation pipeline | Complete (code exists, prior results available) |
| Interactive inference | Complete (CLI chat script exists) |
| MLflow integration | Available but currently disabled (`--no-mlflow`) |

### Data sufficiency assessment

| Training Stage | Data Available | Recommended Minimum | Assessment |
|---------------|---------------|-------------------|------------|
| Stage 1 (Latent Predictor, 0.6B) | 2,110 train samples | 3,000–5,000 | Borderline — may work for proof-of-concept |
| Stage 2 (Response Generator, 4B) | 2,110 train samples | 5,000–10,000 | Insufficient — expect repetitive outputs |
| Stage 3 (Joint, 4B) | 2,110 each | 5,000+ | Too early — depends on Stages 1 & 2 |

### Hardware constraints

- **Single GPU** (12 GB VRAM)
- Cannot run data generation and training simultaneously
- Data generation stopped at 449 episodes to free GPU for training
- Local teacher (Qwen3-8B 4-bit) generated at ~3.8 episodes/hour
- Azure API blocked by content filters on siege world context
