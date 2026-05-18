# Technical Paper: Social-State Models for NPC Dialogue

## Abstract

This project studies dialogue models that separate social reasoning from surface realization. Instead of training a model only to predict the next token, the system supervises an explicit per-turn latent state and then conditions response generation on that state. The goal is believable non-player character dialogue in which relationship stance, affect, secrecy, obligations, and response policy remain coherent across multi-turn interactions.

The repository contains two complementary implementations. The first fine-tunes Qwen3 causal language models with 29 social-state classification heads and a staged response-generation pipeline. The second trains small language models from scratch, including recurrent, transformer, mixture-of-experts, prefix-conditioned, and state-space variants. Both systems share the same research target: make dialogue behavior inspectable, measurable, and controllable through structured intermediate variables.

## 1. Problem Setting

NPC dialogue has three failure modes that pure next-token imitation often hides until runtime: social inconsistency, unsafe disclosure, and weak long-horizon relationship tracking. A character may reveal a secret to a stranger, apologize without preserving status dynamics, or abruptly change tone because the training objective only rewards local fluency.

This project models each player turn as an input $x_t$, dialogue history $H_{t-1}$, persistent world/persona state $W$, structured social state $Z_t$, and NPC response $y_t$. The central modeling assumption is:

$$p(y_t \mid H_{t-1}, x_t, W) \approx p(y_t \mid H_{t-1}, x_t, W, Z_t) \cdot p(Z_t \mid H_{t-1}, x_t, W).$$

The structured state is not intended as a psychological claim. It is an engineering interface for measuring and controlling behavior.

## 2. Data Model

The latent state is:

$$Z_t = \{C_t, A_t, M_t, R_t, N_t, D_t\}.$$

`W` is persistent world/persona state and is conditioned on rather than predicted. The six predicted components are:

| Component | Role | Fields |
|---|---|---|
| $C_t$ Context | Interprets the player utterance | `dialogue_act`, `tone`, `risk_type` |
| $A_t$ Affect | Appraises emotional state | `valence`, `arousal`, `threat`, `control` |
| $M_t$ Mental model | Tracks inferred player state | `player_intent`, `player_knowledge`, `player_credibility` |
| $R_t$ Stance | Tracks relationship dynamics | level and delta for affection, respect, dominance, familiarity, trust, obligation |
| $N_t$ Norms | Tracks obligations and conflict | `duty_pressure`, `secrecy_pressure`, `face_pressure`, `value_conflict` |
| $D_t$ Policy | Selects response strategy | `response_policy`, `reveal_decision`, `repair_strategy` |

There are 29 classification targets per turn. `dialogue_act` is multi-label; all other heads are single-label.

## 3. Data Generation Pipeline

The LLM fine-tuning pipeline generates synthetic episodes from scenario templates:

```text
ScenarioBank -> StateInit -> EpisodePlanner -> TurnGenerator
    -> Labeler(C, A/M, R/N/D) -> Validator
    -> CounterfactualAugmenter -> Packager -> Splitter
```

The current checked-in repository contains the scenario bank and world context, plus compact evaluation outputs under `eval_results/`. Generated packaged/split JSONL artifacts and model checkpoints are not present in this checkout. The scenario bank has seven scenario families and 35 concrete templates:

| Scenario family | Templates | Core dynamics |
|---|---:|---|
| `secret_extraction` | 5 | trust, secrecy, disclosure |
| `apology_repair` | 5 | guilt, face-saving, forgiveness |
| `alliance_negotiation` | 5 | reciprocity, leverage, deception |
| `rumor_confrontation` | 5 | reputation, credibility, anger |
| `threat_escalation` | 5 | fear, dominance, de-escalation |
| `trust_building` | 5 | warmth, self-disclosure, bonding |
| `deception_detection` | 5 | intent inference, lie detection |

Teacher configurations support local Hugging Face models, Azure/OpenAI-compatible endpoints, and OpenRouter-style endpoints. The active default in `llm_finetuning/configs/data_gen.yaml` is `google/gemma-4-4b-it` with 4-bit loading. API generation in `data_gen_api.yaml` targets `gpt-5.4-mini` through an Azure-compatible endpoint. A lightweight local Qwen3-0.6B config is available for smaller runs.

## 4. LLM Fine-Tuning Architecture

The LLM path uses a Qwen3 causal LM backbone with LoRA/QLoRA and a `LatentStatePredictor` wrapper. The predictor pools the final hidden states and routes the pooled vector into independent classification heads:

$$\text{head}_i(h) = W_2^{(i)}\text{Dropout}(\text{GELU}(W_1^{(i)}h)).$$

Supported pooling modes are `last`, `mean`, and `attention`; `train_latent.yaml` currently uses `last`.

Backbone configurations documented by the registry:

| Model | Layers | Hidden | Attention heads | FFN |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 28 | 896 | 14 | 2432 |
| Qwen3-1.7B | 28 | 2048 | 16 | 5504 |
| Qwen3-4B | 36 | 2560 | 32 | 6912 |

The generated registry documents Qwen3-4B as the production configuration and Qwen3-0.6B as the CPU/debug option. The checked-in manuscript and reassessment describe the current reported experimental checkpoints as Qwen3-1.7B runs. Treat model scale as an artifact-backed claim: cite the specific run summary or adapter metadata when reporting a final number.

## 5. LLM Training Objective

Training is staged:

| Stage | Config | Objective | Base model |
|---|---|---|---|
| Stage 1 | `train_latent.yaml` | Predict 29 latent heads from context | Qwen/Qwen3-4B |
| Stage 2 | `train_response.yaml` | Supervised response generation conditioned on gold latent state | Qwen/Qwen3-4B |
| Stage 3 | `train_joint.yaml` | Joint response and latent optimization with consistency loss | Qwen/Qwen3-1.7B |

Stage 1 uses weighted cross-entropy for single-label heads and BCE-with-logits for multi-label dialogue acts. The configured group weights are:

| Group | Weight |
|---|---:|
| $C_t$ | 1.0 |
| $A_t$ | 1.0 |
| $M_t$ | 1.5 |
| $R_t$ | 2.0 |
| $N_t$ | 1.0 |
| $D_t$ | 2.0 |

Joint training adds response loss with weight 1.0 and consistency loss with weight 0.5.

## 6. Small Language Model Architectures

The SLM path benchmarks compact models trained from scratch or lightly fine-tuned for dialogue. It also includes DistilBERT-based encoders for personality and affect, whose outputs can condition response generation.

Exact parameter counts are generated by `scripts/gen_model_registry.py`:

| Architecture | m1_small | rtx4070_small | Main idea |
|---|---:|---:|---|
| `gru` | 42.9M | 94.5M | Multi-layer GRU baseline with tied output projection when dimensions permit |
| `awdlstm` | 42.3M | 106.2M | LSTM with locked dropout, embedding dropout, and DropConnect |
| `gpt` | 16.1M | 51.2M | Decoder-only transformer with causal attention |
| `prefix_gpt` | 16.6M | 53.3M | GPT with learned soft-prefix conditioning from OCEAN and VAD vectors |
| `moe` | 22.4M | 168.8M | Transformer with sparse top-k expert routing in feed-forward blocks |
| `mamba_like` | 15.4M | 45.4M | Pure-PyTorch state-space-style sequence model |

The SLM config `small_lm.yaml` defaults to `gpt` with the `m1_small` profile, sequence length 256, effective batch size 64, learning rate `3e-4`, and three epochs.

## 7. Inference Architecture

The intended inference path is:

```text
Player utterance + history + world/persona state
    -> latent-state predictor
    -> structured state Z_t
    -> policy-aware response generator
    -> NPC response + updated tracked state
```

For the LLM path, `llm_finetuning/src/inference/interactive.py` provides interactive inference around trained checkpoints. For the SLM path, `slm_training/src/infer/` contains chat/demo/service modules and an optional memory store. The SLM service layer is useful for fast iteration and architecture comparison; the LLM path is the higher-fidelity structured-state target.

## 8. Evaluation

Evaluation is split by function. The current checked-in artifacts are intentionally modest: `latent_eval_metrics.json`, `response_eval_metrics.json`, `routing_eval_metrics.json`, sample generations, and latent-head confusion-matrix PNGs.

| Area | Metrics |
|---|---|
| Latent state | per-head accuracy/F1, response-policy F1, stance-delta accuracy |
| Safety/policy | gated and ungated secret leakage rates, contradiction rate, router false-positive rate |
| Response quality | ROUGE-L with confidence interval, BLEU-1/2/4, distinct-n, repetition rate, length ratio, sample generations, keyword leakage, contradiction checks |
| SLM language modeling | validation/test perplexity, BLEU-1/2, Distinct-1/2 |
| Personality encoder | MSE and R-squared per OCEAN trait |
| Affect encoder | CCC, MSE, MAE, and R-squared per VAD dimension |

Configured LLM thresholds are response-policy F1 >= 0.75, stance-delta accuracy >= 0.70, secret leakage <= 0.05, contradiction <= 0.08, ROUGE-L drop <= 0.05, and router false positive rate <= 0.15.

The updated evaluator writes BLEU, bootstrap ROUGE-L confidence intervals, repetition and degeneration rates, prompt-artifact rate, generation/reference length ratio, and separate gated/ungated leakage denominators. These fields still need to be regenerated on the remote checkpoints before paper tables should cite them. Routing has two modes: gold-state routing is a deterministic sanity check, while predicted-state routing consumes `predicted_zt.jsonl` from latent evaluation and is the reportable generalization test. See `research_status.md` for the evidence ledger and claim guardrails.

## 9. Reproducibility Notes

The canonical generated reference is `model_registry.md`, produced by:

```bash
uv run python scripts/gen_model_registry.py
uv run python scripts/gen_model_registry.py --check
```

The registry is generated from configs, code, and local data files. Training and evaluation runs also emit structured JSON and Markdown metric bundles automatically, so paper tables can be derived from run artifacts instead of hand-maintained notes. If packaged JSONL or SLM training data are absent from the checkout, the data sections intentionally show unknown or empty counts rather than preserving stale numbers.

## 10. Research Contribution

The core contribution is an inspectable dialogue-modeling stack where social behavior is represented by explicit intermediate variables, not only hidden activations. This makes the system easier to audit, ablate, and control:

- The data model exposes what the NPC believes, feels, and intends to do before it speaks.
- The training design allows latent prediction, response generation, and joint training to be evaluated separately.
- The SLM benchmark suite gives low-cost architecture comparisons for researchers who do not need a full pre-trained LLM fine-tuning loop.
- The generated registry keeps architecture, hyperparameter, and data references synchronized with code.
