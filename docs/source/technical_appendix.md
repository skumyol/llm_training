# Technical Appendix: Compact Social-State Models for Believable NPC Dialogue

**Supplementary material for paper submission.**

---

## A. Social State Transition Model

### A.1 Formal Definition

We define the NPC latent state at turn $t$ as a 7-component structured vector:

$$Z_t = \{C_t, A_t, M_t, R_t, N_t, D_t, W\}$$

where $W$ is persistent (world/persona state) and the remaining six components update per turn through explicit prediction heads. This replaces monolithic next-token generation with **structured social-state prediction followed by state-conditioned response generation**.

### A.2 Component Taxonomy by Timescale

| Timescale | Components | Update Frequency | Examples |
|-----------|-----------|-----------------|----------|
| **Persistent** | $W$ (world, persona, goals, secrets, values) | Never (conditioned, not predicted) | NPC role, faction, secret ownership |
| **Medium** | $R_t$ (stance), $N_t$ (norms) | Multi-turn, gradual | Trust accumulation, relationship repair |
| **Fast** | $C_t, A_t, M_t, D_t$ | Per turn | Threat appraisal, dialogue act selection |

### A.3 Update Dynamics

The computational graph at each turn follows:

$$C_t = f_C(x_t, H_{t-1}, W)$$
$$A_t = f_A(C_t, R_{t-1}, N_{t-1}, W)$$
$$M_t = f_M(C_t, H_{t-1}, R_{t-1}, W)$$
$$R_t = f_R(R_{t-1}, A_t, M_t, C_t)$$
$$N_t = f_N(N_{t-1}, C_t, W)$$
$$D_t = f_D(A_t, M_t, R_t, N_t, W)$$
$$y_t = f_Y(H_{t-1}, W, D_t)$$

where $x_t$ is the player utterance, $H_{t-1}$ is dialogue history, and $y_t$ is the generated NPC response. In our implementation, all $f$ functions are predictions from classification heads sharing a common transformer backbone, and $f_Y$ is an autoregressive language model conditioned on $D_t$.

---

## B. Full Label Inventory

### B.1 Component C_t — Conversational Interpretation (3 targets)

| Field | Cardinality | Type | Values |
|-------|-----------|------|--------|
| `dialogue_act` | 10 | Multi-label | ask, accuse, threaten, flatter, apologize, negotiate, joke, confess, probe, command |
| `tone` | 6 | Single-label | warm, neutral, confrontational, sarcastic, fearful, evasive |
| `risk_type` | 5 | Single-label | none, secret-risk, face-risk, status-risk, conflict-risk |

**Loss:** BCEWithLogits for `dialogue_act`, weighted cross-entropy for others.

### B.2 Component A_t — Affective Appraisal (4 targets)

Following componential appraisal theory (Scherer, 2001; Smith & Ellsworth, 1985):

| Field | Cardinality | Values |
|-------|-----------|--------|
| `valence` | 3 | negative, neutral, positive |
| `arousal` | 3 | low, medium, high |
| `threat` | 3 | low, medium, high |
| `control` | 3 | low, medium, high |

Rather than predicting a single emotion label, we decompose affect into orthogonal appraisal dimensions. This avoids the instability of free-text emotion prediction and provides more actionable conditioning signals for response generation.

**Loss:** Weighted cross-entropy per head.

### B.3 Component M_t — Player Mental Model (3 targets)

An operationalized Theory of Mind model — not claiming human-like mentalizing, but explicit structured partner modeling:

| Field | Cardinality | Values |
|-------|-----------|--------|
| `player_intent` | 9 | seek-info, trap, bond, manipulate, test, persuade, intimidate, probe, negotiate |
| `player_knowledge` | 4 | unaware, partial, informed, knows-secret |
| `player_credibility` | 3 | low, medium, high |

**Loss:** Weighted cross-entropy.

### B.4 Component R_t — Relational Stance (12 targets)

Six interpersonal dimensions, each with absolute level and turn-level delta:

| Dimension | Level Classes | Delta Classes | Semantics |
|-----------|--------------|---------------|-----------|
| `affection` | VL, L, N, H, VH | --, -, 0, +, ++ | Warmth/liking |
| `respect` | VL, L, N, H, VH | --, -, 0, +, ++ | Status recognition |
| `dominance` | VL, L, N, H, VH | --, -, 0, +, ++ | Power balance |
| `familiarity` | VL, L, N, H, VH | --, -, 0, +, ++ | Acquaintance depth |
| `trust` | VL, L, N, H, VH | --, -, 0, +, ++ | Honesty/reliability |
| `obligation` | VL, L, N, H, VH | --, -, 0, +, ++ | Perceived duty |

The dual level+delta encoding enables the model to predict both the absolute stance state and its rate of change, providing richer supervision than static classification alone.

**Loss:** Cross-entropy per head (24 total stance targets = 12 level + 12 delta).

### B.5 Component N_t — Norm/Value Constraints (4 targets)

| Field | Cardinality | Values |
|-------|-----------|--------|
| `duty_pressure` | 3 | low, medium, high |
| `secrecy_pressure` | 3 | low, medium, high |
| `face_pressure` | 3 | low, medium, high |
| `value_conflict` | 3 | none, mild, strong |

This component encodes the NPC's awareness of role obligations, disclosure constraints, and internal value conflicts. It directly governs secret-keeping, evasion, and repair behavior.

### B.6 Component D_t — Response Policy (3 targets)

The bridge from latent state to utterance generation:

| Field | Cardinality | Values |
|-------|-----------|--------|
| `response_policy` | 10 | answer, partial, withhold, deflect, challenge, soothe, test, threaten, negotiate, clarify |
| `reveal_decision` | 4 | none, hint, partial, full |
| `repair_strategy` | 5 | none, soften, apologize, clarify, redirect |

**Total: 29 classification targets per conversational turn.**

---

## C. Model Architecture

### C.1 LatentStatePredictor (LLM Fine-Tuning)

**Backbone:** Qwen3 family (0.6B, 1.7B, or 4B parameters) with Low-Rank Adaptation.

**Quantization:** 4-bit QLoRA (nf4 quantization type, double quantization, bfloat16 compute).

**Pooling:** Last-token hidden state extraction from the transformer backbone (alternatives: mean pooling, learnable attention pooling).

**Classification Heads:** Each of the 29 targets maps through:

$$\text{head}_i(x) = W_2^{(i)} \cdot \text{GELU}(W_1^{(i)} \cdot \text{Dropout}(x))$$

where $W_1^{(i)} \in \mathbb{R}^{256 \times h}$, $W_2^{(i)} \in \mathbb{R}^{c_i \times 256}$, $h$ is the backbone hidden size, and $c_i$ is the number of classes for head $i$.

**Parameter counts:**

| Model | Backbone | LoRA Params | Head Params | Total Trainable |
|-------|----------|-------------|-------------|-----------------|
| Qwen3-0.6B | 595M | ~1M (r=16) | ~700K | ~1.7M |
| Qwen3-4B | 3.9B | ~4M (r=16) | ~700K | ~4.7M |

### C.2 Small Language Models (SLM Training)

Six architectures trained from scratch for A/B comparison:

#### GRU (SmallGRULM)
$$\text{GRU}(x) = \text{GRU}(\text{Dropout}(\text{Embed}(x)))$$

| Config | m1_small | rtx4070_small |
|--------|----------|---------------|
| embed_dim | 256 | 512 |
| hidden_size | 512 | 1024 |
| num_layers | 3 | 3 |
| Parameters | ~12M | ~40M |

#### AWD-LSTM (AWDLSTMLM)
Implements three regularization techniques (Merity et al., 2018):

- **LockedDropout:** Variational dropout with shared mask across timesteps:
  $$\text{mask} \sim \text{Bernoulli}(1-p)^{B \times 1 \times D}$$

- **DropConnect:** Bernoulli dropout applied to LSTM hidden-to-hidden weight matrices, re-sampled each forward pass.

- **Embedding dropout:** Applied before LSTM input.

| Config | m1_small | rtx4070_small |
|--------|----------|---------------|
| embed_dim | 256 | 400 |
| hidden_size | 512 | 1150 |
| num_layers | 2 | 3 |
| wdrop | 0.5 | 0.5 |
| Parameters | ~10M | ~28M |

#### GPT (TinyGPTLM)
Standard decoder-only transformer with causal self-attention:

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

where $M$ is a causal mask. FFN expansion: 4× with GELU activation.

| Config | m1_small | rtx4070_small |
|--------|----------|---------------|
| n_embd | 256 | 512 |
| n_head | 4 | 8 |
| n_layer | 4 | 8 |
| Parameters | ~8M | ~55M |

#### Prefix GPT (PrefixTinyGPTLM)
Extends TinyGPT with conditioned soft-prefix injection:

$$\text{prefix} = \text{Linear}(\text{Tanh}(\text{Linear}(\text{cond\_vec})))$$
$$h_0 = [\text{prefix}; \text{TokEmb}(x)] + \text{PosEmb}$$

where $\text{cond\_vec} \in \mathbb{R}^8$ concatenates OCEAN personality (5-dim) and VAD affect (3-dim) vectors.

| Config | m1_small | rtx4070_small |
|--------|----------|---------------|
| prefix_length | 8 | 8 |
| cond_dim | 8 | 8 |
| Parameters | ~9M | ~56M |

#### Mixture-of-Experts (TinyMoELM)
Sparse MoE FFN per transformer block with top-k routing:

$$y = \sum_{i \in \text{top-k}} g_i(x) \cdot E_i(x)$$

where $g(x) = \text{softmax}(\text{Router}(x))$ and $E_i$ are independent FFN experts. Load-balancing auxiliary loss:

$$L_{aux} = E \cdot \sum_{e=1}^E f_e \cdot r_e$$

where $f_e$ is the fraction of tokens dispatched to expert $e$ (soft), and $r_e$ is the fraction routed to expert $e$ (hard).

| Config | m1_small | rtx4070_small |
|--------|----------|---------------|
| num_experts | 4 | 8 |
| top_k | 2 | 2 |
| Parameters | ~12M | ~58M |

#### Mamba-like SSM (MambaLikeLM)
Simplified state-space model (pure PyTorch, no external dependency):

State update:
$$h_t = \bar{A}_t \odot h_{t-1} + \bar{B}_t \cdot u_t$$
Output:
$$y_t = C_t \cdot h_t + D \cdot u_t$$

Discretization via Zero-Order Hold:
$$\bar{A} = \exp(\Delta t \cdot A), \quad \bar{B} = \Delta t \cdot B$$

where $A \in \mathbb{R}^{d_{inner} \times d_{state}}$ is a learned diagonal matrix (initialized as $\log(i)$ for stable decay), and $B, C, \Delta t$ are input-dependent projections. Local convolution of kernel width 4 precedes the SSM. Parallel scan via cumulative products enables O(L) vectorized computation without sequential recurrence.

| Config | m1_small | rtx4070_small |
|--------|----------|---------------|
| n_embd | 256 | 512 |
| n_layer | 6 | 12 |
| d_state | 16 | 16 |
| d_conv | 4 | 4 |
| expand | 2 | 2 |
| Parameters | ~8M | ~50M |

---

## D. Training Methodology

### D.1 Three-Stage Fine-Tuning (LLM)

**Stage 1: Latent State Prediction**

| Hyperparameter | Value |
|----------------|-------|
| Base model | Qwen3-4B |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Learning rate (backbone) | $2 \times 10^{-4}$ |
| Learning rate (heads) | $4 \times 10^{-4}$ |
| LR schedule | Cosine with 5% linear warmup |
| Epochs | 5 |
| Effective batch size | 32 |
| Label smoothing | 0.1 |
| Class weighting | Inverse-frequency, clamped [0.2, 5.0] |
| Optimizer | AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay=0.01) |
| Max gradient norm | 1.0 |
| Gradient checkpointing | Enabled |

**Multi-task loss:**

$$\mathcal{L}_{\text{stage1}} = \sum_{G \in \{C,A,M,R,N,D\}} \lambda_G \cdot \frac{1}{|G|} \sum_{f \in G} w_f \cdot \text{CE}(\text{pred}_f, \text{gold}_f)$$

where $w_f$ are inverse-frequency class weights clamped to [0.2, 5.0] and $\lambda$ values are:

| Group | $\lambda$ | Rationale |
|-------|-----------|-----------|
| C (Context) | 1.0 | Baseline |
| A (Affect) | 1.0 | Core signal |
| M (Mental) | 1.5 | Theory of Mind is harder |
| R (Stance) | 2.0 | Most critical for NPC behavior |
| N (Norms) | 1.0 | Auxiliary |
| D (Policy) | 2.0 | Most critical for response selection |

Oversampling of minority classes via WeightedRandomSampler based on the maximum class weight across all labels present in a sample.

**Stage 2: Response Generation (SFT)**

| Hyperparameter | Value |
|----------------|-------|
| Base model | Qwen3-4B |
| LoRA rank | 32 |
| LoRA alpha | 64 |
| Learning rate | $1 \times 10^{-4}$ |
| Epochs | 3 |
| Effective batch size | 32 |
| Conditioning mode | Gold latent state labels |

Standard causal LM cross-entropy loss with prompt masking (input tokens set to -100). The input format concatenates: scene description, prior stance, dialogue history, player utterance, and gold latent state, followed by the NPC response.

**Stage 3: Joint Fine-Tuning**

| Hyperparameter | Value |
|----------------|-------|
| Learning rate | $5 \times 10^{-5}$ |
| Epochs | 3 |
| Effective batch size | 8 |
| Initialization | Stage 1 heads + Stage 2 adapter |

**Joint loss:**

$$\mathcal{L}_{\text{joint}} = \mathcal{L}_{\text{heads}} + \lambda_Y \cdot \mathcal{L}_{\text{LM}} + \lambda_{\text{cons}} \cdot \mathcal{L}_{\text{consistency}}$$

where $\lambda_Y = 1.0$, $\lambda_{\text{cons}} = 0.5$, and the consistency penalty is:

$$\mathcal{L}_{\text{consistency}} = \frac{1}{B} \sum_{i=1}^B P(\text{reveal=full})_i \cdot P(\text{secrecy=high})_i$$

This penalizes the model for simultaneously predicting full disclosure and high secrecy pressure — a logical contradiction that the structured state representation makes detectable.

### D.2 Small LM Training from Scratch

**HPO Phase:** 20 Optuna trials × 5 epochs per architecture. Search space:

| Parameter | Distribution | Range |
|-----------|-------------|-------|
| lr | Log-uniform | $[10^{-4}, 5 \times 10^{-3}]$ |
| weight_decay | Log-uniform | [0.01, 0.5] |
| batch_size | Categorical | {8, 16, 32} |
| grad_accum | Categorical | {1, 2, 4} |
| dropout | Uniform | [0.0, 0.5] |
| embed_dim | Categorical | {128, 256, 512} |
| n_layer | Categorical | {2, 3, 4} (small) / {4, 6, 8} (large) |

**Final Phase:** 3 seeds (42, 43, 44) × 30 epochs using best hyperparameters from HPO. Standard causal LM loss: $\mathcal{L}_{\text{LM}} = \text{CE}(\text{logits}, \text{targets})$ with ignore_index=-100 for padding.

**MoE auxiliary loss:** $\mathcal{L} = \mathcal{L}_{\text{LM}} + 0.01 \cdot \frac{1}{L} \sum_{l=1}^L \mathcal{L}_{\text{aux}}^{(l)}$

**Conditioning (Prefix GPT only):** $\text{cond\_vec} = [\text{OCEAN}(5) \oplus \text{VAD}(3)]$, projected through Linear→Tanh→Linear to produce 8 prefix tokens prepended to the input sequence.

### D.3 Encoder Training (SLM)

**Personality Encoder (OCEAN):**

| Hyperparameter | Value |
|----------------|-------|
| Backbone | distilbert-base-uncased (66M) |
| Pooling | Mean ⊕ Max → 1536-dim |
| Head | 1536→768→384→5 (3-layer MLP, GELU) |
| Output | 5 continuous: O, C, E, A, N |
| Loss | MSE |
| LR | $2 \times 10^{-5}$ |
| Epochs | 3 |
| Batch size | 16 |
| Dropout | 0.3 |

**Affect Encoder (VAD):**

| Hyperparameter | Value |
|----------------|-------|
| Backbone | distilbert-base-uncased |
| Pooling | Mean → 768-dim |
| Head | 768 → 3 (single linear, sigmoid output) |
| Output | 3 continuous: V, A, D ∈ [0,1] |
| Loss | $(1-0.3) \cdot \text{MSE} + 0.3 \cdot (1 - \text{CCC})$ |
| LR | $2 \times 10^{-5}$ (encoder $2 \times 10^{-6}$) |
| Epochs | 15 |
| Batch size | 16 |
| Dropout | 0.3 |

Concordance Correlation Coefficient:

$$\text{CCC}(x,y) = \frac{2\rho\sigma_x\sigma_y}{\sigma_x^2 + \sigma_y^2 + (\mu_x - \mu_y)^2}$$

---

## E. Data Generation Protocol

### E.1 Scenario Bank

7 scenario types × 5 templates = 35 YAML configurations:

| Scenario | Templates | Stakes | Key Social Dynamics |
|----------|-----------|--------|---------------------|
| secret_extraction | 5 | medium | Trust, secrecy, disclosure |
| apology_repair | 5 | low | Face-saving, guilt, forgiveness |
| alliance_negotiation | 5 | high | Trust, reciprocity, deception |
| rumor_confrontation | 5 | medium | Reputation, credibility, anger |
| threat_escalation | 5 | high | Fear, dominance, de-escalation |
| trust_building | 5 | low | Warmth, self-disclosure, bonding |
| deception_detection | 5 | medium | Theory of mind, lie detection |

### E.2 Teacher LLM Pipeline

Each turn requires up to 10 structured API calls:

1. `label_C()` → dialogue_act, tone, risk_type
2. `label_A_M()` → valence, arousal, threat, control, player_intent, player_knowledge, player_credibility
3. `label_R_N_D()` → 12 stance fields + 4 norm fields + 3 policy fields
4. `generate_response()` → NPC utterance text

Teacher models: GPT-4o (Azure), Qwen3-8B (local), Qwen3-0.6B (testing).

### E.3 Counterfactual Augmentation

Five dimensions flipped per episode to test state-sensitive behavior:

| Variable | Flip | Tests |
|----------|------|-------|
| trust | high ↔ low | State-sensitive withholding |
| secrecy_pressure | low ↔ high | Secret-keeping under pressure |
| player_intent | bond ↔ manipulate | Threat detection |
| reveal_decision | none ↔ full | Disclosure policy |
| value_conflict | none ↔ strong | Norm compliance |

### E.4 Dataset Statistics

| Split | Episodes | Turns | % |
|-------|----------|-------|---|
| Train | 587 | 6,175 | 80% |
| Val | 69 | 683 | 10% |
| Test | 80 | 884 | 10% |
| **Total** | **736** | **7,742** | **100%** |

Stratified split by scenario type to maintain distributional balance across splits. Counterfactual variants comprise 4,596 of 7,742 turns (59.4%).

---

## F. Evaluation Protocol

### F.1 Latent State Prediction Metrics

| Metric | Target | Formula |
|--------|--------|---------|
| response_policy_f1 | ≥ 0.75 | Macro F1 over 10 response policy classes |
| stance_delta_accuracy | ≥ 0.70 | Fraction of correctly predicted deltas across 6 dimensions |
| secret_leakage_rate | ≤ 0.05 | Fraction of turns where NPC reveals secrets despite `reveal_decision=none` |
| mean_accuracy | — | Average per-head classification accuracy |
| mean_f1 | — | Average per-head macro F1 |

### F.2 Response Generation Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| ROUGE-L | maximize | Longest common subsequence overlap with gold response |
| contradiction_rate | ≤ 0.08 | Pattern-based detection of contradictory statements |
| secret_leakage_rate | ≤ 0.05 | Semantic keyword-based leakage detection |

### F.3 Selective Routing Metrics

The router decides fast-path vs. slow-path (reflective generation). Evaluated as binary classifier:

| Metric | Target | Description |
|--------|--------|-------------|
| routing_f1 | maximize | Harmonic mean of precision/recall |
| false_positive_rate | ≤ 0.15 | Unnecessary slow-path invocations |
| routing_precision | maximize | Correct slow-path activations |

**Slow-path triggers:** `value_conflict=strong`, `response_policy ∈ {threaten, negotiate}`, `secrecy_pressure=high ∧ reveal≠none`.

### F.4 SLM Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Validation PPL | Perplexity on held-out dialogue text |
| Test PPL | Fresh PPL with loaded checkpoint |
| BLEU-1/2 | N-gram overlap with reference (via sacrebleu) |
| Distinct-1/2 | Token type diversity ratio $\frac{|\text{unique tokens}|}{|\text{total tokens}|}$ |
| CCC | Concordance correlation (affect encoder only) |
| $R^2$ | Coefficient of determination (personality encoder only) |

---

## G. Proposed Ablation Studies

### Study A: Component Importance
Compare response-only vs. response + each latent component added incrementally:
1. Response-only baseline
2. + C_t (context interpretation)
3. + A_t (affect)
4. + M_t (player model)
5. + R_t (stance)
   → Full model with N_t, D_t

**Hypothesis H1:** Stance (R_t) and policy (D_t) components contribute more to perceived believability than affect alone.

### Study B: Architecture Comparison at Fixed Parameter Budget
Compare 6 SLM architectures at matched parameter counts:
- GRU vs. AWD-LSTM (recurrent baselines)
- GPT vs. Prefix GPT (with/without conditioning)
- GPT vs. MoE (dense vs. sparse FFN)
- GPT vs. Mamba-like (attention vs. SSM)

**Hypothesis H4:** Shared-backbone compact models outperform fully separate specialist models at sub-billion scale on the quality-efficiency frontier.

### Study C: Causal Role of Explicit State
Budget-match: same token budget, context window, parameters. Vary only presence/absence of explicit social-state supervision.

**Hypothesis H2:** Explicit socio-emotional latent state improves perceived social coherence more than response-only fine-tuning at matched compute.

### Study D: Selective Routing Efficiency
Compare always-on reflective pipeline vs. selective fast-path/slow-path routing.

**Hypothesis H3:** Selective routing preserves most quality gains while reducing compute relative to always-on reflective pipelines.

---

## H. Reproducibility Checklist

| Item | Status |
|------|--------|
| Code repository | ✅ GitHub (private) |
| Dependency versions | ✅ `requirements.txt` + `setup_*_env.sh` |
| Random seeds | ✅ 42, 43, 44 for all experiments |
| Dataset generation | ✅ Reproducible via scenario bank + teacher LLM configs |
| Hyperparameters | ✅ Full YAML configs per training stage |
| Evaluation protocol | ✅ Thresholds in `eval.yaml` |
| MLflow tracking | ✅ All runs logged with params, metrics, artifacts |
| Model checkpoints | ✅ Saved with run IDs in artifact store |
| Environment setup | ✅ `setup_slm_env.sh` + `setup_llm_env.sh` with module loading |
| SLURM job scripts | ✅ Array jobs with seed × arch grid |

---

## I. Limitations

1. **Single-teacher data:** All supervision labels come from one teacher LLM family (GPT-4o / Qwen3). Label bias may propagate to student models.

2. **Synthetic evaluation:** Secret leakage and contradiction metrics use heuristic keyword matching, not full semantic entailment detection.

3. **Domain specificity:** Scenario bank covers 7 social interaction types in a medieval fantasy setting. Generalization to other domains (sci-fi, modern, professional) is untested.

4. **Binary router:** The selective routing module uses threshold-based heuristics rather than learned routing. Future work should explore trainable router modules.

5. **English only:** All data, prompts, and evaluation are in English. Multilingual social state modeling is unexplored.
