LLM Training Pipeline
=====================

Three-stage training pipeline for the latent state predictor.

Stage 1: Latent State Predictor
--------------------------------

**Objective:** Predict all 29 social-state labels from dialogue context.

Configuration
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Hyperparameter
     - Value
     - Description
   * - ``base_model``
     - Qwen/Qwen3-4B
     - Base LLM to fine-tune
   * - ``quantization``
     - 4bit
     - QLoRA quantization
   * - ``pooling``
     - last
     - Hidden state extraction method

LoRA
~~~~

.. list-table::
   :header-rows: 1

   * - Parameter
     - Value
   * - ``r``
     - 16
   * - ``alpha``
     - 32
   * - ``dropout``
     - 0.05
   * - ``target_modules``
     - q_proj, v_proj, k_proj, o_proj, gate_proj, up_proj, down_proj

Training
~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Parameter
     - Value
   * - Learning rate (backbone)
     - 2×10⁻⁴
   * - Learning rate (heads)
     - 4×10⁻⁴
   * - LR schedule
     - Cosine + 5% warmup
   * - Epochs
     - 5
   * - Max sequence length
     - 512
   * - Batch size (effective)
     - 32 (1 × 32 grad_accum)
   * - Label smoothing
     - 0.1
   * - Weighted sampler
     - ✅ (oversample minority)
   * - Gradient checkpointing
     - ✅
   * - Optimizer
     - AdamW (β₁=0.9, β₂=0.999)
   * - Weight decay
     - 0.01
   * - Max grad norm
     - 1.0

Loss Weights
~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Component Group
     - λ
     - Rationale
   * - C (Context)
     - 1.0
     - Baseline
   * - A (Affect)
     - 1.0
     - Core signal
   * - M (Mental Model)
     - 1.5
     - Theory of Mind is harder
   * - R (Stance)
     - 2.0
     - **Most important** for NPC control
   * - N (Norms)
     - 1.0
     - Auxiliary
   * - D (Decision/Policy)
     - 2.0
     - **Most important** for behavior

Class weights are computed via inverse-frequency, clamped to [0.2, 5.0] per head.

Multi-Head Loss
~~~~~~~~~~~~~~~

For each component group :math:`G`:

.. math::

   L_G = \frac{1}{|G|} \sum_{f \in G} w_f \cdot CE(pred_f, gold_f)

where :math:`w_f` is the inverse-frequency class weight for field :math:`f`.

Multi-label heads (``dialogue_act``) use BCEWithLogitsLoss instead of cross-entropy.

Total loss:

.. math::

   L_{total} = \sum_{G} \lambda_G \cdot L_G

Stage 2: Response Generator (SFT)
-----------------------------------

**Objective:** Generate NPC dialogue conditioned on gold latent state labels.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Value
   * - ``base_model``
     - Qwen/Qwen3-4B
   * - LoRA ``r``
     - 32
   * - LoRA ``alpha``
     - 64
   * - Learning rate
     - 1×10⁻⁴
   * - LR schedule
     - Cosine
   * - Epochs
     - 3
   * - Max sequence length
     - 1024
   * - Batch size (effective)
     - 32 (1 × 32 grad_accum)
   * - Conditioning mode
     - gold (teacher labels)

Input Format
~~~~~~~~~~~~

.. code-block:: text

   <scene>
   Setting: Castle siege
   NPC Role: Guard Captain
   Goals: protect the gate, maintain order
   Values: loyalty, duty
   ...
   </scene>

   <prior_stance>
   affection=N  respect=L  dominance=N  familiarity=VL  trust=VL  obligation=VL
   </prior_stance>

   <history>
   [Turn 1] Player: Who goes there?
   [Turn 1] NPC: State your business, stranger.
   </history>

   Player: Have you caught the spy yet?

   <latent_state>
   C_t: dialogue_act=['probe']  tone=neutral  risk=secret-risk
   A_t: valence=neutral  arousal=medium  threat=low  control=medium
   M_t: player_intent=seek-info  player_knowledge=partial  credibility=medium
   R_t: affection=N(0)  respect=L(0)  dominance=N(0)  familiarity=VL(0)  trust=VL(0)  obligation=VL(0)
   N_t: duty=low  secrecy=high  face=low  conflict=none
   D_t: policy=deflect  reveal=none  repair=none
   </latent_state>

   Generate NPC response:

Stage 3: Joint Fine-Tuning
----------------------------

**Objective:** Train latent prediction + response generation jointly with consistency loss.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Value
   * - ``base_model``
     - Qwen/Qwen3-4B
   * - Learning rate
     - 5×10⁻⁵
   * - Epochs
     - 3
   * - Batch size (effective)
     - 8 (1 × 8 grad_accum)
   * - Initialization
     - Stage 1 heads + Stage 2 adapter

Joint Loss
~~~~~~~~~~

.. math::

   L = L_{heads} + \lambda_Y \cdot L_{lm} + \lambda_{consistency} \cdot L_{consistency}

.. list-table::
   :header-rows: 1

   * - Loss Term
     - λ
     - Description
   * - :math:`L_{heads}`
     - 1.0 (per-group)
     - 29-head classification loss
   * - :math:`L_{lm}`
     - 1.0
     - Causal LM cross-entropy
   * - :math:`L_{consistency}`
     - 0.5
     - Penalizes high-secrecy + full-reveal

Consistency Loss
~~~~~~~~~~~~~~~~

.. code-block:: python

   class ConsistencyLoss(nn.Module):
       def forward(self, logits):
           reveal_probs = F.softmax(logits["reveal_decision"], dim=-1)
           full_reveal_prob = reveal_probs[:, 3]  # index 3 = "full"

           secrecy_probs = F.softmax(logits["secrecy_pressure"], dim=-1)
           high_secrecy_prob = secrecy_probs[:, 2]  # index 2 = "high"

           # Penalize: high secrecy AND full reveal simultaneously
           penalty = (full_reveal_prob * high_secrecy_prob).mean()
           return penalty
