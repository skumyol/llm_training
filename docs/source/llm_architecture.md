LLM Fine-Tuning: Latent State Predictor
=========================================

Fine-tunes pre-trained Qwen3 models into a structured social-state dialogue predictor with 29 classification heads and multi-stage training.

Architecture
------------

.. code-block:: text

   ┌────────────────────────────────────────────────────────────────┐
   │                   LatentStatePredictor                         │
   │                                                                │
   │  Input: dialogue context text                                  │
   │     ↓                                                          │
   │  ┌────────────────────────────────────┐                       │
   │  │  Qwen3 Backbone (0.6B–4B) + LoRA   │                       │
   │  │  - 4-bit QLoRA quantization        │                       │
   │  │  - Pooling: last / mean / attention │                       │
   │  └────────────────────────────────────┘                       │
   │     ↓                                                          │
   │  ┌────────────────────────────────────┐                       │
   │  │  29 Classification Heads           │                       │
   │  │  Linear(256)→GELU→Drop0.1→Linear   │                       │
   │  └────────────────────────────────────┘                       │
   │     ↓                                                          │
   │  Output: 29 label predictions per turn                        │
   └────────────────────────────────────────────────────────────────┘

Classification Head
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class ClassificationHead(nn.Module):
       def __init__(self, hidden_size: int, n_classes: int, dropout: float = 0.1):
           self.net = nn.Sequential(
               nn.Linear(hidden_size, 256),  # Bottleneck projection
               nn.GELU(),                     # Non-linearity
               nn.Dropout(dropout),           # Regularization
               nn.Linear(256, n_classes),     # Class projection
           )

Head Specification
-------------------

29 total classification targets organized into 6 component groups:

C_t — Contextual Analysis (3 heads)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Field
     - n_classes
     - Multi-Label
     - Labels
   * - ``dialogue_act``
     - 10
     - ✅ Yes
     - ask, accuse, threaten, flatter, apologize, negotiate, joke, confess, probe, command
   * - ``tone``
     - 6
     - ❌ No
     - warm, neutral, confrontational, sarcastic, fearful, evasive
   * - ``risk_type``
     - 5
     - ❌ No
     - none, secret-risk, face-risk, status-risk, conflict-risk

A_t — Affective Appraisal (4 heads)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Field
     - n_classes
     - Labels
   * - ``valence``
     - 3
     - negative, neutral, positive
   * - ``arousal``
     - 3
     - low, medium, high
   * - ``threat``
     - 3
     - low, medium, high
   * - ``control``
     - 3
     - low, medium, high

M_t — Player Mental Model (3 heads)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Field
     - n_classes
     - Labels
   * - ``player_intent``
     - 9
     - seek-info, trap, bond, manipulate, test, persuade, intimidate, probe, negotiate
   * - ``player_knowledge``
     - 4
     - unaware, partial, informed, knows-secret
   * - ``player_credibility``
     - 3
     - low, medium, high

R_t — Relational Stance (12 heads)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

6 stance dimensions, each with a **level** (5 classes) and **delta** (5 classes):

.. list-table::
   :header-rows: 1

   * - Dimension
     - Level Labels
     - Delta Labels
   * - ``affection``
     - VL, L, N, H, VH
     - --, -, 0, +, ++
   * - ``respect``
     - VL, L, N, H, VH
     - --, -, 0, +, ++
   * - ``dominance``
     - VL, L, N, H, VH
     - --, -, 0, +, ++
   * - ``familiarity``
     - VL, L, N, H, VH
     - --, -, 0, +, ++
   * - ``trust``
     - VL, L, N, H, VH
     - --, -, 0, +, ++
   * - ``obligation``
     - VL, L, N, H, VH
     - --, -, 0, +, ++

N_t — Norm/Value Constraints (4 heads)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Field
     - n_classes
     - Labels
   * - ``duty_pressure``
     - 3
     - low, medium, high
   * - ``secrecy_pressure``
     - 3
     - low, medium, high
   * - ``face_pressure``
     - 3
     - low, medium, high
   * - ``value_conflict``
     - 3
     - none, mild, strong

D_t — Response Policy (3 heads)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Field
     - n_classes
     - Labels
   * - ``response_policy``
     - 10
     - answer, partial, withhold, deflect, challenge, soothe, test, threaten, negotiate, clarify
   * - ``reveal_decision``
     - 4
     - none, hint, partial, full
   * - ``repair_strategy``
     - 5
     - none, soften, apologize, clarify, redirect

Backbone Models
---------------

.. list-table::
   :header-rows: 1

   * - Model
     - Parameters
     - Hidden Size
     - VRAM (4-bit)
     - Use Case
   * - Qwen3-0.6B
     - 0.6B
     - 896
     - ~2 GB
     - Debug / fast iteration
   * - Qwen3-1.7B
     - 1.7B
     - 2048
     - ~4 GB
     - Debug
   * - Qwen3-4B
     - 4B
     - 2560
     - ~6 GB
     - **Production**

Pooling Strategies
------------------

.. list-table::
   :header-rows: 1

   * - Strategy
     - Description
     - Use Case
   * - ``last``
     - Last non-padding hidden state
     - **Default** — best for autoregressive
   * - ``mean``
     - Mean over all non-padding tokens
     - Better for bidirectional context
   * - ``attention``
     - Learnable attention-weighted sum
     - Most expressive, slower

Quantization
--------------

.. list-table::
   :header-rows: 1

   * - Mode
     - BitsAndBytes Config
     - VRAM Saving
   * - ``4bit``
     - nf4, double quant, bfloat16 compute
     - ~75%
   * - ``8bit``
     - 8-bit linear
     - ~50%
   * - ``none``
     - Full precision
     - 0%
