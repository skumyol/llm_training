SLM Encoders Reference
========================

DistilBertRegressor
-------------------

Both personality and affect encoders use the same base class with different head configurations.

.. code-block:: python

   class DistilBertRegressor(nn.Module):
       def __init__(self, model_name: str, out_dim: int, dropout: float,
                    use_sigmoid: bool = False, multi_sample_dropout: int = 0):
           self.encoder = AutoModel.from_pretrained(model_name)
           hidden = self.encoder.config.hidden_size  # 768 for distilbert-base

Personality Encoder
---------------------

.. list-table::
   :header-rows: 1

   * - Component
     - Specification
   * - Backbone
     - distilbert-base-uncased (66M params)
   * - Output dim
     - 5 (OCEAN: Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism)
   * - Pooling
     - Mean ⊕ Max = 1536
   * - Head architecture
     - 1536→LayerNorm→Dropout→768→GELU→LayerNorm→Dropout→384→GELU→Dropout→5
   * - Activation
     - GELU
   * - Final activation
     - None (outputs can exceed [0,1])
   * - Dropout
     - 0.3
   * - Multi-sample dropout
     - 0 (disabled)

Pooling Strategy
~~~~~~~~~~~~~~~~

.. code-block:: python

   def pooled(self, input_ids, attention_mask):
       out = self.encoder(input_ids, attention_mask)
       last_hidden = out.last_hidden_state  # (B, T, 768)
       mask = attention_mask.unsqueeze(-1)
       # Mean pooling
       mean_pooled = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
       # Max pooling
       max_pooled = (last_hidden + (1 - mask) * -1e9).max(1).values
       # Concatenate
       pooled = torch.cat([mean_pooled, max_pooled], dim=-1)  # (B, 1536)
       return self.layer_norm(pooled)

Loss
~~~~

.. math::

   L = \frac{1}{5B} \sum_{i=1}^B \sum_{j=1}^5 (y_{ij} - \hat{y}_{ij})^2

Affect Encoder
---------------

.. list-table::
   :header-rows: 1

   * - Component
     - Specification
   * - Backbone
     - distilbert-base-uncased
   * - Output dim
     - 3 (VAD: Valence, Arousal, Dominance)
   * - Pooling
     - Mean pooling = 768
   * - Head architecture
     - 768→LayerNorm→Dropout→3
   * - Activation
     - None (single linear)
   * - Final activation
     - Sigmoid (values in [0,1])
   * - Dropout
     - 0.1

Loss
~~~~

Combined MSE + Concordance Correlation Coefficient (CCC):

.. math::

   L = (1 - \lambda_{ccc}) \cdot MSE(y, \hat{y}) + \lambda_{ccc} \cdot (1 - CCC(y, \hat{y}))

where:

.. math::

   CCC(x, y) = \frac{2 \cdot \rho \cdot \sigma_x \cdot \sigma_y}{\sigma_x^2 + \sigma_y^2 + (\mu_x - \mu_y)^2}

and :math:`\lambda_{ccc} = 0.3`.

Conditional Dialogue Model
----------------------------

The dialogue model uses TinyLlama with conditional soft-prefix injection.

.. list-table::
   :header-rows: 1

   * - Component
     - Specification
   * - Backbone
     - TinyLlama-1.1B-Chat
   * - LoRA
     - r=16, α=32, dropout=0.05
   * - LoRA targets
     - q_proj, k_proj, v_proj, o_proj
   * - Prefix length
     - 8 tokens
   * - Condition vector
     - 8 dims (OCEAN=5 + VAD=3)
   * - Prefix projection
     - Linear(8→768)→Tanh→Linear(768→8·768)→reshape

ConditionalSoftPrefix
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class ConditionalSoftPrefix(nn.Module):
       def __init__(self, cond_dim: int, hidden_size: int, prefix_length: int):
           self.prefix_length = prefix_length
           self.proj = nn.Sequential(
               nn.Linear(cond_dim, hidden_size),
               nn.Tanh(),
               nn.Linear(hidden_size, prefix_length * hidden_size),
           )

       def forward(self, cond_vec: torch.Tensor) -> PrefixOutput:
           B = cond_vec.size(0)
           prefix = self.proj(cond_vec).view(B, self.prefix_length, self.hidden_size)
           mask = torch.ones(B, self.prefix_length, dtype=torch.long)
           return PrefixOutput(prefix_embeds=prefix, prefix_mask=mask)

Data Sources
------------

.. list-table::
   :header-rows: 1

   * - Dataset
     - Source
     - Records
     - Description
   * - Synthetic-Persona-Chat
     - Sam Edwards (HuggingFace)
     - 162k turns
     - Multi-turn dialogue with persona profiles
   * - Generated NPC data
     - LLM teacher pipeline
     - 2,734 turns
     - Structured social state annotations
   * - External narrative
     - BookCorpus + OpenSubtitles
     - 107M tokens
     - Domain-adaptive pretraining corpus
