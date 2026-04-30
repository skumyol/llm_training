SLM Architecture Reference
============================

Detailed architecture specifications for all 6 small language models.

1. GRU — SmallGRULM
--------------------

.. list-table::
   :header-rows: 1

   * - Parameter
     - m1_small
     - rtx4070_small
   * - ``embed_dim``
     - 256
     - 512
   * - ``hidden_size``
     - 512
     - 1024
   * - ``num_layers``
     - 3
     - 3
   * - ``dropout``
     - 0.3
     - 0.3
   * - ``tie_weights``
     - True
     - True
   * - ~Parameters
     - ~12M
     - ~40M

.. code-block:: python

   class SmallGRULM(nn.Module):
       def __init__(self, cfg: GRUConfig):
           self.embed = nn.Embedding(vocab_size, embed_dim)  # Token embeddings
           self.gru = nn.GRU(embed_dim, hidden_size, num_layers,
                             batch_first=True, dropout=dropout)
           self.drop = nn.Dropout(dropout)
           self.head = nn.Linear(hidden_size, vocab_size, bias=False)
           # Weight tying: head.weight = embed.weight (if dims match)

2. AWD-LSTM — AWDLSTMLM
-------------------------

.. list-table::
   :header-rows: 1

   * - Parameter
     - m1_small
     - rtx4070_small
   * - ``embed_dim``
     - 256
     - 400
   * - ``hidden_size``
     - 512
     - 1150
   * - ``num_layers``
     - 2
     - 3
   * - ``dropout`` (output)
     - 0.4
     - 0.4
   * - ``dropouth`` (hidden variational)
     - 0.25
     - 0.25
   * - ``dropouti`` (embedding)
     - 0.4
     - 0.65
   * - ``wdrop`` (DropConnect)
     - 0.5
     - 0.5
   * - ~Parameters
     - ~10M
     - ~28M

Key components:

- **LockedDropout:** Variational dropout — same mask across all timesteps
- **WeightDropLSTM:** Applies Bernoulli dropout to LSTM hidden-to-hidden weights every forward pass

.. code-block:: python

   class LockedDropout(nn.Module):
       def forward(self, x, p):
           if not self.training or p == 0: return x
           mask = x.new_empty(B, 1, D).bernoulli_(1-p).div_(1-p)
           return x * mask.expand_as(x)

3. GPT — TinyGPTLM
--------------------

.. list-table::
   :header-rows: 1

   * - Parameter
     - m1_small
     - rtx4070_small
   * - ``n_embd``
     - 256
     - 512
   * - ``n_head``
     - 4
     - 8
   * - ``n_layer``
     - 4
     - 8
   * - ``dropout``
     - 0.1
     - 0.1
   * - ``max_seq_len``
     - 256
     - 512
   * - ``tie_weights``
     - True
     - True
   * - ~Parameters
     - ~8M
     - ~55M

Standard decoder-only transformer with causal self-attention and 4× FFN expansion.

4. Prefix GPT — PrefixTinyGPTLM
-----------------------------------

.. list-table::
   :header-rows: 1

   * - Parameter
     - m1_small
     - rtx4070_small
   * - ``n_embd``
     - 256
     - 512
   * - ``n_head``
     - 4
     - 8
   * - ``n_layer``
     - 4
     - 8
   * - ``prefix_length``
     - 8
     - 8
   * - ``cond_dim``
     - 8
     - 8
   * - ~Parameters
     - ~9M
     - ~56M

.. code-block:: python

   class PrefixTinyGPTLM(nn.Module):
       def forward(self, x, cond_vec, targets=None):
           # Project condition vector to soft-prefix tokens
           prefix = self.prefix_proj(cond_vec).view(B, P, E)
           # Prepend prefix to token embeddings
           h = torch.cat([prefix, tok_emb(x)], dim=1)
           h = h + pos_emb  # Add positional encoding
           # Standard transformer blocks
           for block in self.blocks: h = block(h)
           # Drop prefix positions from output
           logits = self.head(self.ln_f(h[:, P:, :]))

Conditioning: ``cond_vec(8) = [OCEAN(5) | VAD(3)]`` projected through:
``Linear(8→E) → Tanh → Linear(E→P·E) → reshape(B, P, E)``

5. Mixture-of-Experts — TinyMoELM
------------------------------------

.. list-table::
   :header-rows: 1

   * - Parameter
     - m1_small
     - rtx4070_small
   * - ``n_embd``
     - 256
     - 512
   * - ``n_head``
     - 4
     - 8
   * - ``n_layer``
     - 4
     - 8
   * - ``num_experts``
     - 4
     - 8
   * - ``top_k``
     - 2
     - 2
   * - ~Parameters
     - ~12M
     - ~58M

Sparse MoE FFN per transformer block:

- **Router:** ``Linear(E → num_experts)`` with top-k softmax gating
- **Experts:** Each is ``Linear(E→4E)→GELU→Linear(4E→E)→Dropout``
- **Aux loss:** Load-balancing: ``(frac_per_expert · route_per_expert)·sum × num_experts``

6. Mamba-like SSM — MambaLikeLM
----------------------------------

.. list-table::
   :header-rows: 1

   * - Parameter
     - m1_small
     - rtx4070_small
   * - ``n_embd``
     - 256
     - 512
   * - ``n_layer``
     - 6
     - 12
   * - ``d_state``
     - 16
     - 16
   * - ``d_conv``
     - 4
     - 4
   * - ``expand``
     - 2
     - 2
   * - ``dropout``
     - 0.1
     - 0.1
   * - ~Parameters
     - ~8M
     - ~50M

Pure PyTorch implementation, no external dependencies.

.. math::

   h_t &= \bar{A}_t \cdot h_{t-1} + \bar{B}_t \cdot u_t \\
   y_t &= C_t \cdot h_t + D \cdot u_t

Key components:

- **Discretization:** :math:`\Delta t = \text{softplus}(\text{Linear}(\log\Delta t))`
- **A-matrix:** Learned diagonal, input-independent, initialized as :math:`\log(i)`
- **B, C:** Input-dependent projections
- **Parallel scan:** Cumulative product (O(L) vectorized, no sequential recurrence)
- **Gating:** Output multiplied by SiLU(z-branch), then linear projection back to embedding dim

Architecture Registry
----------------------

.. code-block:: python

   def build_model(arch: str, cfg_dict: dict) -> nn.Module:
       """Instantiate from arch name and flat config dict."""
       arch_map = {
           "gru":        (SmallGRULM,       GRUConfig),
           "awdlstm":    (AWDLSTMLM,        AWDLSTMConfig),
           "gpt":        (TinyGPTLM,        GPTConfig),
           "prefix_gpt": (PrefixTinyGPTLM,  PrefixGPTConfig),
           "moe":        (TinyMoELM,        MoEConfig),
           "mamba_like": (MambaLikeLM,      MambaLikeConfig),
       }
       model_cls, cfg_cls = arch_map[arch.lower()]
       return model_cls(cfg_cls(**cfg_dict))
