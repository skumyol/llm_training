"""
small_lm_architectures.py
==========================
Six from-scratch small language model architectures for NPC dialogue A/B testing.

Models:
  SmallGRULM      – multi-layer GRU
  AWDLSTMLM       – AWD-LSTM (DropConnect + variational dropout)
  TinyGPTLM       – transformer decoder
  PrefixTinyGPTLM – transformer decoder + soft-prefix conditioning (cond_vec → prefix tokens)
  TinyMoELM       – GPT with Mixture-of-Experts FFN layers
  MambaLikeLM     – simplified SSM (pure PyTorch, no mamba_ssm dependency)

All models share the same interface:
  out = model(x, targets)          # x: (B, T) int64, targets: (B, T) int64
  out = model(x, cond_vec, targets) # PrefixTinyGPTLM only

Returns LMOutput(loss, logits).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Shared output ─────────────────────────────────────────────────────────────

@dataclass
class LMOutput:
    loss:   Optional[torch.Tensor]
    logits: torch.Tensor
    hidden_states: Optional[torch.Tensor] = None  # last-layer hidden, before head


def select_device(preferred: str = "auto") -> torch.device:
    """Select an accelerator, or fail clearly when an explicit one is unavailable."""
    preferred = str(preferred).lower()
    if preferred not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(f"Unknown device {preferred!r}; expected auto, cuda, mps, or cpu")
    cuda_available = torch.cuda.is_available()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if preferred == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but is not available in this PyTorch runtime")
    if preferred == "mps" and not mps_available:
        raise RuntimeError("MPS was requested but is not available in this PyTorch runtime")
    if preferred != "auto":
        return torch.device(preferred)
    if cuda_available:
        return torch.device("cuda")
    if mps_available:
        return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# 1. GRU LM
# =============================================================================

@dataclass
class GRUConfig:
    vocab_size:  int   = 50257
    embed_dim:   int   = 256
    hidden_size: int   = 512
    num_layers:  int   = 3
    dropout:     float = 0.3
    tie_weights: bool  = True


class SmallGRULM(nn.Module):
    def __init__(self, cfg: GRUConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.gru   = nn.GRU(
            cfg.embed_dim, cfg.hidden_size, cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_weights and cfg.embed_dim == cfg.hidden_size:
            self.head.weight = self.embed.weight
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.uniform_(self.embed.weight, -0.1, 0.1)
        nn.init.zeros_(self.head.bias) if hasattr(self.head, "bias") and self.head.bias is not None else None

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> LMOutput:
        emb    = self.drop(self.embed(x))
        out, _ = self.gru(emb)
        logits = self.head(self.drop(out))
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size), targets.reshape(-1), ignore_index=-100)
        return LMOutput(loss=loss, logits=logits)


# =============================================================================
# 2. AWD-LSTM
# =============================================================================

@dataclass
class AWDLSTMConfig:
    vocab_size:  int   = 50257
    embed_dim:   int   = 256
    hidden_size: int   = 512
    num_layers:  int   = 3
    dropout:     float = 0.4    # output dropout
    dropouth:    float = 0.25   # hidden dropout (variational, between layers)
    dropouti:    float = 0.4    # embedding dropout
    wdrop:       float = 0.5    # DropConnect on hidden-to-hidden weights
    tie_weights: bool  = True


class LockedDropout(nn.Module):
    """Variational dropout — same mask for every timestep in the sequence."""
    def forward(self, x: torch.Tensor, p: float) -> torch.Tensor:
        if not self.training or p == 0.0:
            return x
        # x: (B, T, D)
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - p).div_(1 - p)
        return x * mask.expand_as(x)


class WeightDropLSTM(nn.Module):
    """LSTM with DropConnect applied to hidden-to-hidden weight matrices."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, wdrop: float) -> None:
        super().__init__()
        self.lstm      = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.wdrop     = wdrop
        self.num_layers = num_layers
        # Replace weight_hh with raw parameters we control
        for i in range(num_layers):
            name = f"weight_hh_l{i}"
            w    = getattr(self.lstm, name).data.clone()
            del self.lstm._parameters[name]
            self.register_parameter(f"weight_hh_raw_l{i}", nn.Parameter(w))

    def _apply_weight_drop(self) -> None:
        for i in range(self.num_layers):
            raw = getattr(self, f"weight_hh_raw_l{i}")
            if self.training and self.wdrop > 0:
                mask = raw.new_empty(raw.shape).bernoulli_(1 - self.wdrop).div_(1 - self.wdrop)
                w    = raw * mask
            else:
                w = raw
            self.lstm._parameters[f"weight_hh_l{i}"] = w
        # Reset flat-weights cache so cuDNN picks up the new weight_hh tensors
        self.lstm._flat_weights = [
            getattr(self.lstm, w) if hasattr(self.lstm, w) else self.lstm._parameters.get(w)
            for w in self.lstm._flat_weights_names
        ]

    def forward(self, x: torch.Tensor, hx=None):
        self._apply_weight_drop()
        if hx is None:
            return self.lstm(x)
        return self.lstm(x, hx)


class AWDLSTMLM(nn.Module):
    def __init__(self, cfg: AWDLSTMConfig) -> None:
        super().__init__()
        self.cfg         = cfg
        self.locked_drop = LockedDropout()
        self.embed       = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.lstm        = WeightDropLSTM(cfg.embed_dim, cfg.hidden_size, cfg.num_layers, cfg.wdrop)
        self.head        = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_weights and cfg.embed_dim == cfg.hidden_size:
            self.head.weight = self.embed.weight
        nn.init.uniform_(self.embed.weight, -0.1, 0.1)

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> LMOutput:
        emb    = self.locked_drop(self.embed(x), self.cfg.dropouti)
        out, _ = self.lstm(emb)
        out    = self.locked_drop(out, self.cfg.dropouth)
        out    = F.dropout(out, p=self.cfg.dropout, training=self.training)
        logits = self.head(out)
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size), targets.reshape(-1), ignore_index=-100)
        return LMOutput(loss=loss, logits=logits)


# =============================================================================
# 3. TinyGPT
# =============================================================================

@dataclass
class GPTConfig:
    vocab_size:  int   = 50257
    n_embd:      int   = 256
    n_head:      int   = 4
    n_layer:     int   = 4
    dropout:     float = 0.1
    max_seq_len: int   = 256
    tie_weights: bool  = True


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head   = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv      = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj     = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.drop     = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len))
        self.register_buffer("mask", mask.view(1, 1, cfg.max_seq_len, cfg.max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        Q, K, V = self.qkv(x).split(C, dim=2)
        def reshape(t):
            return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        Q, K, V = reshape(Q), reshape(K), reshape(V)
        # Lets PyTorch select FlashAttention/memory-efficient CUDA kernels and
        # the best available MPS/CPU implementation without changing the model.
        y = F.scaled_dot_product_attention(
            Q,
            K,
            V,
            dropout_p=self.drop.p if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class GPTBlock(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.n_embd)
        self.ffn  = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TinyGPTLM(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg     = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.n_embd)
        self.drop    = nn.Dropout(cfg.dropout)
        self.blocks  = nn.ModuleList([GPTBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f    = nn.LayerNorm(cfg.n_embd)
        self.head    = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None, return_hidden: bool = False) -> LMOutput:
        B, T = x.shape
        pos    = torch.arange(T, device=x.device)
        h      = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        for block in self.blocks:
            h  = block(h)
        hidden = self.ln_f(h)
        logits = self.head(hidden)
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size), targets.reshape(-1), ignore_index=-100)
        return LMOutput(loss=loss, logits=logits, hidden_states=hidden if return_hidden else None)


# =============================================================================
# 4. PrefixTinyGPT (conditioning via soft-prefix tokens)
# =============================================================================

@dataclass
class PrefixGPTConfig:
    vocab_size:    int   = 50257
    n_embd:        int   = 256
    n_head:        int   = 4
    n_layer:       int   = 4
    dropout:       float = 0.1
    max_seq_len:   int   = 256
    prefix_length: int   = 8
    cond_dim:      int   = 8    # OCEAN(5) + VAD(3)
    condition_mode: str   = "ocean_vad"
    tie_weights:   bool  = True


class PrefixTinyGPTLM(nn.Module):
    """TinyGPT with a learned soft-prefix conditioned on personality+affect vectors.

    The prefix is prepended to the token embedding sequence before the first
    transformer block — identical conditioning interface to ConditionalDialogueModel.
    """
    def __init__(self, cfg: PrefixGPTConfig) -> None:
        super().__init__()
        self.cfg          = cfg
        self.tok_emb      = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb      = nn.Embedding(cfg.max_seq_len + cfg.prefix_length, cfg.n_embd)
        self.drop         = nn.Dropout(cfg.dropout)
        self.blocks       = nn.ModuleList([GPTBlock(GPTConfig(
            vocab_size=cfg.vocab_size, n_embd=cfg.n_embd, n_head=cfg.n_head,
            n_layer=cfg.n_layer, dropout=cfg.dropout,
            max_seq_len=cfg.max_seq_len + cfg.prefix_length,
        )) for _ in range(cfg.n_layer)])
        self.ln_f         = nn.LayerNorm(cfg.n_embd)
        self.head         = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight
        self.prefix_proj  = nn.Sequential(
            nn.Linear(cfg.cond_dim, cfg.n_embd),
            nn.Tanh(),
            nn.Linear(cfg.n_embd, cfg.prefix_length * cfg.n_embd),
        )
        self.missing_condition = nn.Parameter(torch.zeros(cfg.cond_dim))
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        cond_vec: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        cond_mask: Optional[torch.Tensor] = None,
    ) -> LMOutput:
        B, T = x.shape
        P    = self.cfg.prefix_length

        tok_h  = self.tok_emb(x)                                       # (B, T, E)
        if cond_mask is not None:
            cond_vec = (
                cond_vec * cond_mask
                + self.missing_condition.unsqueeze(0) * (1.0 - cond_mask)
            )
        prefix = self.prefix_proj(cond_vec).view(B, P, self.cfg.n_embd) # (B, P, E)
        h      = torch.cat([prefix, tok_h], dim=1)                     # (B, P+T, E)
        pos    = torch.arange(P + T, device=x.device)
        h      = self.drop(h + self.pos_emb(pos))

        for block in self.blocks:
            h  = block(h)

        logits = self.head(self.ln_f(h[:, P:, :]))                    # (B, T, V) — drop prefix positions
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size), targets.reshape(-1), ignore_index=-100)
        return LMOutput(loss=loss, logits=logits)


# =============================================================================
# 5. TinyMoE (GPT with Mixture-of-Experts FFN)
# =============================================================================

@dataclass
class MoEConfig:
    vocab_size:   int   = 50257
    n_embd:       int   = 256
    n_head:       int   = 4
    n_layer:      int   = 4
    num_experts:  int   = 4
    top_k:        int   = 2
    dropout:      float = 0.1
    max_seq_len:  int   = 256
    tie_weights:  bool  = True


class MoEFeedForward(nn.Module):
    """Sparse Mixture-of-Experts FFN with top-k routing and load-balance aux loss."""
    def __init__(self, n_embd: int, num_experts: int, top_k: int, dropout: float) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k       = top_k
        self.router      = nn.Linear(n_embd, num_experts, bias=False)
        self.experts     = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
                nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout),
            )
            for _ in range(num_experts)
        ])
        self.aux_loss: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, E  = x.shape
        flat     = x.reshape(B * T, E)
        scores   = self.router(flat)                       # (BT, num_experts)
        probs    = F.softmax(scores, dim=-1)               # (BT, num_experts)

        # Top-k selection
        top_w, top_idx = torch.topk(probs, self.top_k, dim=-1)  # (BT, k)
        top_w          = top_w / top_w.sum(dim=-1, keepdim=True) # normalize

        # Load-balance aux loss (fraction of tokens per expert)
        frac_per_expert = probs.mean(dim=0)
        route_per_expert = (probs == probs.max(dim=-1, keepdim=True).values).float().mean(dim=0)
        self.aux_loss   = (frac_per_expert * route_per_expert).sum() * self.num_experts

        out = torch.zeros_like(flat)
        for k in range(self.top_k):
            idx      = top_idx[:, k]                  # (BT,) expert index
            weight   = top_w[:, k].unsqueeze(-1)      # (BT, 1)
            for e in range(self.num_experts):
                mask    = idx == e
                if mask.any():
                    out[mask] = out[mask] + weight[mask] * self.experts[e](flat[mask])

        return out.reshape(B, T, E)


class MoEBlock(nn.Module):
    def __init__(self, cfg: MoEConfig) -> None:
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(GPTConfig(
            vocab_size=cfg.vocab_size, n_embd=cfg.n_embd, n_head=cfg.n_head,
            dropout=cfg.dropout, max_seq_len=cfg.max_seq_len,
        ))
        self.ln2  = nn.LayerNorm(cfg.n_embd)
        self.moe  = MoEFeedForward(cfg.n_embd, cfg.num_experts, cfg.top_k, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.moe(self.ln2(x))
        return x


class TinyMoELM(nn.Module):
    def __init__(self, cfg: MoEConfig) -> None:
        super().__init__()
        self.cfg     = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.n_embd)
        self.drop    = nn.Dropout(cfg.dropout)
        self.blocks  = nn.ModuleList([MoEBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f    = nn.LayerNorm(cfg.n_embd)
        self.head    = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None, return_hidden: bool = False) -> LMOutput:
        B, T = x.shape
        pos    = torch.arange(T, device=x.device)
        h      = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        aux    = torch.tensor(0.0, device=x.device)
        for block in self.blocks:
            h  = block(h)
            if isinstance(block, MoEBlock) and block.moe.aux_loss is not None:
                aux = aux + block.moe.aux_loss
        hidden = self.ln_f(h)
        logits = self.head(hidden)
        loss   = None
        if targets is not None:
            ce   = F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size), targets.reshape(-1), ignore_index=-100)
            loss = ce + 0.01 * aux / self.cfg.n_layer
        return LMOutput(loss=loss, logits=logits, hidden_states=hidden if return_hidden else None)


# =============================================================================
# 6. Mamba-like SSM (pure PyTorch, no mamba_ssm dependency)
# =============================================================================

@dataclass
class MambaLikeConfig:
    vocab_size:  int   = 50257
    n_embd:      int   = 256
    n_layer:     int   = 6
    d_state:     int   = 16    # SSM state dimension
    d_conv:      int   = 4     # local conv kernel
    expand:      int   = 2     # inner expansion factor
    dropout:     float = 0.1
    max_seq_len: int   = 256
    tie_weights: bool  = True


class SelectiveSSM(nn.Module):
    """Input-selective state space model layer (simplified Mamba).

    Computes: h_t = A_bar_t * h_{t-1} + B_t * u_t
              y_t = C_t * h_t + D * u_t

    A is diagonal, learned as log-A; B, C, dt are input-dependent projections.
    Uses sequential scan — no external dependency.
    """
    def __init__(self, d_inner: int, d_state: int, d_conv: int, expand: int) -> None:
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state

        self.in_proj   = nn.Linear(d_inner // expand, d_inner * 2)
        self.conv1d    = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv - 1, groups=d_inner)
        self.x_proj    = nn.Linear(d_inner, d_state * 2 + 1)          # B, C, log_dt
        self.dt_proj   = nn.Linear(1, d_inner, bias=True)
        self.out_proj  = nn.Linear(d_inner, d_inner // expand)

        A = torch.arange(1, d_state + 1, dtype=torch.float).unsqueeze(0).expand(d_inner, -1)
        self.A_log     = nn.Parameter(torch.log(A))
        self.D         = nn.Parameter(torch.ones(d_inner))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        xz      = self.in_proj(x)                                      # (B, L, 2*d_inner)
        x_in, z = xz.split(self.d_inner, dim=-1)

        # Local convolution
        x_conv  = self.conv1d(x_in.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_conv  = F.silu(x_conv)

        # SSM parameters from input
        ssm_in  = self.x_proj(x_conv)                                  # (B, L, 2N+1)
        B_mat, C_mat, log_dt = ssm_in.split([self.d_state, self.d_state, 1], dim=-1)
        dt      = F.softplus(self.dt_proj(log_dt))                     # (B, L, d_inner)

        A       = -torch.exp(self.A_log)                               # (d_inner, N) negative (stable: A<0)

        # Discretize: log_A_bar = dt * A  (A<0 so log_A_bar<0, A_bar in (0,1))
        log_A_bar = dt.unsqueeze(-1) * A                               # (B, L, d_inner, N)
        B_bar     = dt.unsqueeze(-1) * B_mat.unsqueeze(2)              # (B, L, d_inner, N)
        u         = B_bar * x_conv.unsqueeze(-1)                       # (B, L, d_inner, N) = B_bar * x

        # Parallel scan via cumulative products (exact, O(L) vectorized)
        #   h_t = Σ_{s≤t} exp(log_A_bar[s+1:t+1].sum()) * u_s
        # Let P_t = cumsum(log_A_bar) along L -> cumulative log-product
        # Then h_t = exp(P_t) * Σ_{s≤t} exp(-P_s) * u_s
        # Compute in stable form by clamping max.
        log_A_cum = torch.cumsum(log_A_bar, dim=1)                     # (B, L, d_inner, N) ≤ 0
        # Stable: h_t = exp(log_A_cum_t) * cumsum(u_s * exp(-log_A_cum_s))
        # Since log_A_cum ≤ 0, exp(-log_A_cum) can be large; use shifted form:
        # h_t = Σ_{s≤t} exp(log_A_cum_t - log_A_cum_s) * u_s
        # Approximate efficiently: since A_bar ∈ (0,1), build decay weights and matmul per-seq.
        # Use the identity h = exp(log_A_cum) * cumsum(u * exp(-log_A_cum))
        # For numerical stability subtract running max over s≤t:
        A_cum_exp = torch.exp(log_A_cum)                               # P_t = Π A_bar_k for k≤t, in (0,1]
        # To avoid divide-by-small: use the shifted cumsum with an epsilon.
        eps = 1e-20
        inv_A_cum = torch.exp(-log_A_cum.clamp(min=-30))               # bounded
        cum       = torch.cumsum(u * inv_A_cum, dim=1)                 # (B, L, d_inner, N)
        h_all     = A_cum_exp * cum                                    # (B, L, d_inner, N)

        # Output: y_t = (h_t * C_t).sum(N) + D * x_t
        y       = (h_all * C_mat.unsqueeze(2)).sum(-1)                 # (B, L, d_inner)
        y       = y + self.D * x_conv
        y       = y * F.silu(z)
        return self.out_proj(y)


class MambaLikeBlock(nn.Module):
    def __init__(self, cfg: MambaLikeConfig) -> None:
        super().__init__()
        d_inner   = cfg.n_embd * cfg.expand
        self.ln   = nn.LayerNorm(cfg.n_embd)
        self.ssm  = SelectiveSSM(d_inner, cfg.d_state, cfg.d_conv, cfg.expand)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop(self.ssm(self.ln(x)))


class MambaLikeLM(nn.Module):
    def __init__(self, cfg: MambaLikeConfig) -> None:
        super().__init__()
        self.cfg     = cfg
        self.embed   = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks  = nn.ModuleList([MambaLikeBlock(cfg) for _ in range(cfg.n_layer)])
        self.ln_f    = nn.LayerNorm(cfg.n_embd)
        self.head    = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.embed.weight
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None, return_hidden: bool = False) -> LMOutput:
        h      = self.embed(x)
        for block in self.blocks:
            h  = block(h)
        hidden = self.ln_f(h)
        logits = self.head(hidden)
        loss   = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size), targets.reshape(-1), ignore_index=-100)
        return LMOutput(loss=loss, logits=logits, hidden_states=hidden if return_hidden else None)


# =============================================================================
# Hardware profiles + registry
# =============================================================================

RECOMMENDED_CONFIGS = {
    "paper_16m": {
        # Parameter-matched for the shared 8,192-token dialogue BPE.
        # Actual counts are logged because changing vocab_size changes totals.
        "gru":        dict(vocab_size=8192, embed_dim=512, hidden_size=512, num_layers=7, dropout=0.2),
        "awdlstm":    dict(vocab_size=8192, embed_dim=512, hidden_size=512, num_layers=5, wdrop=0.3, dropout=0.2, dropouth=0.15, dropouti=0.15),
        "gpt":        dict(vocab_size=8192, n_embd=384, n_head=6, n_layer=7, dropout=0.1, max_seq_len=256),
        "prefix_gpt": dict(vocab_size=8192, n_embd=384, n_head=6, n_layer=6, dropout=0.1, max_seq_len=256, prefix_length=8, cond_dim=8, condition_mode="aligned"),
        "moe":        dict(vocab_size=8192, n_embd=248, n_head=4, n_layer=6, num_experts=4, top_k=2, dropout=0.1, max_seq_len=256),
        "mamba_like": dict(vocab_size=8192, n_embd=400, n_layer=12, d_state=16, d_conv=4, expand=2, dropout=0.1, max_seq_len=256),
    },
    "m1_small": {
        # ~5-15M params; fits in 2 GB RAM, fast on MPS
        "gru":        dict(vocab_size=50257, embed_dim=256,  hidden_size=512,  num_layers=3, dropout=0.3),
        "awdlstm":    dict(vocab_size=50257, embed_dim=256,  hidden_size=512,  num_layers=2, wdrop=0.5, dropout=0.4, dropouth=0.25, dropouti=0.4),
        "gpt":        dict(vocab_size=50257, n_embd=256, n_head=4, n_layer=4,  dropout=0.1, max_seq_len=256),
        "prefix_gpt": dict(vocab_size=50257, n_embd=256, n_head=4, n_layer=4,  dropout=0.1, max_seq_len=256, prefix_length=8, cond_dim=8, condition_mode="ocean_vad"),
        "moe":        dict(vocab_size=50257, n_embd=256, n_head=4, n_layer=4,  num_experts=4, top_k=2, dropout=0.1, max_seq_len=256),
        "mamba_like": dict(vocab_size=50257, n_embd=256, n_layer=6, d_state=16, d_conv=4, expand=2, dropout=0.1, max_seq_len=256),
    },
    "rtx4070_small": {
        # ~20-50M params; good VRAM utilization on 8 GB
        "gru":        dict(vocab_size=50257, embed_dim=512,  hidden_size=1024, num_layers=3, dropout=0.3),
        "awdlstm":    dict(vocab_size=50257, embed_dim=400,  hidden_size=1150, num_layers=3, wdrop=0.5, dropout=0.4, dropouth=0.25, dropouti=0.65),
        "gpt":        dict(vocab_size=50257, n_embd=512, n_head=8, n_layer=8,  dropout=0.1, max_seq_len=512),
        "prefix_gpt": dict(vocab_size=50257, n_embd=512, n_head=8, n_layer=8,  dropout=0.1, max_seq_len=512, prefix_length=8, cond_dim=8, condition_mode="ocean_vad"),
        "moe":        dict(vocab_size=50257, n_embd=512, n_head=8, n_layer=8,  num_experts=8, top_k=2, dropout=0.1, max_seq_len=512),
        "mamba_like": dict(vocab_size=50257, n_embd=512, n_layer=12, d_state=16, d_conv=4, expand=2, dropout=0.1, max_seq_len=512),
    },
}


def build_model(arch: str, cfg_dict: dict) -> nn.Module:
    """Instantiate a model from arch name and a flat config dict."""
    a = arch.lower()
    if a == "gru":        return SmallGRULM(GRUConfig(**cfg_dict))
    if a == "awdlstm":    return AWDLSTMLM(AWDLSTMConfig(**cfg_dict))
    if a == "gpt":        return TinyGPTLM(GPTConfig(**cfg_dict))
    if a == "prefix_gpt": return PrefixTinyGPTLM(PrefixGPTConfig(**cfg_dict))
    if a == "moe":        return TinyMoELM(MoEConfig(**cfg_dict))
    if a == "mamba_like": return MambaLikeLM(MambaLikeConfig(**cfg_dict))
    raise ValueError(f"Unknown arch: {arch!r}. Choose: gru, awdlstm, gpt, prefix_gpt, moe, mamba_like")
