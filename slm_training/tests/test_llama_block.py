"""TinyLlamaLM must be a drop-in for TinyGPTLM at matched parameter count.

Guards the three things most likely to be silently wrong in a hand-written
modern block: RoPE tensor shapes, causality, and whether SwiGLU actually keeps
the parameter budget level with the 4x GELU MLP it replaces (if it does not,
any comparison between the two measures capacity, not architecture).

    python slm_training/tests/test_llama_block.py
"""
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "src" / "train")]

import torch
from small_lm_architectures import build_model

CFG = dict(vocab_size=16384, max_seq_len=256, n_embd=512, n_head=8, n_layer=6, dropout=0.0)


def main():
    gpt = build_model("gpt", dict(CFG))
    lla = build_model("llama", dict(CFG))

    n_gpt = sum(p.numel() for p in gpt.parameters())
    n_lla = sum(p.numel() for p in lla.parameters())
    # llama drops the learned position table (max_seq_len x n_embd) and all biases
    drift = abs(n_lla - n_gpt) / n_gpt
    assert drift < 0.03, f"parameter budgets differ by {drift:.1%}: {n_gpt} vs {n_lla}"

    x = torch.randint(0, CFG["vocab_size"], (2, 64))
    y = torch.randint(0, CFG["vocab_size"], (2, 64))
    out = lla(x, y)
    assert out.logits.shape == (2, 64, CFG["vocab_size"]), out.logits.shape
    assert torch.isfinite(out.loss), "non-finite loss on init"
    out.loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in lla.parameters() if p.requires_grad), "bad grads"

    lla.eval()
    with torch.no_grad():
        # causality: changing a later token must not move an earlier position
        a = lla(x).logits
        x2 = x.clone()
        x2[:, -1] = (x2[:, -1] + 1) % CFG["vocab_size"]
        b = lla(x2).logits
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-4), "attention is not causal"

    # RoPE has no learned length limit, so a longer context must still run
    with torch.no_grad():
        long = lla(torch.randint(0, CFG["vocab_size"], (1, 384))).logits
    assert long.shape[1] == 384 and torch.isfinite(long).all(), "RoPE failed to extend"

    print(f"ok: gpt={n_gpt:,}  llama={n_lla:,}  ({drift:+.1%})  causal, finite, extends to 384")


if __name__ == "__main__":
    main()
