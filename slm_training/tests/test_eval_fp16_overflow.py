"""evaluate() sums per-token losses outside the autocast block.

If those losses stay in fp16, the sum overflows 65504 for any validation set of
a few thousand tokens at a normal early-training loss, and val_loss — which
drives checkpoint selection — silently becomes inf.

    python slm_training/tests/test_eval_fp16_overflow.py
"""
import torch
import torch.nn.functional as F

VOCAB, N_TOKENS = 50257, 12455


def _losses(dtype):
    torch.manual_seed(0)
    logits = torch.randn(N_TOKENS, VOCAB, dtype=torch.float32) * 2.0
    y = torch.randint(0, VOCAB, (N_TOKENS,))
    return F.cross_entropy(logits.to(dtype), y, reduction="none")


def main():
    fp16_sum = float(_losses(torch.float16).sum())
    fp32_sum = float(_losses(torch.float16).float().sum())

    assert not torch.isfinite(torch.tensor(fp16_sum)), (
        f"expected fp16 accumulation to overflow, got {fp16_sum}; "
        "if this ever stops overflowing the guard in evaluate() can go"
    )
    assert torch.isfinite(torch.tensor(fp32_sum)), "fp32 accumulation must be finite"
    assert 0 < fp32_sum / N_TOKENS < 30, f"implausible mean nats: {fp32_sum / N_TOKENS}"
    print(f"ok: fp16 sum={fp16_sum}  fp32 sum={fp32_sum:.1f} "
          f"({fp32_sum / N_TOKENS:.2f} nats/token)")


if __name__ == "__main__":
    main()
