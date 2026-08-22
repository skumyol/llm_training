"""bits-per-byte must be comparable across tokenizers; perplexity is not.

Two tokenizers that encode the same text at different granularities give
different per-token cross-entropy for the same underlying model quality. Dividing
by bytes removes the tokenizer from the metric — which is the only reason the
16k-vocab runs can be compared against the GPT-2-vocab ones at all.

    python slm_training/tests/test_bpc_invariance.py
"""
import math

LN2 = math.log(2)


def bpc(mean_nll_per_token: float, bytes_per_token: float) -> float:
    return mean_nll_per_token / (LN2 * bytes_per_token)


def main():
    text_bytes = 100_000
    # same model quality, expressed at two tokenizer granularities
    true_bpc = 1.30
    for bpt in (4.08, 4.10, 2.0, 8.0):
        n_tokens = text_bytes / bpt
        total_nll = true_bpc * LN2 * text_bytes
        mean_nll = total_nll / n_tokens
        got = bpc(mean_nll, bpt)
        assert abs(got - true_bpc) < 1e-9, f"bpc drifted at {bpt} bytes/token: {got}"
        # perplexity, by contrast, must move a lot
        ppl = math.exp(mean_nll)
        print(f"bytes/token={bpt:5.2f}  ppl={ppl:9.2f}  bpc={got:.4f}")

    # coarser tokenization must give higher per-token ppl at equal bpc
    p_fine = math.exp(true_bpc * LN2 * 2.0)
    p_coarse = math.exp(true_bpc * LN2 * 8.0)
    assert p_coarse > p_fine, "sanity: coarser tokens carry more information each"
    print("ok: bpc invariant across tokenizers, ppl is not")


if __name__ == "__main__":
    main()
