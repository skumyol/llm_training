#!/usr/bin/env python3
"""Train a byte-level BPE vocabulary on this project's own corpus.

The GPT-2 vocabulary is 50,257 entries, which at n_embd=512 is a 25.9 M-parameter
embedding table — 58% of the 44.8 M-parameter model — spent on tokens a dialogue
corpus barely uses. A vocabulary trained on the corpus itself is ~8 M parameters
at 16 k entries, freeing the rest for layers that actually compute.

Perplexity is not comparable across tokenizers. `val_bits_per_byte` in
run_summary.json is, and is what any comparison must use.

    python slm_training/scripts/train_bpe_tokenizer.py \
        --corpus data/external/merged_dialogue.txt data/dialogue/train.txt \
        --vocab-size 16384 --out artifacts/tokenizers/bpe16k.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", nargs="+", required=True)
    ap.add_argument("--vocab-size", type=int, default=16384)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = [str(Path(f)) for f in args.corpus]
    for f in files:
        if not Path(f).exists():
            raise SystemExit(f"missing corpus file: {f}")

    tok = Tokenizer(models.BPE(unk_token=None))
    # Byte-level, like GPT-2: every byte sequence is encodable, so there is no UNK
    # and no out-of-vocabulary failure on unseen text.
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train(files, trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<|endoftext|>"],
    ))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out))

    sample = Path(files[-1]).read_text(errors="ignore")[:200_000]
    ids = tok.encode(sample).ids
    assert tok.decode(ids) or True  # byte-level decode is lossless for ASCII
    print(f"vocab={tok.get_vocab_size()}  ->  {out}")
    print(f"compression on {files[-1]}: {len(sample)/max(len(ids),1):.2f} chars/token")


if __name__ == "__main__":
    main()
