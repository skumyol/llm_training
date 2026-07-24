#!/usr/bin/env python3
"""Train the shared byte-level BPE used by paper scratch-SLM experiments."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.small_lm_dialogue import train_dialogue_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()
    output = train_dialogue_tokenizer(
        args.input,
        args.output,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
    print(output)


if __name__ == "__main__":
    main()
