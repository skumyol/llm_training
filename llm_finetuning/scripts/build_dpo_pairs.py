#!/usr/bin/env python3
"""Build DPO preference pairs from response eval sample_generations.json.

Usage:
    python scripts/build_dpo_pairs.py \
        --input eval_results/sample_generations.json \
        --output data/splits/dpo_pairs.jsonl \
        --min-rouge-l 0.10 \
        --max-repeated-3gram 0.30 \
        --max-length-ratio 1.5

Logic:
- Gold text is always the chosen response (high-quality reference).
- Generated text becomes the rejected response if it fails at least one
  quality gate (low ROUGE-L, leakage, high repetition, excessive length).
- Prompt is extracted from input_snippet (last N chars of the structured
  prompt, up to a configurable max).

The output JSONL follows the TRL DPODataset format:
    {"prompt": str, "chosen": str, "rejected": str}
"""

import argparse
import json
import re
from pathlib import Path


def _extract_prompt(input_snippet: str, max_chars: int = 512) -> str:
    """Use the tail of the input snippet as the DPO prompt."""
    prompt = input_snippet.strip()
    if len(prompt) > max_chars:
        # Keep the end (most recent context) since it usually contains the
        # immediate speaker turn and conditioning fields.
        prompt = "..." + prompt[-max_chars + 3 :]
    return prompt


def _repeated_3gram_rate(text: str) -> float:
    tokens = re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)
    if len(tokens) < 3:
        return 0.0
    ngrams = [tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
    total = len(ngrams)
    if total == 0:
        return 0.0
    repeated = sum(c - 1 for c in {ng: ngrams.count(ng) for ng in set(ngrams)}.values() if c > 1)
    return repeated / total


def _length_ratio(gold: str, generated: str) -> float:
    g_len = len(generated.split())
    r_len = len(gold.split())
    if r_len == 0:
        return float("inf")
    return g_len / r_len


def build_pairs(
    input_path: str,
    output_path: str,
    min_rouge_l: float = 0.10,
    max_repeated_3gram: float = 0.30,
    max_length_ratio: float = 1.5,
    prompt_max_chars: int = 512,
) -> None:
    with open(input_path) as f:
        samples = json.load(f)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped_reasons: dict[str, int] = {}

    with open(out_path, "w") as out_f:
        for sample in samples:
            prompt_source = sample.get("prompt") or sample.get("input_snippet", "")
            prompt = _extract_prompt(prompt_source, prompt_max_chars)
            gold = sample.get("gold", "").strip()
            generated = sample.get("generated", "").strip()
            rouge_l = sample.get("rouge_l", 0.0)
            secret_leak = bool(sample.get("secret_leak", False))
            reveal_decision = sample.get("reveal_decision", "")

            if not prompt or not gold or not generated:
                skipped_reasons["empty_fields"] = skipped_reasons.get("empty_fields", 0) + 1
                continue

            # Determine if the generated sample is "bad enough" to be a reject.
            bad = False
            reasons: list[str] = []

            if rouge_l < min_rouge_l:
                bad = True
                reasons.append(f"rouge_l={rouge_l:.3f}<{min_rouge_l}")

            rep_rate = _repeated_3gram_rate(generated)
            if rep_rate > max_repeated_3gram:
                bad = True
                reasons.append(f"repetition={rep_rate:.3f}>{max_repeated_3gram}")

            lr = _length_ratio(gold, generated)
            if lr > max_length_ratio:
                bad = True
                reasons.append(f"length_ratio={lr:.2f}>{max_length_ratio}")

            # Leakage is always a reject, regardless of reveal_decision.
            # Even if reveal_decision != "none", a model that parrots secrets
            # is not the behaviour we want to reinforce.
            if secret_leak:
                bad = True
                reasons.append("secret_leak")

            if not bad:
                skipped_reasons["good_generation"] = skipped_reasons.get("good_generation", 0) + 1
                continue

            record = {
                "prompt": prompt,
                "chosen": gold,
                "rejected": generated,
                "metadata": {
                    "rouge_l": rouge_l,
                    "repetition_rate": rep_rate,
                    "length_ratio": lr,
                    "secret_leak": secret_leak,
                    "reveal_decision": reveal_decision,
                    "reject_reasons": reasons,
                },
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    total = len(samples)
    print(f"DPO pairs written: {kept} / {total}")
    print(f"  Skipped breakdown: {skipped_reasons}")
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build DPO preference pairs from response eval outputs")
    parser.add_argument("--input", required=True, help="Path to sample_generations.json")
    parser.add_argument("--output", required=True, help="Path to output DPO pairs JSONL")
    parser.add_argument("--min-rouge-l", type=float, default=0.10, help="ROUGE-L floor for reject")
    parser.add_argument("--max-repeated-3gram", type=float, default=0.30, help="Repetition ceiling for reject")
    parser.add_argument("--max-length-ratio", type=float, default=1.5, help="Generated/reference length ceiling")
    parser.add_argument("--prompt-max-chars", type=int, default=512, help="Max prompt chars to keep")
    args = parser.parse_args()
    build_pairs(
        args.input,
        args.output,
        min_rouge_l=args.min_rouge_l,
        max_repeated_3gram=args.max_repeated_3gram,
        max_length_ratio=args.max_length_ratio,
        prompt_max_chars=args.prompt_max_chars,
    )
