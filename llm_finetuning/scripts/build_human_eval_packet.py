#!/usr/bin/env python3
"""Build a blinded human-evaluation packet from sample_generations.json.

The packet randomizes gold/generated response order per item and writes:
- items JSONL for annotation
- CSV for spreadsheet-based rating
- answer key JSON for later analysis

Suggested rating dimensions:
- fluency
- relevance
- character_consistency
- social_state_consistency
- secrecy_safety
- overall_preference
"""

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


RATING_COLUMNS = [
    "fluency_a",
    "fluency_b",
    "relevance_a",
    "relevance_b",
    "character_consistency_a",
    "character_consistency_b",
    "social_state_consistency_a",
    "social_state_consistency_b",
    "secrecy_safety_a",
    "secrecy_safety_b",
    "overall_preference",
    "notes",
]


def _item_id(sample: dict, idx: int) -> str:
    raw = f"{idx}|{sample.get('prompt', '')}|{sample.get('gold', '')}|{sample.get('generated', '')}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _stratum(sample: dict) -> str:
    reveal = sample.get("reveal_decision") or "unknown"
    leak = "leak" if sample.get("secret_leak") else "clean"
    return f"{reveal}:{leak}"


def _prompt_snippet(prompt: str, max_chars: int) -> str:
    prompt = (prompt or "").strip()
    return prompt if len(prompt) <= max_chars else "..." + prompt[-max_chars + 3 :]


def build_packet(
    input_path: str,
    output_dir: str,
    n_items: int,
    seed: int,
    prompt_chars: int,
) -> None:
    with open(input_path) as f:
        samples = json.load(f)

    strata: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        if sample.get("gold") and sample.get("generated"):
            strata[_stratum(sample)].append((idx, sample))

    rng = random.Random(seed)
    selected: list[tuple[int, dict]] = []
    stratum_names = sorted(strata)
    while len(selected) < n_items and any(strata.values()):
        for name in stratum_names:
            if len(selected) >= n_items:
                break
            bucket = strata[name]
            if not bucket:
                continue
            pick_idx = rng.randrange(len(bucket))
            selected.append(bucket.pop(pick_idx))

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items_path = out_dir / "human_eval_items.jsonl"
    csv_path = out_dir / "human_eval_sheet.csv"
    key_path = out_dir / "human_eval_answer_key.json"

    answer_key = {}
    csv_rows = []

    with open(items_path, "w") as f:
        for idx, sample in selected:
            item_id = _item_id(sample, idx)
            gold_first = rng.random() < 0.5
            response_a = sample["gold"] if gold_first else sample["generated"]
            response_b = sample["generated"] if gold_first else sample["gold"]
            answer_key[item_id] = {
                "source_index": idx,
                "a": "gold" if gold_first else "generated",
                "b": "generated" if gold_first else "gold",
                "rouge_l": sample.get("rouge_l"),
                "secret_leak": sample.get("secret_leak"),
                "reveal_decision": sample.get("reveal_decision"),
            }
            item = {
                "item_id": item_id,
                "prompt_snippet": _prompt_snippet(sample.get("prompt") or sample.get("input_snippet", ""), prompt_chars),
                "response_a": response_a,
                "response_b": response_b,
                "reveal_decision": sample.get("reveal_decision"),
                "rating_scale": "1=bad, 3=acceptable, 5=excellent",
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            csv_rows.append({**item, **{col: "" for col in RATING_COLUMNS}})

    with open(csv_path, "w", newline="") as f:
        fieldnames = ["item_id", "prompt_snippet", "response_a", "response_b", "reveal_decision", "rating_scale", *RATING_COLUMNS]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    key_path.write_text(json.dumps(answer_key, indent=2, ensure_ascii=False))

    print(f"Human eval packet: {len(selected)} items")
    print(f"  Items: {items_path}")
    print(f"  CSV:   {csv_path}")
    print(f"  Key:   {key_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build blinded human-evaluation packet")
    parser.add_argument("--input", default="eval_results/sample_generations.json")
    parser.add_argument("--output-dir", default="eval_results/human_eval")
    parser.add_argument("--n-items", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt-chars", type=int, default=900)
    args = parser.parse_args()
    build_packet(args.input, args.output_dir, args.n_items, args.seed, args.prompt_chars)


if __name__ == "__main__":
    main()
