# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Preprocess test_heads.jsonl + test_sft.jsonl into the format expected by human_audit_app.py.

The audit app expects separate fields:
    scene, dialogue_history, player_utterance, npc_response

Our test_heads.jsonl stores everything in a single 'context' field and lacks the NPC response.
This script parses the context and merges in the NPC response from test_sft.jsonl.

Usage:
    uv run prepare_audit_data.py \
        --heads ../data/splits/test_heads.jsonl \
        --sft ../data/splits/test_sft.jsonl \
        --output ./audit_input.jsonl
"""

import argparse
import json
import sys
import re
from pathlib import Path


def parse_context(context: str) -> dict[str, str]:
    """Extract scene, history, and player utterance from the context string."""
    result = {"scene": "", "dialogue_history": "", "player_utterance": ""}

    # Extract scene
    scene_match = re.search(r"<scene>(.*?)</scene>", context, re.DOTALL)
    if scene_match:
        result["scene"] = scene_match.group(1).strip()

    # Extract history
    history_match = re.search(r"<history>(.*?)</history>", context, re.DOTALL)
    if history_match:
        result["dialogue_history"] = history_match.group(1).strip()

    # Extract player utterance (the "Player: ..." that appears AFTER </history>)
    # Split on </history> and look for the last "Player:" in the remainder
    parts = context.split("</history>", 1)
    after_history = parts[-1] if len(parts) > 1 else context
    player_match = re.search(r"Player:\s*(.+)$", after_history, re.DOTALL)
    if player_match:
        result["player_utterance"] = player_match.group(1).strip()

    return result


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description="Prepare audit data from test splits")
    parser.add_argument("--heads", required=True, type=Path, help="Path to test_heads.jsonl")
    parser.add_argument("--sft", required=True, type=Path, help="Path to test_sft.jsonl")
    parser.add_argument("--output", required=True, type=Path, help="Output path for audit_input.jsonl")
    args = parser.parse_args()

    heads_records = load_jsonl(args.heads)
    sft_records = load_jsonl(args.sft)

    if len(heads_records) != len(sft_records):
        raise ValueError(
            f"heads/sft length mismatch: {len(heads_records)} heads vs {len(sft_records)} sft"
        )

    output_records = []
    seen_records: set[str] = set()
    duplicate_count = 0
    for rec, sft_rec in zip(heads_records, sft_records):
        key = (rec["episode_id"], rec["turn_idx"])
        sft_key = (sft_rec["episode_id"], sft_rec["turn_idx"])
        if key != sft_key:
            raise ValueError(f"heads/sft alignment mismatch: heads={key} sft={sft_key}")

        npc_response = sft_rec.get("target", "")
        if not npc_response:
            print(f"Warning: empty NPC response for {key}", file=sys.stderr)

        parsed = parse_context(rec["context"])

        out_rec = {
            "episode_id": rec["episode_id"],
            "turn_id": rec.get("record_id", f"{rec['episode_id']}_{rec['turn_idx']}_{len(output_records)}"),
            "turn_number": rec["turn_idx"],
            "scenario_type": rec["scenario_type"],
            "scene": parsed["scene"],
            "dialogue_history": parsed["dialogue_history"],
            "player_utterance": parsed["player_utterance"],
            "npc_response": npc_response,
            "labels": rec["labels"],
            "counterfactual": rec.get("counterfactual", False),
        }
        fingerprint_rec = dict(out_rec)
        fingerprint_rec.pop("turn_id", None)
        fingerprint = json.dumps(fingerprint_rec, sort_keys=True, ensure_ascii=False)
        if fingerprint in seen_records:
            duplicate_count += 1
            continue
        seen_records.add(fingerprint)
        output_records.append(out_rec)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in output_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(output_records)} records to {args.output}")
    if duplicate_count:
        print(f"Skipped {duplicate_count} exact duplicate records", file=sys.stderr)


if __name__ == "__main__":
    main()
