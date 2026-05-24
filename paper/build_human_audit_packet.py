# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Build the human-audit input file from the prepared audit JSONL.

The audit interface should show annotators coherent natural dialogue turns, not
counterfactual variants with duplicated turn IDs or stale histories. This script:

1. keeps only non-counterfactual rows,
2. removes exact duplicates and player/NPC echo rows,
3. normalizes teacher labels to the choices exposed by human_audit_app.py,
4. rebuilds dialogue_history from canonical prior turns within each episode,
5. writes stable unique turn_id values.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


STANCE_DIMS = ("trust_level", "familiarity_level")
VALID_LEVELS = {"VL", "L", "N", "H", "VH"}
VALID_SECRECY = {"low", "medium", "high"}
VALID_RESPONSE_POLICY = {
    "answer",
    "withhold",
    "deflect",
    "clarify",
    "soothe",
    "challenge",
    "threaten",
    "negotiate",
    "test",
    "partial",
}
VALID_REPAIR = {"none", "soften", "apologize", "clarify", "redirect"}

RESPONSE_POLICY_MAP = {
    "defect": "deflect",
    "redirect": "deflect",
    "reveal": "answer",
    "ignore": "withhold",
    "evade": "deflect",
    "avoid": "deflect",
    "confront": "challenge",
    "reassure": "soothe",
    "comfort": "soothe",
    "bargain": "negotiate",
    "warn": "threaten",
    "deny": "withhold",
}


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip().lower())
    return text.strip("\"' ")


def normalize_stance(value: object) -> str:
    raw = str(value or "N").strip().upper()
    match = re.match(r"^(VL|VH|[LNHV])", raw)
    if not match:
        return "N"
    level = match.group(1)
    if level == "V":
        return "VH"
    return level if level in VALID_LEVELS else "N"


def normalize_labels(labels: dict) -> dict:
    labels = dict(labels or {})

    secrecy = str(labels.get("secrecy_pressure", "")).strip().lower()
    labels["secrecy_pressure"] = secrecy if secrecy in VALID_SECRECY else "medium"

    policy = str(labels.get("response_policy", "")).strip().lower()
    policy = RESPONSE_POLICY_MAP.get(policy, policy)
    labels["response_policy"] = policy if policy in VALID_RESPONSE_POLICY else "deflect"

    repair = str(labels.get("repair_strategy", "")).strip().lower()
    labels["repair_strategy"] = repair if repair in VALID_REPAIR else "none"

    for dim in STANCE_DIMS:
        labels[dim] = normalize_stance(labels.get(dim))

    return labels


def rebuild_histories(records: list[dict]) -> list[dict]:
    by_episode: dict[str, list[dict]] = {}
    for rec in records:
        by_episode.setdefault(rec["episode_id"], []).append(rec)

    rebuilt = []
    for episode_id, turns in by_episode.items():
        turns = sorted(turns, key=lambda r: int(r.get("turn_number", 0)))
        history_lines: list[str] = []
        for rec in turns:
            rec = dict(rec)
            rec["dialogue_history"] = "\n".join(history_lines)
            rec["turn_id"] = f"{episode_id}_{rec.get('turn_number')}"
            rebuilt.append(rec)
            history_lines.append(f"Player: {rec.get('player_utterance', '')}")
            history_lines.append(f"NPC: {rec.get('npc_response', '')}")
    return rebuilt


def build_packet(input_path: Path, output_path: Path) -> dict:
    records = load_jsonl(input_path)

    kept = []
    seen = set()
    stats = Counter()
    for rec in records:
        stats["input_rows"] += 1
        if rec.get("counterfactual"):
            stats["counterfactual_skipped"] += 1
            continue
        if normalize_text(rec.get("player_utterance", "")) == normalize_text(rec.get("npc_response", "")):
            stats["echo_skipped"] += 1
            continue

        rec = dict(rec)
        rec["labels"] = normalize_labels(rec.get("labels", {}))
        fingerprint = json.dumps(rec, sort_keys=True, ensure_ascii=False)
        if fingerprint in seen:
            stats["exact_duplicates_skipped"] += 1
            continue
        seen.add(fingerprint)
        kept.append(rec)

    kept = rebuild_histories(kept)
    kept.sort(key=lambda r: (r.get("scenario_type", ""), r.get("episode_id", ""), int(r.get("turn_number", 0))))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats["output_rows"] = len(kept)
    stats["unique_episodes"] = len({r["episode_id"] for r in kept})
    for scenario, count in Counter(r.get("scenario_type", "unknown") for r in kept).items():
        stats[f"scenario_{scenario}"] = count
    return dict(stats)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("audit_input.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("audit_input_clean.jsonl"))
    args = parser.parse_args()

    stats = build_packet(args.input, args.output)
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
