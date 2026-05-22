#!/usr/bin/env python3
"""Analyze SLM training data distributions."""
import json
from pathlib import Path
from collections import Counter
import statistics

SLM_DATA = Path(__file__).resolve().parent.parent / "slm_training" / "data" / "dialogue"

def analyze_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records

def main():
    # Analyze from_gen_train.jsonl which has rich metadata
    train = analyze_jsonl(SLM_DATA / "from_gen_train.jsonl")
    val = analyze_jsonl(SLM_DATA / "from_gen_val.jsonl")

    print(f"from_gen_train records: {len(train)}")
    print(f"from_gen_val records: {len(val)}")

    for name, records in [("train", train), ("val", val)]:
        print(f"\n=== {name.upper()} ===")
        npcs = Counter(r["npc_id"] for r in records)
        scenarios = Counter(r.get("metadata", {}).get("scenario_type", "unknown") for r in records)
        settings = Counter(r.get("metadata", {}).get("setting", "unknown") for r in records)
        policies = Counter(r.get("metadata", {}).get("response_policy", "unknown") for r in records)
        reveals = Counter(r.get("metadata", {}).get("reveal_decision", "unknown") for r in records)
        valences = Counter(r.get("metadata", {}).get("valence", "unknown") for r in records)
        arousals = Counter(r.get("metadata", {}).get("arousal", "unknown") for r in records)

        response_lens = [len(r["target_response"].split()) for r in records]
        ctx_lens = [sum(len(turn["text"].split()) for turn in r["dialogue_context"]) for r in records]

        print(f"Records: {len(records)}")
        print(f"Unique NPCs: {len(npcs)}")
        print(f"Scenario distribution:")
        for s, c in scenarios.most_common():
            print(f"  {s:25s}: {c:5d} ({100*c/len(records):5.1f}%)")
        print(f"Setting distribution:")
        for s, c in settings.most_common():
            print(f"  {s:25s}: {c:5d} ({100*c/len(records):5.1f}%)")
        print(f"Response policy:")
        for s, c in policies.most_common():
            print(f"  {s:25s}: {c:5d} ({100*c/len(records):5.1f}%)")
        print(f"Reveal decision:")
        for s, c in reveals.most_common():
            print(f"  {s:25s}: {c:5d} ({100*c/len(records):5.1f}%)")
        print(f"Valence: {dict(valences)}")
        print(f"Arousal: {dict(arousals)}")
        print(f"Response length: mean={statistics.mean(response_lens):.1f}, max={max(response_lens)}, min={min(response_lens)}")
        print(f"Context length: mean={statistics.mean(ctx_lens):.1f}, max={max(ctx_lens)}, min={min(ctx_lens)}")

        # Save report
        report = {
            "n_records": len(records),
            "n_npcs": len(npcs),
            "scenario_types": dict(scenarios),
            "settings": dict(settings),
            "response_policy": dict(policies),
            "reveal_decision": dict(reveals),
            "valence": dict(valences),
            "arousal": dict(arousals),
            "response_length": {"mean": statistics.mean(response_lens), "min": min(response_lens), "max": max(response_lens)},
            "context_length": {"mean": statistics.mean(ctx_lens), "min": min(ctx_lens), "max": max(ctx_lens)},
        }
        out = Path(__file__).resolve().parent.parent / "eval_results" / f"slm_{name}_distribution.json"
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Saved report to {out}")

if __name__ == "__main__":
    main()
