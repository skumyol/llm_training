#!/usr/bin/env python3
"""Compute training/testing data distribution metrics for the paper."""
import json
from pathlib import Path
from collections import Counter, defaultdict
import statistics

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "splits"
EVAL_DIR = Path(__file__).resolve().parent.parent / "eval_results"

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def analyze_split(turns, name):
    stats = {
        "name": name,
        "n_turns": len(turns),
        "n_episodes": len(set(t["episode_id"] for t in turns)),
        "scenario_types": Counter(t["scenario_type"] for t in turns),
        "turns_per_episode": Counter(t["episode_id"] for t in turns),
        "turn_lengths": [],
        "label_counts": defaultdict(Counter),
    }
    for turn in turns:
        ctx = turn.get("context", "")
        stats["turn_lengths"].append(len(ctx.split()))
        labels = turn.get("labels", {})
        for field, val in labels.items():
            if isinstance(val, list):
                for v in val:
                    stats["label_counts"][field][v] += 1
            else:
                stats["label_counts"][field][val] += 1
    return stats

def main():
    splits = {}
    for split in ["train", "val", "test"]:
        path = DATA_DIR / f"{split}_heads.jsonl"
        if not path.exists():
            print(f"WARNING: {path} not found, skipping")
            continue
        turns = load_jsonl(path)
        splits[split] = analyze_split(turns, split)
        print(f"\n=== {split.upper()} ===")
        print(f"Turns: {splits[split]['n_turns']}, Episodes: {splits[split]['n_episodes']}")
        print("Scenario distribution:")
        for scen, cnt in sorted(splits[split]['scenario_types'].items()):
            pct = 100 * cnt / splits[split]['n_turns']
            print(f"  {scen:25s}: {cnt:4d} ({pct:5.1f}%)")
        tpe = list(splits[split]['turns_per_episode'].values())
        print(f"Turns per episode: mean={statistics.mean(tpe):.1f}, max={max(tpe)}, min={min(tpe)}")
        print(f"Context word length: mean={statistics.mean(splits[split]['turn_lengths']):.0f}, max={max(splits[split]['turn_lengths'])}")

    # Label distributions across all splits combined
    all_label_counts = defaultdict(Counter)
    for split_stats in splits.values():
        for field, counter in split_stats["label_counts"].items():
            all_label_counts[field].update(counter)

    print("\n=== LABEL DISTRIBUTIONS (all splits) ===")
    for field in sorted(all_label_counts.keys()):
        counter = all_label_counts[field]
        total = sum(counter.values())
        print(f"\n{field} (n={total}):")
        for val, cnt in counter.most_common():
            print(f"  {val:20s}: {cnt:5d} ({100*cnt/total:5.1f}%)")

    # Save structured report
    report = {
        "splits": {
            name: {
                "n_turns": s["n_turns"],
                "n_episodes": s["n_episodes"],
                "scenario_types": dict(s["scenario_types"]),
                "turns_per_episode": {
                    "mean": statistics.mean(list(s["turns_per_episode"].values())),
                    "min": min(s["turns_per_episode"].values()),
                    "max": max(s["turns_per_episode"].values()),
                },
                "context_words": {
                    "mean": statistics.mean(s["turn_lengths"]),
                    "min": min(s["turn_lengths"]),
                    "max": max(s["turn_lengths"]),
                },
            }
            for name, s in splits.items()
        },
        "label_distributions": {
            field: dict(counter.most_common())
            for field, counter in all_label_counts.items()
        },
    }
    out_path = EVAL_DIR / "data_distribution_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report to {out_path}")

if __name__ == "__main__":
    main()
