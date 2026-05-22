#!/usr/bin/env python3
"""Generate prediction examples and data examples for the paper."""
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "splits"
EVAL_DIR = Path(__file__).resolve().parent.parent / "eval_results"

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

def find_examples():
    gold = { (t["episode_id"], t["turn_idx"]): t for t in load_jsonl(DATA_DIR / "test_heads.jsonl") }
    pred = { (t["episode_id"], t["turn_idx"]): t for t in load_jsonl(EVAL_DIR / "predicted_zt_test.jsonl") }

    # Check available fields in predictions
    sample_pred = next(iter(pred.values()))
    pred_fields = set(k for k in sample_pred.keys() if k not in ("record_idx", "episode_id", "turn_idx"))
    print(f"Available prediction fields: {pred_fields}")

    # Fields to show in examples (subset of what predictions have)
    display_fields = sorted(pred_fields)

    examples = []
    for key, g in gold.items():
        p = pred.get(key)
        if not p:
            continue
        gl = g.get("labels", {})
        correct = {}
        for field in display_fields:
            if field in gl:
                correct[field] = gl[field] == p[field]

        score = sum(correct.values())
        total = len(correct)
        if total == 0:
            continue

        # We want examples with some correct and some incorrect (mixed difficulty)
        if 2 <= score < total - 1:
            examples.append({
                "key": key,
                "context": g["context"],
                "scenario": g["scenario_type"],
                "correct": correct,
                "score": score,
                "total": total,
                "labels": {k: {"gold": gl.get(k, "N/A"), "pred": p.get(k, "N/A")} for k in display_fields if k in gl},
            })

    # Sort by score descending, then pick diverse scenarios
    examples.sort(key=lambda x: (-x["score"], x["scenario"]))
    seen_scenarios = set()
    diverse = []
    for ex in examples:
        if ex["scenario"] not in seen_scenarios and len(diverse) < 4:
            diverse.append(ex)
            seen_scenarios.add(ex["scenario"])
    for ex in examples:
        if len(diverse) < 6 and ex not in diverse:
            diverse.append(ex)

    return diverse[:6]

def main():
    examples = find_examples()
    print(f"Found {len(examples)} diverse prediction examples")
    for i, ex in enumerate(examples):
        print(f"\nExample {i+1}: {ex['key']} scenario={ex['scenario']} score={ex['score']}/{ex['total']}")
        for field, vals in ex["labels"].items():
            mark = "OK" if vals["gold"] == vals["pred"] else "WRONG"
            print(f"  {field:20s}: gold={vals['gold']:15s} pred={vals['pred']:15s} {mark}")

    out_path = EVAL_DIR / "paper_prediction_examples.json"
    with open(out_path, "w") as f:
        json.dump(examples, f, indent=2)
    print(f"\nSaved to {out_path}")

if __name__ == "__main__":
    main()
