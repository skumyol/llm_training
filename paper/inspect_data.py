# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
Inspect a JSONL file to guess field names and print the first record.
Use this to verify your data matches what human_audit_app.py expects.

Usage:
    uv run inspect_data.py path/to/test_heads.jsonl
"""

import json
import sys
from pathlib import Path


def inspect(path: Path):
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Total records: {len(records)}")
    if not records:
        return

    first = records[0]
    print("\n--- Top-level keys ---")
    for k in sorted(first.keys()):
        v = first[k]
        preview = str(v)[:120].replace("\n", " ")
        print(f"  {k}: {type(v).__name__} = {preview}")

    # Guess scenario distribution
    if "scenario_type" in first:
        from collections import Counter
        counts = Counter(r.get("scenario_type", "unknown") for r in records)
        print("\n--- Scenario distribution ---")
        for sc, n in counts.most_common():
            print(f"  {sc}: {n}")

    # Guess labels keys
    if "labels" in first and isinstance(first["labels"], dict):
        print("\n--- Label keys ---")
        for k in sorted(first["labels"].keys()):
            print(f"  {k}")

    print("\n--- Sample record (pretty-printed) ---")
    print(json.dumps(first, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path.jsonl>")
        sys.exit(1)
    inspect(Path(sys.argv[1]))
