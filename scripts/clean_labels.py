#!/usr/bin/env python3
"""Clean corrupted labels in existing packaged/split JSONL files.

Applies the same normalize_labels() logic used during data generation
to fix known teacher LLM output errors:
  - response_policy typos: defect→deflect, challenged→challenge
  - stance level corruption: H+→level=H/delta=+, VH(-)→level=VH/delta=-
  - whitespace/trailing garbage in label values

Usage:
    python scripts/clean_labels.py [--dry-run]
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_gen.labeler import normalize_labels, _normalize_stance_entry, STANCE_DIMS


def clean_record(record: dict) -> tuple[dict, int]:
    """Normalize labels in a single JSONL record. Returns (record, n_fixes)."""
    fixes = 0
    labels = record.get("labels", {})
    if not labels:
        return record, 0

    original = json.dumps(labels, sort_keys=True)
    normalized = normalize_labels(labels)
    record["labels"] = normalized

    if json.dumps(normalized, sort_keys=True) != original:
        fixes = 1

    return record, fixes


def clean_file(path: Path, dry_run: bool = False) -> dict:
    """Clean all records in a JSONL file. Returns stats dict."""
    records = []
    total = 0
    fixed = 0
    field_fixes = Counter()

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            total += 1

            labels = record.get("labels", {})
            old_labels = json.dumps(labels, sort_keys=True)
            record, n = clean_record(record)
            new_labels = json.dumps(record.get("labels", {}), sort_keys=True)

            if old_labels != new_labels:
                fixed += 1
                # Track which fields changed
                old = json.loads(old_labels)
                new = json.loads(new_labels)
                for k in set(list(old.keys()) + list(new.keys())):
                    if json.dumps(old.get(k), sort_keys=True) != json.dumps(new.get(k), sort_keys=True):
                        field_fixes[k] += 1

            records.append(record)

    if not dry_run and fixed > 0:
        with open(path, "w") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {"total": total, "fixed": fixed, "field_fixes": dict(field_fixes)}


def main():
    parser = argparse.ArgumentParser(description="Clean corrupted labels in JSONL files")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing")
    args = parser.parse_args()

    files_to_clean = []
    for pattern in ["data/packaged/*.jsonl", "data/splits/*.jsonl"]:
        files_to_clean.extend(Path(".").glob(pattern))

    if not files_to_clean:
        print("No JSONL files found in data/packaged/ or data/splits/")
        return

    total_fixes = 0
    for path in sorted(files_to_clean):
        stats = clean_file(path, dry_run=args.dry_run)
        if stats["fixed"] > 0:
            mode = "[DRY RUN] " if args.dry_run else ""
            print(f"{mode}{path}: {stats['fixed']}/{stats['total']} records fixed")
            for field, count in sorted(stats["field_fixes"].items(), key=lambda x: -x[1]):
                print(f"    {field}: {count} fixes")
            total_fixes += stats["fixed"]
        else:
            print(f"  {path}: clean ({stats['total']} records)")

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Total: {total_fixes} records fixed across {len(files_to_clean)} files")


if __name__ == "__main__":
    main()
