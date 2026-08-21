#!/usr/bin/env python3
"""
make_val_test_split.py — carve a reporting test set out of val.txt.

The SLM pipeline previously had only train/val, so every reported val_ppl was
also the number that chose the checkpoint (early stopping + best-model saving).
This splits val.txt into two disjoint files:

    val_sel.txt   used for checkpoint selection during training
    test.txt      never seen by selection; the number to report

Blocks are grouped by their opening PLAYER line before splitting. The corpus
expands each dialogue into growing-prefix blocks that share an opener, so a
naive per-block split would put a block and its own prefix on opposite sides.

Usage:
    python scripts/make_val_test_split.py [--val data/dialogue/val.txt] [--seed 42]
"""
from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path


def split_blocks(val_path: Path, seed: int, test_frac: float) -> tuple[list[str], list[str]]:
    blocks = [b.strip() for b in val_path.read_text().split("\n\n") if b.strip()]

    groups: dict[str, list[str]] = defaultdict(list)
    for b in blocks:
        groups[b.splitlines()[0]].append(b)

    keys = sorted(groups)                 # sorted first => deterministic given seed
    random.Random(seed).shuffle(keys)

    # Assign whole groups, greedily filling test until it reaches test_frac of
    # the character mass. Balancing on characters (not group count) keeps the two
    # sides comparable in size even though groups vary a lot in length.
    total_chars = sum(len(b) for b in blocks)
    target = total_chars * test_frac

    test_keys: set[str] = set()
    acc = 0
    for k in keys:
        if acc >= target:
            break
        test_keys.add(k)
        acc += sum(len(b) for b in groups[k])

    test = [b for k in keys if k in test_keys for b in groups[k]]
    sel = [b for k in keys if k not in test_keys for b in groups[k]]
    return sel, test


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=Path, default=Path("data/dialogue/val.txt"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-frac", type=float, default=0.5)
    args = ap.parse_args()

    sel, test = split_blocks(args.val, args.seed, args.test_frac)

    out_dir = args.val.parent
    (out_dir / "val_sel.txt").write_text("\n\n".join(sel) + "\n")
    (out_dir / "test.txt").write_text("\n\n".join(test) + "\n")

    sel_openers = {b.splitlines()[0] for b in sel}
    test_openers = {b.splitlines()[0] for b in test}
    overlap = sel_openers & test_openers
    assert not overlap, f"opener leaked across split: {list(overlap)[:3]}"

    sel_npc = {l for b in sel for l in b.splitlines() if l.startswith("NPC:")}
    test_npc = {l for b in test for l in b.splitlines() if l.startswith("NPC:")}
    assert not (sel_npc & test_npc), "NPC line leaked across split"

    print(f"val_sel.txt  blocks={len(sel):4d}  chars={sum(len(b) for b in sel):,}  openers={len(sel_openers)}")
    print(f"test.txt     blocks={len(test):4d}  chars={sum(len(b) for b in test):,}  openers={len(test_openers)}")
    print("disjoint openers and NPC lines: OK")


if __name__ == "__main__":
    main()
