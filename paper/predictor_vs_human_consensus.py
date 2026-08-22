#!/usr/bin/env python3
"""
predictor_vs_human_consensus.py — predictor agreement where humans actually agree.

Overall predictor-human kappa is uninterpretable in this study: inter-annotator
kappa is ~0.01, so there is no stable human judgement for any model to agree
with, and the comparison has no ceiling.

This conditions on the subset where the two annotators independently chose the
SAME label. On those turns a reliable human judgement exists, so "does the
predictor match the humans" becomes answerable, and the teacher can be scored on
the same subset as a reference point.

Reported per field:
  n_consensus     turns where Human A == Human B
  consensus_rate  n_consensus / n_common (how often a human label exists at all)
  predictor_acc   predictor == human consensus, with a bootstrap CI
  teacher_acc     teacher == human consensus, on the same turns
  majority_acc    always predicting the most frequent consensus label

majority_acc is the floor: any accuracy at or below it is not evidence of
agreement, only of matching the marginal distribution.

Usage:
    python paper/predictor_vs_human_consensus.py \
        --predictor paper/audit_results/audit_predictor.jsonl \
        --human-a paper/audit_results/audit_654cfad67f990b0393b85132.jsonl \
        --human-b paper/audit_results/audit_67c87fc1b3ba111d0e1526a0.jsonl \
        --teacher paper/audit_input_clean.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

HEADS = [
    "valence", "arousal", "secrecy_pressure", "reveal_decision",
    "response_policy", "repair_strategy", "trust_level", "familiarity_level",
]


def load(path: Path) -> dict[str, dict]:
    out = {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        tid = r.get("turn_id")
        if tid is None and r.get("episode_id") is not None:
            tid = f"{r['episode_id']}_{r.get('turn_number', r.get('turn_idx'))}"
        if tid:
            out[str(tid)] = r.get("labels") or {}
    return out


def _norm(v):
    """Labels appear as scalars or single-element lists across sources."""
    if isinstance(v, list):
        v = v[0] if v else ""
    return str(v).strip().lower()


def boot_ci(hits: list[int], n_boot: int = 2000, seed: int = 13) -> tuple[float, float]:
    if not hits:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(hits)
    means = sorted(sum(hits[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictor", required=True)
    ap.add_argument("--human-a", required=True)
    ap.add_argument("--human-b", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--out", default="paper/audit_results/predictor_vs_consensus.json")
    args = ap.parse_args()

    P, A, B, T = (load(Path(p)) for p in (args.predictor, args.human_a, args.human_b, args.teacher))
    common = sorted(set(A) & set(B) & set(P) & set(T))
    print(f"turns common to predictor, both annotators and teacher: {len(common)}\n")

    print(f"{'field':19s} {'n_cons':>7s} {'cons_rate':>9s} {'pred_acc':>9s} "
          f"{'95% CI':>15s} {'teach_acc':>10s} {'major_acc':>10s}")
    print("-" * 86)

    results = {}
    for h in HEADS:
        pairs = [(t, _norm(A[t].get(h))) for t in common
                 if _norm(A[t].get(h)) == _norm(B[t].get(h)) and _norm(A[t].get(h)) != ""]
        n = len(pairs)
        if n == 0:
            print(f"{h:19s} {0:>7d}  (no turns where the annotators agree)")
            results[h] = {"n_consensus": 0}
            continue

        pred_hits = [int(_norm(P[t].get(h)) == lab) for t, lab in pairs]
        teach_hits = [int(_norm(T[t].get(h)) == lab) for t, lab in pairs]
        maj = Counter(lab for _, lab in pairs).most_common(1)[0][1] / n

        pa = sum(pred_hits) / n
        ta = sum(teach_hits) / n
        lo, hi = boot_ci(pred_hits)
        results[h] = {
            "n_consensus": n, "consensus_rate": n / len(common),
            "predictor_acc": pa, "predictor_ci": [lo, hi],
            "teacher_acc": ta, "majority_acc": maj,
        }
        print(f"{h:19s} {n:>7d} {n/len(common):>9.2f} {pa:>9.3f} "
              f"  [{lo:.3f},{hi:.3f}] {ta:>10.3f} {maj:>10.3f}")

    got = [r for r in results.values() if r.get("n_consensus")]
    if got:
        print("-" * 86)
        w = sum(r["n_consensus"] for r in got)
        for label, key in (("MEAN (unweighted)", None), ("POOLED (by n)", "pooled")):
            if key is None:
                p = sum(r["predictor_acc"] for r in got) / len(got)
                t = sum(r["teacher_acc"] for r in got) / len(got)
                m = sum(r["majority_acc"] for r in got) / len(got)
            else:
                p = sum(r["predictor_acc"] * r["n_consensus"] for r in got) / w
                t = sum(r["teacher_acc"] * r["n_consensus"] for r in got) / w
                m = sum(r["majority_acc"] * r["n_consensus"] for r in got) / w
            print(f"{label:19s} {'':>7s} {'':>9s} {p:>9.3f} {'':>15s} {t:>10.3f} {m:>10.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"n_common": len(common), "fields": results}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
