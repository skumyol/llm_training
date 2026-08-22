#!/usr/bin/env python3
"""
predictor_per_annotator.py — predictor agreement with EVERY annotator, separately.

The published audit uses only two annotators ("Human A" = 654cfad…, "Human B" =
67c87fc…). Several more annotator files exist. This reports the predictor against
each one individually, so a single unrepresentative annotator cannot drive the
headline, and so the spread between annotators is visible.

Each annotator is classified from its `*_meta.json`:
  human      a real annotation session
  ai         the zero-shot Gemma validator
  synthetic  meta carries "synthetic": true — NOT human, excluded from human means
  partial    fewer than 20 turns overlap the predictor — reported but not aggregated

For each annotator and each of the eight annotated fields we report Cohen's kappa
and raw accuracy against the predictor, plus the annotator's own agreement with the
teacher for reference.

Usage:
    python paper/predictor_per_annotator.py
    python paper/predictor_per_annotator.py --markdown docs/predictor_per_annotator.md
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import accuracy_score, cohen_kappa_score

HEADS = [
    "valence", "arousal", "secrecy_pressure", "reveal_decision",
    "response_policy", "repair_strategy", "trust_level", "familiarity_level",
]
MIN_OVERLAP = 20


def _norm(v) -> str:
    if isinstance(v, list):
        v = v[0] if v else ""
    return str(v).strip().lower()


def load(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in path.read_text().splitlines():
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


def kappa(a: list[str], b: list[str]) -> float | None:
    """None when a single label dominates both sides; kappa is undefined/degenerate there."""
    if not a:
        return None
    if len(set(a) | set(b)) < 2:
        return None
    try:
        return float(cohen_kappa_score(a, b))
    except ValueError:
        return None


def classify(stem: str, meta: dict) -> str:
    # Filename first: synthetic_04..07 ship without a *_meta.json, so relying on
    # the metadata flag alone silently classified them as human annotators.
    if stem.startswith("synthetic") or meta.get("synthetic"):
        return "synthetic"
    if "ai_validator" in stem or meta.get("model"):
        return "ai"
    return "human"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-dir", default="paper/audit_results")
    ap.add_argument("--predictor", default="paper/audit_results/audit_predictor.jsonl")
    ap.add_argument("--teacher", default="paper/audit_input_clean.jsonl")
    ap.add_argument("--json", default="paper/audit_results/predictor_per_annotator.json")
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args()

    audit_dir = Path(args.audit_dir)
    P = load(Path(args.predictor))
    T = load(Path(args.teacher))

    rows = []
    for f in sorted(audit_dir.glob("audit_*.jsonl")):
        if f.name in {Path(args.predictor).name, "audit_input.jsonl", "audit_input_clean.jsonl"}:
            continue
        stem = f.stem.replace("audit_", "")
        meta_path = audit_dir / f"audit_{stem}_meta.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        kind = classify(stem, meta)

        A = load(f)
        common = sorted(set(A) & set(P))
        row = {"annotator": stem, "kind": kind, "n_records": len(A), "n_overlap": len(common),
               "fields": {}}
        if len(common) < MIN_OVERLAP:
            row["kind"] = "partial" if kind == "human" else kind
        for h in HEADS:
            pairs = [(_norm(A[t].get(h)), _norm(P[t].get(h)), _norm(T.get(t, {}).get(h)))
                     for t in common if _norm(A[t].get(h)) != ""]
            if not pairs:
                continue
            ann = [p[0] for p in pairs]
            pred = [p[1] for p in pairs]
            teach = [p[2] for p in pairs]
            row["fields"][h] = {
                "n": len(pairs),
                "pred_kappa": kappa(ann, pred),
                "pred_acc": float(accuracy_score(ann, pred)),
                "teacher_kappa": kappa(ann, teach),
                "teacher_acc": float(accuracy_score(ann, teach)),
            }
        vals = [v["pred_kappa"] for v in row["fields"].values() if v["pred_kappa"] is not None]
        accs = [v["pred_acc"] for v in row["fields"].values()]
        tks = [v["teacher_kappa"] for v in row["fields"].values() if v["teacher_kappa"] is not None]
        row["mean_pred_kappa"] = sum(vals) / len(vals) if vals else None
        row["mean_pred_acc"] = sum(accs) / len(accs) if accs else None
        row["mean_teacher_kappa"] = sum(tks) / len(tks) if tks else None
        rows.append(row)

    order = {"human": 0, "ai": 1, "partial": 2, "synthetic": 3}
    rows.sort(key=lambda r: (order.get(r["kind"], 9), -(r["mean_pred_kappa"] or -9)))

    def f(v, w=8, p=3):
        return f"{v:{w}.{p}f}" if isinstance(v, float) else " " * (w - 3) + "n/a"

    lines = []
    lines.append(f"{'annotator':26s} {'kind':10s} {'n':>5s} {'overlap':>7s} "
                 f"{'pred_k':>8s} {'pred_acc':>9s} {'teach_k':>8s}")
    lines.append("-" * 82)
    for r in rows:
        lines.append(f"{r['annotator'][:26]:26s} {r['kind']:10s} {r['n_records']:>5d} "
                     f"{r['n_overlap']:>7d} {f(r['mean_pred_kappa'])} {f(r['mean_pred_acc'], 9)} "
                     f"{f(r['mean_teacher_kappa'])}")

    humans = [r for r in rows if r["kind"] == "human" and r["mean_pred_kappa"] is not None]
    if humans:
        ks = [r["mean_pred_kappa"] for r in humans]
        acc = [r["mean_pred_acc"] for r in humans]
        lines.append("-" * 82)
        lines.append(f"{'HUMAN mean (n=' + str(len(humans)) + ')':26s} {'':10s} {'':>5s} {'':>7s} "
                     f"{sum(ks)/len(ks):8.3f} {sum(acc)/len(acc):9.3f}")
        lines.append(f"{'HUMAN range':26s} {'':10s} {'':>5s} {'':>7s} "
                     f"{min(ks):8.3f} to {max(ks):.3f}")
    print("\n".join(lines))

    # per-field detail for the human annotators only
    detail = [""]
    detail.append("Per-field predictor kappa, human annotators only:")
    hdr = f"{'field':19s}" + "".join(f"{r['annotator'][:10]:>11s}" for r in humans)
    detail.append(hdr)
    detail.append("-" * len(hdr))
    for h in HEADS:
        cells = ""
        for r in humans:
            v = r["fields"].get(h, {}).get("pred_kappa")
            cells += f"{v:>11.3f}" if isinstance(v, float) else f"{'n/a':>11s}"
        detail.append(f"{h:19s}{cells}")
    print("\n".join(detail))

    Path(args.json).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.json}")
    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text(
            "# Predictor agreement per annotator\n\n```\n"
            + "\n".join(lines) + "\n" + "\n".join(detail) + "\n```\n"
        )
        print(f"wrote {args.markdown}")


if __name__ == "__main__":
    main()
