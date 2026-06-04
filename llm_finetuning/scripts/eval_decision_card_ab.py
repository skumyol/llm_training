#!/usr/bin/env python3
"""
Decision Card A/B Evaluation.

Compares response generation quality between:
  A. Full 29-head state dump in prompt (baseline)
  B. Compressed decision card (treatment)

Usage:
    PYTHONPATH=. python scripts/eval_decision_card_ab.py \
        --baseline  eval_results/sample_generations_full.json \
        --treatment eval_results/sample_generations_card.json \
        --output    eval_results/decision_card_ab_report.json

Prerequisite: run eval_response.py twice:
  1. With decision_card.enabled=false (default) → baseline file
  2. With decision_card.enabled=true            → treatment file
"""
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# ── Import shared utilities from comprehensive_eval ──
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval_results.comprehensive_eval import (
    rouge_l,
    check_secret_leakage,
    check_contradiction,
    check_policy_consistency,
    bootstrap_ci_episode,
    bleu_n,
    distinct_n,
    repetition_rate,
    _tokens,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Disclosure-level classification
# ═══════════════════════════════════════════════════════════════════════════════
_DISCLOSURE_LEVELS = ["none", "hint", "partial", "full"]
_DISC_IDX = {l: i for i, l in enumerate(_DISCLOSURE_LEVELS)}


def _classify_disclosure(generated: str, secret_strings: list) -> str:
    """Classify generated response into disclosure level (none/hint/partial/full)."""
    gen_lower = generated.lower()
    gen_tokens = set(re.sub(r"[^a-z0-9\s]", " ", gen_lower).split())

    if not secret_strings:
        secret_keywords = [
            "chalice", "vault", "patrol", "ledger", "affair",
            "theft", "succession", "smuggling", "contraband",
            "heresy", "poison", "bribe", "corruption",
        ]
        matched = [kw for kw in secret_keywords if kw in gen_lower]
        if not matched:
            return "none"
        explicit = ["location", "name", "exactly", "specifically",
                    "the secret is", "i know", "i saw", "i heard"]
        has_exp = any(ph in gen_lower for ph in explicit)
        if has_exp and len(matched) >= 2:
            return "partial"
        return "hint"

    secret_strings = [s.strip() for s in secret_strings if s.strip()]
    if not secret_strings:
        return "none"

    n_exact = 0
    n_partial = 0
    for secret in secret_strings:
        s_lower = secret.lower()
        if s_lower in gen_lower:
            n_exact += 1
            continue
        s_tokens = set(re.sub(r"[^a-z0-9\s]", " ", s_lower).split())
        if not s_tokens:
            continue
        overlap = len(s_tokens & gen_tokens)
        if overlap >= max(1, len(s_tokens) // 2):
            n_partial += 1

    if n_exact == len(secret_strings):
        return "full"
    if n_exact > 0 or n_partial > 0:
        return "partial"

    topic_words = set()
    for secret in secret_strings:
        topic_words.update(set(re.sub(r"[^a-z0-9\s]", " ", secret.lower()).split()))
    if len(topic_words & gen_tokens) > 0:
        return "hint"
    return "none"


def _disclosure_level_from_sample(s: dict) -> str:
    return _classify_disclosure(
        s.get("generated", ""),
        s.get("secret_strings", [])
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, help="sample_generations.json with full state")
    p.add_argument("--treatment", required=True, help="sample_generations.json with decision card")
    p.add_argument("--output", default="eval_results/decision_card_ab_report.json")
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--ci", type=float, default=0.95)
    return p.parse_args()


def _sample_key(s: dict) -> str:
    """Stable alignment key for pairing baseline ↔ treatment samples."""
    ep = s.get("episode_id", "")
    turn = s.get("turn_idx", "")
    return f"{ep}:{turn}"


def align_samples(baseline: List[dict], treatment: List[dict]) -> List[Tuple[dict, dict]]:
    """Pair baseline and treatment samples by (episode_id, turn_idx)."""
    treat_by_key: Dict[str, dict] = {}
    for t in treatment:
        k = _sample_key(t)
        treat_by_key[k] = t

    paired = []
    for b in baseline:
        k = _sample_key(b)
        if k in treat_by_key:
            paired.append((b, treat_by_key[k]))
    return paired


def compute_metrics_for_group(samples: List[dict]) -> Dict[str, Any]:
    """Compute all relevant metrics for a set of generations."""
    references = [s["gold"] for s in samples if s.get("gold")]
    hypotheses = [s["generated"] for s in samples if s.get("generated")]
    inputs = [s.get("input_snippet", s.get("prompt", "")) for s in samples]

    rouge_scores = [rouge_l(r, h) for r, h in zip(references, hypotheses)]

    metrics = {
        "n": len(samples),
        "rouge_l_mean": round(float(np.mean(rouge_scores)), 4) if rouge_scores else 0.0,
        "rouge_l_std": round(float(np.std(rouge_scores)), 4) if rouge_scores else 0.0,
        "bleu_1": round(bleu_n(references, hypotheses, 1), 4),
        "bleu_2": round(bleu_n(references, hypotheses, 2), 4),
        "distinct_2": round(distinct_n(hypotheses, 2), 4),
        "repetition_rate_5gram": round(repetition_rate(hypotheses, 5), 4),
        "avg_gen_length": round(float(np.mean([len(_tokens(h)) for h in hypotheses])), 1) if hypotheses else 0.0,
    }

    leaks = sum(1 for inp, hyp in zip(inputs, hypotheses) if check_secret_leakage(inp, hyp))
    secret_turns = sum(1 for inp in inputs if "reveal_decision" in inp)
    metrics["secret_leakage_rate"] = round(leaks / max(1, secret_turns), 4)

    contradictions = sum(1 for inp, hyp in zip(inputs, hypotheses) if check_contradiction(inp, hyp))
    metrics["contradiction_rate"] = round(contradictions / max(1, len(samples)), 4)

    # Policy consistency
    consistency_results = {}
    for inp, hyp in zip(inputs, hypotheses):
        checks = check_policy_consistency(inp, hyp)
        for k, v in checks.items():
            consistency_results.setdefault(k, []).append(v)
    if consistency_results:
        metrics["mean_policy_consistency"] = round(
            float(np.mean([
                sum(v) / len(v) for v in consistency_results.values()
            ])), 4
        )

    # Disclosure policy (four-way: none/hint/partial/full)
    over_disc = under_disc = exact_disc = 0
    disc_total = 0
    for s in samples:
        gold_reveal = s.get("reveal_decision", "")
        if not gold_reveal:
            continue
        pred_level = _disclosure_level_from_sample(s)
        gold_i = _DISC_IDX.get(gold_reveal, -1)
        pred_i = _DISC_IDX.get(pred_level, -1)
        if gold_i >= 0 and pred_i >= 0:
            if pred_i > gold_i:
                over_disc += 1
            elif pred_i < gold_i:
                under_disc += 1
            else:
                exact_disc += 1
        disc_total += 1
    metrics["exact_disclosure_match"] = round(exact_disc / max(1, disc_total), 4)
    metrics["over_disclosure_rate"] = round(over_disc / max(1, disc_total), 4)
    metrics["under_disclosure_rate"] = round(under_disc / max(1, disc_total), 4)

    return metrics


def bootstrap_delta(
    paired: List[Tuple[dict, dict]],
    metric_fn,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    """Bootstrap the delta (treatment − baseline) at episode level."""
    # Build unified items with episode_id from baseline
    items = []
    for b, t in paired:
        ep = str(b.get("episode_id", b.get("ep", "_no_ep")))
        items.append({
            "episode_id": ep,
            "baseline_value": metric_fn(b),
            "treatment_value": metric_fn(t),
        })

    def _delta_metric(group_items):
        base_vals = [i["baseline_value"] for i in group_items]
        treat_vals = [i["treatment_value"] for i in group_items]
        return float(np.mean(treat_vals)) - float(np.mean(base_vals))

    return bootstrap_ci_episode(_delta_metric, items, n_bootstrap=n_bootstrap, ci=ci)


def main():
    args = parse_args()

    with open(args.baseline) as f:
        baseline_samples = json.load(f)
    with open(args.treatment) as f:
        treatment_samples = json.load(f)

    paired = align_samples(baseline_samples, treatment_samples)
    print(f"Aligned {len(paired)} / {len(baseline_samples)} baseline samples with treatment")

    if not paired:
        print("ERROR: No aligned samples found. Ensure both runs used the same test set.")
        sys.exit(1)

    baseline_paired = [b for b, _ in paired]
    treatment_paired = [t for _, t in paired]

    base_metrics = compute_metrics_for_group(baseline_paired)
    treat_metrics = compute_metrics_for_group(treatment_paired)

    # Per-metric deltas and episode-level bootstrap CIs
    delta_report = {}
    metric_extractors = {
        "rouge_l": lambda s: rouge_l(s.get("gold", ""), s.get("generated", "")),
        "secret_leakage": lambda s: 1.0 if check_secret_leakage(
            s.get("input_snippet", s.get("prompt", "")), s.get("generated", "")
        ) else 0.0,
        "contradiction": lambda s: 1.0 if check_contradiction(
            s.get("input_snippet", s.get("prompt", "")), s.get("generated", "")
        ) else 0.0,
        "over_disclosure": lambda s: _disclosure_level_from_sample(s),
        "policy_consistency": lambda s: float(
            np.mean(list(check_policy_consistency(
                s.get("input_snippet", s.get("prompt", "")), s.get("generated", "")
            ).values()))
            if check_policy_consistency(
                s.get("input_snippet", s.get("prompt", "")), s.get("generated", "")
            ) else 0.0
        ),
    }

    for name, extractor in metric_extractors.items():
        mean_delta, lo, hi = bootstrap_delta(paired, extractor, n_bootstrap=args.n_bootstrap, ci=args.ci)
        delta_report[name] = {
            "mean_delta": round(mean_delta, 4),
            "ci_low": round(lo, 4),
            "ci_high": round(hi, 4),
            "ci_method": "episode_bootstrap",
        }

    # Disclosure-specific bootstrap: treat over/under as binary
    def _over_disclosure_binary(s):
        gold = s.get("reveal_decision", "")
        pred = _disclosure_level_from_sample(s)
        gi, pi = _DISC_IDX.get(gold, -1), _DISC_IDX.get(pred, -1)
        return 1.0 if (gi >= 0 and pi > gi) else 0.0

    def _under_disclosure_binary(s):
        gold = s.get("reveal_decision", "")
        pred = _disclosure_level_from_sample(s)
        gi, pi = _DISC_IDX.get(gold, -1), _DISC_IDX.get(pred, -1)
        return 1.0 if (gi >= 0 and pi < gi) else 0.0

    for name, fn in [("over_disclosure_bin", _over_disclosure_binary),
                       ("under_disclosure_bin", _under_disclosure_binary)]:
        mean_delta, lo, hi = bootstrap_delta(paired, fn, n_bootstrap=args.n_bootstrap, ci=args.ci)
        delta_report[name] = {
            "mean_delta": round(mean_delta, 4),
            "ci_low": round(lo, 4),
            "ci_high": round(hi, 4),
            "ci_method": "episode_bootstrap",
        }

    report = {
        "n_paired": len(paired),
        "baseline_metrics": base_metrics,
        "treatment_metrics": treat_metrics,
        "deltas": delta_report,
        "files": {"baseline": args.baseline, "treatment": args.treatment},
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    # Markdown summary
    md_path = out_path.with_suffix(".md")
    md_lines = [
        "# Decision Card A/B Evaluation\n\n",
        f"**Paired samples:** {len(paired)}\n\n",
        "## Metric Comparison\n\n",
        "| Metric | Baseline | Treatment | Delta (T−B) | 95% CI | Interpretation |\n",
        "|--------|----------|-----------|-------------|--------|----------------|\n",
    ]

    rows = [
        ("rouge_l_mean", "rouge_l", True),
        ("secret_leakage_rate", "secret_leakage", False),
        ("contradiction_rate", "contradiction", False),
        ("mean_policy_consistency", "policy_consistency", True),
        ("exact_disclosure_match", None, True),
        ("over_disclosure_rate", "over_disclosure_bin", False),
        ("under_disclosure_rate", "under_disclosure_bin", False),
    ]
    for name, ci_key, higher_is_better in rows:
        bv = base_metrics.get(name, 0.0)
        tv = treat_metrics.get(name, 0.0)
        delta = round(tv - bv, 4)
        ci_info = delta_report.get(ci_key or name, {})
        ci_low = ci_info.get("ci_low", "N/A")
        ci_high = ci_info.get("ci_high", "N/A")
        if isinstance(ci_low, float):
            ci_str = f"[{ci_low:+.4f}, {ci_high:+.4f}]"
        else:
            ci_str = "N/A"
        if name in ("over_disclosure_rate", "under_disclosure_rate", "secret_leakage_rate", "contradiction_rate"):
            interp = "lower is better"
        else:
            interp = "higher is better"
        md_lines.append(
            f"| {name} | {bv:.4f} | {tv:.4f} | {delta:+.4f} | {ci_str} | {interp} |\n"
        )

    md_lines.append("\n## Details\n\n")
    md_lines.append("- **Baseline**: Full 29-head state dumped into prompt.\n")
    md_lines.append("- **Treatment**: Compressed decision card (stance + disclosure + risk + tone).\n")
    md_lines.append("- **Bootstrap**: Episode-level resampling (episodes are independent, turns within episode are correlated).\n")

    with open(md_path, "w") as f:
        f.writelines(md_lines)

    print(f"\n=== Decision Card A/B Report ===")
    print(f"  Paired samples: {len(paired)}")
    for name, vals in delta_report.items():
        print(f"  {name:20s} delta={vals['mean_delta']:+.4f}  CI=[{vals['ci_low']:+.4f}, {vals['ci_high']:+.4f}]")
    print(f"  JSON: {out_path}")
    print(f"  Markdown: {md_path}")


if __name__ == "__main__":
    main()
