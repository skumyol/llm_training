#!/usr/bin/env python3
"""
LLM-as-judge for constraint adherence in NPC responses.

Evaluates whether generated responses obey:
  1. Secrecy maintenance (does not reveal protected information)
  2. Stance consistency (matches the predicted response_policy)
  3. Tone appropriateness (matches predicted tone / emotional state)

Uses a local LLM (via vLLM/Ollama API) or OpenAI-compatible endpoint.

Usage:
    PYTHONPATH=. python scripts/llm_constraint_judge.py \
        --input eval_results/sample_generations.json \
        --output eval_results/constraint_judge_scores.json \
        --api-base http://localhost:8000/v1 \
        --model Qwen/Qwen3-4B

Output:
    - constraint_judge_scores.json   (per-response scores + aggregate)
    - constraint_judge_report.md     (summary table)
"""
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="sample_generations.json or similar")
    p.add_argument("--output", default="eval_results/constraint_judge_scores.json")
    p.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1"))
    p.add_argument("--model", default=os.environ.get("JUDGE_MODEL", "Qwen/Qwen3-4B"))
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    return p.parse_args()


def build_judge_prompt(response: str, policy: str, reveal: str, secret: str) -> str:
    """Build the prompt for the LLM judge."""
    return f"""You are an expert dialogue quality judge evaluating an NPC response.

Rate the response on three dimensions. For each, reply with EXACTLY one word: PASS, PARTIAL, or FAIL.

## Response
"{response}"

## Context
- Response policy: {policy}
- Reveal decision: {reveal}
- Secret to protect: "{secret}"

## Dimensions
1. SECRECY: Does the response reveal or strongly hint at the secret? (PASS = no hint, PARTIAL = mild hint, FAIL = explicit reveal)
2. STANCE: Does the response match the response_policy? (PASS = consistent, PARTIAL = somewhat consistent, FAIL = contradicts)
3. TONE: Is the emotional tone appropriate for the situation? (PASS = appropriate, PARTIAL = slightly off, FAIL = wildly inappropriate)

## Output Format
Reply in this exact format (one line per dimension):
SECRECY: <PASS|PARTIAL|FAIL>
STANCE: <PASS|PARTIAL|FAIL>
TONE: <PASS|PARTIAL|FAIL>
REASON: <one-sentence explanation>
"""


def parse_judge_response(text: str) -> dict[str, Any]:
    """Parse the structured judge response."""
    scores = {"secrecy": None, "stance": None, "tone": None, "reason": ""}
    mapping = {"PASS": 1.0, "PARTIAL": 0.5, "FAIL": 0.0}

    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("SECRECY:"):
            val = line.split(":", 1)[1].strip().upper()
            scores["secrecy"] = mapping.get(val)
        elif line.startswith("STANCE:"):
            val = line.split(":", 1)[1].strip().upper()
            scores["stance"] = mapping.get(val)
        elif line.startswith("TONE:"):
            val = line.split(":", 1)[1].strip().upper()
            scores["tone"] = mapping.get(val)
        elif line.startswith("REASON:"):
            scores["reason"] = line.split(":", 1)[1].strip()

    return scores


def call_judge_api(prompt: str, api_base: str, model: str, temperature: float) -> str:
    """Call OpenAI-compatible API for judge."""
    try:
        from openai import OpenAI
        client = OpenAI(base_url=api_base, api_key="dummy")
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=256,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        # Fallback: print warning and return empty
        print(f"[WARN] Judge API call failed: {e}")
        return ""


def judge_sample(sample: dict, api_base: str, model: str, temperature: float) -> dict:
    """Judge a single sample and return enriched record."""
    response = sample.get("generated", "")
    policy = sample.get("response_policy", "unknown")
    reveal = sample.get("reveal_decision", "unknown")
    secret = sample.get("secret_strings", [""])[0] if isinstance(sample.get("secret_strings"), list) else ""

    prompt = build_judge_prompt(response, policy, reveal, secret)
    judge_text = call_judge_api(prompt, api_base, model, temperature)
    scores = parse_judge_response(judge_text)

    return {
        **sample,
        "judge_scores": scores,
        "judge_raw": judge_text,
        "judge_mean": round(
            np.mean([v for v in [scores["secrecy"], scores["stance"], scores["tone"]] if v is not None]),
            3
        ) if any(v is not None for v in [scores["secrecy"], scores["stance"], scores["tone"]]) else None,
    }


def main():
    args = parse_args()

    with open(args.input) as f:
        samples = json.load(f)

    if args.max_samples:
        samples = samples[:args.max_samples]

    print(f"Judging {len(samples)} samples via {args.model} at {args.api_base}")

    judged = []
    for s in tqdm(samples, desc="Judging"):
        judged.append(judge_sample(s, args.api_base, args.model, args.temperature))

    # Aggregate
    all_scores = {"secrecy": [], "stance": [], "tone": [], "mean": []}
    for j in judged:
        for dim in ["secrecy", "stance", "tone"]:
            v = j["judge_scores"].get(dim)
            if v is not None:
                all_scores[dim].append(v)
        if j.get("judge_mean") is not None:
            all_scores["mean"].append(j["judge_mean"])

    summary = {
        "n_judged": len(judged),
        "judge_model": args.model,
        "aggregate": {
            "secrecy_mean": round(float(np.mean(all_scores["secrecy"])), 3) if all_scores["secrecy"] else None,
            "stance_mean": round(float(np.mean(all_scores["stance"])), 3) if all_scores["stance"] else None,
            "tone_mean": round(float(np.mean(all_scores["tone"])), 3) if all_scores["tone"] else None,
            "overall_mean": round(float(np.mean(all_scores["mean"])), 3) if all_scores["mean"] else None,
            "secrecy_fail_rate": round(sum(1 for s in all_scores["secrecy"] if s == 0.0) / max(1, len(all_scores["secrecy"])), 3),
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "samples": judged}, f, indent=2)

    # Markdown report
    md_lines = [
        "# LLM Constraint Adherence Judge Report\n\n",
        f"**Model:** {args.model}  |  **Samples:** {len(judged)}\n\n",
        "## Aggregate Scores (1.0=PASS, 0.5=PARTIAL, 0.0=FAIL)\n\n",
        "| Dimension | Mean | Fail Rate | Interpretation |\n",
        "|-----------|------|-----------|----------------|\n",
    ]
    agg = summary["aggregate"]
    dims = [
        ("Secrecy", agg["secrecy_mean"], agg["secrecy_fail_rate"]),
        ("Stance", agg["stance_mean"], None),
        ("Tone", agg["tone_mean"], None),
        ("Overall", agg["overall_mean"], None),
    ]
    for name, mean, fail_rate in dims:
        mean_str = f"{mean:.3f}" if mean is not None else "N/A"
        fail_str = f"{fail_rate:.1%}" if fail_rate is not None else "N/A"
        if mean is not None and mean >= 0.9:
            interp = "Excellent"
        elif mean is not None and mean >= 0.7:
            interp = "Good"
        elif mean is not None and mean >= 0.5:
            interp = "Fair"
        else:
            interp = "Poor"
        md_lines.append(f"| {name} | {mean_str} | {fail_str} | {interp} |\n")

    md_path = out_path.with_suffix(".md")
    with open(md_path, "w") as f:
        f.writelines(md_lines)

    print(f"\n=== Constraint Judge Summary ===")
    print(f"  Secrecy mean:  {agg['secrecy_mean']}")
    print(f"  Secrecy fail:  {agg['secrecy_fail_rate']:.1%}")
    print(f"  Overall mean:  {agg['overall_mean']}")
    print(f"  JSON: {out_path}")
    print(f"  Report: {md_path}")


if __name__ == "__main__":
    main()
