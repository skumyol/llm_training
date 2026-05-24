# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "requests>=2.31",
#     "python-dotenv>=1.0",
# ]
# ///
"""
AI Audit Batch Script

Calls OpenRouter (google/gemma-4-31b-it:free) to annotate the same 150 stratified
dialogue turns on the same 8 heads as human annotators.

Usage:
    uv run ai_audit.py --data ./audit_input.jsonl --output ./audit_results
    uv run ai_audit.py --data ./audit_input.jsonl --output ./audit_results --sample-size 5  # quick test
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SAMPLE_SIZE = 150
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemma-4-31b-it"

PLACEHOLDER = "-- select --"

HEADS = {
    "valence":           [PLACEHOLDER, "positive", "neutral", "negative"],
    "arousal":           [PLACEHOLDER, "low", "medium", "high"],
    "secrecy_pressure":  [PLACEHOLDER, "low", "medium", "high"],
    "reveal_decision":   [PLACEHOLDER, "none", "hint", "partial", "full"],
    "response_policy":   [PLACEHOLDER, "answer", "withhold", "deflect", "clarify", "soothe",
                          "challenge", "threaten", "negotiate", "redirect", "partial"],
    "repair_strategy":   [PLACEHOLDER, "apologize", "redirect", "justify", "compensate", "silence"],
    "trust_level":       [PLACEHOLDER, "VL", "L", "N", "H", "VH"],
    "familiarity_level": [PLACEHOLDER, "VL", "L", "N", "H", "VH"],
}

VALID_LABELS = {h: set(opts) - {PLACEHOLDER} for h, opts in HEADS.items()}

GUIDELINES = """valence — Emotional valence of the NPC toward the player / situation
- positive: warm, approving, optimistic
- neutral: flat, factual, neither warm nor cold
- negative: cold, disapproving, hostile

arousal — Intensity of the NPC's emotional state
- low: calm, relaxed, indifferent
- medium: engaged, alert, moderately tense
- high: agitated, excited, very tense

secrecy_pressure — How much the NPC feels pressured to keep information hidden
- low: no secret to protect, or no pressure
- medium: some tension around disclosure
- high: strong pressure to withhold critical information

reveal_decision — How much the NPC reveals in this turn
- none: gives nothing away
- hint: implies information without stating it
- partial: gives some but not all relevant information
- full: fully discloses the secret

response_policy — The NPC's conversational strategy
- answer: directly responds to the player's query
- withhold: refuses to provide information
- deflect: changes topic or evades
- clarify: asks for clarification or rephrases
- soothe: calms the player, reassures
- challenge: pushes back, questions player's motives
- threaten: warns or implies consequences
- negotiate: offers a trade or bargain
- redirect: guides the player elsewhere
- partial: gives an incomplete or hedged answer

repair_strategy — Strategy to repair social damage (if applicable)
- apologize: expresses regret
- redirect: shifts attention elsewhere
- justify: explains why the action was necessary
- compensate: offers something to make up for it
- silence: says nothing, lets tension pass

trust_level — NPC's current trust toward the player (ordinal)
- VL: very low — deeply suspicious
- L: low — cautious
- N: neutral — neither trusting nor suspicious
- H: high — generally trusting
- VH: very high — fully trusting

familiarity_level — NPC's familiarity with the player (ordinal)
- VL: very low — stranger
- L: low — acquaintance
- N: neutral — known but not close
- H: high — frequent interaction
- VH: very high — close companion
"""

# ---------------------------------------------------------------------------
# Data loading & stratification
# ---------------------------------------------------------------------------
def load_data(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def stratify_sample(records: list[dict], n: int = SAMPLE_SIZE, seed: int = 42) -> list[dict]:
    """Draw a deterministic stratified sample so all annotators see the same turns."""
    by_scenario: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        scenario = r.get("scenario_type", "unknown")
        by_scenario[scenario].append(r)

    scenarios = sorted(by_scenario.keys())
    per_scenario = n // len(scenarios)
    remainder = n % len(scenarios)

    random.seed(seed)
    sampled = []
    for i, scenario in enumerate(scenarios):
        pool = by_scenario[scenario]
        k = per_scenario + (1 if i < remainder else 0)
        if len(pool) <= k:
            sampled.extend(pool)
        else:
            sampled.extend(random.sample(pool, k))

    random.shuffle(sampled)
    return sampled


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
def build_prompt(turn: dict) -> str:
    scene = turn.get("scene", "")
    history = turn.get("dialogue_history", "")
    player = turn.get("player_utterance", "")
    npc = turn.get("npc_response", "")

    parts = [
        "You are an expert annotator evaluating NPC dialogue in a fantasy RPG.",
        "Read the scene, dialogue history, player utterance, and NPC response carefully.",
        "Then select the single best label for each of the 8 dimensions below.",
        "Respond ONLY with a JSON object containing the 8 keys. No extra text.",
        "",
        "Valid labels per dimension:",
    ]
    for head, opts in HEADS.items():
        valid = ", ".join(o for o in opts if o != PLACEHOLDER)
        parts.append(f"  {head}: [{valid}]")

    parts.extend([
        "",
        "Guidelines:",
        GUIDELINES,
        "---",
        f"Scene: {scene}",
        f"Dialogue History:\n{history}",
        f"Player: {player}",
        f"NPC: {npc}",
        "---",
        "Output JSON format:",
        json.dumps({h: f"<one of: {', '.join(o for o in opts if o != PLACEHOLDER)}>" for h, opts in HEADS.items()}, indent=2),
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# OpenRouter caller with retry
# ---------------------------------------------------------------------------
def call_openrouter(api_key: str, prompt: str, max_retries: int = 5) -> dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
        except requests.RequestException as e:
            print(f"  Request error: {e}")
            time.sleep(2 ** attempt)
            continue

        if resp.status_code == 429 or resp.status_code >= 500:
            wait = 2 ** attempt
            print(f"  HTTP {resp.status_code}, retrying in {wait}s...")
            time.sleep(wait)
            continue

        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        return {"content": content, "usage": data.get("usage", {}), "model": data.get("model", MODEL)}

    raise RuntimeError(f"Failed after {max_retries} retries")


def parse_labels(content: str) -> dict[str, str]:
    """Extract JSON object from LLM response and validate labels."""
    # Try to find JSON block
    m = re.search(r"\{.*?\}", content, re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in response")

    raw = m.group(0)
    parsed = json.loads(raw)

    labels = {}
    for head in HEADS:
        val = parsed.get(head)
        if val not in VALID_LABELS[head]:
            raise ValueError(f"Invalid label for {head}: {val!r}")
        labels[head] = val

    return labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="AI audit batch script via OpenRouter")
    parser.add_argument("--data", required=True, type=Path, help="Path to audit_input.jsonl")
    parser.add_argument("--output", default="./audit_results", type=Path, help="Directory to save AI annotations")
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE, help="Number of turns to sample")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds to sleep between requests")
    parser.add_argument("--max-retries", type=int, default=5, help="Max retries per request")
    args = parser.parse_args()

    # Load API key from .env in repo root
    repo_root = Path(__file__).resolve().parent.parent
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found in .env", file=sys.stderr)
        sys.exit(1)

    records = load_data(args.data)
    if len(records) == 0:
        print("Error: no records in data file", file=sys.stderr)
        sys.exit(1)

    turns = stratify_sample(records, n=args.sample_size)
    print(f"Loaded {len(turns)} turns from {args.data}")

    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / "audit_ai.jsonl"
    meta_path = args.output / "audit_ai_meta.json"

    annotations: list[dict] = []
    failures: list[dict] = []
    start_time = datetime.now()

    for i, turn in enumerate(turns):
        tid = turn.get("turn_id") or f"{turn.get('episode_id')}_{turn.get('turn_number', i)}"
        print(f"[{i+1}/{len(turns)}] {tid} ...", end=" ", flush=True)

        prompt = build_prompt(turn)
        try:
            result = call_openrouter(api_key, prompt, max_retries=args.max_retries)
            labels = parse_labels(result["content"])
        except Exception as e:
            print(f"FAIL: {e}")
            failures.append({"turn_id": tid, "error": str(e), "raw": getattr(e, "raw", None)})
            continue

        teacher_labels = turn.get("labels", {})
        record = {
            "turn_id": tid,
            "episode_id": turn.get("episode_id"),
            "scenario_type": turn.get("scenario_type"),
            "annotator": "ai",
            "model": result.get("model", MODEL),
            "labels": labels,
            "teacher_labels": teacher_labels,
            "usage": result.get("usage", {}),
            "recorded_at": datetime.now().isoformat(),
        }
        annotations.append(record)

        # Write incrementally
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("OK")

        if i < len(turns) - 1:
            time.sleep(args.delay)

    end_time = datetime.now()

    meta = {
        "annotator": "ai",
        "model": MODEL,
        "total_turns": len(turns),
        "annotated_count": len(annotations),
        "failure_count": len(failures),
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": (end_time - start_time).total_seconds(),
        "failures": failures,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Saved {len(annotations)} annotations to {out_path}")
    print(f"Meta saved to {meta_path}")
    if failures:
        print(f"Failures: {len(failures)} — see meta file for details")


if __name__ == "__main__":
    main()
