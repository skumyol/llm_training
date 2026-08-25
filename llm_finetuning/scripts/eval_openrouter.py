"""Zero-shot evaluation of frontier LLMs via OpenRouter on the latent-state task.

Sends each test record's dialogue context to a frontier model (e.g. Gemini 3.5
Flash, GPT-4o, Claude) and asks it to predict the 29-field latent state.  Parses
the response and computes the same metrics as the SFT eval, so numbers are
directly comparable to the fine-tuned Qwen3-4B results.

This script is self-contained — it does NOT import torch or the training code.
All label maps and parsing logic are inlined from the training codebase.

Usage:
    python llm_finetuning/scripts/eval_openrouter.py \
        --model google/gemini-3.5-flash \
        --test-file data/splits/test_heads.jsonl \
        --output-dir eval_results/test_openrouter_gemini35flash \
        --limit 0          # 0 = all 884 records
        --concurrent 4     # parallel API calls
        --reasoning        # enable reasoning/thinking mode if supported
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

# ════════════════════════════════════════════════════════════════════════════
# Label maps — inlined from src/training/dataset.py to avoid torch dependency
# ════════════════════════════════════════════════════════════════════════════

LABEL_MAPS: dict[str, list[str]] = {
    "dialogue_act": ["ask", "accuse", "threaten", "flatter", "apologize", "negotiate", "joke", "confess", "probe", "command"],
    "tone": ["warm", "neutral", "confrontational", "sarcastic", "fearful", "evasive"],
    "risk_type": ["none", "secret-risk", "face-risk", "status-risk", "conflict-risk"],
    "valence": ["negative", "neutral", "positive"],
    "arousal": ["low", "medium", "high"],
    "threat": ["low", "medium", "high"],
    "control": ["low", "medium", "high"],
    "player_intent": ["seek-info", "trap", "bond", "manipulate", "test", "persuade", "intimidate", "probe", "negotiate"],
    "player_knowledge": ["unaware", "partial", "informed", "knows-secret"],
    "player_credibility": ["low", "medium", "high"],
    "duty_pressure": ["low", "medium", "high"],
    "secrecy_pressure": ["low", "medium", "high"],
    "face_pressure": ["low", "medium", "high"],
    "value_conflict": ["none", "mild", "strong"],
    "response_policy": ["answer", "partial", "withhold", "deflect", "challenge", "soothe", "test", "threaten", "negotiate", "clarify"],
    "reveal_decision": ["none", "hint", "partial", "full"],
    "repair_strategy": ["none", "soften", "apologize", "clarify", "redirect"],
    "affection_level": ["VL", "L", "N", "H", "VH"],
    "affection_delta": ["--", "-", "0", "+", "++"],
    "respect_level": ["VL", "L", "N", "H", "VH"],
    "respect_delta": ["--", "-", "0", "+", "++"],
    "dominance_level": ["VL", "L", "N", "H", "VH"],
    "dominance_delta": ["--", "-", "0", "+", "++"],
    "familiarity_level": ["VL", "L", "N", "H", "VH"],
    "familiarity_delta": ["--", "-", "0", "+", "++"],
    "trust_level": ["VL", "L", "N", "H", "VH"],
    "trust_delta": ["--", "-", "0", "+", "++"],
    "obligation_level": ["VL", "L", "N", "H", "VH"],
    "obligation_delta": ["--", "-", "0", "+", "++"],
}

LABEL_TO_IDX: dict[str, dict[str, int]] = {
    f: {label: i for i, label in enumerate(classes)}
    for f, classes in LABEL_MAPS.items()
}

FIELD_ORDER: list[str] = list(LABEL_MAPS.keys())

BEGIN, END = "<state>", "</state>"

# Fields that are grouped into "social stance" vs "schema" for reporting
STANCE_FIELDS = {
    "affection_level", "affection_delta",
    "respect_level", "respect_delta",
    "dominance_level", "dominance_delta",
    "familiarity_level", "familiarity_delta",
    "trust_level", "trust_delta",
    "obligation_level", "obligation_delta",
}

# ════════════════════════════════════════════════════════════════════════════
# Parsing — inlined from src/training/latent_sft.py
# ════════════════════════════════════════════════════════════════════════════


def parse(text: str) -> dict[str, object]:
    """Parse a generated state block. Tolerant of truncation and stray prose."""
    out: dict[str, object] = {}
    body = text.split(BEGIN, 1)[-1].split(END, 1)[0]
    for line in body.splitlines():
        line = line.strip()
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip()
        if k not in LABEL_TO_IDX:
            continue
        out[k] = (
            [p for p in (s.strip() for s in v.split(",")) if p]
            if k == "dialogue_act"
            else v
        )
    return out


def to_indices(parsed: dict) -> dict[str, int]:
    """Map parsed strings to class indices. Missing/invalid → len(classes)."""
    idx: dict[str, int] = {}
    for f in FIELD_ORDER:
        if f == "dialogue_act":
            continue
        classes = LABEL_MAPS[f]
        v = parsed.get(f)
        if isinstance(v, list):
            v = v[0] if v else None
        idx[f] = LABEL_TO_IDX[f].get(str(v), len(classes))
    return idx


# ════════════════════════════════════════════════════════════════════════════
# Metrics — inlined from src/metrics_report.py (torch-free subset)
# ════════════════════════════════════════════════════════════════════════════

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
)


def compute_metrics(
    all_preds: dict[str, list],
    all_golds: dict[str, list],
    extra_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fields: dict[str, dict[str, float]] = {}

    for field, golds in all_golds.items():
        preds = all_preds.get(field, [])
        if not golds or not preds:
            continue

        n_classes = len(LABEL_MAPS.get(field, []))
        gold_labels = sorted(set(golds))

        fields[field] = {
            "accuracy": float(accuracy_score(golds, preds)),
            "macro_f1": float(f1_score(golds, preds, average="macro", labels=gold_labels, zero_division=0)),
            "weighted_f1": float(f1_score(golds, preds, average="weighted", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(golds, preds)),
            "mcc": float(matthews_corrcoef(golds, preds) if len(gold_labels) > 1 else 0.0),
            "support": float(len(golds)),
            "n_classes": float(n_classes),
        }

    # Aggregate
    schema_fields = [f for f in FIELD_ORDER if f != "dialogue_act" and f in fields]
    stance_fields = [f for f in schema_fields if f in STANCE_FIELDS]
    non_stance = [f for f in schema_fields if f not in STANCE_FIELDS]

    def _mean(key: str, fs: list[str]) -> float:
        vals = [fields[f].get(key, 0) for f in fs]
        return sum(vals) / len(vals) if vals else 0.0

    summary = {
        "mean_accuracy": _mean("accuracy", schema_fields),
        "mean_macro_f1": _mean("macro_f1", schema_fields),
        "mean_balanced_accuracy": _mean("balanced_accuracy", schema_fields),
        "mean_mcc": _mean("mcc", schema_fields),
        "mean_weighted_f1": _mean("weighted_f1", schema_fields),
        "stance_accuracy": _mean("accuracy", stance_fields) if stance_fields else 0,
        "stance_macro_f1": _mean("macro_f1", stance_fields) if stance_fields else 0,
        "schema_accuracy": _mean("accuracy", non_stance) if non_stance else 0,
        "schema_macro_f1": _mean("macro_f1", non_stance) if non_stance else 0,
        "response_policy_f1": fields.get("response_policy", {}).get("macro_f1", 0),
        "response_policy_accuracy": fields.get("response_policy", {}).get("accuracy", 0),
        "response_policy_weighted_f1": fields.get("response_policy", {}).get("weighted_f1", 0),
    }

    # Stance delta accuracy (level vs delta averaged separately)
    delta_fields = [f for f in stance_fields if f.endswith("_delta")]
    level_fields = [f for f in stance_fields if f.endswith("_level")]
    summary["trust_delta_accuracy"] = fields.get("trust_delta", {}).get("accuracy", 0)
    summary["trust_level_accuracy"] = fields.get("trust_level", {}).get("accuracy", 0)

    if extra_summary:
        summary.update(extra_summary)

    return {"summary": summary, "fields": fields}


# ════════════════════════════════════════════════════════════════════════════
# Prompt construction
# ════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are a narrative AI analyst. Given a dialogue scene from a medieval-fantasy \
RPG, you must predict the NPC's internal latent state as a structured block.

The state has {n_fields} fields. Each field has a fixed set of valid values. \
You must output ONLY a block delimited by <state> and </state> tags, with one \
field per line in the format field=value. Do not output anything outside the \
block. If you are unsure, pick the most plausible value from the valid set.

Valid values for each field:
{field_specs}

Output format:
<state>
dialogue_act=<one or more comma-separated values from the valid set>
tone=<one value>
risk_type=<one value>
valence=<one value>
arousal=<one value>
threat=<one value>
control=<one value>
player_intent=<one value>
player_knowledge=<one value>
player_credibility=<one value>
duty_pressure=<one value>
secrecy_pressure=<one value>
face_pressure=<one value>
value_conflict=<one value>
response_policy=<one value>
reveal_decision=<one value>
repair_strategy=<one value>
affection_level=<one value>
affection_delta=<one value>
respect_level=<one value>
respect_delta=<one value>
dominance_level=<one value>
dominance_delta=<one value>
familiarity_level=<one value>
familiarity_delta=<one value>
trust_level=<one value>
trust_delta=<one value>
obligation_level=<one value>
obligation_delta=<one value>
</state>
"""

USER_PROMPT_TEMPLATE = """\
Analyse the following dialogue context and predict the NPC's latent state.

{context}

Predict the latent state.
"""


def _build_field_specs() -> str:
    lines = []
    for f in FIELD_ORDER:
        classes = LABEL_MAPS[f]
        if f == "dialogue_act":
            lines.append(f"  {f}: {', '.join(classes)} (one or more)")
        else:
            lines.append(f"  {f}: {', '.join(classes)}")
    return "\n".join(lines)


def build_system_prompt() -> str:
    return SYSTEM_PROMPT.format(
        n_fields=len(FIELD_ORDER),
        field_specs=_build_field_specs(),
    )


def build_user_prompt(record: dict) -> str:
    return USER_PROMPT_TEMPLATE.format(context=record["context"])


# ════════════════════════════════════════════════════════════════════════════
# API call
# ════════════════════════════════════════════════════════════════════════════

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def call_openrouter(
    model: str,
    system: str,
    user: str,
    api_key: str,
    reasoning: bool = False,
    max_tokens: int = 1024,
    timeout: int = 120,
) -> str:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    # Some models (Qwen3, DeepSeek V4) do chain-of-thought reasoning by
    # default, which consumes max_tokens.  Either enable it explicitly with
    # a higher budget, or disable it to get direct answers.
    if reasoning:
        payload["reasoning"] = {"enabled": True, "max_tokens": max_tokens}
    else:
        # Disable reasoning for models that enable it by default
        payload["reasoning"] = {"enabled": False}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    for attempt in range(3):
        try:
            resp = requests.post(
                OPENROUTER_URL, headers=headers, json=payload, timeout=timeout
            )
            resp.raise_for_status()
            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                # Some providers return content in different structure
                content = data.get("content") or data.get("output") or ""
                if content:
                    return content
                print(f"  no choices in response: {str(data)[:200]}")
                return ""
            msg = choices[0].get("message", {})
            return msg.get("content", "") or ""
        except requests.exceptions.HTTPError:
            if resp.status_code == 429:
                wait = 2 ** (attempt + 2)
                print(f"  rate-limited, waiting {wait}s…")
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                wait = 2 ** (attempt + 1)
                print(f"  server error {resp.status_code}, retrying in {wait}s…")
                time.sleep(wait)
                continue
            print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
            return ""
        except Exception as e:
            print(f"  error: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
            return ""
    return ""


# ════════════════════════════════════════════════════════════════════════════
# Evaluation
# ════════════════════════════════════════════════════════════════════════════


def evaluate_one(
    idx: int,
    record: dict,
    model: str,
    system: str,
    api_key: str,
    reasoning: bool,
    max_tokens: int,
) -> dict:
    user = build_user_prompt(record)
    try:
        raw = call_openrouter(
            model, system, user, api_key,
            reasoning=reasoning, max_tokens=max_tokens,
        )
    except Exception as e:
        print(f"  [{idx}] API error: {e}")
        raw = ""
    parsed = parse(raw) if raw else {}
    idx_map = to_indices(parsed)
    return {
        "idx": idx,
        "raw": raw[:500],
        "parsed": parsed,
        "indices": idx_map,
        "n_fields_parsed": len(parsed),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="OpenRouter zero-shot latent eval")
    ap.add_argument("--model", required=True, help="OpenRouter model slug")
    ap.add_argument("--test-file", default="data/splits/test_heads.jsonl")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--concurrent", type=int, default=4)
    ap.add_argument("--reasoning", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument(
        "--env-file", default=".env",
        help="Path to .env file with OPENROUTER_API_KEY",
    )
    args = ap.parse_args()

    # Load API key
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key and Path(args.env_file).exists():
        for line in Path(args.env_file).read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                api_key = line.split("=", 1)[1].strip()
                break
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not found in env or .env file")
        sys.exit(1)

    # Load test data
    records: list[dict] = []
    with open(args.test_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if args.limit:
        records = records[: args.limit]
    print(f"Loaded {len(records)} test records")
    print(f"Model: {args.model}")
    print(f"Reasoning: {args.reasoning}")
    print(f"Concurrent: {args.concurrent}")

    system = build_system_prompt()

    # Run eval with thread pool
    preds: dict[str, list] = {}
    golds: dict[str, list] = {}
    n_parsed = 0
    field_hits = 0
    raw_samples: list[str] = []
    results_meta: list[dict] = []

    t0 = time.time()
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrent) as pool:
        futures = {
            pool.submit(
                evaluate_one, i, rec, args.model, system, api_key,
                args.reasoning, args.max_tokens,
            ): (i, rec)
            for i, rec in enumerate(records)
        }
        for fut in concurrent.futures.as_completed(futures):
            i, rec = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                print(f"  [{i}] exception: {e}")
                res = {"idx": i, "raw": "", "parsed": {}, "indices": {}, "n_fields_parsed": 0}

            done += 1
            if done % 50 == 0 or done == len(records):
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (len(records) - done) / rate if rate > 0 else 0
                print(f"  {done}/{len(records)} done  ({rate:.1f}/s, ETA {eta:.0f}s)")

            if res["parsed"]:
                n_parsed += 1
            field_hits += res["n_fields_parsed"]
            if len(raw_samples) < 5:
                raw_samples.append(f"=== Record {i} ===\n{res['raw']}")

            idx_map = res["indices"]
            for f in FIELD_ORDER:
                if f == "dialogue_act":
                    continue
                gv = rec["labels"].get(f)
                if isinstance(gv, list):
                    gv = gv[0] if gv else None
                g = LABEL_TO_IDX[f].get(str(gv), -1)
                if g == -1:
                    continue
                preds.setdefault(f, []).append(idx_map.get(f, len(LABEL_MAPS[f])))
                golds.setdefault(f, []).append(g)

            results_meta.append({
                "idx": i,
                "episode_id": rec.get("episode_id"),
                "turn_idx": rec.get("turn_idx"),
                "n_fields_parsed": res["n_fields_parsed"],
                "response_policy_gold": rec["labels"].get("response_policy"),
                "response_policy_pred": res["parsed"].get("response_policy", ""),
            })

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({len(records)/elapsed:.1f} records/s)")
    print(f"parse_rate (>=1 field): {n_parsed}/{len(records)} = {n_parsed/len(records):.4f}")
    print(f"field_coverage: {field_hits}/{len(records)*len(FIELD_ORDER)} = {field_hits/(len(records)*len(FIELD_ORDER)):.4f}")

    # Compute metrics
    metrics = compute_metrics(preds, golds, extra_summary={
        "parse_rate": n_parsed / max(len(records), 1),
        "field_coverage": field_hits / max(len(records) * len(FIELD_ORDER), 1),
        "n_records": len(records),
        "model": args.model,
        "reasoning": args.reasoning,
        "elapsed_s": elapsed,
    })

    s = metrics.get("summary", {})
    print(f"\n{'='*60}")
    print(f"RESULTS: {args.model}")
    print(f"{'='*60}")
    print(f"mean_accuracy       {s.get('mean_accuracy', 0):.4f}")
    print(f"mean_macro_f1       {s.get('mean_macro_f1', 0):.4f}")
    print(f"mean_balanced_acc   {s.get('mean_balanced_accuracy', 0):.4f}")
    print(f"mean_mcc            {s.get('mean_mcc', 0):.4f}")
    print(f"mean_weighted_f1    {s.get('mean_weighted_f1', 0):.4f}")
    print(f"response_policy_f1  {s.get('response_policy_f1', 0):.4f}")
    print(f"response_policy_acc {s.get('response_policy_accuracy', 0):.4f}")
    print(f"response_policy_wf1 {s.get('response_policy_weighted_f1', 0):.4f}")
    print(f"trust_delta_acc     {s.get('trust_delta_accuracy', 0):.4f}")

    # Per-field breakdown
    print(f"\n{'field':<25} {'acc':>7} {'macro_f1':>8} {'w_f1':>7} {'mcc':>7}")
    print("-" * 58)
    for f in FIELD_ORDER:
        if f == "dialogue_act":
            continue
        fm = metrics.get("fields", {}).get(f, {})
        print(f"{f:<25} {fm.get('accuracy', 0):>7.4f} {fm.get('macro_f1', 0):>8.4f} {fm.get('weighted_f1', 0):>7.4f} {fm.get('mcc', 0):>7.4f}")

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "latent_eval_metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "samples.txt").write_text("\n\n---\n\n".join(raw_samples))
    (out_dir / "per_record.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results_meta)
    )
    print(f"\nWrote results to {out_dir}/")


if __name__ == "__main__":
    main()
