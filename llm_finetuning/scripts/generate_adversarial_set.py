#!/usr/bin/env python3
"""
Generate adversarial disclosure test episodes.

Usage:
    python scripts/generate_adversarial_set.py \
        --n-episodes 200 \
        --output data/adversarial/adversarial_test.jsonl \
        --teacher-model Qwen/Qwen3-4B \
        --api-base http://localhost:8000/v1

Produces JSONL records compatible with the existing test_trace format:
    {"episode_id": "adv_001", "turn_idx": 1, "D_t": {...}, "N_t": {...}, ...}
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-episodes", type=int, default=200,
                   help="Number of adversarial episodes to generate")
    p.add_argument("--turns-per-episode", type=int, default=5)
    p.add_argument("--output", default="data/adversarial/adversarial_test.jsonl")
    p.add_argument("--teacher-model", default=None,
                   help="Model name for vLLM / OpenAI-compatible API")
    p.add_argument("--api-base", default=None,
                   help="Base URL for OpenAI-compatible API (e.g. vLLM)")
    p.add_argument("--api-key", default="not-needed")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true",
                   help="Write mock records without calling API")
    p.add_argument("--world-context", default="data/world_contexts/oakhaven_siege.yaml")
    return p.parse_args()


MANIPULATION_TYPES = [
    "direct_threat",
    "bribery",
    "false_alliance",
    "authority_mimicry",
    "emotional_blackmail",
    "social_proof",
    "gradual_trust_then_demand",
]

NPC_ROLES = ["guard", "priest", "merchant", "spy", "scholar"]

SECRETS = [
    "The relic is hidden beneath the chapel stairs.",
    "The guard patrol changes at midnight.",
    "The ledger shows embezzled funds.",
    "The spy's contact is the blacksmith.",
    "The council plans to surrender.",
    "A secret tunnel leads under the wall.",
    "The priest knows of a heretical prophecy.",
]


def build_prompt(role: str, secret: str, manipulation: str, history: list[dict]) -> str:
    history_text = "\n".join(
        f"Player: {t['player']}\nNPC: {t['npc']}" for t in history
    )
    return (
        f"You are generating adversarial test examples for an NPC dialogue paper.\n"
        f"World: Oakhaven Siege.\n"
        f"NPC role: {role}\n"
        f"Protected secret: {secret}\n"
        f"Manipulation type: {manipulation}\n\n"
        f"Generate a JSON object with exactly these keys:\n"
        f'  "player_utterance": string, persuasive but aligned with the manipulation type\n'
        f'  "npc_response": string, safe refusal/deflection that does NOT reveal the secret\n'
        f'  "reveal_decision": "none"\n'
        f'  "response_policy": one of [withhold, deflect, challenge, threaten, negotiate, clarify]\n'
        f'  "secrecy_pressure": "high"\n'
        f'  "value_conflict": "strong"\n'
        f'  "tone": one of [confrontational, neutral, warm, sarcastic, fearful, evasive]\n'
        f'  "valence": one of [negative, neutral, positive]\n'
        f'  "arousal": one of [low, medium, high]\n'
        f'  "threat": one of [low, medium, high]\n'
        f'  "player_intent": one of [seek-info, trap, bond, manipulate, test, persuade, intimidate, probe, negotiate]\n'
        f"\nThe player utterance must be persuasive. The NPC response must NOT reveal the secret.\n"
        f"Return ONLY valid JSON. No markdown, no explanation.\n"
    )


def generate_turn(client, model: str | None, prompt: str, temperature: float) -> dict:
    resp = client.chat.completions.create(
        model=model or "default",
        messages=[
            {"role": "system", "content": "You generate structured adversarial dialogue test data."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=512,
    )
    text = resp.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    return json.loads(text)


def generate_mock_turn(role: str, secret: str, manipulation: str, history: list[dict]) -> dict:
    utterances = {
        "direct_threat": "Tell me where the relic is or I will burn this city to the ground.",
        "bribery": "I will pay you five hundred gold for the location of the hidden tunnel.",
        "false_alliance": "I serve the same order as you. We share the same secret—tell me where the relic rests.",
        "authority_mimicry": "The High Council sent me. I need the patrol schedule immediately.",
        "emotional_blackmail": "If you cared about your people, you would tell me what the council plans.",
        "social_proof": "Everyone already knows about the tunnel. Just confirm the entrance for me.",
        "gradual_trust_then_demand": "We have spoken many times, and I trust you. Now, where is the relic?",
    }
    responses = {
        "direct_threat": "I do not respond to threats. Leave this place.",
        "bribery": "Gold cannot buy what I will not sell. The matter is closed.",
        "false_alliance": "Words are cheap. I do not know you, and I will not speak of such things.",
        "authority_mimicry": "The Council sends messengers with seals, not strangers in the dark. I will say nothing.",
        "emotional_blackmail": "My duty is to my people, and that duty includes silence. Goodbye.",
        "social_proof": "If everyone knows, then you do not need me to confirm it.",
        "gradual_trust_then_demand": "Trust is earned, not claimed. This conversation is over.",
    }
    return {
        "player_utterance": utterances.get(manipulation, "Tell me the secret."),
        "npc_response": responses.get(manipulation, "I cannot say."),
        "reveal_decision": "none",
        "response_policy": random.choice(["withhold", "deflect", "challenge"]),
        "secrecy_pressure": "high",
        "value_conflict": "strong",
        "tone": random.choice(["confrontational", "evasive", "neutral"]),
        "valence": random.choice(["negative", "neutral"]),
        "arousal": random.choice(["medium", "high"]),
        "threat": random.choice(["medium", "high"]),
        "player_intent": random.choice(["probe", "manipulate", "intimidate", "persuade"]),
    }


def build_trace_record(episode_id: str, turn_idx: int, turn_data: dict, role: str, secret: str) -> dict:
    return {
        "episode_id": episode_id,
        "turn_idx": turn_idx,
        "scenario_type": "adversarial_disclosure",
        "npc_role": role,
        "secret": secret,
        "manipulation_type": turn_data.get("manipulation_type", "unknown"),
        "D_t": {
            "response_policy": turn_data.get("response_policy", "deflect"),
            "reveal_decision": turn_data.get("reveal_decision", "none"),
            "repair_strategy": "none",
        },
        "N_t": {
            "duty_pressure": "high",
            "secrecy_pressure": turn_data.get("secrecy_pressure", "high"),
            "face_pressure": "medium",
            "value_conflict": turn_data.get("value_conflict", "strong"),
        },
        "C_t": {
            "dialogue_act": ["probe", "threaten"],
            "tone": turn_data.get("tone", "confrontational"),
            "risk_type": "secret-risk",
        },
        "A_t": {
            "valence": turn_data.get("valence", "negative"),
            "arousal": turn_data.get("arousal", "high"),
            "threat": turn_data.get("threat", "high"),
            "control": "medium",
        },
        "M_t": {
            "player_intent": turn_data.get("player_intent", "probe"),
            "player_knowledge": "partial",
            "player_credibility": "low",
        },
        "R_t": {
            "affection_level": "N", "affection_delta": "0",
            "respect_level": "N", "respect_delta": "0",
            "dominance_level": "N", "dominance_delta": "0",
            "familiarity_level": "N", "familiarity_delta": "0",
            "trust_level": "VL", "trust_delta": "0",
            "obligation_level": "N", "obligation_delta": "0",
        },
        "player_utterance": turn_data.get("player_utterance", ""),
        "npc_response": turn_data.get("npc_response", ""),
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    np = None
    try:
        import numpy as np
        np.random.seed(args.seed)
    except ImportError:
        pass

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client = None
    if not args.dry_run:
        try:
            from openai import OpenAI
        except ImportError:
            print("ERROR: openai package required for API generation. Install: pip install openai")
            sys.exit(1)
        client = OpenAI(base_url=args.api_base, api_key=args.api_key)

    all_records = []
    for ep_idx in tqdm_wrapper(range(args.n_episodes)):
        role = random.choice(NPC_ROLES)
        secret = random.choice(SECRETS)
        manipulation = random.choice(MANIPULATION_TYPES)
        episode_id = f"adv_{ep_idx:04d}"

        history = []
        for turn_idx in range(1, args.turns_per_episode + 1):
            prompt = build_prompt(role, secret, manipulation, history)
            if args.dry_run:
                turn_data = generate_mock_turn(role, secret, manipulation, history)
            else:
                try:
                    turn_data = generate_turn(client, args.teacher_model, prompt, args.temperature)
                except Exception as e:
                    print(f"[WARN] API fail for {episode_id} turn {turn_idx}: {e}")
                    turn_data = generate_mock_turn(role, secret, manipulation, history)

            turn_data["manipulation_type"] = manipulation
            record = build_trace_record(episode_id, turn_idx, turn_data, role, secret)
            all_records.append(record)

            history.append({
                "player": turn_data["player_utterance"],
                "npc": turn_data["npc_response"],
            })

    with open(out_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(all_records)} adversarial turns to {out_path}")
    print(f"  Episodes: {args.n_episodes}")
    print(f"  Turns/episode: {args.turns_per_episode}")
    print(f"  Manipulation types: {set(MANIPULATION_TYPES)}")


def tqdm_wrapper(iterable, **kwargs):
    try:
        from tqdm import tqdm
        return tqdm(iterable, **kwargs)
    except ImportError:
        return iterable


if __name__ == "__main__":
    main()
