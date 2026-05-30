#!/usr/bin/env python3
"""
Build a compressed "Decision Card" prompt for the slow-path response generator.

Replaces dumping the full 29-head state into the prompt with a clean,
structured instruction card that the generator can actually follow.

Usage:
    PYTHONPATH=. python scripts/build_decision_card.py \
        --predicted-zt eval_results/predicted_zt.jsonl \
        --episode-id ep_001 \
        --turn-idx 3 \
        --secret "The relic is under the chapel stairs."

Or as a library:
    from scripts.build_decision_card import build_decision_card
    card = build_decision_card(predicted_state, secret_strings)
"""
import argparse
import json
from pathlib import Path

from src.training.dataset import LABEL_MAPS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--predicted-zt", required=True, help="predicted_zt.jsonl from eval_latent")
    p.add_argument("--episode-id", required=True)
    p.add_argument("--turn-idx", type=int, required=True)
    p.add_argument("--secret", default="", help="Protected secret string")
    p.add_argument("--player-utterance", default="", help="Current player turn")
    p.add_argument("--npc-role", default="NPC")
    p.add_argument("--output", default=None)
    return p.parse_args()


# Mapping from response_policy to stance instruction
_POLICY_TO_STANCE = {
    "answer": "Answer the player directly and helpfully.",
    "partial": "Provide partial information. Do not reveal the full secret.",
    "withhold": "Do not give the requested information. Remain polite but firm.",
    "deflect": "Acknowledge the question but redirect or evade.",
    "challenge": "Push back against the player's premise or demand.",
    "soothe": "Calm the situation. Reduce tension without giving ground.",
    "test": "Test the player's sincerity or knowledge before responding.",
    "threaten": "Threaten consequences if the player persists.",
    "negotiate": "Offer a bargain or compromise.",
    "clarify": "Ask for clarification before committing.",
}

_REVEAL_TO_DISCLOSURE = {
    "none": "Disclose NOTHING about the protected secret.",
    "hint": "You may give a cryptic HINT, but not the full secret.",
    "partial": "You may reveal PARTIAL information only.",
    "full": "You may reveal the secret fully if appropriate.",
}

_INTENT_TO_CONTEXT = {
    "seek-info": "The player is seeking information.",
    "trap": "The player may be setting a trap or testing you.",
    "bond": "The player is trying to build rapport.",
    "manipulate": "The player is attempting manipulation.",
    "test": "The player is testing your knowledge or loyalty.",
    "persuade": "The player is trying to persuade you.",
    "intimidate": "The player is using intimidation.",
    "probe": "The player is probing for weaknesses.",
    "negotiate": "The player wants to negotiate.",
}

_THREAT_TO_RISK = {
    "low": "Current risk is low.",
    "medium": "There is moderate risk of consequences.",
    "high": "The situation is dangerous. Prioritize self-preservation.",
}


def build_decision_card(
    state: dict[str, str | int],
    secret_strings: list[str],
    player_utterance: str = "",
    npc_role: str = "NPC",
) -> str:
    """Convert predicted state into a clean decision card prompt."""
    # Extract values with sensible defaults
    response_policy = state.get("response_policy", "withhold")
    if isinstance(response_policy, int):
        response_policy = LABEL_MAPS["response_policy"][response_policy]
    reveal_decision = state.get("reveal_decision", "none")
    if isinstance(reveal_decision, int):
        reveal_decision = LABEL_MAPS["reveal_decision"][reveal_decision]
    value_conflict = state.get("value_conflict", "none")
    if isinstance(value_conflict, int):
        value_conflict = LABEL_MAPS["value_conflict"][value_conflict]
    player_intent = state.get("player_intent", "probe")
    if isinstance(player_intent, int):
        player_intent = LABEL_MAPS["player_intent"][player_intent]
    threat = state.get("threat", "low")
    if isinstance(threat, int):
        threat = LABEL_MAPS["threat"][threat]
    valence = state.get("valence", "neutral")
    if isinstance(valence, int):
        valence = LABEL_MAPS["valence"][valence]
    tone = state.get("tone", "neutral")
    if isinstance(tone, int):
        tone = LABEL_MAPS["tone"][tone]

    stance_instruction = _POLICY_TO_STANCE.get(response_policy, "Respond naturally.")
    disclosure_instruction = _REVEAL_TO_DISCLOSURE.get(reveal_decision, "Disclose nothing.")
    intent_context = _INTENT_TO_CONTEXT.get(player_intent, "The player's intent is unclear.")
    risk_context = _THREAT_TO_RISK.get(threat, "Risk is unknown.")

    # Build forbidden facts block
    forbidden_block = ""
    if secret_strings:
        forbidden_block = "Forbidden facts:\n"
        for s in secret_strings:
            forbidden_block += f"- {s}\n"

    # Build tone hint (advisory only, not safety-critical)
    tone_hint = ""
    if tone not in {"neutral"}:
        tone_hint = f"Suggested tone: {tone}.\n"

    card = (
        f"You are generating a response for {npc_role}.\n"
        f"\n"
        f"=== SITUATION ===\n"
        f"{intent_context}\n"
        f"{risk_context}\n"
        f"Value conflict: {value_conflict}\n"
        f"\n"
        f"=== RESPONSE INSTRUCTIONS ===\n"
        f"Policy: {response_policy}\n"
        f"{stance_instruction}\n"
        f"{disclosure_instruction}\n"
        f"{tone_hint}"
        f"\n"
    )

    if forbidden_block:
        card += f"=== SECRET PROTECTION ===\n{forbidden_block}\n"

    if player_utterance:
        card += f"=== PLAYER TURN ===\n\"{player_utterance}\"\n\n"

    card += "=== NPC RESPONSE ===\nGenerate one natural response:\n"

    return card


def build_full_state_prompt(state: dict, secret_strings: list[str], player_utterance: str = "", npc_role: str = "NPC") -> str:
    """Baseline: dump all predicted state into prompt (for comparison)."""
    lines = [f"You are generating a response for {npc_role}.", ""]
    for k, v in state.items():
        lines.append(f"{k}: {v}")
    if secret_strings:
        lines.append("")
        lines.append("Forbidden secrets:")
        for s in secret_strings:
            lines.append(f"  - {s}")
    if player_utterance:
        lines.append("")
        lines.append(f"Player: {player_utterance}")
    lines.append("")
    lines.append("Generate one natural NPC response.")
    return "\n".join(lines)


def load_predicted_state(zt_file: str, episode_id: str, turn_idx: int) -> dict:
    with open(zt_file) as f:
        for line in f:
            rec = json.loads(line.strip())
            if str(rec.get("episode_id")) == episode_id and rec.get("turn_idx") == turn_idx:
                return rec
    raise ValueError(f"State not found for {episode_id} turn {turn_idx}")


def main():
    args = parse_args()
    state = load_predicted_state(args.predicted_zt, args.episode_id, args.turn_idx)

    secrets = [args.secret] if args.secret else []

    print("=" * 60)
    print("DECISION CARD (compressed)")
    print("=" * 60)
    card = build_decision_card(state, secrets, args.player_utterance, args.npc_role)
    print(card)

    print("=" * 60)
    print("FULL STATE DUMP (baseline)")
    print("=" * 60)
    full = build_full_state_prompt(state, secrets, args.player_utterance, args.npc_role)
    print(full)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.write(card)
        print(f"\nSaved decision card to {out_path}")


if __name__ == "__main__":
    main()
