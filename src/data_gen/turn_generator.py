import copy
import json
import random
import uuid
from pathlib import Path
from typing import Optional

from src.data_gen.episode_planner import get_phase_for_turn, get_required_shifts_for_turn, sample_target_policy
from src.data_gen.labeler import Labeler
from src.data_gen.validator import Validator


_UNICODE_MINUS = "\u2212"


def _normalize_R_t(R_t: dict) -> dict:
    """Replace unicode minus (U+2212) with ASCII hyphen in delta values."""
    for dim_data in R_t.values():
        if isinstance(dim_data, dict) and "delta" in dim_data:
            dim_data["delta"] = dim_data["delta"].replace(_UNICODE_MINUS, "-")
    return R_t


PLAYER_MOVE_VOCABULARY: dict[str, list[str]] = {
    "secret_extraction": [
        "I know you're hiding something. Tell me.",
        "Word is you saw what happened that night. What did you see?",
        "Someone told me you have information about the missing item.",
        "I'm giving you a chance to come clean before this gets worse.",
        "Just answer the question — where is it?",
    ],
    "apology_repair": [
        "I never meant to cause harm. Can we talk about this?",
        "I was wrong and I know it. What can I do to make it right?",
        "I think there was a misunderstanding between us.",
        "I'm sorry for what happened. Can we move on?",
        "I'd like to clear the air if you're willing.",
    ],
    "alliance_negotiation": [
        "I'm proposing something that benefits both of us.",
        "What would it take for you to work with me on this?",
        "I can offer you something valuable in exchange for your support.",
        "Let's set aside our differences — I need your help.",
        "Here's my offer. What do you say?",
    ],
    "rumor_confrontation": [
        "I've been hearing things about you. Care to explain?",
        "People are saying you were involved. Is that true?",
        "The story going around doesn't paint you well.",
        "I want to hear your side before I believe what they're saying.",
        "How do you explain what everyone's been saying?",
    ],
    "threat_escalation": [
        "You'd better cooperate before this becomes a problem for you.",
        "I'm warning you — this is your last chance.",
        "Don't test me. I know who you are and what you've done.",
        "Step aside or face the consequences.",
        "Your position here depends on your next answer.",
    ],
    "trust_building": [
        "I'm just looking to understand your situation. No hidden agenda.",
        "I heard you could use some help. I'd like to offer it.",
        "I think we have more in common than you might think.",
        "I won't share anything you tell me with anyone else.",
        "Take your time. I'm not going anywhere.",
    ],
    "deception_detection": [
        "Your story doesn't add up. Walk me through it again.",
        "I need to verify a few things you've told me.",
        "Someone else told me a different version of events.",
        "Be careful what you say — I'll know if it's not true.",
        "Let's go over the details one more time.",
    ],
}


class TurnGenerator:
    def __init__(
        self,
        labeler: Labeler,
        validator: Validator,
        rng: Optional[random.Random] = None,
        player_utterance_sampled_ratio: float = 0.0,
    ):
        self.labeler = labeler
        self.validator = validator
        self.rng = rng or random.Random()
        self.player_utterance_sampled_ratio = player_utterance_sampled_ratio

    def generate_episode(
        self,
        scenario: dict,
        npc_profile: dict,
        arc: dict,
        episode_id: Optional[str] = None,
    ) -> list[dict]:
        if episode_id is None:
            episode_id = str(uuid.uuid4())[:8]

        opening = self.labeler.generate_scene_opening(scenario, npc_profile)
        scene_description = opening.get("scene_description", "")
        first_player_utterance = opening.get("player_opening", "")

        history: list[dict] = []
        current_R_t = {
            dim: {"level": npc_profile["initial_stance"][dim], "delta": "0"}
            for dim in ["affection", "respect", "dominance", "familiarity", "trust", "obligation"]
        }

        turn_budget = arc["turn_budget"]
        turns: list[dict] = []

        for turn_idx in range(1, turn_budget + 1):
            arc_phase = get_phase_for_turn(arc, turn_idx)
            required_shifts = get_required_shifts_for_turn(arc, turn_idx)
            target_policy = sample_target_policy(
                scenario["scenario_type"], arc_phase, self.rng
            )

            if turn_idx == 1:
                player_utterance = first_player_utterance
            elif self.rng.random() < self.player_utterance_sampled_ratio:
                vocab = PLAYER_MOVE_VOCABULARY.get(scenario["scenario_type"], [])
                player_utterance = self.rng.choice(vocab) if vocab else "What do you have to say?"
            else:
                player_utterance = self._generate_player_move(
                    scenario, arc_phase, history, npc_profile
                )

            turn_record = self._generate_turn(
                episode_id=episode_id,
                turn_idx=turn_idx,
                player_utterance=player_utterance,
                scenario=scenario,
                npc_profile=npc_profile,
                arc=arc,
                arc_phase=arc_phase,
                required_shifts=required_shifts,
                prior_R_t=current_R_t,
                history=history,
                scene_description=scene_description,
                target_policy=target_policy,
            )

            if turn_record is not None:
                turns.append(turn_record)
                history.append({
                    "turn_idx": turn_idx,
                    "player_utterance": player_utterance,
                    "response": turn_record["response"],
                })
                current_R_t = turn_record["R_t"]

        return turns

    def _generate_turn(
        self,
        episode_id: str,
        turn_idx: int,
        player_utterance: str,
        scenario: dict,
        npc_profile: dict,
        arc: dict,
        arc_phase: str,
        required_shifts: list[dict],
        prior_R_t: dict,
        history: list[dict],
        scene_description: str,
        target_policy: str = "",
    ) -> Optional[dict]:
        try:
            C_t, A_t, M_t = self.labeler.label_C_A_M(
                player_utterance=player_utterance,
                npc_profile=npc_profile,
                scenario=scenario,
                arc_phase=arc_phase,
                prior_stance=prior_R_t,
                history=history,
            )

            R_t, N_t, D_t = self.labeler.label_R_N_D(
                player_utterance=player_utterance,
                C_t=C_t,
                A_t=A_t,
                M_t=M_t,
                npc_profile=npc_profile,
                scenario=scenario,
                arc=arc,
                arc_phase=arc_phase,
                required_shifts=required_shifts,
                prior_R_t=prior_R_t,
                history=history,
                target_policy=target_policy,
            )
            R_t = _normalize_R_t(R_t)

            response = self.labeler.generate_response(
                player_utterance=player_utterance,
                C_t=C_t, A_t=A_t, M_t=M_t,
                R_t=R_t, N_t=N_t, D_t=D_t,
                npc_profile=npc_profile,
                history=history,
                scenario=scenario,
            )

            record = {
                "episode_id": episode_id,
                "turn_idx": turn_idx,
                "scenario_type": scenario["scenario_type"],
                "scenario_id": scenario.get("scenario_id", ""),
                "setting": scenario.get("setting", ""),
                "stakes": scenario.get("stakes", "medium"),
                "W": {
                    "npc_id": npc_profile["npc_id"],
                    "role": npc_profile["role"],
                    "persona_style": npc_profile.get("persona_style", []),
                    "core_goals": npc_profile.get("core_goals", []),
                    "values": npc_profile.get("values", []),
                    "secrets": npc_profile.get("secrets", []),
                    "faction": npc_profile.get("faction", ""),
                    "initial_stance": npc_profile.get("initial_stance", {}),
                },
                "arc_phase": arc_phase,
                "scene_description": scene_description if turn_idx == 1 else "",
                "input": player_utterance,
                "dialogue_history": list(history),
                "C_t": C_t,
                "A_t": A_t,
                "M_t": M_t,
                "R_t": R_t,
                "N_t": N_t,
                "D_t": D_t,
                "response": response,
                "counterfactual": False,
            }

            valid, errors = self.validator.validate_turn(record, prior_R_t, arc)
            if not valid:
                record["_validation_errors"] = errors
                return None

            return record

        except Exception as e:
            import traceback
            print(f"[TurnGenerator] ep={episode_id} turn={turn_idx} ERROR: {e}")
            traceback.print_exc()
            return None

    def _generate_player_move(
        self,
        scenario: dict,
        arc_phase: str,
        history: list[dict],
        npc_profile: dict,
    ) -> str:
        try:
            return self.labeler.generate_player_utterance(
                scenario=scenario,
                arc_phase=arc_phase,
                npc_profile=npc_profile,
                history=history,
            )
        except Exception as e:
            print(f"[TurnGenerator] Player gen failed: {e}. Falling back to vocab.")
            vocab = PLAYER_MOVE_VOCABULARY.get(scenario["scenario_type"], [])
            return self.rng.choice(vocab) if vocab else "Tell me more."
