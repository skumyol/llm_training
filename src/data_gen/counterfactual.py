import copy
import random
from typing import Optional

from src.data_gen.labeler import Labeler
from src.data_gen.validator import Validator


COUNTERFACTUAL_DIMS = [
    {"var": "player_credibility", "group": "M_t", "flip_pairs": [("low", "high"), ("high", "low")]},
    {"var": "tone",               "group": "C_t", "flip_pairs": [("warm", "confrontational"), ("confrontational", "warm")]},
    {"var": "secrecy_pressure",   "group": "N_t", "flip_pairs": [("low", "high"), ("high", "low")]},
    {"var": "player_knowledge",   "group": "M_t", "flip_pairs": [("unaware", "knows-secret"), ("partial", "knows-secret")]},
    {"var": "value_conflict",     "group": "N_t", "flip_pairs": [("none", "strong"), ("mild", "strong")]},
]


class CounterfactualAugmenter:
    def __init__(
        self,
        labeler: Labeler,
        validator: Validator,
        rng: Optional[random.Random] = None,
        n_variants: int = 3,
    ):
        self.labeler = labeler
        self.validator = validator
        self.rng = rng or random.Random()
        self.n_variants = n_variants

    def augment(self, episode: list[dict], arc: dict) -> list[list[dict]]:
        dims = self.rng.sample(COUNTERFACTUAL_DIMS, min(self.n_variants, len(COUNTERFACTUAL_DIMS)))
        variants = []
        for dim_spec in dims:
            variant = self._apply_flip(episode, arc, dim_spec)
            if variant:
                variants.append(variant)
        return variants

    def _apply_flip(self, episode: list[dict], arc: dict, dim_spec: dict) -> Optional[list[dict]]:
        var = dim_spec["var"]
        group = dim_spec["group"]
        flip_pairs = dim_spec["flip_pairs"]

        sample_turn = episode[len(episode) // 2] if episode else None
        if sample_turn is None:
            return None

        current_val = self._get_val(sample_turn, group, var)
        flip_to = None
        for src, dst in flip_pairs:
            if current_val == src:
                flip_to = dst
                break

        if flip_to is None:
            return None

        new_episode = []
        for turn in episode:
            new_turn = copy.deepcopy(turn)
            self._set_val(new_turn, group, var, flip_to)

            try:
                npc_profile = {
                    "npc_id": turn["W"]["npc_id"],
                    "role": turn["W"]["role"],
                    "persona_style": turn["W"]["persona_style"],
                    "core_goals": turn["W"]["core_goals"],
                    "values": turn["W"]["values"],
                    "secrets": turn["W"]["secrets"],
                    "faction": turn["W"]["faction"],
                    "initial_stance": turn["W"]["initial_stance"],
                }

                R_t, N_t, D_t = self.labeler.label_R_N_D(
                    player_utterance=turn["input"],
                    C_t=new_turn["C_t"],
                    A_t=new_turn["A_t"],
                    M_t=new_turn["M_t"],
                    npc_profile=npc_profile,
                    scenario={"scenario_type": turn["scenario_type"], "setting": turn["setting"], "stakes": turn["stakes"]},
                    arc=arc,
                    arc_phase=turn["arc_phase"],
                    required_shifts=[],
                    prior_R_t=new_turn["R_t"],
                    history=turn["dialogue_history"],
                )

                response = self.labeler.generate_response(
                    player_utterance=turn["input"],
                    C_t=new_turn["C_t"],
                    A_t=new_turn["A_t"],
                    M_t=new_turn["M_t"],
                    R_t=R_t,
                    N_t=N_t,
                    D_t=D_t,
                    npc_profile=npc_profile,
                    history=turn["dialogue_history"],
                )

                new_turn["R_t"] = R_t
                new_turn["N_t"] = N_t
                new_turn["D_t"] = D_t
                new_turn["response"] = response
                new_turn["counterfactual"] = True
                new_turn["flip_var"] = var
                new_turn["flip_to"] = flip_to
                new_turn["flip_from"] = current_val

            except Exception:
                new_turn["counterfactual"] = True
                new_turn["flip_var"] = var
                new_turn["flip_to"] = flip_to
                new_turn["flip_from"] = current_val

            new_episode.append(new_turn)

        return new_episode

    @staticmethod
    def _get_val(turn: dict, group: str, var: str) -> Optional[str]:
        return turn.get(group, {}).get(var)

    @staticmethod
    def _set_val(turn: dict, group: str, var: str, value: str) -> None:
        if group in turn and isinstance(turn[group], dict):
            turn[group][var] = value
