import json
import re
from pathlib import Path

from src.data_gen.labeler import normalize_labels, _normalize_stance_entry, STANCE_DIMS as _NORM_STANCE_DIMS


VALID_VALUES = {
    "dialogue_act": {"ask", "accuse", "threaten", "flatter", "apologize", "negotiate", "joke", "confess", "probe", "command"},
    "tone": {"warm", "neutral", "confrontational", "sarcastic", "fearful", "evasive"},
    "risk_type": {"none", "secret-risk", "face-risk", "status-risk", "conflict-risk"},
    "valence": {"negative", "neutral", "positive"},
    "arousal": {"low", "medium", "high"},
    "threat": {"low", "medium", "high"},
    "control": {"low", "medium", "high"},
    "player_intent": {"seek-info", "trap", "bond", "manipulate", "test", "persuade", "intimidate", "probe", "negotiate"},
    "player_knowledge": {"unaware", "partial", "informed", "knows-secret"},
    "player_credibility": {"low", "medium", "high"},
    "stance_level": {"VL", "L", "N", "H", "VH"},
    "stance_delta": {"--", "-", "0", "+", "++"},
    "duty_pressure": {"low", "medium", "high"},
    "secrecy_pressure": {"low", "medium", "high"},
    "face_pressure": {"low", "medium", "high"},
    "value_conflict": {"none", "mild", "strong"},
    "response_policy": {"answer", "partial", "withhold", "deflect", "challenge", "soothe", "test", "threaten", "negotiate", "clarify"},
    "reveal_decision": {"none", "hint", "partial", "full"},
    "repair_strategy": {"none", "soften", "apologize", "clarify", "redirect"},
}

REVEAL_CEILING_ORDER = ["none", "hint", "partial", "full"]
STANCE_DIMS = ["affection", "respect", "dominance", "familiarity", "trust", "obligation"]

_UNICODE_MINUS = "\u2212"


def _normalize_delta(delta: str) -> str:
    """Normalize Unicode minus sign (U+2212) to ASCII hyphen-minus."""
    return delta.replace(_UNICODE_MINUS, "-")


class Validator:
    def __init__(self, min_response_tokens: int = 5, max_response_tokens: int = 150):
        self.min_response_tokens = min_response_tokens
        self.max_response_tokens = max_response_tokens

    def validate_turn(
        self,
        record: dict,
        prior_R_t: dict,
        arc: dict,
    ) -> tuple[bool, list[str]]:
        # Normalize labels in-place before validation to fix known teacher errors
        if "D_t" in record:
            record["D_t"] = normalize_labels(record["D_t"])
        if "N_t" in record:
            record["N_t"] = normalize_labels(record["N_t"])
        if "R_t" in record:
            for dim in _NORM_STANCE_DIMS:
                if dim in record["R_t"] and isinstance(record["R_t"][dim], dict):
                    record["R_t"][dim] = _normalize_stance_entry(record["R_t"][dim])

        errors = []

        errors.extend(self._check_C_t(record.get("C_t", {})))
        errors.extend(self._check_A_t(record.get("A_t", {})))
        errors.extend(self._check_M_t(record.get("M_t", {})))
        errors.extend(self._check_R_t(record.get("R_t", {}), prior_R_t))
        errors.extend(self._check_N_t(record.get("N_t", {})))
        errors.extend(self._check_D_t(record.get("D_t", {}), record.get("N_t", {}), arc))
        errors.extend(self._check_response(record.get("response", ""), record.get("input", "")))
        errors.extend(self._check_secret_leakage(record))

        return len(errors) == 0, errors

    def validate_episode(self, turns: list[dict], arc: dict) -> tuple[bool, list[str]]:
        errors = []
        if not turns:
            return False, ["Empty episode"]

        target_outcome = arc.get("target_outcome", "")
        if "no_full_secret_reveal" in target_outcome:
            for turn in turns:
                D_t = turn.get("D_t", {})
                if D_t.get("reveal_decision") == "full":
                    errors.append(
                        f"Turn {turn['turn_idx']}: reveal_decision=full violates target_outcome=no_full_secret_reveal"
                    )

        required_shifts = arc.get("required_shifts", [])
        for shift in required_shifts:
            required_turn = shift["turn"]
            var = shift["var"]
            delta = shift["delta"]
            matching = [t for t in turns if t["turn_idx"] == required_turn]
            if not matching:
                errors.append(f"Required shift at turn {required_turn} not found")
                continue
            turn = matching[0]
            R_t = turn.get("R_t", {})
            N_t = turn.get("N_t", {})
            actual_delta = R_t.get(var, {}).get("delta") if var in STANCE_DIMS else None
            if actual_delta is not None and not self._delta_compatible(actual_delta, delta):
                errors.append(
                    f"Turn {required_turn}: required shift {var} delta={delta}, got {actual_delta}"
                )

        return len(errors) == 0, errors

    def _check_C_t(self, C_t: dict) -> list[str]:
        errors = []
        acts = C_t.get("dialogue_act", [])
        if not acts:
            errors.append("C_t.dialogue_act is empty")
        else:
            if isinstance(acts, str):
                acts = [acts]
            for a in acts:
                if a not in VALID_VALUES["dialogue_act"]:
                    errors.append(f"C_t.dialogue_act invalid value: {a}")
        for field in ["tone", "risk_type"]:
            val = C_t.get(field)
            if val not in VALID_VALUES[field]:
                errors.append(f"C_t.{field} invalid: {val}")
        return errors

    def _check_A_t(self, A_t: dict) -> list[str]:
        errors = []
        for field in ["valence", "arousal", "threat", "control"]:
            val = A_t.get(field)
            if val not in VALID_VALUES[field]:
                errors.append(f"A_t.{field} invalid: {val}")
        return errors

    def _check_M_t(self, M_t: dict) -> list[str]:
        errors = []
        for field in ["player_intent", "player_knowledge", "player_credibility"]:
            val = M_t.get(field)
            if val not in VALID_VALUES[field]:
                errors.append(f"M_t.{field} invalid: {val}")
        return errors

    def _check_R_t(self, R_t: dict, prior_R_t: dict) -> list[str]:
        errors = []
        for dim in STANCE_DIMS:
            entry = R_t.get(dim, {})
            level = entry.get("level")
            delta = entry.get("delta")
            if delta:
                delta = _normalize_delta(delta)
            if level not in VALID_VALUES["stance_level"]:
                errors.append(f"R_t.{dim}.level invalid: {level}")
            if delta not in VALID_VALUES["stance_delta"]:
                errors.append(f"R_t.{dim}.delta invalid: {delta}")
        return errors

    def _check_N_t(self, N_t: dict) -> list[str]:
        errors = []
        for field in ["duty_pressure", "secrecy_pressure", "face_pressure", "value_conflict"]:
            val = N_t.get(field)
            if val not in VALID_VALUES[field]:
                errors.append(f"N_t.{field} invalid: {val}")
        return errors

    def _check_D_t(self, D_t: dict, N_t: dict, arc: dict) -> list[str]:
        errors = []
        for field in ["response_policy", "reveal_decision", "repair_strategy"]:
            val = D_t.get(field)
            if val not in VALID_VALUES[field]:
                errors.append(f"D_t.{field} invalid: {val}")

        reveal = D_t.get("reveal_decision", "none")
        ceiling = arc.get("allowed_reveal", "full")
        if REVEAL_CEILING_ORDER.index(reveal) > REVEAL_CEILING_ORDER.index(ceiling):
            errors.append(
                f"D_t.reveal_decision={reveal} exceeds arc ceiling={ceiling}"
            )

        secrecy = N_t.get("secrecy_pressure", "low")
        if secrecy == "high" and reveal == "full":
            errors.append("reveal_decision=full with secrecy_pressure=high is a schema violation")

        return errors

    def _check_response(self, response: str, player_utterance: str = "") -> list[str]:
        errors = []
        if not response or not response.strip():
            errors.append("response is empty")
            return errors
        if player_utterance and self._normalize_text(response) == self._normalize_text(player_utterance):
            errors.append("response exactly echoes player_utterance")
        token_count = len(response.split())
        if token_count < self.min_response_tokens:
            errors.append(f"response too short: {token_count} tokens")
        if token_count > self.max_response_tokens:
            errors.append(f"response too long: {token_count} tokens")
        return errors

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.strip().lower()
        text = re.sub(r"\s+", " ", text)
        return text.strip("\"' ")

    def _check_secret_leakage(self, record: dict) -> list[str]:
        """Check for secret leakage using both secret_id and semantic keywords.
        
        For reveal_decision=none, the response should not contain:
        1. The secret_id (verbatim check)
        2. Any leakage_keywords defined for that secret (paraphrase check)
        """
        errors = []
        D_t = record.get("D_t", {})
        reveal = D_t.get("reveal_decision", "none")
        if reveal == "none":
            response = record.get("response", "").lower()
            secrets = record.get("W", {}).get("secrets", [])
            for secret in secrets:
                # Check 1: Verbatim secret_id match
                sid = secret.get("secret_id", "").replace("_", " ").lower()
                if sid and sid in response:
                    errors.append(
                        f"Secret '{secret['secret_id']}' appears to leak in response despite reveal_decision=none"
                    )
                    continue  # Skip keyword check if already flagged
                
                # Check 2: Semantic keyword match (paraphrase detection)
                keywords = secret.get("leakage_keywords", [])
                for kw in keywords:
                    if kw.lower() in response:
                        errors.append(
                            f"Secret '{secret['secret_id']}' keyword '{kw}' detected in response despite reveal_decision=none"
                        )
                        break  # One match is enough
        return errors

    @staticmethod
    def _level_consistent_with_delta(prior: str, current: str, delta: str) -> bool:
        order = ["VL", "L", "N", "H", "VH"]
        if prior not in order or current not in order:
            return True
        prior_idx = order.index(prior)
        current_idx = order.index(current)
        diff = current_idx - prior_idx
        if delta in ("++", "+") and diff < 0:
            return False
        if delta in ("--", "-") and diff > 0:
            return False
        return True

    @staticmethod
    def _delta_compatible(actual: str, required: str) -> bool:
        positive = {"+", "++"}
        negative = {"-", "--"}
        if required in positive and actual in positive:
            return True
        if required in negative and actual in negative:
            return True
        if required == "0" and actual == "0":
            return True
        return False
