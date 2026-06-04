# Auto-generated calibration wrapper (temperature scaling)
import numpy as np
import torch

TEMPERATURES = {
    "tone": 1.188614,
    "risk_type": 1.136736,
    "valence": 1.200250,
    "arousal": 1.034059,
    "threat": 1.300021,
    "control": 1.457094,
    "player_intent": 1.126821,
    "player_knowledge": 1.018112,
    "player_credibility": 1.281534,
    "duty_pressure": 0.954340,
    "secrecy_pressure": 1.229504,
    "face_pressure": 0.743995,
    "value_conflict": 0.951125,
    "response_policy": 0.903963,
    "reveal_decision": 1.036396,
    "repair_strategy": 1.080096,
    "affection_level": 1.033568,
    "affection_delta": 1.191707,
    "respect_level": 1.272118,
    "respect_delta": 1.008125,
    "dominance_level": 1.179103,
    "dominance_delta": 1.178840,
    "familiarity_level": 1.378543,
    "familiarity_delta": 1.411942,
    "trust_level": 1.210900,
    "trust_delta": 1.097600,
    "obligation_level": 1.352067,
    "obligation_delta": 1.259922,
}

def apply_calibration(field: str, logits: np.ndarray | torch.Tensor):
    T = TEMPERATURES.get(field, 1.0)
    if isinstance(logits, torch.Tensor):
        return logits / T
    return logits / T
