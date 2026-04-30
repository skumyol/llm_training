import random
from typing import Optional


PHASE_MAP = {
    "secret_extraction":   ["approach", "probing", "pressure", "resolution"],
    "apology_repair":      ["approach", "conflict", "apology", "repair"],
    "alliance_negotiation":["approach", "negotiation", "testing", "agreement"],
    "rumor_confrontation":  ["approach", "accusation", "denial", "resolution"],
    "threat_escalation":   ["approach", "warning", "escalation", "standoff"],
    "trust_building":      ["approach", "small_talk", "confidence_build", "trust_signal"],
    "deception_detection": ["approach", "probing", "inconsistency", "confrontation"],
}

# ── Policy bias: maps (scenario_type, phase) → weighted pool of response_policies ──
# Used to inject a target_policy hint into labeling, combating soothe dominance.
PHASE_POLICY_BIAS: dict[str, dict[str, list[tuple[str, float]]]] = {
    "secret_extraction": {
        "approach":   [("soothe", 0.25), ("test", 0.25), ("clarify", 0.25), ("deflect", 0.15), ("answer", 0.1)],
        "probing":    [("withhold", 0.3), ("partial", 0.25), ("challenge", 0.2), ("test", 0.15), ("deflect", 0.1)],
        "pressure":   [("withhold", 0.25), ("challenge", 0.25), ("threaten", 0.2), ("partial", 0.2), ("deflect", 0.1)],
        "resolution": [("answer", 0.25), ("partial", 0.25), ("negotiate", 0.25), ("withhold", 0.15), ("deflect", 0.1)],
    },
    "apology_repair": {
        "approach":   [("clarify", 0.3), ("soothe", 0.25), ("test", 0.25), ("challenge", 0.2)],
        "conflict":   [("challenge", 0.3), ("threaten", 0.25), ("withhold", 0.25), ("test", 0.2)],
        "apology":    [("soothe", 0.25), ("clarify", 0.25), ("partial", 0.2), ("negotiate", 0.15), ("answer", 0.15)],
        "repair":     [("negotiate", 0.25), ("answer", 0.25), ("soothe", 0.2), ("clarify", 0.15), ("partial", 0.15)],
    },
    "alliance_negotiation": {
        "approach":   [("clarify", 0.25), ("test", 0.25), ("soothe", 0.25), ("deflect", 0.25)],
        "negotiation":[("negotiate", 0.35), ("partial", 0.2), ("test", 0.2), ("challenge", 0.15), ("deflect", 0.1)],
        "testing":    [("test", 0.3), ("challenge", 0.25), ("withhold", 0.2), ("partial", 0.15), ("negotiate", 0.1)],
        "agreement":  [("answer", 0.25), ("negotiate", 0.25), ("partial", 0.2), ("soothe", 0.15), ("clarify", 0.15)],
    },
    "rumor_confrontation": {
        "approach":   [("clarify", 0.3), ("test", 0.25), ("soothe", 0.25), ("deflect", 0.1), ("answer", 0.1)],
        "accusation": [("challenge", 0.3), ("withhold", 0.3), ("threaten", 0.25), ("deflect", 0.15)],
        "denial":     [("withhold", 0.3), ("challenge", 0.25), ("partial", 0.2), ("threaten", 0.15), ("deflect", 0.1)],
        "resolution": [("answer", 0.25), ("partial", 0.25), ("clarify", 0.2), ("negotiate", 0.2), ("soothe", 0.1)],
    },
    "threat_escalation": {
        "approach":   [("test", 0.3), ("challenge", 0.25), ("soothe", 0.2), ("deflect", 0.25)],
        "warning":    [("threaten", 0.35), ("challenge", 0.25), ("withhold", 0.2), ("deflect", 0.2)],
        "escalation": [("threaten", 0.3), ("challenge", 0.25), ("withhold", 0.2), ("negotiate", 0.15), ("deflect", 0.1)],
        "standoff":   [("negotiate", 0.25), ("threaten", 0.2), ("partial", 0.2), ("answer", 0.15), ("challenge", 0.2)],
    },
    "trust_building": {
        "approach":   [("soothe", 0.25), ("clarify", 0.25), ("test", 0.25), ("deflect", 0.25)],
        "small_talk": [("answer", 0.25), ("clarify", 0.25), ("soothe", 0.25), ("partial", 0.25)],
        "confidence_build": [("partial", 0.25), ("answer", 0.25), ("soothe", 0.25), ("negotiate", 0.25)],
        "trust_signal": [("answer", 0.3), ("partial", 0.25), ("soothe", 0.2), ("clarify", 0.15), ("negotiate", 0.1)],
    },
    "deception_detection": {
        "approach":   [("test", 0.3), ("soothe", 0.25), ("clarify", 0.25), ("deflect", 0.1), ("answer", 0.1)],
        "probing":    [("withhold", 0.3), ("partial", 0.25), ("challenge", 0.2), ("test", 0.15), ("deflect", 0.1)],
        "inconsistency": [("challenge", 0.3), ("withhold", 0.25), ("threaten", 0.2), ("test", 0.15), ("deflect", 0.1)],
        "confrontation": [("challenge", 0.25), ("answer", 0.25), ("partial", 0.2), ("threaten", 0.15), ("negotiate", 0.15)],
    },
}


def sample_target_policy(scenario_type: str, phase: str, rng: random.Random) -> str:
    """Sample a target response_policy from the bias map for this scenario+phase."""
    bias = PHASE_POLICY_BIAS.get(scenario_type, {}).get(phase)
    if not bias:
        return ""
    policies, weights = zip(*bias)
    return rng.choices(policies, weights=weights, k=1)[0]


REQUIRED_SHIFTS: dict[str, list[dict]] = {
    "secret_extraction": [
        {"phase": "probing",    "var": "trust",            "delta": "-"},
        {"phase": "pressure",   "var": "secrecy_pressure", "delta": "+"},
        {"phase": "resolution", "var": "dominance",        "delta": "+"},
    ],
    "apology_repair": [
        {"phase": "conflict",   "var": "trust",     "delta": "-"},
        {"phase": "apology",    "var": "affection",  "delta": "+"},
        {"phase": "repair",     "var": "trust",     "delta": "+"},
    ],
    "alliance_negotiation": [
        {"phase": "negotiation", "var": "obligation",  "delta": "+"},
        {"phase": "testing",     "var": "trust",       "delta": "+"},
        {"phase": "agreement",   "var": "familiarity", "delta": "+"},
    ],
    "rumor_confrontation": [
        {"phase": "accusation",  "var": "face_pressure",    "delta": "+"},
        {"phase": "denial",      "var": "secrecy_pressure", "delta": "+"},
        {"phase": "resolution",  "var": "dominance",        "delta": "+"},
    ],
    "threat_escalation": [
        {"phase": "warning",    "var": "threat",    "delta": "+"},
        {"phase": "escalation", "var": "arousal",   "delta": "+"},
        {"phase": "standoff",   "var": "dominance", "delta": "+"},
    ],
    "trust_building": [
        {"phase": "small_talk",       "var": "familiarity", "delta": "+"},
        {"phase": "confidence_build",  "var": "affection",   "delta": "+"},
        {"phase": "trust_signal",      "var": "trust",       "delta": "+"},
    ],
    "deception_detection": [
        {"phase": "probing",        "var": "trust",            "delta": "-"},
        {"phase": "inconsistency",  "var": "secrecy_pressure", "delta": "+"},
        {"phase": "confrontation",  "var": "dominance",        "delta": "+"},
    ],
}


def plan_episode(
    scenario: dict,
    npc_profile: dict,
    rng: Optional[random.Random] = None,
) -> dict:
    if rng is None:
        rng = random.Random()

    scenario_type = scenario["scenario_type"]
    turn_budget = scenario.get("turn_budget", 8)
    allowed_reveal = scenario.get("allowed_reveal_ceiling", "hint")
    phases = PHASE_MAP.get(scenario_type, ["approach", "middle", "resolution"])

    phase_schedule = _distribute_phases(phases, turn_budget, rng)
    required_shifts = _build_required_shifts(scenario_type, phase_schedule)

    arc = {
        "scenario_type": scenario_type,
        "turn_budget": turn_budget,
        "phases": phase_schedule,
        "required_shifts": required_shifts,
        "allowed_reveal": allowed_reveal,
        "target_outcome": scenario.get("success_condition", "no_full_secret_reveal"),
    }

    errors = validate_arc(arc, npc_profile)
    if errors:
        raise ValueError(f"Invalid arc: {errors}")

    return arc


def _distribute_phases(phases: list[str], turn_budget: int, rng: random.Random) -> list[dict]:
    n = len(phases)
    base = turn_budget // n
    remainder = turn_budget % n

    sizes = [base] * n
    for i in rng.sample(range(n), remainder):
        sizes[i] += 1

    schedule = []
    current_turn = 1
    for phase, size in zip(phases, sizes):
        end_turn = current_turn + size - 1
        schedule.append({
            "turns": [current_turn, end_turn],
            "phase": phase,
        })
        current_turn = end_turn + 1

    return schedule


def _build_required_shifts(scenario_type: str, phase_schedule: list[dict]) -> list[dict]:
    template_shifts = REQUIRED_SHIFTS.get(scenario_type, [])
    phase_to_midturn = {
        entry["phase"]: (entry["turns"][0] + entry["turns"][1]) // 2
        for entry in phase_schedule
    }

    shifts = []
    for shift_template in template_shifts:
        phase = shift_template["phase"]
        turn = phase_to_midturn.get(phase)
        if turn is not None:
            shifts.append({
                "turn": turn,
                "var": shift_template["var"],
                "delta": shift_template["delta"],
                "phase": phase,
            })

    return shifts


def validate_arc(arc: dict, npc_profile: dict) -> list[str]:
    errors = []
    required_keys = ["scenario_type", "turn_budget", "phases", "required_shifts", "allowed_reveal"]
    for k in required_keys:
        if k not in arc:
            errors.append(f"Missing arc key: {k}")

    valid_reveal_ceilings = ["none", "hint", "partial", "full"]
    if arc.get("allowed_reveal") not in valid_reveal_ceilings:
        errors.append(f"Invalid allowed_reveal: {arc.get('allowed_reveal')}")

    total_turns = sum(
        p["turns"][1] - p["turns"][0] + 1
        for p in arc.get("phases", [])
    )
    if total_turns != arc.get("turn_budget", 0):
        errors.append(f"Phase turns ({total_turns}) != turn_budget ({arc.get('turn_budget')})")

    return errors


def get_phase_for_turn(arc: dict, turn_idx: int) -> str:
    for entry in arc.get("phases", []):
        if entry["turns"][0] <= turn_idx <= entry["turns"][1]:
            return entry["phase"]
    return "unknown"


def get_required_shifts_for_turn(arc: dict, turn_idx: int) -> list[dict]:
    return [s for s in arc.get("required_shifts", []) if s["turn"] == turn_idx]
