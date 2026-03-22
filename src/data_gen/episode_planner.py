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
