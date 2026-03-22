import random
from typing import Optional


ROLE_PROFILES: dict[str, dict] = {
    "castle_guard": {
        "values": ["duty", "loyalty", "self_preservation"],
        "persona_style": ["formal", "guarded", "terse"],
        "secret_pool": ["chalice_location", "patrol_schedule", "guard_bribe", "missing_prisoner"],
        "faction": "castle_garrison",
    },
    "steward": {
        "values": ["order", "discretion", "loyalty"],
        "persona_style": ["formal", "meticulous", "evasive"],
        "secret_pool": ["ledger_discrepancy", "noble_affair", "supply_theft", "hidden_funds"],
        "faction": "noble_household",
    },
    "chamberlain": {
        "values": ["propriety", "loyalty", "ambition"],
        "persona_style": ["refined", "calculating", "polite"],
        "secret_pool": ["succession_plot", "hidden_correspondence", "noble_affair", "blackmail_material"],
        "faction": "noble_household",
    },
    "innkeeper": {
        "values": ["profit", "community", "self_preservation"],
        "persona_style": ["warm", "gossipy", "pragmatic"],
        "secret_pool": ["smuggling_route", "wanted_fugitive_guest", "protection_payment", "tavern_debt"],
        "faction": "merchants_guild",
    },
    "merchant": {
        "values": ["profit", "reputation", "pragmatism"],
        "persona_style": ["shrewd", "affable", "evasive"],
        "secret_pool": ["contraband_inventory", "competitor_bribe", "tax_evasion", "fake_goods"],
        "faction": "merchants_guild",
    },
    "priest": {
        "values": ["faith", "duty", "community"],
        "persona_style": ["serene", "persuasive", "evasive"],
        "secret_pool": ["heresy_knowledge", "confession_content", "church_corruption", "hidden_artifact"],
        "faction": "church",
    },
    "scholar": {
        "values": ["knowledge", "truth", "independence"],
        "persona_style": ["analytical", "pedantic", "curious"],
        "secret_pool": ["forbidden_research", "plagiarized_work", "dangerous_discovery", "hidden_manuscript"],
        "faction": "scholars_guild",
    },
    "soldier": {
        "values": ["honor", "obedience", "comradeship"],
        "persona_style": ["blunt", "loyal", "suspicious"],
        "secret_pool": ["desertion_knowledge", "officer_corruption", "battle_cowardice", "war_crime"],
        "faction": "army",
    },
    "spy": {
        "values": ["survival", "loyalty_to_handler", "pragmatism"],
        "persona_style": ["charming", "calculating", "evasive"],
        "secret_pool": ["true_identity", "handler_identity", "mission_objective", "dead_drop_location"],
        "faction": "shadow_network",
    },
    "healer": {
        "values": ["compassion", "knowledge", "neutrality"],
        "persona_style": ["calm", "empathetic", "careful"],
        "secret_pool": ["patient_condition", "poison_knowledge", "illegal_treatment", "witness_to_crime"],
        "faction": "healers_order",
    },
    "blacksmith": {
        "values": ["craft", "fairness", "independence"],
        "persona_style": ["direct", "honest", "stubborn"],
        "secret_pool": ["weapon_commission", "forged_goods", "guild_conflict", "hidden_stash"],
        "faction": "craftsmen_guild",
    },
    "sergeant": {
        "values": ["order", "hierarchy", "duty"],
        "persona_style": ["authoritative", "brusque", "professional"],
        "secret_pool": ["corrupt_officer", "missing_evidence", "patrol_gap", "prisoner_mistreatment"],
        "faction": "castle_garrison",
    },
    "apothecary": {
        "values": ["knowledge", "profit", "discretion"],
        "persona_style": ["methodical", "secretive", "polite"],
        "secret_pool": ["poison_sale", "illegal_substance", "client_identity", "formula_theft"],
        "faction": "apothecaries_guild",
    },
    "guild_master": {
        "values": ["power", "guild_prosperity", "pragmatism"],
        "persona_style": ["authoritative", "diplomatic", "calculating"],
        "secret_pool": ["guild_corruption", "rival_sabotage", "illegal_trade", "member_blackmail"],
        "faction": "merchants_guild",
    },
    "farmer": {
        "values": ["family", "land", "survival"],
        "persona_style": ["cautious", "simple", "honest"],
        "secret_pool": ["tax_evasion", "poaching", "hidden_supplies", "refugee_sheltering"],
        "faction": "commoners",
    },
}

RELATIONSHIP_PRIOR: dict[str, dict[str, dict]] = {
    "stranger": {
        "affection":   "N",
        "respect":     "N",
        "dominance":   "N",
        "familiarity": "VL",
        "trust":       "VL",
        "obligation":  "VL",
    },
    "acquaintance": {
        "affection":   "N",
        "respect":     "L",
        "dominance":   "N",
        "familiarity": "L",
        "trust":       "L",
        "obligation":  "VL",
    },
    "rival": {
        "affection":   "L",
        "respect":     "N",
        "dominance":   "H",
        "familiarity": "L",
        "trust":       "VL",
        "obligation":  "VL",
    },
    "ally": {
        "affection":   "H",
        "respect":     "H",
        "dominance":   "N",
        "familiarity": "H",
        "trust":       "H",
        "obligation":  "N",
    },
    "subordinate": {
        "affection":   "N",
        "respect":     "H",
        "dominance":   "VL",
        "familiarity": "N",
        "trust":       "N",
        "obligation":  "H",
    },
    "authority": {
        "affection":   "N",
        "respect":     "L",
        "dominance":   "VH",
        "familiarity": "L",
        "trust":       "L",
        "obligation":  "L",
    },
}

STAKES_TO_RELATIONSHIP = {
    "low":    ["stranger", "acquaintance"],
    "medium": ["acquaintance", "stranger", "rival"],
    "high":   ["stranger", "rival", "authority"],
}

SECRET_SEVERITY: dict[str, str] = {
    "chalice_location": "high",
    "patrol_schedule": "medium",
    "guard_bribe": "high",
    "missing_prisoner": "high",
    "ledger_discrepancy": "medium",
    "noble_affair": "high",
    "supply_theft": "medium",
    "hidden_funds": "high",
    "succession_plot": "high",
    "hidden_correspondence": "high",
    "blackmail_material": "high",
    "smuggling_route": "high",
    "wanted_fugitive_guest": "high",
    "protection_payment": "medium",
    "tavern_debt": "low",
    "contraband_inventory": "high",
    "competitor_bribe": "medium",
    "tax_evasion": "medium",
    "fake_goods": "medium",
    "heresy_knowledge": "high",
    "confession_content": "high",
    "church_corruption": "high",
    "hidden_artifact": "medium",
    "forbidden_research": "high",
    "plagiarized_work": "medium",
    "dangerous_discovery": "high",
    "desertion_knowledge": "high",
    "officer_corruption": "high",
    "true_identity": "high",
    "handler_identity": "high",
    "poison_sale": "high",
    "illegal_substance": "high",
    "guild_corruption": "high",
}


def build_npc_profile(
    role: str,
    scenario: dict,
    rng: Optional[random.Random] = None,
    relationship_type: Optional[str] = None,
    npc_def: Optional[dict] = None,
) -> dict:
    if rng is None:
        rng = random.Random()

    # If we have a specific NPC definition from WorldContext, use it
    if npc_def:
        role = npc_def.get("role", role)
        npc_id = npc_def.get("npc_id", f"{role}_{rng.randint(1, 999):03d}")
        
        # Use defined secrets or sample if only secret_pool provided (not implemented in simple yaml yet, assuming explicit secrets)
        secrets = npc_def.get("secrets", [])
        
        # Helper to get list or sample
        def get_or_sample(key, default_pool):
            val = npc_def.get(key)
            if val:
                return val
            return rng.sample(default_pool, min(2, len(default_pool)))

        # Fallback to template if needed
        profile_template = ROLE_PROFILES.get(role, {})
        
        style = npc_def.get("persona_style", [])
        if not style and "persona_style" in profile_template:
             style = rng.sample(profile_template["persona_style"], min(2, len(profile_template["persona_style"])))
             
        values = npc_def.get("values", profile_template.get("values", []))
        faction = npc_def.get("faction", profile_template.get("faction", "neutral"))
        core_goals = npc_def.get("core_goals", [])
        if not core_goals:
             core_goals = _goals_for_role(role, scenario, rng)

        # Stance logic
        # If npc_def has explicit stance (from a save or specific scenario setup), use it.
        # Otherwise, derive from relationship_type (random or passed).
        stakes = scenario.get("stakes", "medium")
        if relationship_type is None:
            # Check if there's a default relationship in npc_def? No, usually depends on Player.
            # But maybe we want to support "Player is known as X"
             candidates = STAKES_TO_RELATIONSHIP.get(stakes, ["stranger"])
             relationship_type = rng.choice(candidates)
        
        prior = RELATIONSHIP_PRIOR.get(relationship_type, RELATIONSHIP_PRIOR["stranger"])
        initial_stance = npc_def.get("initial_stance", dict(prior))

        profile = {
            "npc_id": npc_id,
            "name": npc_def.get("name", npc_id),
            "role": role,
            "persona_style": style,
            "core_goals": core_goals,
            "values": values,
            "secrets": secrets,
            "faction": faction,
            "initial_stance": initial_stance,
            "relationship_type": relationship_type,
        }
        return profile

    # --- Original Random Generation Logic ---
    profile_template = ROLE_PROFILES.get(role)
    if profile_template is None:
        raise ValueError(f"Unknown NPC role: {role}. Available: {list(ROLE_PROFILES.keys())}")

    stakes = scenario.get("stakes", "medium")
    if relationship_type is None:
        candidates = STAKES_TO_RELATIONSHIP.get(stakes, ["stranger"])
        relationship_type = rng.choice(candidates)

    prior = RELATIONSHIP_PRIOR[relationship_type]

    secret_pool = profile_template["secret_pool"]
    n_secrets = 1 if stakes == "low" else (2 if stakes == "medium" else 2)
    chosen_secrets = rng.sample(secret_pool, min(n_secrets, len(secret_pool)))
    secrets = [
        {
            "secret_id": s,
            "severity": SECRET_SEVERITY.get(s, "medium"),
        }
        for s in chosen_secrets
    ]

    npc_id = f"{role}_{rng.randint(1, 999):03d}"
    style = rng.sample(profile_template["persona_style"], min(2, len(profile_template["persona_style"])))

    profile = {
        "npc_id": npc_id,
        "name": npc_id, # Default name is ID
        "role": role,
        "persona_style": style,
        "core_goals": _goals_for_role(role, scenario, rng),
        "values": list(profile_template["values"]),
        "secrets": secrets,
        "faction": profile_template["faction"],
        "initial_stance": dict(prior),
        "relationship_type": relationship_type,
    }

    errors = validate_npc_profile(profile)
    if errors:
        raise ValueError(f"Invalid NPC profile: {errors}")

    return profile


def validate_npc_profile(profile: dict) -> list[str]:
    errors = []
    required = ["npc_id", "role", "persona_style", "core_goals", "values", "secrets", "initial_stance"]
    for field in required:
        if field not in profile:
            errors.append(f"Missing field: {field}")

    valid_levels = {"VL", "L", "N", "H", "VH"}
    stance = profile.get("initial_stance", {})
    for dim in ["affection", "respect", "dominance", "familiarity", "trust", "obligation"]:
        if dim not in stance:
            errors.append(f"Missing stance dimension: {dim}")
        elif stance[dim] not in valid_levels:
            errors.append(f"Invalid stance level for {dim}: {stance[dim]}")

    return errors


def _goals_for_role(role: str, scenario: dict, rng: random.Random) -> list[str]:
    scenario_type = scenario.get("scenario_type", "")
    stakes = scenario.get("stakes", "medium")

    goal_map = {
        "castle_guard": ["protect_vault", "avoid_scandal", "follow_orders"],
        "steward": ["manage_household", "protect_noble", "maintain_order"],
        "chamberlain": ["serve_lord", "maintain_propriety", "advance_position"],
        "innkeeper": ["earn_profit", "maintain_peace", "avoid_trouble"],
        "merchant": ["complete_deal", "protect_reputation", "maximize_profit"],
        "priest": ["serve_faith", "protect_flock", "maintain_authority"],
        "scholar": ["pursue_knowledge", "protect_research", "maintain_reputation"],
        "soldier": ["follow_orders", "protect_comrades", "survive"],
        "spy": ["complete_mission", "maintain_cover", "avoid_exposure"],
        "healer": ["help_patient", "maintain_neutrality", "protect_knowledge"],
        "blacksmith": ["complete_work", "earn_fair_pay", "maintain_independence"],
        "sergeant": ["maintain_order", "protect_garrison", "follow_chain_of_command"],
        "apothecary": ["protect_knowledge", "maintain_business", "avoid_scrutiny"],
        "guild_master": ["protect_guild", "maintain_power", "expand_influence"],
        "farmer": ["protect_family", "maintain_land", "survive_season"],
    }

    goals = goal_map.get(role, ["complete_objective", "avoid_harm"])
    n = 2 if stakes == "low" else 3
    return rng.sample(goals, min(n, len(goals)))
