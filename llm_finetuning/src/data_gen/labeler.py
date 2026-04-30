import json
import re
from pathlib import Path
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential


def _load_prompt(path: str) -> str:
    return Path(path).read_text()


# ── Label normalization (fixes known teacher LLM output errors) ──────────────

_VALID_STANCE_LEVELS = {"VL", "L", "N", "H", "VH"}
_VALID_STANCE_DELTAS = {"--", "-", "0", "+", "++"}

_RESPONSE_POLICY_TYPOS = {
    "defect": "deflect",
    "challenged": "challenge",
    "redirect": "deflect",
    "reveal": "answer",
    "ignore": "withhold",
    "evade": "deflect",
    "avoid": "deflect",
    "confront": "challenge",
    "reassure": "soothe",
    "comfort": "soothe",
    "bargain": "negotiate",
    "warn": "threaten",
    "deny": "withhold",
}

_VALID_RESPONSE_POLICIES = {"answer", "partial", "withhold", "deflect", "challenge",
                            "soothe", "test", "threaten", "negotiate", "clarify"}
_VALID_REVEAL = {"none", "hint", "partial", "full"}
_VALID_REPAIR = {"none", "soften", "apologize", "clarify", "redirect"}
_VALID_THREE = {"low", "medium", "high"}
_VALID_VALUE_CONFLICT = {"none", "mild", "strong"}

STANCE_DIMS = ["affection", "respect", "dominance", "familiarity", "trust", "obligation"]


def _normalize_stance_entry(entry: dict) -> dict:
    """Fix stance entries where teacher merged level+delta into one field.

    Examples of corrupted values:
        level="H+"  → level="H", delta="+"
        level="VH(-)" → level="VH", delta="-"
        level="H(++)" → level="H", delta="++"
        level="+"   → level unchanged from prior, delta="+"
        level="- "  → level unchanged from prior, delta="-"
    """
    if not isinstance(entry, dict):
        return {"level": "N", "delta": "0"}

    level = str(entry.get("level", "N")).strip()
    delta = str(entry.get("delta", "0")).strip()

    # Fix delta: normalize unicode minus, strip whitespace/periods
    delta = delta.replace("\u2212", "-").rstrip(".").strip()

    # Case 1: level contains parenthetical delta like "H(+)" or "VH(--)"
    import re
    paren_match = re.match(r"^(VL|VH|[LNHV])\s*\(([^)]+)\)", level)
    if paren_match:
        level = paren_match.group(1)
        delta = paren_match.group(2).strip()

    # Case 2: level has delta appended like "H+", "VH+", "L-"
    suffix_match = re.match(r"^(VL|VH|[LNHV])([+\-]+)$", level)
    if suffix_match:
        level = suffix_match.group(1)
        delta = suffix_match.group(2).strip()

    # Case 3: level is just a delta value like "+", "-", "0"
    if level in _VALID_STANCE_DELTAS and level not in _VALID_STANCE_LEVELS:
        delta = level
        level = "N"  # default to neutral when level is missing

    # Final validation
    if level not in _VALID_STANCE_LEVELS:
        level = "N"
    if delta not in _VALID_STANCE_DELTAS:
        delta = "0"

    return {"level": level, "delta": delta}


def normalize_labels(labels: dict) -> dict:
    """Normalize and fix all label fields in a turn record.

    Call this after extracting JSON from teacher LLM output and before
    validation. Fixes known typos, malformed stance entries, and
    out-of-vocabulary values.
    """
    # ── D_t: response_policy ──
    rp = labels.get("response_policy", "")
    if isinstance(rp, str):
        rp = rp.strip().lower()
        rp = _RESPONSE_POLICY_TYPOS.get(rp, rp)
        if rp not in _VALID_RESPONSE_POLICIES:
            rp = "soothe"  # safe default for unknown policies
        labels["response_policy"] = rp

    rd = labels.get("reveal_decision", "")
    if isinstance(rd, str):
        rd = rd.strip().lower()
        if rd not in _VALID_REVEAL:
            rd = "none"
        labels["reveal_decision"] = rd

    rs = labels.get("repair_strategy", "")
    if isinstance(rs, str):
        rs = rs.strip().lower()
        if rs not in _VALID_REPAIR:
            rs = "none"
        labels["repair_strategy"] = rs

    # ── N_t: pressure fields ──
    for field in ["duty_pressure", "secrecy_pressure", "face_pressure"]:
        val = labels.get(field, "")
        if isinstance(val, str):
            val = val.strip().lower()
            if val not in _VALID_THREE:
                val = "medium"
            labels[field] = val

    vc = labels.get("value_conflict", "")
    if isinstance(vc, str):
        vc = vc.strip().lower()
        if vc not in _VALID_VALUE_CONFLICT:
            vc = "none"
        labels["value_conflict"] = vc

    # ── R_t: stance dimensions ──
    for dim in STANCE_DIMS:
        if dim in labels and isinstance(labels[dim], dict):
            labels[dim] = _normalize_stance_entry(labels[dim])
        # Also handle flat keys like "affection_level", "affection_delta"
        level_key = f"{dim}_level"
        delta_key = f"{dim}_delta"
        if level_key in labels or delta_key in labels:
            entry = {"level": labels.get(level_key, "N"), "delta": labels.get(delta_key, "0")}
            fixed = _normalize_stance_entry(entry)
            labels[level_key] = fixed["level"]
            labels[delta_key] = fixed["delta"]

    return labels


def _format_dialogue_history(history: list[dict]) -> str:
    lines = []
    for turn in history:
        lines.append(f"[Turn {turn['turn_idx']}] Player: {turn['player_utterance']}")
        if turn.get("response"):
            lines.append(f"[Turn {turn['turn_idx']}] NPC: {turn['response']}")
    return "\n".join(lines) if lines else "(no prior dialogue)"


def _format_stance(stance: dict) -> str:
    parts = []
    for dim, val in stance.items():
        if isinstance(val, dict):
            parts.append(f"{dim}={val.get('level', '?')}({val.get('delta', '0')})")
        else:
            parts.append(f"{dim}={val}")
    return "  ".join(parts)


# ── Behaviour description helpers (map coded values → natural language) ──────

_POLICY_DESCRIPTIONS = {
    "withhold": "Dodge or avoid answering — give a vague, evasive response",
    "deflect": "Change the subject without answering the question",
    "challenge": "Push back and demand proof or explanation",
    "soothe": "Be reassuring and calm — reduce tension",
    "threaten": "Assert yourself with a firm warning, implicit or explicit",
    "negotiate": "Offer a conditional exchange or compromise",
    "answer": "Answer honestly and directly",
    "reveal": "Share the relevant information openly",
    "ignore": "Act as if the question was not asked",
}

_REVEAL_DESCRIPTIONS = {
    "none": "Keep ALL secrets completely hidden — do not hint at anything",
    "hint": "You may give only a very vague, deniable hint — no direct confirmation",
    "partial": "You may reveal one minor, non-critical detail only",
    "full": "You may speak openly about this matter",
}

_REPAIR_INSTRUCTIONS = {
    "none": "",
    "apologize": "- Also include a genuine apology.",
    "clarify": "- Also correct a misunderstanding the other person has.",
    "redirect": "- Also steer the conversation toward a different topic.",
    "soften": "- Also soften your tone and approach.",
}


def _describe_policy(policy: str) -> str:
    return _POLICY_DESCRIPTIONS.get(policy, f"Respond with a {policy} approach")


def _describe_reveal(reveal: str) -> str:
    return _REVEAL_DESCRIPTIONS.get(reveal, _REVEAL_DESCRIPTIONS["none"])


def _describe_repair(repair: str) -> str:
    return _REPAIR_INSTRUCTIONS.get(repair, "")


def _split_system_user(template: str):
    """Split a template that contains [SYSTEM] and [USER] markers.
    Returns (system_text, user_text) or (None, full_text) if no markers."""
    if "[SYSTEM]" in template and "[USER]" in template:
        parts = template.split("[USER]", 1)
        system_part = parts[0].replace("[SYSTEM]", "").strip()
        user_part = parts[1].strip()
        return system_part, user_part
    return None, template


def _extract_json(text: str) -> dict:
    text = text.strip()
    
    # Remove markdown code blocks if present
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    
    # Try to find JSON object or array
    for pattern in [r"\{.*\}", r"\[.*\]"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            raw = match.group()
            # Coerce Python-style literals
            raw = re.sub(r"\bNone\b", "null", raw)
            raw = re.sub(r"\bTrue\b", "true", raw)
            raw = re.sub(r"\bFalse\b", "false", raw)
            # Remove trailing commas
            raw = re.sub(r",\s*([}\]])", r"\1", raw)
            
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                # Try fixing single quotes
                try:
                    return json.loads(raw.replace("'", '"'))
                except json.JSONDecodeError:
                    pass
                
                # Try fixing unquoted keys
                try:
                    fixed = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', raw)
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    pass
    
    # If all extraction attempts fail, raise with context
    raise ValueError(f"No valid JSON found in response: {text[:200]}...")


class Labeler:
    def __init__(self, client, prompt_config: dict, model: str = "gpt-4o"):
        self.client = client
        self.model = model
        
        # Detect prompt template family
        self.template_family = "default"
        model_lower = model.lower()
        if "qwen" in model_lower:
            self.template_family = "qwen"
        elif "mistral" in model_lower or "mixtral" in model_lower:
            self.template_family = "mistral"
        elif "gemini" in model_lower:
            self.template_family = "gemini"
        elif "gpt" in model_lower or "openai" in model_lower:
            self.template_family = "openai"
            
        # Load prompts, checking for model-specific overrides first
        self.prompts = {}
        for key, path in prompt_config.items():
            # Check for model-specific template override
            if self.template_family != "default":
                template_path = Path(f"prompts/templates/{self.template_family}_{key}.txt")
                if template_path.exists():
                    self.prompts[key] = template_path.read_text()
                    print(f"[Labeler] Loaded {self.template_family} template for {key}")
                    continue
            
            self.prompts[key] = _load_prompt(path)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _call_llm(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    def label_C_A_M(
        self,
        player_utterance: str,
        npc_profile: dict,
        scenario: dict,
        arc_phase: str,
        prior_stance: dict,
        history: list[dict],
    ) -> tuple[dict, dict, dict]:
        """Merged labeling: produce C_t, A_t, M_t in a single LLM call."""
        if "label_C_A_M" in self.prompts:
            prompt = self.prompts["label_C_A_M"].format(
                setting=scenario.get("setting", "unknown"),
                npc_role=npc_profile["role"],
                npc_goals=", ".join(npc_profile.get("core_goals", [])),
                npc_secrets=", ".join(s["secret_id"] for s in npc_profile.get("secrets", [])),
                npc_values=", ".join(npc_profile.get("values", [])),
                arc_phase=arc_phase,
                prior_stance=_format_stance(prior_stance),
                dialogue_history=_format_dialogue_history(history),
                player_utterance=player_utterance,
            )
            raw = self._call_llm(prompt)
            result = _extract_json(raw)
            C_t = normalize_labels(result.get("C_t", {}))
            A_t = normalize_labels(result.get("A_t", {}))
            M_t = normalize_labels(result.get("M_t", {}))
            return C_t, A_t, M_t
        else:
            # Fallback: two separate calls (backwards compatible)
            C_t = self.label_C(
                player_utterance=player_utterance,
                npc_profile=npc_profile,
                scenario=scenario,
                arc_phase=arc_phase,
                history=history,
            )
            A_t, M_t = self.label_A_M(
                player_utterance=player_utterance,
                C_t=C_t,
                npc_profile=npc_profile,
                scenario=scenario,
                prior_stance=prior_stance,
                history=history,
            )
            return C_t, A_t, M_t

    def label_C(
        self,
        player_utterance: str,
        npc_profile: dict,
        scenario: dict,
        arc_phase: str,
        history: list[dict],
    ) -> dict:
        prompt = self.prompts["label_C"].format(
            setting=scenario.get("setting", "unknown"),
            npc_role=npc_profile["role"],
            npc_goals=", ".join(npc_profile.get("core_goals", [])),
            npc_secrets=", ".join(s["secret_id"] for s in npc_profile.get("secrets", [])),
            arc_phase=arc_phase,
            dialogue_history=_format_dialogue_history(history),
            player_utterance=player_utterance,
        )
        raw = self._call_llm(prompt)
        result = _extract_json(raw)
        return normalize_labels(result)

    def label_A_M(
        self,
        player_utterance: str,
        C_t: dict,
        npc_profile: dict,
        scenario: dict,
        prior_stance: dict,
        history: list[dict],
    ) -> dict:
        prompt = self.prompts["label_A_M"].format(
            setting=scenario.get("setting", "unknown"),
            npc_role=npc_profile["role"],
            npc_goals=", ".join(npc_profile.get("core_goals", [])),
            npc_secrets=", ".join(s["secret_id"] for s in npc_profile.get("secrets", [])),
            npc_values=", ".join(npc_profile.get("values", [])),
            prior_stance=_format_stance(prior_stance),
            dialogue_history=_format_dialogue_history(history),
            player_utterance=player_utterance,
            C_t=json.dumps(C_t),
        )
        raw = self._call_llm(prompt)
        result = _extract_json(raw)
        A_t = normalize_labels(result.get("A_t", result))
        M_t = normalize_labels(result.get("M_t", {}))
        return A_t, M_t

    def label_R_N_D(
        self,
        player_utterance: str,
        C_t: dict,
        A_t: dict,
        M_t: dict,
        npc_profile: dict,
        scenario: dict,
        arc: dict,
        arc_phase: str,
        required_shifts: list[dict],
        prior_R_t: dict,
        history: list[dict],
        target_policy: str = "",
    ) -> dict:
        prompt = self.prompts["label_R_N_D"].format(
            setting=scenario.get("setting", "unknown"),
            npc_role=npc_profile["role"],
            npc_goals=", ".join(npc_profile.get("core_goals", [])),
            npc_secrets=", ".join(s["secret_id"] for s in npc_profile.get("secrets", [])),
            npc_values=", ".join(npc_profile.get("values", [])),
            npc_faction=npc_profile.get("faction", "none"),
            prior_R_t=_format_stance(prior_R_t),
            arc_phase=arc_phase,
            required_shifts=json.dumps(required_shifts),
            allowed_reveal_ceiling=arc.get("allowed_reveal", "hint"),
            dialogue_history=_format_dialogue_history(history),
            player_utterance=player_utterance,
            C_t=json.dumps(C_t),
            A_t=json.dumps(A_t),
            M_t=json.dumps(M_t),
            target_policy=target_policy,
        )
        raw = self._call_llm(prompt)
        result = _extract_json(raw)
        R_t = result.get("R_t", {})
        N_t = normalize_labels(result.get("N_t", {}))
        D_t = normalize_labels(result.get("D_t", {}))
        # Normalize each stance dimension inside R_t
        for dim in STANCE_DIMS:
            if dim in R_t and isinstance(R_t[dim], dict):
                R_t[dim] = _normalize_stance_entry(R_t[dim])
        return R_t, N_t, D_t

    def generate_response(
        self,
        player_utterance: str,
        C_t: dict,
        A_t: dict,
        M_t: dict,
        R_t: dict,
        N_t: dict,
        D_t: dict,
        npc_profile: dict,
        history: list[dict],
        scenario: dict = None,
    ) -> str:
        stance = R_t
        # Extract world context from scenario if available
        world_context = ""
        scenario_type = "unknown"
        setting = "unknown"
        stakes = "medium"
        
        if scenario:
            world_context = scenario.get("world_context", "")
            scenario_type = scenario.get("scenario_type", "unknown")
            setting = scenario.get("setting", "unknown")
            stakes = scenario.get("stakes", "medium")

        response_policy = D_t.get("response_policy", "answer")
        reveal_decision = D_t.get("reveal_decision", "none")
        repair_strategy = D_t.get("repair_strategy", "none")

        # Computed natural-language descriptions (prevents state-label echo)
        policy_desc = _describe_policy(response_policy)
        reveal_desc = _describe_reveal(reveal_decision)
        repair_instr = _describe_repair(repair_strategy)

        template = self.prompts["response_generation"]
        sys_tmpl, usr_tmpl = _split_system_user(template)

        fmt_kwargs = dict(
            npc_role=npc_profile["role"],
            persona_style=", ".join(npc_profile.get("persona_style", [])),
            npc_goals=", ".join(npc_profile.get("core_goals", [])),
            npc_values=", ".join(npc_profile.get("values", [])),
            npc_secrets="; ".join(
                f"{s['secret_id']} (severity={s['severity']})"
                for s in npc_profile.get("secrets", [])
            ),
            world_context=world_context,
            scenario_type=scenario_type,
            setting=setting,
            stakes=stakes,
            affection_level=stance.get("affection", {}).get("level", "N"),
            respect_level=stance.get("respect", {}).get("level", "N"),
            dominance_level=stance.get("dominance", {}).get("level", "N"),
            familiarity_level=stance.get("familiarity", {}).get("level", "VL"),
            trust_level=stance.get("trust", {}).get("level", "VL"),
            obligation_level=stance.get("obligation", {}).get("level", "VL"),
            dialogue_history=_format_dialogue_history(history),
            player_utterance=player_utterance,
            dialogue_act=", ".join(C_t.get("dialogue_act", [])),
            tone=C_t.get("tone", "neutral"),
            risk_type=C_t.get("risk_type", "none"),
            valence=A_t.get("valence", "neutral"),
            arousal=A_t.get("arousal", "medium"),
            threat=A_t.get("threat", "low"),
            player_intent=M_t.get("player_intent", "seek-info"),
            player_knowledge=M_t.get("player_knowledge", "partial"),
            secrecy_pressure=N_t.get("secrecy_pressure", "low"),
            face_pressure=N_t.get("face_pressure", "low"),
            duty_pressure=N_t.get("duty_pressure", "low"),
            response_policy=response_policy,
            reveal_decision=reveal_decision,
            repair_strategy=repair_strategy,
            # Natural-language descriptions for Qwen template
            response_policy_description=policy_desc,
            reveal_description=reveal_desc,
            repair_instruction=repair_instr,
        )

        if sys_tmpl is not None:
            # System+user split: send as two messages
            sys_content = sys_tmpl.format(**fmt_kwargs)
            usr_content = usr_tmpl.format(**fmt_kwargs)
            messages_template = [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": usr_content},
            ]
        else:
            full_prompt = template.format(**fmt_kwargs)
            messages_template = [{"role": "user", "content": full_prompt}]

        # State-label pattern: detects variable=value pairs in any form
        _STATE_LABEL_RE = re.compile(
            r"(\b(?:valence|arousal|threat|secrecy|face|duty|policy|response_policy|"
            r"reveal|repair|dialogue_act|player_intent|player_knowledge|tone|risk|"
            r"response|[cnamrd]_t)\s*[=:]"
            r"|\b(?:C_t|A_t|M_t|N_t|D_t|R_t)\s*[=:]"
            r"|/no_input|/d_t|/keep_|/clear)",
            re.IGNORECASE,
        )
        # Single-token state-key pattern (for line-start checks)
        _STATE_KEY_LINE_RE = re.compile(
            r"^(?:secrets|stakes|setting|arc_phase)\s*:",
            re.IGNORECASE,
        )

        for attempt in range(3):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages_template,
                temperature=0.7 + (attempt * 0.1),
                max_completion_tokens=200,
            )
            content = response.choices[0].message.content.strip()

            # ── Cleaning pipeline ────────────────────────────────────────────────────

            # 0. Strip /no_think inline (before per-line loop to avoid false drops)
            content = re.sub(r"\s*/no_think\b", "", content, flags=re.IGNORECASE).strip()

            # 1. Strip role-prefix labels (NPC:, Response:, Me:, etc.)
            content = re.sub(
                r"^(NPC RESPONSE|NPC|Response|Me|You)(\s*\(.*?\))?:\s*",
                "", content, flags=re.IGNORECASE
            ).strip()

            # 2. Remove markdown bold state labels: **tone=neutral** etc.
            content = re.sub(r"\*\*[^*]*=[^*]*\*\*", "", content)
            # 2b. Strip single-asterisk emphasis *text* (keep inner text)
            content = re.sub(r"\*([^*\n]+)\*", r"\1", content)
            # Strip comma-only residue left after bold-label removal
            content = re.sub(r"^[\s,]+$", "", content, flags=re.MULTILINE)
            content = content.strip()

            # 3. Strip trailing parenthetical state annotations: (C_t: ...) or (A_t: ...)
            content = re.sub(r"\s*\([CNAMRD]_t:[^)]*\)", "", content, flags=re.IGNORECASE).strip()

            # 4. Per-line filtering — drop any line that smells like state leak
            lines = content.split("\n")
            clean_lines = []
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                # 4a. Try to salvage quoted dialogue from "state_label=val: 'quote'" lines
                if _STATE_LABEL_RE.search(stripped):
                    salvage = re.search(r'["\u2018\u2019\u201c\u201d](.+?)["\u2018\u2019\u201c\u201d]', stripped)
                    if salvage:
                        clean_lines.append(salvage.group(1).strip())
                    continue
                if re.match(r"^/", stripped):  # any slash-command line
                    continue
                if re.search(r"\*\*[a-z_]+=" , stripped):  # **var=val** leaks
                    continue
                # Drop single-token state-key lines: "secrets: ...", "stakes: ..."
                if _STATE_KEY_LINE_RE.match(stripped):
                    continue
                if re.search(r"^[a-z_]+\s*:\s*[a-z_]+", stripped):  # "tone: neutral"
                    # Only drop if it looks like a state dump (multiple key: value pairs)
                    if re.search(r"[a-z_]+\s*:\s*[a-z_]+.*[a-z_]+\s*:", stripped):
                        continue
                clean_lines.append(line)
            content = "\n".join(clean_lines).strip()

            # 5. Strip any residual /no_input markers
            content = re.sub(r"\*\*/no_input\*\*", "", content).strip()
            content = re.sub(r"\s*/no_input\b", "", content, flags=re.IGNORECASE).strip()

            # 5b. Strip third-person narrative wrapping
            # Detect: content starts with generic subject + 3rd-person verb
            # e.g. "She says", "He nods", "The Blacksmith turns", "Elara looks"
            _NARRATION_START_RE = re.compile(
                r"^(?:She|He|They|It)\s+"
                r"(?:places|rests|traces|speaks|says|looks|nods|gestures|sighs|"
                r"pauses|turns|steps|walks|raises|lowers|smiles|frowns|whispers|"
                r"murmurs|glances|narrows|straightens|squares|leans|shifts|clasps|"
                r"reaches|moves|stands|sits|crosses|draws|holds|lets|shakes|tilts|"
                r"debates|deliberates|considers|awaits|guards|ensures|remains|continues)"
                r"|"
                r"^(?:The|A)\s+[A-Z][a-z]+\s+"  # The Blacksmith ...
                r"(?:places|rests|traces|speaks|says|looks|nods|gestures|sighs|"
                r"pauses|turns|steps|walks|raises|lowers|smiles|frowns|whispers|"
                r"murmurs|glances|narrows|straightens|squares|leans|shifts|clasps|"
                r"reaches|moves|stands|sits|crosses|draws|holds|lets|shakes|tilts|"
                r"debates|deliberates|considers|awaits|guards|ensures|remains|continues)",
                re.MULTILINE,
            )
            if _NARRATION_START_RE.search(content):
                # Extract all quoted strings from the narration
                quotes = re.findall(
                    r'["\u201c\u2018](.+?)["\u201d\u2019]', content, re.DOTALL
                )
                if quotes:
                    content = " ".join(q.strip() for q in quotes if len(q.strip()) > 5)
                else:
                    content = ""  # no salvageable dialogue — force retry

            # 6. Strip speech-attribution suffixes: ," I replied. / ," she said.
            content = re.sub(
                r'[,.]?["\u201d]\s*(?:I|he|she|they)\s+(?:said|replied|responded|added|noted|answered|'  
                r'murmured|whispered|continued|insisted|asked|explained|told them|told him|told her)'
                r'[^.]*\.?\s*$',
                '"', content, flags=re.IGNORECASE
            ).strip()

            # 6b. Remove surrounding quotes if the whole string is quoted
            if content.startswith('"') and content.endswith('"'):
                content = content[1:-1].strip()

            if content:
                return content

        # Fallback
        return "..."

    def generate_player_utterance(
        self,
        scenario: dict,
        arc_phase: str,
        npc_profile: dict,
        history: list[dict],
    ) -> str:
        prompt = self.prompts["player_generation"].format(
            world_context=scenario.get("world_context", ""),
            scenario_type=scenario["scenario_type"],
            setting=scenario.get("setting", "unknown"),
            stakes=scenario.get("stakes", "medium"),
            arc_phase=arc_phase,
            npc_role=npc_profile["role"],
            persona_style=", ".join(npc_profile.get("persona_style", [])),
            dialogue_history=_format_dialogue_history(history),
        )

        # Retry loop to avoid repetition artifacts
        max_retries = 3
        last_npc_response = history[-1].get("response", "").strip() if history else ""
        last_player_utterance = history[-1].get("player_utterance", "").strip() if history else ""
        if not last_player_utterance and len(history) > 1:
             last_player_utterance = history[-2].get("player_utterance", "").strip()

        for attempt in range(max_retries):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7 + (attempt * 0.1), # Increase temp slightly on retries
                max_completion_tokens=150,
            )
            content = response.choices[0].message.content.strip()
            
            # Cleaning artifacts
            # 1. Remove meta-labels (Player, NPC, Me, etc.)
            content = re.sub(r"^(PLAYER UTTERANCE|Player|Me|You|NPC|Response)(\s*\(.*?\))?:", "", content, flags=re.IGNORECASE).strip()
            
            # 1b. Remove leaked tags/actions starting with / or *
            content = re.sub(r"^/.*?\n", "", content, flags=re.MULTILINE).strip()
            content = re.sub(r"^\*+.*?\*+", "", content, flags=re.MULTILINE).strip()
            content = re.sub(r"\*\*/no_input\*\*", "", content).strip()
            
            if content.startswith('"') and content.endswith('"'):
                content = content[1:-1].strip()
                
            # 3. Check for repetition of the last NPC response (hallucination/completion mode)
            is_npc_repeat = False
            if last_npc_response:
                # Check if content is a significant substring of last_response
                if len(content) > 10 and (content in last_npc_response):
                    is_npc_repeat = True
                # Check if last_response is in content
                elif len(last_npc_response) > 10 and (last_npc_response in content):
                    # Remove the repeated NPC text
                    content = content.replace(last_npc_response, "").strip()
                    # Re-clean any labels that might have been exposed (e.g. "Player:")
                    content = re.sub(r"^(PLAYER UTTERANCE|Player|Me|You)(\s*\(.*?\))?:", "", content, flags=re.IGNORECASE).strip()
                    if not content: 
                        is_npc_repeat = True
                
                # Fallback: Check for case-insensitive match if exact match failed
                if not is_npc_repeat and len(last_npc_response) > 10:
                    lower_content = content.lower()
                    lower_last = last_npc_response.lower()
                    if lower_last in lower_content:
                         # This is harder to strip cleanly without regex, so we'll just flag it if it's the whole thing
                         if len(content) < len(last_npc_response) * 1.5: # content is mostly the repetition
                             is_npc_repeat = True
                         else:
                             # Try to strip via regex
                             content = re.sub(re.escape(last_npc_response), "", content, flags=re.IGNORECASE).strip()
                             content = re.sub(r"^(PLAYER UTTERANCE|Player|Me|You)(\s*\(.*?\))?:", "", content, flags=re.IGNORECASE).strip()
                             if not content:
                                 is_npc_repeat = True

            # 4. Check for repetition of self (looping)

            # 4. Check for repetition of self (looping)
            is_self_repeat = False
            if last_player_utterance and len(content) > 10:
                # Simple check: if identical or contained
                if content == last_player_utterance or (content in last_player_utterance) or (last_player_utterance in content):
                     is_self_repeat = True

            if not is_npc_repeat and not is_self_repeat and content:
                return content
            
            # If we are here, we failed a check. Loop again.
            # print(f"DEBUG: Rejected player utterance (attempt {attempt}): {content!r} (NPC_Repeat={is_npc_repeat}, Self_Repeat={is_self_repeat})")

        # Fallback if all retries fail
        return "I remain silent, watching carefully."

    def generate_scene_opening(
        self,
        scenario: dict,
        npc_profile: dict,
    ) -> dict:
        prompt = self.prompts["scenario_instantiation"].format(
            world_context=scenario.get("world_context", ""),
            scenario_type=scenario["scenario_type"],
            setting=scenario.get("setting", "unknown"),
            stakes=scenario.get("stakes", "medium"),
            required_events=", ".join(scenario.get("required_events", [])),
            npc_role=npc_profile["role"],
            persona_style=", ".join(npc_profile.get("persona_style", [])),
            npc_goals=", ".join(npc_profile.get("core_goals", [])),
            npc_values=", ".join(npc_profile.get("values", [])),
            npc_secrets="; ".join(s["secret_id"] for s in npc_profile.get("secrets", [])),
        )
        raw = self._call_llm(prompt)
        return _extract_json(raw)
