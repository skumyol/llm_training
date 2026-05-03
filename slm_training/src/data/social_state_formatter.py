#!/usr/bin/env python3
"""
social_state_formatter.py
=========================
Format social state metadata for injection into LLM prompts.

Supports multiple serialization formats for social state conditioning:
  - xml: Structured XML tags (default)
  - json: JSON object
  - text: Plain text description
  - yaml: YAML frontmatter

Usage:
    from social_state_formatter import SocialStateFormatter
    
    formatter = SocialStateFormatter(format_type="xml")
    state_text = formatter.format_state({
        "scenario_type": "secret_extraction",
        "response_policy": "deflect",
        "reveal_decision": "none",
        "trust_level": "low"
    })
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SocialState:
    """Structured social state for NPC dialogue."""
    scenario_type: str = "unknown"
    response_policy: str = "neutral"
    reveal_decision: str = "none"
    valence: str = "neutral"
    arousal: str = "medium"
    trust_level: Optional[str] = None
    secrecy_level: Optional[str] = None
    
    @classmethod
    def from_metadata(cls, metadata: Dict[str, Any]) -> "SocialState":
        """Extract social state from dialogue metadata."""
        # Derive trust level from scenario and reveal patterns
        trust_level = cls._derive_trust_level(metadata)
        secrecy_level = cls._derive_secrecy_level(metadata)
        
        return cls(
            scenario_type=metadata.get("scenario_type", "unknown"),
            response_policy=metadata.get("response_policy", "neutral"),
            reveal_decision=metadata.get("reveal_decision", "none"),
            valence=metadata.get("valence", "neutral"),
            arousal=metadata.get("arousal", "medium"),
            trust_level=trust_level,
            secrecy_level=secrecy_level,
        )
    
    @staticmethod
    def _derive_trust_level(metadata: Dict[str, Any]) -> str:
        """Derive trust level from scenario type and response policy."""
        scenario = metadata.get("scenario_type", "")
        policy = metadata.get("response_policy", "")
        
        if scenario == "trust_building":
            return "medium" if policy in ["soothe", "clarify"] else "low"
        elif scenario == "secret_extraction":
            return "low" if policy in ["deflect", "evade", "threaten"] else "medium"
        return "medium"
    
    @staticmethod
    def _derive_secrecy_level(metadata: Dict[str, Any]) -> str:
        """Derive secrecy level from reveal decision and scenario."""
        reveal = metadata.get("reveal_decision", "none")
        scenario = metadata.get("scenario_type", "")
        
        if reveal == "none":
            return "high" if scenario == "secret_extraction" else "medium"
        elif reveal == "partial":
            return "medium"
        return "low"


class SocialStateFormatter:
    """Format social state for prompt injection."""
    
    def __init__(self, format_type: str = "xml", include_descriptions: bool = True):
        """
        Args:
            format_type: Output format - "xml", "json", "text", or "yaml"
            include_descriptions: Include human-readable descriptions
        """
        self.format_type = format_type.lower()
        self.include_descriptions = include_descriptions
        
        # Policy descriptions for human readability
        self.policy_desc = {
            "deflect": "Avoid direct answer, steer conversation away",
            "evade": "Act as if question wasn't asked",
            "clarify": "Ask for clarification before responding",
            "soothe": "Calm the situation, reduce tension",
            "threaten": "Warn or intimidate the player",
            "test": "Test player's intentions before trusting",
            "reveal": "Share information openly",
            "negotiate": "Offer information for something in return",
        }
        
        self.reveal_desc = {
            "none": "Keep all secrets, reveal nothing",
            "partial": "Hint at information without full disclosure",
            "full": "Reveal complete information",
        }
    
    def format_state(self, metadata: Dict[str, Any]) -> str:
        """Format social state from metadata dict."""
        state = SocialState.from_metadata(metadata)
        
        if self.format_type == "xml":
            return self._format_xml(state)
        elif self.format_type == "json":
            return self._format_json(state)
        elif self.format_type == "text":
            return self._format_text(state)
        elif self.format_type == "yaml":
            return self._format_yaml(state)
        else:
            raise ValueError(f"Unknown format: {self.format_type}")
    
    def _format_xml(self, state: SocialState) -> str:
        """Format as XML tags."""
        lines = ["<social_state>"]
        lines.append(f'  <scenario>{state.scenario_type}</scenario>')
        lines.append(f'  <response_policy>{state.response_policy}</response_policy>')
        if self.include_descriptions and state.response_policy in self.policy_desc:
            lines.append(f'    <!-- {self.policy_desc[state.response_policy]} -->')
        lines.append(f'  <reveal_decision>{state.reveal_decision}</reveal_decision>')
        if self.include_descriptions and state.reveal_decision in self.reveal_desc:
            lines.append(f'    <!-- {self.reveal_desc[state.reveal_decision]} -->')
        lines.append(f'  <emotional_state valence="{state.valence}" arousal="{state.arousal}" />')
        if state.trust_level:
            lines.append(f'  <trust_level>{state.trust_level}</trust_level>')
        if state.secrecy_level:
            lines.append(f'  <secrecy_level>{state.secrecy_level}</secrecy_level>')
        lines.append("</social_state>")
        return "\n".join(lines)
    
    def _format_json(self, state: SocialState) -> str:
        """Format as JSON."""
        data = {
            "scenario_type": state.scenario_type,
            "response_policy": state.response_policy,
            "reveal_decision": state.reveal_decision,
            "emotional_state": {"valence": state.valence, "arousal": state.arousal},
        }
        if state.trust_level:
            data["trust_level"] = state.trust_level
        if state.secrecy_level:
            data["secrecy_level"] = state.secrecy_level
        if self.include_descriptions:
            data["policy_description"] = self.policy_desc.get(state.response_policy, "")
            data["reveal_description"] = self.reveal_desc.get(state.reveal_decision, "")
        return json.dumps(data, indent=2)
    
    def _format_text(self, state: SocialState) -> str:
        """Format as plain text."""
        lines = ["Current Situation:"]
        lines.append(f"- This is a {state.scenario_type.replace('_', ' ')} scenario")
        lines.append(f"- Response approach: {state.response_policy}")
        if self.include_descriptions and state.response_policy in self.policy_desc:
            lines.append(f"  ({self.policy_desc[state.response_policy]})")
        lines.append(f"- Information sharing: {state.reveal_decision}")
        if self.include_descriptions and state.reveal_decision in self.reveal_desc:
            lines.append(f"  ({self.reveal_desc[state.reveal_decision]})")
        lines.append(f"- Emotional state: {state.valence}, {state.arousal} intensity")
        if state.trust_level:
            lines.append(f"- Trust with player: {state.trust_level}")
        return "\n".join(lines)
    
    def _format_yaml(self, state: SocialState) -> str:
        """Format as YAML frontmatter."""
        lines = ["---"]
        lines.append(f'scenario_type: {state.scenario_type}')
        lines.append(f'response_policy: {state.response_policy}')
        lines.append(f'reveal_decision: {state.reveal_decision}')
        lines.append(f'valence: {state.valence}')
        lines.append(f'arousal: {state.arousal}')
        if state.trust_level:
            lines.append(f'trust_level: {state.trust_level}')
        if state.secrecy_level:
            lines.append(f'secrecy_level: {state.secrecy_level}')
        lines.append("---")
        return "\n".join(lines)


def build_conditioned_prompt(
    npc_profile: str,
    dialogue_context: List[Dict[str, str]],
    metadata: Dict[str, Any],
    formatter: Optional[SocialStateFormatter] = None,
    include_social_state: bool = True,
) -> str:
    """
    Build a complete conditioned prompt for LLM training/inference.
    
    Args:
        npc_profile: NPC description and personality
        dialogue_context: List of {speaker, text} turns
        metadata: Episode metadata containing social state
        formatter: SocialStateFormatter instance (uses XML if None)
        include_social_state: Whether to inject social state
        
    Returns:
        Complete formatted prompt string
    """
    if formatter is None:
        formatter = SocialStateFormatter(format="xml")
    
    parts = []
    
    # NPC Profile section
    parts.append("<npc_profile>")
    parts.append(npc_profile)
    parts.append("</npc_profile>")
    
    # Social State section (conditional)
    if include_social_state:
        parts.append("")
        parts.append(formatter.format_state(metadata))
    
    # Dialogue History section
    parts.append("")
    parts.append("<dialogue_history>")
    for turn in dialogue_context:
        speaker = turn.get("speaker", "unknown")
        text = turn.get("text", "")
        parts.append(f"<{speaker}>{text}</{speaker}>")
    parts.append("</dialogue_history>")
    
    return "\n".join(parts)


# Template prompts for different conditioning modes
# Templates with {npc_profile} are for SFT-only mode
# Templates with {context} are for social state modes (context includes profile + social state)
SYSTEM_PROMPT_TEMPLATES = {
    "none": (
        "You are roleplaying an NPC in a dialogue simulation. Stay in character, "
        "reply naturally, and use the NPC profile below as a hard constraint.\n\n"
        "NPC Profile: {npc_profile}"
    ),
    
    "social_state_xml": (
        "You are roleplaying an NPC in a dialogue simulation. "
        "The <social_state> block below defines your current situation, "
        "response policy, and emotional state. Follow it strictly.\n\n"
        "{context}"
    ),
    
    "social_state_text": (
        "You are roleplaying an NPC in a dialogue simulation. "
        "Pay attention to the Current Situation instructions below.\n\n"
        "{context}"
    ),
    
    "social_state_json": (
        "You are roleplaying an NPC in a dialogue simulation. "
        "Use the JSON social state to guide your response.\n\n"
        "{context}"
    ),
    
    "social_state_yaml": (
        "You are roleplaying an NPC in a dialogue simulation. "
        "Use the YAML social state to guide your response.\n\n"
        "{context}"
    ),
}


def format_chat_messages(
    npc_profile: str,
    dialogue_context: List[Dict[str, str]],
    metadata: Dict[str, Any],
    system_template: str = "social_state_xml",
    formatter: Optional[SocialStateFormatter] = None,
) -> List[Dict[str, str]]:
    """
    Format as chat messages for apply_chat_template.
    
    Returns:
        List of {"role": str, "content": str} messages
    """
    # Build context with social state
    include_social = system_template != "none"
    context = build_conditioned_prompt(
        npc_profile=npc_profile,
        dialogue_context=dialogue_context,
        metadata=metadata,
        formatter=formatter,
        include_social_state=include_social,
    )
    
    # Get system prompt
    template = SYSTEM_PROMPT_TEMPLATES.get(system_template, SYSTEM_PROMPT_TEMPLATES["social_state_xml"])
    system_content = template.format(context=context)
    
    # Build messages
    messages = [{"role": "system", "content": system_content}]
    
    # Add dialogue history as user/assistant alternating
    # Last player message is user, expected NPC response will be assistant
    for i, turn in enumerate(dialogue_context):
        if turn.get("speaker") == "player":
            messages.append({"role": "user", "content": turn["text"]})
        elif turn.get("speaker") == "npc":
            messages.append({"role": "assistant", "content": turn["text"]})
    
    return messages
