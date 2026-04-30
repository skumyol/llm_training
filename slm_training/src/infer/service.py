from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoTokenizer

from src.common.config import InferenceConfig
from src.data.datasets import PersonalityCache
from src.infer.memory_store import EpisodicMemoryStore
from src.models.affect import DistilBertRegressor
from src.models.dialogue import ConditionalDialogueModel
from src.models.personality import DistilBertRegressor as PersonalityRegressor


@dataclass
class NPCState:
    npc_id: str
    profile_text: str
    conversation_window: List[Dict[str, str]] = field(default_factory=list)


class NPCInferenceService:
    def __init__(self, cfg: InferenceConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.personality_cache = PersonalityCache(cfg.personality_cache_path)
        self.memory = EpisodicMemoryStore(cfg.sentence_transformer_name)

        self.affect_tokenizer = AutoTokenizer.from_pretrained(cfg.affect_encoder_path)
        self.affect_encoder = DistilBertRegressor(cfg.affect_encoder_path, out_dim=3).to(self.device)
        self.affect_encoder.load_state_dict(
            torch.load(Path(cfg.affect_encoder_path) / "pytorch_model.bin", map_location=self.device), strict=False
        )
        self.affect_encoder.eval()

        cond_dim = 5 + 3  # OCEAN + VAD
        self.dialogue = ConditionalDialogueModel(
            base_model_name=cfg.dialogue_model_dir,
            cond_dim=cond_dim,
            prefix_length=8,
            lora_r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ).to(self.device)
        try:
            self.dialogue.model.load_adapter(cfg.dialogue_model_dir, adapter_name="default")
        except Exception:
            # Fine for an untrained scaffold.
            pass
        prefix_path = Path(cfg.dialogue_model_dir) / "prefix_encoder.pt"
        if prefix_path.exists():
            self.dialogue.prefix.load_state_dict(torch.load(prefix_path, map_location=self.device))
        self.states: Dict[str, NPCState] = {}

    def register_npc(self, npc_id: str, profile_text: str) -> None:
        self.states[npc_id] = NPCState(npc_id=npc_id, profile_text=profile_text)
        if self.personality_cache.get(npc_id) is None:
            raise ValueError(f"No cached personality vector found for NPC '{npc_id}'. Build the cache first.")

    def _personality_vec(self, npc_id: str) -> torch.Tensor:
        vec = self.personality_cache.get(npc_id)
        if vec is None:
            raise KeyError(f"No personality vector for '{npc_id}'.")
        return torch.tensor(vec, dtype=torch.float, device=self.device).unsqueeze(0)

    @torch.no_grad()
    def _affect_vec(self, conversation_window: List[Dict[str, str]]) -> torch.Tensor:
        text = "\n".join(f"{m['speaker']}: {m['text']}" for m in conversation_window[-8:]) or "neutral"
        enc = self.affect_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        enc.pop("token_type_ids", None)  # DistilBERT doesn't use these
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.affect_encoder(**enc)
        return out["preds"]

    def _build_prompt(self, npc: NPCState, player_message: str) -> str:
        memories = self.memory.search(npc.npc_id, player_message, k=self.cfg.memory_top_k)
        mem_block = "\n".join(f"- {m.text}" for m in memories)
        convo = "\n".join(f"{m['speaker']}: {m['text']}" for m in npc.conversation_window[-10:])
        return (
            f"You are roleplaying a persistent RPG NPC.\n"
            f"NPC PROFILE:\n{npc.profile_text}\n\n"
            f"RETRIEVED MEMORIES:\n{mem_block or '- none'}\n\n"
            f"RECENT CONVERSATION:\n{convo or '(empty)'}\n"
            f"player: {player_message}\n"
            f"npc:"
        )

    def respond(self, npc_id: str, player_message: str) -> str:
        npc = self.states[npc_id]
        npc.conversation_window.append({"speaker": "player", "text": player_message})

        cond_vec = torch.cat([self._personality_vec(npc_id), self._affect_vec(npc.conversation_window)], dim=-1)
        prompt = self._build_prompt(npc, player_message)
        text = self.dialogue.generate(
            prompt=prompt,
            cond_vec=cond_vec,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
        )
        reply = text.split("npc:")[-1].strip()
        npc.conversation_window.append({"speaker": "npc", "text": reply})
        self.memory.add(npc_id, f"player said: {player_message}; npc replied: {reply}")
        return reply
