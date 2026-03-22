from pathlib import Path
from typing import Optional
import yaml
import random

class WorldContext:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.data = self._load()
        self.name = self.data.get("world_name", "Unknown World")
        self.description = self.data.get("description", "")
        self.factions = {f["name"]: f for f in self.data.get("factions", [])}
        self.npcs = {n["npc_id"]: n for n in self.data.get("npcs", [])}
        self.relationships = self.data.get("relationships", [])

    def _load(self) -> dict:
        if not self.config_path.exists():
            raise FileNotFoundError(f"World context not found: {self.config_path}")
        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def get_npc_template(self, npc_id: str) -> Optional[dict]:
        return self.npcs.get(npc_id)

    def get_relationships_for_npc(self, npc_id: str) -> list[dict]:
        """Return all relationships where this NPC is the source."""
        return [r for r in self.relationships if r["source"] == npc_id]
    
    def get_inbound_relationships(self, npc_id: str) -> list[dict]:
        """Return all relationships where this NPC is the target."""
        return [r for r in self.relationships if r["target"] == npc_id]

    def get_random_named_npc(self, rng: random.Random) -> dict:
        npc_id = rng.choice(list(self.npcs.keys()))
        return self.npcs[npc_id]
