from __future__ import annotations

from src.common.config import InferenceConfig
from src.infer.service import NPCInferenceService


if __name__ == "__main__":
    cfg = InferenceConfig()
    service = NPCInferenceService(cfg)
    service.register_npc(
        npc_id="npc_001",
        profile_text=(
            "Iria Voss is a cautious but defiant navigator. She distrusts the priesthood, "
            "protects hidden charts, and speaks in compact, suspicious sentences."
        ),
    )
    while True:
        msg = input("player> ").strip()
        if msg.lower() in {"quit", "exit"}:
            break
        reply = service.respond("npc_001", msg)
        print(f"npc> {reply}")
