from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class MemoryItem:
    text: str
    metadata: Dict[str, str]


class EpisodicMemoryStore:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.embedder = SentenceTransformer(model_name)
        self.indices: Dict[str, faiss.IndexFlatIP] = {}
        self.items: Dict[str, List[MemoryItem]] = {}
        self.dim = self.embedder.get_sentence_embedding_dimension()

    def _ensure(self, npc_id: str) -> None:
        if npc_id not in self.indices:
            self.indices[npc_id] = faiss.IndexFlatIP(self.dim)
            self.items[npc_id] = []

    def add(self, npc_id: str, text: str, metadata: Dict[str, str] | None = None) -> None:
        self._ensure(npc_id)
        vec = self.embedder.encode([text], normalize_embeddings=True)
        self.indices[npc_id].add(np.asarray(vec, dtype=np.float32))
        self.items[npc_id].append(MemoryItem(text=text, metadata=metadata or {}))

    def search(self, npc_id: str, query: str, k: int = 3) -> List[MemoryItem]:
        if npc_id not in self.indices or len(self.items[npc_id]) == 0:
            return []
        vec = self.embedder.encode([query], normalize_embeddings=True)
        scores, idx = self.indices[npc_id].search(np.asarray(vec, dtype=np.float32), k)
        out: List[MemoryItem] = []
        for i in idx[0].tolist():
            if 0 <= i < len(self.items[npc_id]):
                out.append(self.items[npc_id][i])
        return out
