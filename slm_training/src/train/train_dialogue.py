from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from src.common.config import DialogueTrainConfig, ensure_dir
from src.data.dialogue_data import DialogueExample, DialogueJsonlDataset, PersonalityCache
from src.infer.memory_store import EpisodicMemoryStore
from src.models.affect import DistilBertRegressor
from src.models.dialogue import ConditionalDialogueModel


@dataclass
class DialogueBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    cond_vec: torch.Tensor


class DialogueCollator:
    def __init__(self, cfg: DialogueTrainConfig, tokenizer: AutoTokenizer, device: torch.device) -> None:
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.device = device
        self.personality_cache = PersonalityCache(cfg.personality_cache_path)
        self.randomize_vad = getattr(cfg, 'randomize_vad', False)

        self.affect_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.affect_encoder = DistilBertRegressor("distilbert-base-uncased", out_dim=3).to(device)
        self.affect_encoder.load_state_dict(
            torch.load(f"{cfg.affect_encoder_path}/pytorch_model.bin", map_location=device), strict=False
        )
        self.affect_encoder.eval()
        self.memory = EpisodicMemoryStore(cfg.sentence_transformer_name)

    @torch.no_grad()
    def _affect(self, example: DialogueExample) -> torch.Tensor:
        text = "\n".join(f"{m['speaker']}: {m['text']}" for m in example.dialogue_context[-8:]) or "neutral"
        enc = self.affect_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        # DistilBERT doesn't use token_type_ids
        enc.pop("token_type_ids", None)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.affect_encoder(**enc)
        return out["preds"].squeeze(0).cpu()

    def _prompt(self, ex: DialogueExample) -> str:
        query = ex.dialogue_context[-1]["text"] if ex.dialogue_context else ex.target_response
        memories = self.memory.search(ex.npc_id, query, k=self.cfg.memory_top_k)
        mem_block = "\n".join(f"- {m.text}" for m in memories)
        convo = "\n".join(f"{m['speaker']}: {m['text']}" for m in ex.dialogue_context)
        return (
            f"NPC PROFILE:\n{ex.npc_profile}\n\n"
            f"RETRIEVED MEMORIES:\n{mem_block or '- none'}\n\n"
            f"RECENT CONVERSATION:\n{convo}\n"
            f"npc:"
        )

    def __call__(self, batch: List[DialogueExample]) -> DialogueBatch:
        prompts: List[str] = []
        labels_text: List[str] = []
        cond_vecs: List[torch.Tensor] = []
        for ex in batch:
            prompts.append(self._prompt(ex))
            labels_text.append(ex.target_response)
            p = self.personality_cache.get(ex.npc_id)
            if p is None:
                raise KeyError(f"Missing cached personality vector for {ex.npc_id}")
            p_vec = torch.tensor(p, dtype=torch.float)
            if self.randomize_vad:
                a_vec = torch.rand(3, dtype=torch.float)
            else:
                a_vec = self._affect(ex)
            cond_vecs.append(torch.cat([p_vec, a_vec], dim=-1))

        model_inputs = self.tokenizer(
            [p + t for p, t in zip(prompts, labels_text)],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.cfg.max_source_length + self.cfg.max_target_length,
        )
        labels = model_inputs["input_ids"].clone()
        return DialogueBatch(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            labels=labels,
            cond_vec=torch.stack(cond_vecs, dim=0),
        )


def run(cfg: DialogueTrainConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalDialogueModel(
        base_model_name=cfg.base_model_name,
        cond_dim=8,
        prefix_length=cfg.prefix_length,
        lora_r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
    ).to(device)
    tokenizer = model.tokenizer

    train_ds = DialogueJsonlDataset(cfg.train_path)
    val_ds = DialogueJsonlDataset(cfg.val_path)
    collator = DialogueCollator(cfg, tokenizer, device)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, collate_fn=collator)

    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    out_dir = ensure_dir(cfg.output_dir)
    best_val = float("inf")

    for epoch in range(cfg.epochs):
        model.train()
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(train_loader, desc=f"dialogue epoch {epoch+1}"), start=1):
            out = model(
                input_ids=batch.input_ids.to(device),
                attention_mask=batch.attention_mask.to(device),
                cond_vec=batch.cond_vec.to(device),
                labels=batch.labels.to(device),
            )
            loss = out.loss / cfg.grad_accum_steps
            loss.backward()
            if step % cfg.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        model.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for batch in val_loader:
                out = model(
                    input_ids=batch.input_ids.to(device),
                    attention_mask=batch.attention_mask.to(device),
                    cond_vec=batch.cond_vec.to(device),
                    labels=batch.labels.to(device),
                )
                total += out.loss.item()
                n += 1
        val_loss = total / max(n, 1)
        print(f"val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            model.model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            torch.save(model.prefix.state_dict(), out_dir / "prefix_encoder.pt")
            print(f"saved dialogue model to {out_dir}")


if __name__ == "__main__":
    run(DialogueTrainConfig())
