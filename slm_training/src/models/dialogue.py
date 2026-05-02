from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class PrefixOutput:
    prefix_embeds: torch.Tensor
    prefix_mask: torch.Tensor


class ConditionalSoftPrefix(nn.Module):
    """Portable conditional prefix module.

    Your uploaded architecture proposes a per-layer KV prefix encoder. This scaffold keeps
    the same *conditioning interface* but implements a model-agnostic soft-prefix variant
    that prepends learned prefix embeddings to the input stream. That keeps training and
    inference runnable while preserving a clean upgrade path to true KV-prefix injection. fileciteturn12file1L9-L24
    """

    def __init__(self, cond_dim: int, hidden_size: int, prefix_length: int) -> None:
        super().__init__()
        self.prefix_length = prefix_length
        self.proj = nn.Sequential(
            nn.Linear(cond_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, prefix_length * hidden_size),
        )
        self.hidden_size = hidden_size

    def forward(self, cond_vec: torch.Tensor) -> PrefixOutput:
        batch_size = cond_vec.size(0)
        prefix = self.proj(cond_vec).view(batch_size, self.prefix_length, self.hidden_size)
        mask = torch.ones(batch_size, self.prefix_length, device=cond_vec.device, dtype=torch.long)
        return PrefixOutput(prefix_embeds=prefix, prefix_mask=mask)


class ConditionalDialogueModel(nn.Module):
    def __init__(
        self,
        base_model_name: str,
        cond_dim: int,
        prefix_length: int,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        target_modules: List[str],
    ) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(base_model_name)
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base, lora_cfg)
        hidden_size = self.model.config.hidden_size
        self.prefix = ConditionalSoftPrefix(cond_dim=cond_dim, hidden_size=hidden_size, prefix_length=prefix_length)

    def _combine_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cond_vec: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        input_embeds = self.model.get_input_embeddings()(input_ids)
        target_dtype = input_embeds.dtype
        prefix = self.prefix(cond_vec)
        # Cast prefix embeddings to match model dtype (handles bfloat16/float16 models)
        prefix_embeds = prefix.prefix_embeds.to(dtype=target_dtype)
        full_embeds = torch.cat([prefix_embeds, input_embeds], dim=1)
        full_mask = torch.cat([prefix.prefix_mask, attention_mask], dim=1)
        out: Dict[str, torch.Tensor] = {
            "inputs_embeds": full_embeds,
            "attention_mask": full_mask,
        }
        if labels is not None:
            prefix_ignore = torch.full(
                (labels.size(0), prefix_embeds.size(1)),
                -100,
                device=labels.device,
                dtype=labels.dtype,
            )
            out["labels"] = torch.cat([prefix_ignore, labels], dim=1)
        return out

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        cond_vec: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        model_inputs = self._combine_inputs(input_ids, attention_mask, cond_vec, labels)
        return self.model(**model_inputs)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        cond_vec: torch.Tensor,
        max_new_tokens: int = 120,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> str:
        enc = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}
        cond_vec = cond_vec.to(device)
        model_inputs = self._combine_inputs(enc["input_ids"], enc["attention_mask"], cond_vec)
        # Note: generate with inputs_embeds is supported, but decoded output includes prompt tokens.
        generated = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return text
