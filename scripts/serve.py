#!/usr/bin/env python3
"""
serve.py — Unified OpenAI-compatible API for LLM + SLM models
===============================================================
Single server exposing both trained systems behind a standard /v1/chat/completions
endpoint. Works with any OpenAI-compatible frontend (ChatGPT UI, Continue.dev, etc.)

Usage:
  # Serve LLM fine-tuned model (Qwen3 + LoRA adapter)
  ./scripts/serve.py --system llm --checkpoint checkpoints/joint_model_best

  # Serve SLM from scratch
  ./scripts/serve.py --system slm --model-dir artifacts/small_lm/final_gpt_s42

  # Serve with vLLM backend (high throughput)
  ./scripts/serve.py --system llm --backend vllm --checkpoint checkpoints/response_generator_best

  # Serve everything on one port
  ./scripts/serve.py --system both --port 8765

Endpoints:
  GET  /v1/models                    List available models
  POST /v1/chat/completions          Chat completion (OpenAI format)
  GET  /health                       Health check
  GET  /docs                         Swagger UI
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "llm_finetuning"))
sys.path.insert(0, str(ROOT / "slm_training"))
sys.path.insert(0, str(ROOT / "slm_training" / "src" / "train"))


# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic models (OpenAI-compatible)
# ═══════════════════════════════════════════════════════════════════════════════

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "npc-dialogue"
    messages: List[Message]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=200, ge=1, le=4096)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = False
    system: Optional[str] = None  # extra fields ignored

class ChatChoice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str = "stop"

class ChatUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: ChatUsage

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "npc-project"
    created: int

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# ═══════════════════════════════════════════════════════════════════════════════
# App factory
# ═══════════════════════════════════════════════════════════════════════════════

def create_app(args) -> FastAPI:
    app = FastAPI(
        title="NPC Dialogue API",
        description="OpenAI-compatible chat completions for NPC dialogue models",
        version="2.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Load backends ─────────────────────────────────────────────────────────
    backends: Dict[str, Any] = {}

    if args.system in ("llm", "both"):
        backends["llm"] = _load_llm_backend(args)
    if args.system in ("slm", "both"):
        backends["slm"] = _load_slm_backend(args)

    available_models = list(backends.keys())
    if not available_models:
        raise RuntimeError("No backends loaded. Check --system, --checkpoint, --model-dir.")

    # ── Routes ────────────────────────────────────────────────────────────────
    @app.get("/health")
    async def health():
        return {"status": "ok", "backends": list(backends.keys())}

    @app.get("/v1/models", response_model=ModelList)
    async def list_models():
        return ModelList(data=[
            ModelInfo(id=name, created=int(datetime.now().timestamp()))
            for name in available_models
        ])

    @app.post("/v1/chat/completions", response_model=ChatResponse)
    async def chat_completions(req: ChatRequest):
        model_name = req.model
        if model_name not in backends:
            raise HTTPException(404, f"Model '{model_name}' not found. Available: {available_models}")

        backend = backends[model_name]
        prompt = _build_prompt(req.messages)

        t0 = time.time()
        try:
            response_text = backend.generate(
                prompt=prompt,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                top_p=req.top_p,
            )
        except Exception as e:
            raise HTTPException(500, f"Generation failed: {e}")

        elapsed = time.time() - t0

        return ChatResponse(
            id=f"chatcmpl-{int(t0)}",
            created=int(t0),
            model=model_name,
            choices=[ChatChoice(message=Message(role="assistant", content=response_text))],
            usage=ChatUsage(
                prompt_tokens=len(prompt.split()),
                completion_tokens=len(response_text.split()),
                total_tokens=len(prompt.split()) + len(response_text.split()),
            ),
        )

    return app


# ═══════════════════════════════════════════════════════════════════════════════
# LLM Backend (LatentStatePredictor)
# ═══════════════════════════════════════════════════════════════════════════════

class LLMBackend:
    def __init__(self, checkpoint: str, base_model: str, quantization: str):
        from src.training.model import load_predictor, STANCE_DIMS as _SD
        from src.training.dataset import LABEL_MAPS as _LM, IDX_TO_LABEL as _I2L

        self.stance_dims = _SD

        print(f"[LLM] Loading {base_model} + checkpoint {checkpoint} (quant={quantization})...")
        self.predictor, self.tokenizer = load_predictor(
            checkpoint, base_model,
            quantization=quantization if quantization != "none" else None,
            torch_dtype="bfloat16" if quantization != "none" else "float16",
        )
        self.predictor.eval()
        self.device = next(self.predictor.backbone.parameters()).device
        print(f"[LLM] Ready on {self.device}")

    def generate(self, prompt: str, temperature: float, max_tokens: int, top_p: float) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)

        with torch.no_grad():
            out = self.predictor(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)

            # Build latent state string from predictions
            latent_lines = ["<latent_state>"]
            for name, logit in out["logits"].items():
                idx = logit.argmax(dim=-1).item()
                # We don't have IDX_TO_LABEL in scope but we can use default
                latent_lines.append(f"  {name}: cls_{idx}")
            latent_lines.append("</latent_state>")
            latent_str = "\n".join(latent_lines)

            full_input = prompt + "\n\n" + latent_str + "\n\nResponse:"
            gen_inputs = self.tokenizer(full_input, return_tensors="pt", truncation=True, max_length=2048).to(self.device)

            generated = self.predictor.backbone.generate(
                input_ids=gen_inputs.input_ids,
                attention_mask=gen_inputs.attention_mask,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            new_tokens = generated[0, gen_inputs.input_ids.shape[1]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            return text


# ═══════════════════════════════════════════════════════════════════════════════
# SLM Backend (small LM architectures)
# ═══════════════════════════════════════════════════════════════════════════════

class SLMBackend:
    def __init__(self, model_dir: str, arch: str):
        import torch.nn.functional as F

        model_dir = Path(model_dir)
        self.arch = arch

        # Load model
        config_path = model_dir / "config.json"

        # Try multiple checkpoint naming conventions
        ckpt_path = None
        for pattern in [f"{arch}_best.pt", "best_model.pt", "model.pt"]:
            candidate = model_dir / pattern
            if candidate.exists():
                ckpt_path = candidate
                break
        if ckpt_path is None:
            candidates = list(model_dir.glob("*.pt"))
            if candidates:
                ckpt_path = candidates[0]
            else:
                raise FileNotFoundError(f"No .pt checkpoint found in {model_dir}")

        from small_lm_architectures import build_model, RECOMMENDED_CONFIGS

        # Load config from file or use hardware profile defaults
        if config_path.exists():
            import json
            cfg_dict = json.loads(config_path.read_text())
        else:
            profile = "rtx4070_small" if torch.cuda.is_available() else "m1_small"
            cfg_dict = RECOMMENDED_CONFIGS.get(profile, {}).get(
                arch, RECOMMENDED_CONFIGS["m1_small"][arch])

        self.model = build_model(arch, cfg_dict)

        state = torch.load(ckpt_path, map_location="cpu")
        self.model.load_state_dict(state)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Simple tokenizer (char-level, no dependencies on external tokenizers that may not match)
        self.vocab_size = cfg_dict.get("vocab_size", 50257)
        self.seq_len = cfg_dict.get("max_seq_len", 256)

        # Use a simple char-level tokenizer for portability
        # Users can substitute with GPT2 tokenizer if desired
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self._tokenize = self._hf_tokenize
            self._detokenize = self._hf_detokenize
        except Exception:
            self._tokenize = self._char_tokenize
            self._detokenize = self._char_detokenize

        print(f"[SLM] Loaded {arch} from {ckpt_path} on {self.device}")

    def _hf_tokenize(self, text: str) -> torch.Tensor:
        return self.tokenizer(text, return_tensors="pt", truncation=True,
                              max_length=self.seq_len).input_ids.to(self.device)

    def _hf_detokenize(self, ids: torch.Tensor) -> str:
        return self.tokenizer.decode(ids[0], skip_special_tokens=True)

    def _char_tokenize(self, text: str) -> torch.Tensor:
        chars = list(text[:self.seq_len])
        return torch.tensor([[ord(c) % self.vocab_size for c in chars]], device=self.device)

    def _char_detokenize(self, ids: torch.Tensor) -> str:
        return "".join(chr(i % 128) for i in ids[0].tolist())

    def generate(self, prompt: str, temperature: float, max_tokens: int, top_p: float) -> str:
        x = self._tokenize(prompt)
        generated = []

        with torch.no_grad():
            for _ in range(max_tokens):
                # Truncate to seq_len if needed
                if x.shape[1] > self.seq_len:
                    x = x[:, -self.seq_len:]

                out = self.model(x)
                logits = out.logits[:, -1, :] / max(temperature, 0.01)

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                    cutoff = cumulative > top_p
                    cutoff[..., 0] = False
                    logits[sorted_indices[cutoff]] = float("-inf")

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                generated.append(next_token.item())
                x = torch.cat([x, next_token], dim=1)

        gen_ids = torch.tensor([generated], device=self.device)
        return self._detokenize(gen_ids).strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _build_prompt(messages: List[Message]) -> str:
    """Convert OpenAI message list to a single prompt string."""
    parts = []
    for msg in messages:
        prefix = {"system": "System", "user": "Player", "assistant": "NPC"}.get(msg.role, msg.role)
        parts.append(f"{prefix}: {msg.content}")
    return "\n".join(parts) + "\nNPC:"


def _load_llm_backend(args) -> LLMBackend:
    if not args.checkpoint:
        raise ValueError("--checkpoint is required for LLM backend")
    return LLMBackend(
        checkpoint=args.checkpoint,
        base_model=args.base_model,
        quantization=args.quantization,
    )


def _load_slm_backend(args) -> SLMBackend:
    if not args.model_dir:
        raise ValueError("--model-dir is required for SLM backend")
    return SLMBackend(
        model_dir=args.model_dir,
        arch=args.arch,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# vLLM integration (subprocess launcher)
# ═══════════════════════════════════════════════════════════════════════════════

def launch_vllm(args) -> None:
    """Launch vLLM server for the Qwen3 + LoRA model."""
    import subprocess

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.base_model,
        "--enable-lora",
        "--lora-modules", f"npc-adapter={args.checkpoint}",
        "--port", str(args.port),
        "--host", args.host,
        "--dtype", "bfloat16",
        "--max-model-len", "4096",
    ]
    print(f"[vLLM] Launching: {' '.join(cmd)}")
    subprocess.run(cmd)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Unified NPC model serving")
    p.add_argument("--system", choices=["llm", "slm", "both"], default="llm",
                   help="Which model system to serve")
    p.add_argument("--backend", choices=["native", "vllm"], default="native",
                   help="Serving backend (native PyTorch or vLLM)")
    p.add_argument("--checkpoint", type=str,
                   help="LLM checkpoint path (e.g., checkpoints/joint_model_best)")
    p.add_argument("--base-model", type=str, default="Qwen/Qwen3-4B",
                   help="Base LLM name for LLM backend")
    p.add_argument("--quantization", type=str, default="4bit",
                   choices=["4bit", "8bit", "none"])
    p.add_argument("--model-dir", type=str,
                   help="SLM model directory (e.g., slm_training/artifacts/small_lm/final_gpt_s42)")
    p.add_argument("--arch", type=str, default="gpt",
                   choices=["gru", "awdlstm", "gpt", "prefix_gpt", "moe", "mamba_like"])
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--host", type=str, default="0.0.0.0")
    args = p.parse_args()

    if args.backend == "vllm":
        launch_vllm(args)
        return

    app = create_app(args)
    print(f"\n{'='*60}")
    print(f"  NPC Dialogue API  (system={args.system}  backend={args.backend})")
    print(f"  Listening on http://{args.host}:{args.port}")
    print(f"  Swagger:  http://localhost:{args.port}/docs")
    print(f"{'='*60}\n")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
