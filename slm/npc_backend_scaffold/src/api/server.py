#!/usr/bin/env python3
"""
NPC Backend — FastAPI Server
=============================
Serves the NPC inference pipeline over HTTP for the React frontend.

Run:
  cd slm/npc_backend_scaffold
  PYTHONPATH=. uvicorn src.api.server:app --reload --port 8765

Endpoints:
  GET  /api/health
  GET  /api/models            list trained model artifacts
  GET  /api/eval              encoder + dialogue metrics summary
  GET  /api/npcs              list registered NPCs
  POST /api/npcs              register new NPC (encodes personality on-the-fly)
  DELETE /api/npcs/{id}       remove NPC
  POST /api/chat/{id}         chat turn → response + full state
  POST /api/reset/{id}        clear conversation history
  POST /api/load-world        load all NPCs from a world YAML
"""
from __future__ import annotations

import glob
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="NPC Backend", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Globals (lazy-loaded) ─────────────────────────────────────────────────────

_service = None                # NPCInferenceService
_personality_encoder = None    # DistilBertRegressor for on-the-fly encoding
_personality_tokenizer = None
_device = None
_load_error: Optional[str] = None
_registered_profiles: Dict[str, str] = {}  # npc_id → profile_text (in-memory)

OCEAN_DIMS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
VAD_DIMS   = ["valence", "arousal", "dominance"]

SCAFFOLD_ROOT = Path(__file__).parent.parent.parent  # slm/npc_backend_scaffold/

# ── Full model catalog (static definitions) ───────────────────────────────────

SMALL_LM_ARCHS: Dict[str, Dict] = {
    "gru":        {"name": "Small GRU LM",           "desc": "Gated Recurrent Unit — lightweight, fast to train",            "train_cmd": "python train_benchmark_small_lms_example.py --arch gru --train-text data/dialogue/train.txt --out-dir runs/gru"},
    "awdlstm":    {"name": "AWD-LSTM",                "desc": "ASGD Weight-Dropped LSTM — strong regularisation",             "train_cmd": "python train_benchmark_small_lms_example.py --arch awdlstm --train-text data/dialogue/train.txt --out-dir runs/awdlstm"},
    "gpt":        {"name": "Tiny GPT",                "desc": "Causal transformer, no conditioning",                          "train_cmd": "python train_benchmark_small_lms_example.py --arch gpt --train-text data/dialogue/train.txt --out-dir runs/gpt"},
    "prefix_gpt": {"name": "Prefix-Conditioned GPT",  "desc": "GPT with soft personality+affect prefix (condition-aware)",   "train_cmd": "python train_benchmark_small_lms_example.py --arch prefix_gpt --train-text data/dialogue/train.txt --out-dir runs/prefix_gpt"},
    "moe":        {"name": "Tiny MoE LM",              "desc": "Mixture-of-Experts transformer, sparse routing",              "train_cmd": "python train_benchmark_small_lms_example.py --arch moe --train-text data/dialogue/train.txt --out-dir runs/moe"},
    "mamba_like": {"name": "Mamba-like SSM",           "desc": "State-space model, linear-time attention alternative",        "train_cmd": "python train_benchmark_small_lms_example.py --arch mamba_like --train-text data/dialogue/train.txt --out-dir runs/mamba_like"},
}

GEMMA_UNSLOTH_MODEL: Dict[str, str] = {
    "id": "gemma4_unsloth",
    "name": "Gemma 4 E2B + Unsloth",
    "desc": "Gemma 4 fine-tuned with Unsloth LoRA on the scaffold dialogue JSONL format",
    "train_cmd": "python -m src.train.run_gemma_unsloth --config configs/dialogue_gemma_unsloth.yaml",
}

# active model IDs per pipeline stage (None = auto-select best available)
_active_models: Dict[str, Optional[str]] = {
    "personality_encoder":  None,
    "affect_encoder":       None,
    "dialogue_generator":   None,
}


# ── Startup: load models ───────────────────────────────────────────────────────

@app.on_event("startup")
async def _startup() -> None:
    global _service, _personality_encoder, _personality_tokenizer, _device, _load_error

    _device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    # Find best available artifacts
    cache_paths   = sorted(SCAFFOLD_ROOT.glob("artifacts/personality_cache*.jsonl"))
    affect_paths  = sorted(SCAFFOLD_ROOT.glob("artifacts/affect_encoder/*/best_model"))
    personality_paths = sorted(SCAFFOLD_ROOT.glob("artifacts/personality_encoder/*/best_model"))
    dialogue_paths = sorted(SCAFFOLD_ROOT.glob("artifacts/dialogue_model*/*/best_model"))

    print(f"[startup] device={_device}")
    print(f"[startup] personality cache: {cache_paths[-1] if cache_paths else 'NOT FOUND'}")
    print(f"[startup] affect encoder  : {affect_paths[-1] if affect_paths else 'NOT FOUND'}")
    print(f"[startup] personality enc : {personality_paths[-1] if personality_paths else 'NOT FOUND'}")
    print(f"[startup] dialogue model  : {dialogue_paths[-1] if dialogue_paths else 'NOT FOUND'}")

    # Load personality encoder (for on-the-fly NPC registration)
    if personality_paths:
        try:
            from transformers import AutoTokenizer
            from src.models.personality import DistilBertRegressor
            ppath = str(personality_paths[-1])
            _personality_tokenizer = AutoTokenizer.from_pretrained(ppath)
            _personality_encoder = DistilBertRegressor(ppath, out_dim=5).to(_device)
            _personality_encoder.load_state_dict(
                torch.load(Path(ppath) / "pytorch_model.bin", map_location=_device), strict=False
            )
            _personality_encoder.eval()
            print("[startup] Personality encoder loaded ✓")
        except Exception as e:
            print(f"[startup] Personality encoder load failed: {e}")

    # Load full service (requires all components)
    if cache_paths and affect_paths and dialogue_paths:
        try:
            from src.common.config import InferenceConfig
            from src.infer.service import NPCInferenceService

            cfg = InferenceConfig(
                dialogue_model_dir   = str(dialogue_paths[-1]),
                affect_encoder_path  = str(affect_paths[-1]),
                personality_cache_path = str(cache_paths[-1]),
            )
            _service = NPCInferenceService(cfg)
            print("[startup] NPCInferenceService loaded ✓")
        except Exception as e:
            _load_error = str(e)
            print(f"[startup] Service load failed: {e}")
    else:
        missing = []
        if not cache_paths:    missing.append("personality_cache")
        if not affect_paths:   missing.append("affect_encoder")
        if not dialogue_paths: missing.append("dialogue_model")
        _load_error = f"Missing: {', '.join(missing)}"
        print(f"[startup] Service not loaded — {_load_error}")


# ── Helpers ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def _encode_personality(text: str) -> List[float]:
    """Encode profile text → OCEAN vector using the personality encoder."""
    if _personality_encoder is None or _personality_tokenizer is None:
        return [0.5] * 5
    enc = _personality_tokenizer(
        text, return_tensors="pt", truncation=True,
        padding=True, max_length=256,
    )
    enc.pop("token_type_ids", None)
    enc = {k: v.to(_device) for k, v in enc.items()}
    out = _personality_encoder(**enc)
    return out["preds"][0].tolist()


@torch.no_grad()
def _extract_state(npc_id: str) -> Dict[str, Any]:
    """Extract personality + affect state after a chat turn."""
    if _service is None:
        return {}
    try:
        pvec = _service._personality_vec(npc_id)
        personality = {k: round(float(v), 4) for k, v in zip(OCEAN_DIMS, pvec[0].tolist())}

        npc_state = _service.states.get(npc_id)
        window = npc_state.conversation_window if npc_state else []
        avec = _service._affect_vec(window)
        affect = {k: round(float(v), 4) for k, v in zip(VAD_DIMS, avec[0].tolist())}

        return {"personality": personality, "affect": affect}
    except Exception as e:
        return {"error": str(e)}


def _ensure_service() -> None:
    if _service is None:
        detail = f"Inference service not loaded. {_load_error or ''}"
        detail += " Train models first: ./train_all.sh"
        raise HTTPException(status_code=503, detail=detail)


def _build_model_catalog() -> Dict[str, Any]:
    """Build the full pipeline model catalog with live training status."""

    def _read_summary(path: Path) -> Dict:
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}

    def _slm_status(arch: str) -> Dict[str, Any]:
        """Find training artifacts for a small-LM arch."""
        # Search common output locations from the benchmark training script
        candidates = [
            *SCAFFOLD_ROOT.glob(f"runs/**/{arch}_best.pt"),
            *SCAFFOLD_ROOT.glob(f"runs/{arch}/{arch}_best.pt"),
            *SCAFFOLD_ROOT.glob(f"artifacts/small_lm/{arch}/{arch}_best.pt"),
        ]
        if not candidates:
            return {"status": "not_trained", "artifact": None, "metrics": {}}
        ckpt_path = sorted(candidates)[-1]
        summary_path = ckpt_path.parent / f"{arch}_summary.json"
        summary = _read_summary(summary_path)
        return {
            "status": "trained",
            "artifact": str(ckpt_path),
            "metrics": {
                "val_ppl":    summary.get("final_val_ppl"),
                "val_loss":   summary.get("final_val_loss"),
                "test_ppl":   summary.get("final_test_ppl"),
                "num_params": summary.get("num_params"),
            },
        }

    def _run_summary_status(pattern: str) -> Dict[str, Any]:
        paths = sorted(SCAFFOLD_ROOT.glob(pattern))
        if not paths:
            return {"status": "not_trained", "artifact": None, "metrics": {}}
        summary = _read_summary(paths[-1])
        return {
            "status": "trained",
            "artifact": str(paths[-1].parent / "best_model"),
            "metrics": summary.get("best", {}),
        }

    # ── Stage 1: Personality Encoder ─────────────────────────────────────────
    pers_paths = sorted(SCAFFOLD_ROOT.glob("artifacts/personality_encoder/*/run_summary.json"))
    if pers_paths:
        ps = _read_summary(pers_paths[-1])
        pers_status = "trained"
        pers_metrics = ps.get("best", {})
    else:
        pers_status  = "not_trained"
        pers_metrics = {}

    personality_options = [{
        "id":          "distilbert_ocean",
        "name":        "DistilBERT OCEAN",
        "desc":        "Fine-tuned DistilBERT regressor predicting Big-5 (OCEAN) from text",
        "status":      pers_status,
        "metrics":     pers_metrics,
        "train_cmd":   "python -m src.train.run_personality",
        "artifact":    str(pers_paths[-1].parent) if pers_paths else None,
        "is_active":   True,
    }]

    # ── Stage 2: Affect Encoder ───────────────────────────────────────────────
    aff_paths = sorted(SCAFFOLD_ROOT.glob("artifacts/affect_encoder/*/run_summary.json"))
    if aff_paths:
        af = _read_summary(aff_paths[-1])
        aff_status  = "trained"
        aff_metrics = af.get("best", {})
    else:
        aff_status  = "not_trained"
        aff_metrics = {}

    affect_options = [{
        "id":        "distilbert_vad",
        "name":      "DistilBERT VAD",
        "desc":      "Fine-tuned DistilBERT regressor predicting Valence-Arousal-Dominance",
        "status":    aff_status,
        "metrics":   aff_metrics,
        "train_cmd": "python -m src.train.run_affect",
        "artifact":  str(aff_paths[-1].parent) if aff_paths else None,
        "is_active": True,
    }]

    # ── Stage 3: Dialogue Generator ───────────────────────────────────────────
    dlg_paths = sorted(SCAFFOLD_ROOT.glob("artifacts/dialogue_model*/*/best_model"))
    active_dlg = _active_models.get("dialogue_generator")

    dialogue_options = [{
        "id":          "tinyllama_lora",
        "name":        "TinyLlama + LoRA",
        "desc":        "TinyLlama-1.1B with LoRA adapters + soft personality/affect prefix conditioning",
        "status":      "trained" if dlg_paths else "not_trained",
        "metrics":     {},
        "train_cmd":   "python -m src.train.run_dialogue  (or ./train_all.sh)",
        "artifact":    str(dlg_paths[-1]) if dlg_paths else None,
        "is_active":   (active_dlg is None and bool(dlg_paths)) or active_dlg == "tinyllama_lora",
        "conditioned": True,
    }]

    gemma = _run_summary_status("artifacts/gemma_unsloth/**/run_summary.json")
    dialogue_options.append({
        "id": GEMMA_UNSLOTH_MODEL["id"],
        "name": GEMMA_UNSLOTH_MODEL["name"],
        "desc": GEMMA_UNSLOTH_MODEL["desc"],
        "status": gemma["status"],
        "metrics": gemma["metrics"],
        "train_cmd": GEMMA_UNSLOTH_MODEL["train_cmd"],
        "artifact": gemma["artifact"],
        "is_active": active_dlg == GEMMA_UNSLOTH_MODEL["id"],
        "conditioned": False,
    })

    for arch, meta in SMALL_LM_ARCHS.items():
        slm = _slm_status(arch)
        dialogue_options.append({
            "id":          arch,
            "name":        meta["name"],
            "desc":        meta["desc"],
            "status":      slm["status"],
            "metrics":     slm["metrics"],
            "train_cmd":   meta["train_cmd"],
            "artifact":    slm["artifact"],
            "is_active":   active_dlg == arch,
            "conditioned": arch == "prefix_gpt",
        })

    return {
        "stages": [
            {
                "id":          "personality_encoder",
                "label":       "Personality Encoder",
                "description": "Encodes NPC profile text → OCEAN Big-5 personality vector",
                "output":      "5-dim OCEAN vector",
                "options":     personality_options,
                "active":      _active_models.get("personality_encoder") or "distilbert_ocean",
            },
            {
                "id":          "affect_encoder",
                "label":       "Affect Encoder",
                "description": "Reads conversation history → Valence-Arousal-Dominance affect state",
                "output":      "3-dim VAD vector",
                "options":     affect_options,
                "active":      _active_models.get("affect_encoder") or "distilbert_vad",
            },
            {
                "id":          "dialogue_generator",
                "label":       "Dialogue Generator",
                "description": "Generates NPC response conditioned on personality + affect + memory",
                "output":      "NPC reply text",
                "options":     dialogue_options,
                "active":      _active_models.get("dialogue_generator") or
                               ("tinyllama_lora" if dlg_paths else None),
            },
        ]
    }


# ── Routes: health & models ───────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {
        "status":        "ok" if _service else "degraded",
        "service_ready": _service is not None,
        "personality_encoder_ready": _personality_encoder is not None,
        "load_error":    _load_error,
        "device":        str(_device),
        "registered_npcs": list(_registered_profiles.keys()),
        "active_models":   _active_models,
    }


@app.get("/api/models/catalog")
async def model_catalog():
    """Full pipeline catalog: all 9 models with training status."""
    return _build_model_catalog()


class SelectModelRequest(BaseModel):
    stage:    str   # personality_encoder | affect_encoder | dialogue_generator
    model_id: str


@app.post("/api/models/select")
async def select_model(req: SelectModelRequest):
    valid_stages = {"personality_encoder", "affect_encoder", "dialogue_generator"}
    if req.stage not in valid_stages:
        raise HTTPException(400, f"Unknown stage '{req.stage}'. Must be one of {valid_stages}")
    _active_models[req.stage] = req.model_id
    return {"success": True, "stage": req.stage, "model_id": req.model_id}


@app.get("/api/models")
async def list_models():
    """List all trained model artifacts available under artifacts/."""
    models = []
    for summary in sorted(SCAFFOLD_ROOT.glob("artifacts/**/run_summary.json")):
        try:
            d = json.loads(summary.read_text())
            best_model_dir = summary.parent / "best_model"
            models.append({
                "run_id":   d.get("run_id", "?"),
                "task":     d.get("task", "?"),
                "arch":     d.get("arch") or d.get("backbone") or d.get("model", "?"),
                "best":     d.get("best", {}),
                "path":     str(summary.parent),
                "has_model": best_model_dir.exists(),
            })
        except Exception:
            continue
    return {"models": models}


@app.get("/api/eval")
async def eval_summary():
    """Return latest encoder + dialogue metrics from run_summary.json files."""
    result: Dict[str, Any] = {}

    for tag, pattern in [
        ("personality", "artifacts/personality_encoder/**/run_summary.json"),
        ("affect",      "artifacts/affect_encoder/**/run_summary.json"),
        ("dialogue",    "artifacts/dialogue_model*/**/run_summary.json"),
    ]:
        paths = sorted(SCAFFOLD_ROOT.glob(pattern))
        if paths:
            try:
                result[tag] = json.loads(paths[-1].read_text())
            except Exception:
                pass

    return result


# ── Routes: NPC management ────────────────────────────────────────────────────

class RegisterNPCRequest(BaseModel):
    npc_id:       str
    profile_text: str


@app.get("/api/npcs")
async def list_npcs():
    npcs = []
    for npc_id, profile in _registered_profiles.items():
        turn_count = 0
        if _service and npc_id in _service.states:
            turn_count = len(_service.states[npc_id].conversation_window)
        npcs.append({"npc_id": npc_id, "profile_text": profile, "turn_count": turn_count})
    return {"npcs": npcs}


@app.post("/api/npcs")
async def register_npc(req: RegisterNPCRequest):
    _registered_profiles[req.npc_id] = req.profile_text

    # Encode personality vector on-the-fly and inject into cache
    pvec = _encode_personality(req.profile_text)

    if _service is not None:
        # Inject into personality cache so register_npc doesn't raise
        _service.personality_cache.cache[req.npc_id] = pvec
        try:
            _service.register_npc(req.npc_id, req.profile_text)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return {
        "success": True,
        "npc_id": req.npc_id,
        "personality_vector": {k: round(v, 4) for k, v in zip(OCEAN_DIMS, pvec)},
    }


@app.delete("/api/npcs/{npc_id}")
async def remove_npc(npc_id: str):
    _registered_profiles.pop(npc_id, None)
    if _service and npc_id in _service.states:
        del _service.states[npc_id]
    return {"success": True}


class LoadWorldRequest(BaseModel):
    yaml_path: str


@app.post("/api/load-world")
async def load_world(req: LoadWorldRequest):
    yaml_path = Path(req.yaml_path)
    if not yaml_path.exists():
        # Try relative to scaffold root
        yaml_path = SCAFFOLD_ROOT / req.yaml_path
    if not yaml_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.yaml_path}")

    with open(yaml_path, encoding="utf-8") as f:
        world = yaml.safe_load(f)

    loaded = []
    for npc in world.get("npcs", []):
        npc_id  = npc["npc_id"]
        persona = ", ".join(npc.get("persona_style", []))
        values  = ", ".join(npc.get("values", []))
        goals   = ". ".join(npc.get("core_goals", []))
        name    = npc.get("name", npc_id)
        profile = (
            f"{name}, a {npc.get('role', 'character')}. "
            f"Persona: {persona}. Values: {values}. Goals: {goals}."
        )
        # Register via the existing endpoint logic
        pvec = _encode_personality(profile)
        _registered_profiles[npc_id] = profile
        if _service is not None:
            _service.personality_cache.cache[npc_id] = pvec
            try:
                _service.register_npc(npc_id, profile)
            except Exception:
                pass
        loaded.append({"npc_id": npc_id, "name": name, "profile_text": profile})

    return {"loaded": loaded, "world_name": world.get("world_name", "?")}


# ── Routes: Chat ──────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat/{npc_id}")
async def chat(npc_id: str, req: ChatRequest):
    _ensure_service()

    if npc_id not in _registered_profiles:
        raise HTTPException(status_code=404, detail=f"NPC '{npc_id}' not registered")

    # Retrieve memories BEFORE respond() so we can return them
    memories: List[str] = []
    try:
        mem_results = _service.memory.search(npc_id, req.message, k=_service.cfg.memory_top_k)
        memories = [m.text for m in mem_results]
    except Exception:
        pass

    t0 = time.time()
    try:
        response = _service.respond(npc_id, req.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {e}")
    elapsed_ms = int((time.time() - t0) * 1000)

    state = _extract_state(npc_id)

    return {
        "response":   response,
        "elapsed_ms": elapsed_ms,
        "state":      state,
        "memories":   memories,
        "model":      "dialogue_model",
    }


@app.post("/api/reset/{npc_id}")
async def reset(npc_id: str):
    if _service and npc_id in _service.states:
        _service.states[npc_id].conversation_window.clear()
        _service.memory.search  # memory persists intentionally
    return {"success": True, "npc_id": npc_id}


# ── Route: NPC current state (no chat turn) ───────────────────────────────────

@app.get("/api/npcs/{npc_id}/state")
async def npc_state(npc_id: str):
    if npc_id not in _registered_profiles:
        raise HTTPException(status_code=404, detail=f"NPC '{npc_id}' not registered")
    if _service is None:
        pvec = _encode_personality(_registered_profiles[npc_id])
        return {
            "personality": {k: round(v, 4) for k, v in zip(OCEAN_DIMS, pvec)},
            "affect":      {k: 0.5 for k in VAD_DIMS},
            "note":        "Dialogue model not loaded",
        }
    return _extract_state(npc_id)
