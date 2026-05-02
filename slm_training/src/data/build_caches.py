#!/usr/bin/env python3
"""
build_caches.py
===============
Builds the personality cache by encoding NPC profiles using the trained
personality encoder (DistilBERT → OCEAN).

Usage:
    python -m src.data.build_caches \
        --profiles-path data/npc_profiles.csv \
        --encoder-dir artifacts/personality_encoder/run_xxx/best_model \
        --out-path artifacts/personality_cache.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    return logging.getLogger(__name__)


def load_personality_encoder(encoder_dir: str, device: str, log: logging.Logger):
    """Load the trained personality encoder model."""
    from src.models.personality import DistilBertRegressor
    
    model_path = Path(encoder_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Encoder directory not found: {encoder_dir}")
    
    # Load config to get dims
    config_path = model_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        dims = config.get("dims", ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"])
        out_dim = len(dims)
    else:
        dims = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
        out_dim = 5
    
    # Load tokenizer from base model
    base_model = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Load model - instantiate with fresh head
    model = DistilBertRegressor(model_name=base_model, out_dim=out_dim)
    
    # Load encoder weights from safetensors or pytorch (saved model is base DistilBERT)
    weights_path = model_path / "model.safetensors"
    if weights_path.exists():
        from safetensors.torch import load_file
        encoder_state = load_file(str(weights_path))
    else:
        weights_path = model_path / "pytorch_model.bin"
        if weights_path.exists():
            encoder_state = torch.load(str(weights_path), map_location=device)
        else:
            raise FileNotFoundError(f"No weights found in {model_path}")
    
    # Load encoder weights (head remains randomly initialized - not used for caching)
    model.encoder.load_state_dict(encoder_state)
    model.to(device)
    model.eval()
    
    log.info(f"Loaded personality encoder from {encoder_dir}")
    log.info(f"  Tokenizer: {base_model}")
    log.info(f"  Dims: {dims}")
    
    return model, tokenizer, dims


def encode_profiles(
    profiles_path: str,
    encoder_dir: str,
    out_path: str,
    device: Optional[str] = None,
) -> None:
    """Encode NPC profiles and save to cache."""
    log = setup_logging()
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    log.info(f"Device: {device}")
    log.info(f"Loading encoder from: {encoder_dir}")
    log.info(f"Reading profiles from: {profiles_path}")
    
    # Load encoder
    model, tokenizer, dims = load_personality_encoder(encoder_dir, device, log)
    
    # Read profiles
    profiles: List[Dict[str, str]] = []
    with open(profiles_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            profiles.append(row)
    
    log.info(f"Found {len(profiles)} NPC profiles")
    
    # Encode each profile
    cache_entries: List[Dict] = []
    
    with torch.no_grad():
        for profile in profiles:
            npc_id = profile.get("npc_id", profile.get("id", "unknown"))
            description = profile.get("profile_text", profile.get("description", profile.get("profile", "")))
            
            if not description:
                log.warning(f"Empty description for {npc_id}, skipping")
                continue
            
            # Tokenize
            inputs = tokenizer(
                description,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding="max_length",
            )
            inputs = {k: v.to(device) for k, v in inputs.items() if k != "token_type_ids"}
            
            # Encode
            outputs = model(**inputs)
            ocean_vec = outputs["preds"].cpu().numpy().flatten().tolist()
            
            # Build entry
            entry = {
                "npc_id": npc_id,
                "description": description,
                "vector": ocean_vec,
                "dims": dims,
            }
            cache_entries.append(entry)
    
    # Write cache
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_file, "w", encoding="utf-8") as f:
        for entry in cache_entries:
            f.write(json.dumps(entry) + "\n")
    
    log.info(f"✓ Wrote {len(cache_entries)} entries to {out_path}")
    
    # Summary stats
    if cache_entries:
        vecs = np.array([e["vector"] for e in cache_entries])
        log.info(f"  OCEAN means: {dict(zip(dims, np.round(vecs.mean(axis=0), 3).tolist()))}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build personality cache")
    p.add_argument("--profiles-path", required=True, help="Path to NPC profiles CSV")
    p.add_argument("--encoder-dir", required=True, help="Path to trained personality encoder")
    p.add_argument("--out-path", required=True, help="Output path for cache JSONL")
    p.add_argument("--device", default=None, help="Device (cuda/cpu)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    encode_profiles(
        profiles_path=args.profiles_path,
        encoder_dir=args.encoder_dir,
        out_path=args.out_path,
        device=args.device,
    )
