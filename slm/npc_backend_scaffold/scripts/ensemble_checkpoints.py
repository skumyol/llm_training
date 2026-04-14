#!/usr/bin/env python3
"""Create ensemble from multiple checkpoints (stochastic weights averaging)."""

import torch
import json
from pathlib import Path
import numpy as np

def ensemble_personality_checkpoints(checkpoint_paths, output_path):
    """Average weights from multiple checkpoints."""
    state_dicts = []
    for path in checkpoint_paths:
        ckpt = torch.load(path, map_location="cpu")
        state_dicts.append(ckpt["model_state_dict"])
    
    # Average weights
    averaged = {}
    for key in state_dicts[0].keys():
        averaged[key] = torch.stack([sd[key] for sd in state_dicts]).mean(dim=0)
    
    # Save
    torch.save({"model_state_dict": averaged}, output_path)
    print(f"Ensemble saved to {output_path}")

if __name__ == "__main__":
    # Example: Ensemble epochs 3,4,5 from personality training
    # (where F1 was highest and stable)
    base_path = "artifacts/personality_encoder/personality_v2_aggressive"
    checkpoints = [
        f"{base_path}/checkpoint_epoch_3.pt",
        f"{base_path}/checkpoint_epoch_4.pt",  # Best
        f"{base_path}/checkpoint_epoch_5.pt",
    ]
    
    # Filter existing
    existing = [p for p in checkpoints if Path(p).exists()]
    if existing:
        ensemble_personality_checkpoints(
            existing, 
            f"{base_path}/ensemble_345.pt"
        )
