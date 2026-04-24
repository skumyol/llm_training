#!/usr/bin/env python3
"""
Upload trained Small LM models to Hugging Face Hub.
Reads HF_HUB_TOKEN from /home/serkan/llm_training/.env

Usage:
    python scripts/upload_to_hf.py --arch awdlstm --seed 44
    python scripts/upload_to_hf.py --all-completed
"""

import argparse
import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_file

ROOT = Path(__file__).parent.parent
ENV_FILE = Path("/home/serkan/llm_training/.env")


def get_token() -> str:
    """Read HF_HUB_TOKEN from .env file."""
    with open(ENV_FILE) as f:
        for line in f:
            if line.startswith("HF_HUB_TOKEN="):
                return line.strip().split("=", 1)[1]
    raise ValueError("HF_HUB_TOKEN not found in .env")


def find_completed_models(arch: str = None, seed: int = None):
    """Find all completed (30-epoch) models."""
    models = []
    for d in sorted((ROOT / "artifacts" / "small_lm").glob("final_*_*_*")):
        summary = d / "run_summary.json"
        model = d / "best_model.pt"
        config = d / "config.json"
        if not (summary.exists() and model.exists() and config.exists()):
            continue

        data = json.load(open(summary))
        epochs = data.get("epochs", [])
        if isinstance(epochs, list) and len(epochs) >= 30:
            if arch and data.get("arch") != arch:
                continue
            if seed is not None and data.get("seed") != seed:
                continue
            models.append(d)
    return models


def upload_model(run_dir: Path, token: str, org: str = None):
    """Upload one model to Hugging Face Hub."""
    ckpt = json.load(open(run_dir / "config.json"))
    summary = json.load(open(run_dir / "run_summary.json"))

    arch = summary.get("arch", "unknown")
    seed = summary.get("seed", "unknown")
    run_id = run_dir.name
    
    # Extract seed from directory name if missing from summary
    if seed == "unknown":
        parts = run_id.split("_")
        for p in parts:
            if p.startswith("s") and p[1:].isdigit():
                seed = p[1:]
                break

    # Repo ID: username/slm-{arch}-s{seed}
    username = HfApi(token=token).whoami()["name"]
    repo_id = f"{username}/slm-{arch}-s{seed}-{run_id[-6:]}"

    print(f"\nUploading {arch} seed={seed} → {repo_id}")

    # Create repo (private by default)
    try:
        create_repo(repo_id, token=token, repo_type="model", exist_ok=True, private=True)
        print(f"  ✓ Repo created/exists")
    except Exception as e:
        print(f"  ⚠ Repo issue: {e}")

    # Upload files
    api = HfApi(token=token)

    # Model checkpoint
    print(f"  Uploading model.pt ({(run_dir / 'best_model.pt').stat().st_size / 1024 / 1024:.1f} MB)...")
    upload_file(
        path_or_fileobj=str(run_dir / "best_model.pt"),
        path_in_repo="model.pt",
        repo_id=repo_id,
        token=token,
    )

    # Config
    upload_file(
        path_or_fileobj=str(run_dir / "config.json"),
        path_in_repo="config.json",
        repo_id=repo_id,
        token=token,
    )

    # Summary
    upload_file(
        path_or_fileobj=str(run_dir / "run_summary.json"),
        path_in_repo="run_summary.json",
        repo_id=repo_id,
        token=token,
    )

    # README
    readme = f"""# {arch.upper()} Small LM - Seed {seed}

**Architecture:** {arch}  
**Seed:** {seed}  
**Epochs:** 30  
**Best Val PPL:** {summary.get('best_ppl', 'N/A'):.2f}

## Files

- `model.pt` - PyTorch checkpoint (state_dict + config)
- `config.json` - Training configuration
- `run_summary.json` - Training metrics

## Usage

```python
import torch
from train.small_lm_architectures import build_model

ckpt = torch.load("model.pt", map_location="cpu")
model = build_model(ckpt["arch"], ckpt["params"])
model.load_state_dict(ckpt["state"])
model.eval()
```

**Note:** Requires `src/train/small_lm_architectures.py` from the original repo.
"""
    upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        token=token,
    )

    print(f"  ✓ Uploaded to https://huggingface.co/{repo_id}")
    return repo_id


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--all-completed", action="store_true", help="Upload all completed models")
    p.add_argument("--arch", help="Upload specific architecture")
    p.add_argument("--seed", type=int, help="Upload specific seed")
    p.add_argument("--list", action="store_true", help="List completed models without uploading")
    args = p.parse_args()

    token = get_token()
    user = HfApi(token=token).whoami()["name"]
    print(f"Authenticated as: {user}")

    models = find_completed_models(args.arch, args.seed)

    if not models:
        print("No completed (30-epoch) models found.")
        return

    print(f"\nFound {len(models)} completed model(s):")
    for d in models:
        summary = json.load(open(d / "run_summary.json"))
        arch = summary.get('arch', '?')
        seed = summary.get('seed', '?')
        ppl = summary.get('best_ppl', 0)
        print(f"  {arch:<12} s{seed:<3}  PPL={ppl:.2f}  {d.name}")

    if args.list:
        return

    print(f"\nUploading to https://huggingface.co/{user}")
    for d in models:
        try:
            repo_id = upload_model(d, token)
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
