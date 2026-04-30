#!/usr/bin/env python3
"""
Export trained Small LM models for deployment/testing on another machine.

Each model is packaged as a self-contained directory with:
  - model.pt        : checkpoint (state_dict + config)
  - config.json     : training/inference config
  - tokenizer.json  : vocabulary info (for non-tiktoken models)
  - inference.py    : standalone inference script
  - requirements.txt: dependencies

Usage:
  python scripts/export_small_lm_models.py --all
  python scripts/export_small_lm_models.py --arch gpt --seed 42
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from train.small_lm_architectures import build_model

ROOT = Path(__file__).parent.parent
EXPORT_DIR = ROOT / "artifacts" / "exported_models"


def find_best_models(arch: str = None, seed: int = None) -> List[Path]:
    """Find all completed (30-epoch) best_model.pt files."""
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


def export_model(run_dir: Path, out_dir: Path):
    """Export one trained model to a portable directory."""
    ckpt = torch.load(run_dir / "best_model.pt", map_location="cpu")
    config = json.load(open(run_dir / "config.json"))
    summary = json.load(open(run_dir / "run_summary.json"))

    arch = ckpt["arch"]
    run_id = run_dir.name

    export = out_dir / f"{arch}_{run_id}"
    export.mkdir(parents=True, exist_ok=True)

    # 1. Save model checkpoint
    torch.save(ckpt, export / "model.pt")

    # 2. Save inference config (subset of training config)
    inf_config = {
        "arch": arch,
        "vocab_size": config.get("vocab_size", config.get("arch_params", {}).get("vocab_size")),
        "seq_len": config.get("seq_len", 256),
        "device": "cpu",  # Default to CPU for portability
        "params": ckpt["params"],
        "model_info": {
            "run_id": run_id,
            "best_epoch": ckpt.get("epoch"),
            "best_val_loss": ckpt.get("val_loss"),
            "best_val_ppl": summary.get("best_ppl"),
            "seed": summary.get("seed"),
        }
    }
    json.dump(inf_config, open(export / "config.json", "w"), indent=2)

    # 3. Copy tokenizer if it's a custom one (not tiktoken)
    # tiktoken is standard; just need to note which encoding
    tokenizer_name = config.get("tokenizer", "tiktoken:gpt2")
    if tokenizer_name.startswith("tiktoken"):
        inf_config["tokenizer"] = tokenizer_name
    else:
        # Custom BPE tokenizer - copy the vocab/merge files
        tok_dir = run_dir / "tokenizer"
        if tok_dir.exists():
            shutil.copytree(tok_dir, export / "tokenizer", dirs_exist_ok=True)
        inf_config["tokenizer"] = "custom"

    json.dump(inf_config, open(export / "config.json", "w"), indent=2)

    # 4. Write standalone inference script
    (export / "inference.py").write_text(f'''#!/usr/bin/env python3
"""Standalone inference script for {arch} model.
Usage:
    python inference.py --prompt "Hello, how are you?" --max-tokens 50
    python inference.py --interactive
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Add src to path if available (for development)
src = Path(__file__).parent.parent / "src"
if src.exists():
    sys.path.insert(0, str(src))
    from train.small_lm_architectures import build_model
else:
    # Minimal fallback - must have source code
    raise ImportError("Please ensure src/train/small_lm_architectures.py is available")


def load_model(model_dir: Path, device: str = "cpu"):
    ckpt = torch.load(model_dir / "model.pt", map_location=device)
    cfg = json.load(open(model_dir / "config.json"))
    model = build_model(cfg["arch"], cfg["params"])
    model.load_state_dict(ckpt["state"])
    model.to(device)
    model.eval()
    return model, cfg


def encode(text: str, tokenizer_name: str):
    if tokenizer_name.startswith("tiktoken"):
        import tiktoken
        enc = tiktoken.get_encoding(tokenizer_name.split(":")[1])
        return enc.encode(text)
    else:
        # Custom tokenizer loading
        raise NotImplementedError(f"Custom tokenizer not yet implemented: {{tokenizer_name}}")


def decode(tokens: list, tokenizer_name: str):
    if tokenizer_name.startswith("tiktoken"):
        import tiktoken
        enc = tiktoken.get_encoding(tokenizer_name.split(":")[1])
        return enc.decode(tokens)
    else:
        raise NotImplementedError(f"Custom tokenizer not yet implemented: {{tokenizer_name}}")


def generate(model, prompt_ids: list, max_tokens: int = 50, temperature: float = 1.0, top_k: int = 50, device: str = "cpu"):
    ids = list(prompt_ids)
    with torch.no_grad():
        for _ in range(max_tokens):
            x = torch.tensor([ids[-model.cfg.get("seq_len", 256):]], dtype=torch.long, device=device)
            logits = model(x)
            logits = logits[0, -1] / temperature
            # Top-k
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[-1]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1).item()
            ids.append(next_tok)
            if next_tok in (encode("<|endoftext|>", "tiktoken:gpt2") or [50256]):
                break
    return ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default=".", help="Path to exported model directory")
    p.add_argument("--prompt", default="Hello, how are you?", help="Input prompt")
    p.add_argument("--max-tokens", type=int, default=50)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--device", default="cpu")
    p.add_argument("--interactive", action="store_true")
    args = p.parse_args()

    model_dir = Path(args.model_dir)
    cfg = json.load(open(model_dir / "config.json"))
    model, _ = load_model(model_dir, args.device)
    tok = cfg.get("tokenizer", "tiktoken:gpt2")

    if args.interactive:
        print(f"Interactive mode ({{cfg['arch']}}). Type 'quit' to exit.")
        while True:
            prompt = input(">>> ")
            if prompt.lower() in ("quit", "exit", "q"):
                break
            ids = encode(prompt, tok)
            out = generate(model, ids, args.max_tokens, args.temperature, args.top_k, args.device)
            text = decode(out, tok)
            print(text)
    else:
        ids = encode(args.prompt, tok)
        out = generate(model, ids, args.max_tokens, args.temperature, args.top_k, args.device)
        print(decode(out, tok))


if __name__ == "__main__":
    main()
''')

    # 5. Requirements
    (export / "requirements.txt").write_text("""torch>=2.0.0
tiktoken>=0.5.0
""")

    # 6. README
    (export / "README.md").write_text(f"""# Exported {arch.upper()} Model

Run ID: `{run_id}`
Best val PPL: `{summary.get('best_ppl', 'N/A')}`
Seed: {summary.get('seed')}
Epochs: {len(summary.get('epochs', []))}

## Quick Start

```bash
pip install -r requirements.txt
python inference.py --prompt "Hello world" --max-tokens 50
```

## Files

- `model.pt` - PyTorch checkpoint (state_dict + config)
- `config.json` - Model architecture and metadata
- `inference.py` - Standalone inference script
- `requirements.txt` - Python dependencies

## Notes

- This model was trained on a dialogue corpus (~107M tokens).
- For production use, consider quantizing (INT8/INT4) or compiling (torch.compile).
- The model expects `src/train/small_lm_architectures.py` for `build_model()`.
""")

    size_mb = (export / "model.pt").stat().st_size / 1024 / 1024
    print(f"  → Exported {arch} ({run_id}) to {export}")
    print(f"    Model size: {size_mb:.1f} MB")
    print(f"    Best PPL: {summary.get('best_ppl', 'N/A'):.2f}")
    return export


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true", help="Export all completed models")
    p.add_argument("--arch", help="Export specific architecture")
    p.add_argument("--seed", type=int, help="Export specific seed")
    p.add_argument("--out-dir", default=str(EXPORT_DIR))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = find_best_models(args.arch, args.seed)
    if not runs:
        print("No completed (30-epoch) models found.")
        print("Completed models:")
        for d in sorted((ROOT / "artifacts" / "small_lm").glob("final_*_*_*")):
            summary = d / "run_summary.json"
            if summary.exists():
                data = json.load(open(summary))
                epochs = data.get("epochs", [])
                status = "DONE" if isinstance(epochs, list) and len(epochs) >= 30 else f"ep={len(epochs)}"
                print(f"  {d.name}: {status}  arch={data.get('arch')} seed={data.get('seed')}")
        return

    print(f"Found {len(runs)} completed model(s)")
    for run_dir in runs:
        export_model(run_dir, out_dir)

    # Create a combined archive
    if len(runs) > 1:
        archive = out_dir / "all_models.tar.gz"
        import tarfile
        with tarfile.open(archive, "w:gz") as tar:
            for run_dir in runs:
                ckpt = torch.load(run_dir / "best_model.pt", map_location="cpu")
                name = f"{ckpt['arch']}_{run_dir.name}"
                tar.add(out_dir / name, arcname=name)
        size_mb = archive.stat().st_size / 1024 / 1024
        print(f"\n→ Combined archive: {archive} ({size_mb:.1f} MB)")

    print(f"\nExported to: {out_dir}")


if __name__ == "__main__":
    main()
