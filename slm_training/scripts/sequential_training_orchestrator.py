#!/usr/bin/env python3
"""
Sequential training orchestrator - runs all archs one at a time (1 GPU).
Restarts automatically if interrupted. Safe to run with nohup.
"""
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent  # slm_training/
ARCHS = ["gpt", "prefix_gpt", "moe", "mamba_like"]
SEEDS = [42, 43, 44]
EPOCHS = 20
TRAIN_TEXT = "data/external/merged_dialogue.txt"
VAL_TEXT = "data/dialogue/val.txt"

def check_completed(arch: str, seed: int) -> bool:
    """Check if a run is already completed (20 epochs)."""
    # Look for run_summary.json with 20+ epochs
    for d in (ROOT / "artifacts" / "small_lm").glob(f"final_{arch}_s{seed}_*"):
        summary = d / "run_summary.json"
        if summary.exists():
            try:
                data = json.load(open(summary))
                epochs = data.get("epochs", [])
                if len(epochs) >= 20:
                    return True
            except:
                pass
    return False

def run_arch(arch: str):
    """Run all seeds for one architecture sequentially."""
    print(f"\n{'='*60}")
    print(f"  {arch.upper()}  (3 seeds × {EPOCHS} epochs)")
    print(f"{'='*60}")
    
    for seed in SEEDS:
        if check_completed(arch, seed):
            print(f"  [skip] {arch} seed={seed} — already completed")
            continue
        
        print(f"\n  [START] {arch} seed={seed} at {datetime.now():%H:%M:%S}")
        
        cmd = [
            ROOT / ".venv/bin/python",
            "scripts/train_final_small_lms.py",
            "--arch", arch,
            "--seeds", str(seed),
            "--epochs", str(EPOCHS),
            "--timeout", "172800",
            "--train-text", TRAIN_TEXT,
            "--val-text", VAL_TEXT,
            "--skip-existing"
        ]
        
        result = subprocess.run(cmd, cwd=ROOT, capture_output=False)
        
        if result.returncode != 0:
            print(f"  [ERROR] {arch} seed={seed} failed with code {result.returncode}")
        else:
            print(f"  [DONE] {arch} seed={seed}")
        
        # Small delay between runs
        time.sleep(5)

def main():
    print(f"Sequential Training Orchestrator started at {datetime.now()}")
    print(f"Training {len(ARCHS)} architectures × {len(SEEDS)} seeds = {len(ARCHS)*len(SEEDS)} runs")
    print(f"GPU: Single (sequential execution)\n")
    
    for arch in ARCHS:
        run_arch(arch)
    
    print(f"\n{'='*60}")
    print(f"  ALL TRAINING COMPLETED at {datetime.now()}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
