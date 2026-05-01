#!/usr/bin/env python3
"""
Parallel training orchestrator - runs multiple seeds concurrently on single GPU.
Uses process pool to maximize GPU utilization.
"""
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

ROOT = Path("/home/serkan/llm_training/slm_training")
PYTHON = str(ROOT / ".venv/bin/python")

# Remaining runs after current progress
REMAINING = [
    ("mamba_like", 43), ("mamba_like", 44),
    ("prefix_gpt", 42), ("prefix_gpt", 43),
    ("moe", 42), ("moe", 43), ("moe", 44),
]

def check_completed(arch: str, seed: int, min_epochs: int = 20) -> bool:
    for d in (ROOT / "artifacts" / "small_lm").glob(f"final_{arch}_s{seed}_*"):
        summary = d / "run_summary.json"
        if summary.exists():
            try:
                data = json.load(open(summary))
                epochs = data.get("epochs", [])
                if isinstance(epochs, list) and len(epochs) >= min_epochs:
                    return True
            except:
                pass
    return False

def run_single(arch: str, seed: int, epochs: int = 20) -> dict:
    run_id = f"final_{arch}_s{seed}_{datetime.now():%H%M%S}"
    out_dir = ROOT / "artifacts" / "small_lm" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = out_dir / "run.log"
    
    cmd = [
        PYTHON, "scripts/train_final_small_lms.py",
        "--arch", arch,
        "--seeds", str(seed),
        "--epochs", str(epochs),
        "--timeout", "86400",
        "--train-text", "data/external/merged_dialogue.txt",
        "--val-text", "data/dialogue/val.txt",
        "--skip-existing"
    ]
    
    print(f"[START] {arch} seed={seed} at {datetime.now():%H:%M:%S}")
    
    with open(log_file, "w") as f:
        result = subprocess.run(
            cmd, cwd=ROOT,
            stdout=f, stderr=subprocess.STDOUT
        )
    
    status = "DONE" if result.returncode == 0 else "FAILED"
    print(f"[{status}] {arch} seed={seed} (code={result.returncode})")
    
    return {
        "arch": arch, "seed": seed,
        "success": result.returncode == 0,
        "run_id": run_id
    }

def run_sequential(archs_seeds: list, max_workers: int = 1):
    """Run all remaining jobs."""
    print(f"Parallel Orchestrator started at {datetime.now()}")
    print(f"GPU: RTX 2080 Ti (22GB)")
    print(f"Workers: {max_workers}")
    print(f"Jobs remaining: {len(archs_seeds)}\n")
    
    if max_workers == 1:
        # Sequential
        results = []
        for arch, seed in archs_seeds:
            if check_completed(arch, seed):
                print(f"[SKIP] {arch} s{seed} already done")
                continue
            r = run_single(arch, seed)
            results.append(r)
            time.sleep(5)
    else:
        # Parallel - submit all, let GPU time-slice
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for arch, seed in archs_seeds:
                if check_completed(arch, seed):
                    print(f"[SKIP] {arch} s{seed} already done")
                    continue
                f = executor.submit(run_single, arch, seed)
                futures[f] = (arch, seed)
                time.sleep(30)  # Stagger starts to avoid CUDA conflicts
            
            results = []
            for f in as_completed(futures):
                arch, seed = futures[f]
                try:
                    r = f.result()
                    results.append(r)
                except Exception as e:
                    print(f"[ERROR] {arch} s{seed}: {e}")
                    results.append({"arch": arch, "seed": seed, "success": False})
    
    print(f"\n{'='*60}")
    print(f"  ALL JOBS COMPLETED at {datetime.now()}")
    print(f"  Successful: {sum(1 for r in results if r.get('success'))}/{len(results)}")
    print(f"{'='*60}")
    
    return results

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=1,
                   help="Concurrent jobs (1=sequential, 2+=parallel on same GPU)")
    p.add_argument("--epochs", type=int, default=20)
    args = p.parse_args()
    
    # Safety: parallel on 1 GPU is risky for MoE
    if args.workers > 1:
        print("WARNING: Running multiple jobs on 1 GPU may cause OOM")
        print("Recommended: Use --workers=1 or submit to HPC")
        time.sleep(3)
    
    run_sequential(REMAINING, max_workers=args.workers)
