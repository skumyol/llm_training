#!/usr/bin/env python3
"""
run_parallel_pipeline.py
========================
GPU-aware parallel runner for the SLM training pipeline.

Manages a job queue with configurable concurrency, monitors GPU utilization,
and pipelines independent architecture HPO + final training jobs to minimize
wall-clock time on a single (or multi) GPU setup.

Key features:
  - Job queue with max_concurrency limit (default: 1 for single GPU)
  - GPU utilization monitoring — starts next job when GPU is idle
  - Per-architecture parallel HPO studies (independent, no GPU contention)
  - Per-architecture parallel final training (each seed is sequential within arch)
  - Automatic log collection and failure retry

Usage:
  # Full pipeline — HPO + train + eval
  python scripts/run_parallel_pipeline.py --mode full

  # HPO only, all architectures, 1 GPU worker
  python scripts/run_parallel_pipeline.py --mode hpo --workers 1

  # HPO with 2 concurrent GPU workers (if you have 2 GPUs)
  python scripts/run_parallel_pipeline.py --mode hpo --workers 2

  # Single architecture, quick test
  python scripts/run_parallel_pipeline.py --arch prefix_gpt --trials 5 --epochs 2

  # Resume from existing HPO bests, skip to training
  python scripts/run_parallel_pipeline.py --mode train --workers 1
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
LOG = logging.getLogger("parallel_pipeline")

ARCHS = ["gru", "awdlstm", "gpt", "prefix_gpt", "moe", "mamba_like"]


@dataclass
class Job:
    """A single training/HPO job."""
    name: str
    cmd: List[str]
    log_file: Path
    phase: str          # "hpo" | "train" | "eval"
    arch: str
    seed: Optional[int] = None
    timeout: int = 7200
    retries: int = 1

    def run(self) -> Tuple[bool, str]:
        """Execute the job, return (success, log_tail)."""
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        start = time.time()

        with open(self.log_file, "a") as f:
            f.write(f"\n=== {self.name} START {datetime.now().isoformat()} ===\n")
            f.flush()

            result = subprocess.run(
                self.cmd,
                cwd=str(ROOT),
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=self.timeout,
            )

            elapsed = time.time() - start
            f.write(f"\n=== {self.name} DONE rc={result.returncode} "
                   f"time={elapsed:.0f}s {datetime.now().isoformat()} ===\n")
            f.flush()

        success = result.returncode == 0
        # Read last 20 lines for error reporting
        tail = ""
        if self.log_file.exists():
            lines = self.log_file.read_text().splitlines()
            tail = "\n".join(lines[-20:])

        return success, tail


def wait_for_gpu_idle(device: int = 0, threshold_mb: int = 200,
                       poll_interval: float = 2.0, max_wait: float = 300.0) -> bool:
    """Poll nvidia-smi until GPU memory usage drops below threshold."""
    start = time.time()
    while time.time() - start < max_wait:
        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
                text=True, timeout=5
            )
            mem_used = int(out.strip().split("\n")[device].strip())
            if mem_used < threshold_mb:
                return True
        except Exception:
            pass
        time.sleep(poll_interval)
    LOG.warning(f"GPU {device} did not become idle within {max_wait}s")
    return False


def build_jobs(
    archs: List[str],
    mode: str,
    n_trials: int,
    hpo_epochs: int,
    final_epochs: int,
    seeds: List[int],
    train_text: Optional[str] = None,
    val_text: Optional[str] = None,
    skip_existing: bool = True,
) -> List[Job]:
    """Build the full job queue."""
    jobs: List[Job] = []
    run_tag = f"parallel_{datetime.now():%Y%m%d_%H%M%S}"
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    # ── Phase 1: HPO ─────────────────────────────────────────────────────────
    if mode in ("full", "hpo"):
        for arch in archs:
            best_json = ROOT / "artifacts" / "optuna" / f"small_lm_{arch}_best.json"
            if skip_existing and best_json.exists():
                LOG.info(f"  {arch}: best.json exists — skipping HPO")
                continue

            log_file = log_dir / f"parallel_hpo_{arch}_{run_tag}.log"
            cmd = [
                PYTHON, str(ROOT / "scripts" / "optuna_small_lm.py"),
                "--arch", arch,
                "--n-trials", str(n_trials),
                "--epochs", str(hpo_epochs),
                "--timeout", "1800",
            ]
            if train_text:
                cmd += ["--train-text", train_text]
            if val_text:
                cmd += ["--val-text", val_text]

            jobs.append(Job(
                name=f"HPO:{arch}",
                cmd=cmd,
                log_file=log_file,
                phase="hpo",
                arch=arch,
                timeout=n_trials * 300,  # ~5 min per trial max
            ))

    # ── Phase 2: Final Training ──────────────────────────────────────────────
    if mode in ("full", "train"):
        for arch in archs:
            best_json = ROOT / "artifacts" / "optuna" / f"small_lm_{arch}_best.json"
            if not best_json.exists():
                LOG.warning(f"  {arch}: missing best.json — skipping final training")
                continue

            for seed in seeds:
                log_file = log_dir / f"parallel_train_{arch}_s{seed}_{run_tag}.log"
                cmd = [
                    PYTHON, str(ROOT / "scripts" / "train_final_small_lms.py"),
                    "--arch", arch,
                    "--seeds", str(seed),
                    "--epochs", str(final_epochs),
                ]
                if skip_existing:
                    cmd.append("--skip-existing")
                if train_text:
                    cmd += ["--train-text", train_text]
                if val_text:
                    cmd += ["--val-text", val_text]

                jobs.append(Job(
                    name=f"TRAIN:{arch}:seed={seed}",
                    cmd=cmd,
                    log_file=log_file,
                    phase="train",
                    arch=arch,
                    seed=seed,
                    timeout=final_epochs * 600,  # ~10 min per epoch max
                ))

    # ── Phase 3: Evaluation ──────────────────────────────────────────────────
    if mode == "full":
        log_file = log_dir / f"parallel_eval_{run_tag}.log"
        eval_csv = ROOT / "artifacts" / f"slm_parallel_eval_{run_tag}.csv"
        jobs.append(Job(
            name="EVAL:all",
            cmd=[
                PYTHON, str(ROOT / "scripts" / "eval_small_lms.py"),
                "--out-csv", str(eval_csv),
            ],
            log_file=log_file,
            phase="eval",
            arch="all",
            timeout=1800,
        ))

    return jobs


def run_sequential_with_gpu_wait(jobs: List[Job], workers: int = 1) -> Dict[str, Any]:
    """Run jobs sequentially (or with limited concurrency), waiting for GPU idle."""
    results: Dict[str, Any] = {"success": [], "failed": []}

    # For simplicity with single GPU, run jobs one at a time but in a smart order:
    # Group by phase: all HPO first, then all train, then eval
    phases = ["hpo", "train", "eval"]

    for phase in phases:
        phase_jobs = [j for j in jobs if j.phase == phase]
        if not phase_jobs:
            continue

        LOG.info(f"\n{'='*60}")
        LOG.info(f"  Phase: {phase.upper()}  ({len(phase_jobs)} jobs)")
        LOG.info(f"{'='*60}")

        # Within a phase, we can run multiple jobs in parallel if workers > 1
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(j.run): j for j in phase_jobs}
                for future in as_completed(futures):
                    job = futures[future]
                    try:
                        success, tail = future.result()
                        if success:
                            LOG.info(f"  ✓ {job.name}")
                            results["success"].append(job.name)
                        else:
                            LOG.error(f"  ✗ {job.name} FAILED")
                            LOG.debug(f"    Tail:\n{tail}")
                            results["failed"].append(job.name)
                    except Exception as e:
                        LOG.error(f"  ✗ {job.name} EXCEPTION: {e}")
                        results["failed"].append(job.name)
        else:
            # Single worker: run one at a time, wait for GPU idle between jobs
            for job in phase_jobs:
                if phase in ("hpo", "train"):
                    wait_for_gpu_idle()

                LOG.info(f"  Starting: {job.name}")
                success, tail = job.run()

                if success:
                    LOG.info(f"  ✓ {job.name}")
                    results["success"].append(job.name)
                else:
                    LOG.error(f"  ✗ {job.name} FAILED")
                    LOG.debug(f"    Tail:\n{tail}")
                    results["failed"].append(job.name)

    return results


def main():
    p = argparse.ArgumentParser(description="Parallel SLM training pipeline")
    p.add_argument("--mode", choices=["full", "hpo", "train"], default="full",
                   help="Pipeline phase to run")
    p.add_argument("--arch", default="all",
                   help=f"Architecture or 'all'. Choices: {', '.join(ARCHS)}")
    p.add_argument("--workers", type=int, default=1,
                   help="Max concurrent GPU workers (default: 1 for single GPU)")
    p.add_argument("--trials", type=int, default=20,
                   help="HPO trials per architecture")
    p.add_argument("--hpo-epochs", type=int, default=5,
                   help="Epochs per HPO trial")
    p.add_argument("--epochs", type=int, default=30,
                   help="Final training epochs")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44],
                   help="Random seeds for final training")
    p.add_argument("--train-text", default=None,
                   help="Override training text file")
    p.add_argument("--val-text", default=None,
                   help="Override validation text file")
    p.add_argument("--no-skip-existing", action="store_true",
                   help="Re-run even if best.json exists")
    p.add_argument("--dry-run", action="store_true",
                   help="Print job list without executing")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    archs = ARCHS if args.arch == "all" else [args.arch]
    if args.arch not in ARCHS and args.arch != "all":
        p.error(f"Unknown arch '{args.arch}'")

    LOG.info("="*60)
    LOG.info("  PARALLEL SLM PIPELINE")
    LOG.info(f"  Mode    : {args.mode}")
    LOG.info(f"  Archs   : {archs}")
    LOG.info(f"  Workers : {args.workers}")
    LOG.info("="*60)

    jobs = build_jobs(
        archs=archs,
        mode=args.mode,
        n_trials=args.trials,
        hpo_epochs=args.hpo_epochs,
        final_epochs=args.epochs,
        seeds=args.seeds,
        train_text=args.train_text,
        val_text=args.val_text,
        skip_existing=not args.no_skip_existing,
    )

    if not jobs:
        LOG.info("No jobs to run (all skipped or already complete).")
        return

    LOG.info(f"\nJob queue: {len(jobs)} jobs")
    for job in jobs:
        LOG.info(f"  [{job.phase:6}] {job.name:30} → {job.log_file.name}")

    if args.dry_run:
        LOG.info("\n[DRY RUN] Jobs listed above would be executed.")
        return

    # Run the pipeline
    start = time.time()
    results = run_sequential_with_gpu_wait(jobs, workers=args.workers)
    elapsed = time.time() - start

    # Summary
    LOG.info(f"\n{'='*60}")
    LOG.info("  PIPELINE COMPLETE")
    LOG.info(f"  Wall time: {elapsed/60:.1f} min")
    LOG.info(f"  Success  : {len(results['success'])}")
    LOG.info(f"  Failed   : {len(results['failed'])}")
    if results["failed"]:
        LOG.info(f"  Failures : {', '.join(results['failed'])}")
    LOG.info("="*60)

    sys.exit(0 if not results["failed"] else 1)


if __name__ == "__main__":
    main()
