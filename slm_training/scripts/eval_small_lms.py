#!/usr/bin/env python3
"""
eval_small_lms.py
=================
Evaluates all trained Small-LM checkpoints and prints a comparison table.

Metrics computed:
  - val_ppl        : best val perplexity from training (epoch_metrics.csv)
  - test_ppl       : fresh perplexity on held-out val text (loaded model)
  - bleu_1/2       : corpus BLEU on 256 val windows (prompt→generate vs reference)
  - distinct_1/2   : lexical diversity of generated text (higher = less repetitive)
  - gen_sample     : a short generated sample from each model

Usage:
  python scripts/eval_small_lms.py
  python scripts/eval_small_lms.py --gen-len 100 --out-csv artifacts/slm_eval.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import sacrebleu as _sacrebleu
    _SACREBLEU_OK = True
except ImportError:
    _SACREBLEU_OK = False

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "train"))
sys.path.insert(0, str(ROOT))

from small_lm_architectures import build_model

# ── ANSI colours ───────────────────────────────────────────────────────────────
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
CYAN   = "\033[36m"
YELLOW = "\033[33m"
RED    = "\033[31m"
RESET  = "\033[0m"

ARCHS = ["gru", "awdlstm", "gpt", "prefix_gpt", "moe", "mamba_like"]

ARCH_LABELS = {
    "gru":        "GRU-LM",
    "awdlstm":    "AWD-LSTM",
    "gpt":        "TinyGPT",
    "prefix_gpt": "PrefixGPT",
    "moe":        "TinyMoE",
    "mamba_like": "Mamba-like",
}


# ── Tokenizer ──────────────────────────────────────────────────────────────────

def get_tokenizer():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        return enc
    except ImportError:
        raise RuntimeError("tiktoken not installed — pip install tiktoken")


# ── Best run discovery ─────────────────────────────────────────────────────────

def find_best_run(arch: str, slm_dir: Path) -> Optional[Path]:
    """Return the run dir with lowest best_val_ppl for this arch.
    Scans: report_slm_{arch}_*, final_{arch}_*, optuna_{arch}_*
    """
    patterns = [
        f"report_slm_{arch}_s*",
        f"final_{arch}_s*",
    ]
    runs = []
    for pat in patterns:
        runs.extend([d for d in slm_dir.glob(pat) if d.is_dir()])
    runs = sorted(set(runs))

    best_ppl = float("inf")
    best_dir = None
    for run_dir in runs:
        summary_path = run_dir / "run_summary.json"
        if not summary_path.exists():
            continue
        try:
            with open(summary_path) as f:
                data = json.load(f)
        except Exception:
            continue
        ppl = data.get("best", {}).get("val_ppl", float("inf"))
        epochs_completed = len(data.get("epochs", []))
        if epochs_completed < 2:
            continue  # skip incomplete runs
        # Validate checkpoint vocab_size matches tiktoken gpt2 (50,257)
        ckpt_path = run_dir / "best_model.pt"
        if ckpt_path.exists():
            try:
                import torch as _torch
                meta = _torch.load(ckpt_path, map_location="cpu", weights_only=False)
                ckpt_vocab = meta.get("params", {}).get("vocab_size", 50257)
                if ckpt_vocab < 1000:  # obviously wrong (char-level etc.)
                    continue
            except Exception:
                pass
        if ppl < best_ppl:
            best_ppl = ppl
            best_dir = run_dir
    return best_dir


# ── Load model from checkpoint ─────────────────────────────────────────────────

def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch   = ckpt["arch"]
    params = ckpt["params"]
    model  = build_model(arch, params)
    missing, unexpected = model.load_state_dict(ckpt["state"], strict=False)
    if unexpected:
        import logging
        logging.getLogger(__name__).warning(
            f"  [{arch}] Ignoring {len(unexpected)} unexpected checkpoint keys "
            f"(e.g. {unexpected[0]})"
        )
    model.to(device)
    model.eval()
    return model, arch, ckpt.get("val_loss", None)


# ── Compute perplexity on raw text ─────────────────────────────────────────────

@torch.no_grad()
def compute_ppl(model, token_ids: List[int], seq_len: int, device: torch.device,
                cond_dim: int = 8) -> float:
    """Slide a window over token_ids and accumulate cross-entropy."""
    if len(token_ids) < seq_len + 1:
        return float("nan")
    total_loss = 0.0
    n_batches  = 0
    for start in range(0, len(token_ids) - seq_len, seq_len):
        chunk = token_ids[start: start + seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long, device=device).unsqueeze(0)
        y = torch.tensor(chunk[1:],  dtype=torch.long, device=device).unsqueeze(0)
        # Handle prefix_gpt: needs cond vector
        try:
            arch_name = model.__class__.__name__.lower()
            if "prefix" in arch_name:
                cond = torch.zeros(1, cond_dim, device=device)
                out  = model(x, cond, targets=y)
            else:
                out  = model(x, targets=y)
        except Exception:
            out  = model(x, targets=y)
        if out.loss is not None:
            total_loss += out.loss.item()
            n_batches  += 1
    if n_batches == 0:
        return float("nan")
    return math.exp(min(total_loss / n_batches, 20))


# ── Token-level generation ─────────────────────────────────────────────────────

@torch.no_grad()
def generate(model, prompt_ids: List[int], max_new: int, device: torch.device,
             temperature: float = 0.8, top_k: int = 50, cond_dim: int = 8) -> List[int]:
    ids = list(prompt_ids[-128:])  # cap context
    generated = []
    arch_name = model.__class__.__name__.lower()

    for _ in range(max_new):
        x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        try:
            if "prefix" in arch_name:
                cond = torch.zeros(1, cond_dim, device=device)
                out  = model(x, cond)
            else:
                out  = model(x)
        except Exception:
            out  = model(x)
        logits = out.logits[0, -1, :]  # [vocab]
        if temperature > 0:
            logits = logits / temperature
        if top_k > 0:
            topk_vals, _ = torch.topk(logits, top_k)
            logits[logits < topk_vals[-1]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        ids.append(next_id)
        generated.append(next_id)
    return generated


# ── BLEU-1/2 on val windows ───────────────────────────────────────────────────

@torch.no_grad()
def compute_bleu(model, token_ids: List[int], seq_len: int, device: torch.device,
                 cond_dim: int = 8, n_windows: int = 256,
                 prompt_frac: float = 0.5) -> Tuple[float, float]:
    """Slide windows over val tokens: use first half as prompt, generate second half,
    compare to actual second half. Returns (bleu1, bleu2) as percentages."""
    if not _SACREBLEU_OK:
        return float("nan"), float("nan")

    half = max(4, seq_len // 2)
    stride = max(1, (len(token_ids) - seq_len) // n_windows)
    hypotheses: List[str] = []
    references: List[str] = []
    arch_name = model.__class__.__name__.lower()

    enc = get_tokenizer()
    positions = range(0, len(token_ids) - seq_len, stride)

    for start in list(positions)[:n_windows]:
        prompt_ids = token_ids[start: start + half]
        ref_ids    = token_ids[start + half: start + half * 2]
        if not ref_ids:
            continue
        try:
            gen_ids = generate(model, prompt_ids, len(ref_ids), device,
                               temperature=0.0, top_k=1, cond_dim=cond_dim)
            hyp = enc.decode(gen_ids)
            ref = enc.decode(ref_ids)
            hypotheses.append(hyp)
            references.append(ref)
        except Exception:
            continue

    if not hypotheses:
        return float("nan"), float("nan")

    try:
        b1 = _sacrebleu.corpus_bleu(hypotheses, [references], max_ngram_order=1).score
        b2 = _sacrebleu.corpus_bleu(hypotheses, [references], max_ngram_order=2).score
    except Exception:
        return float("nan"), float("nan")
    return round(b1, 2), round(b2, 2)


# ── Diversity metrics ──────────────────────────────────────────────────────────

def distinct_n(token_ids: List[int], n: int) -> float:
    ngrams = list(zip(*[token_ids[i:] for i in range(n)]))
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


# ── Per-arch evaluation ────────────────────────────────────────────────────────

def eval_arch(arch: str, slm_dir: Path, val_tokens: List[int],
              device: torch.device, gen_len: int, seq_len: int) -> Dict[str, Any]:
    run_dir = find_best_run(arch, slm_dir)
    if run_dir is None:
        return {"arch": arch, "error": "no complete run found"}

    ckpt_path = run_dir / "best_model.pt"
    if not ckpt_path.exists():
        return {"arch": arch, "error": "best_model.pt missing"}

    summary_path = run_dir / "run_summary.json"
    with open(summary_path) as f:
        summary = json.load(f)

    best_meta = summary.get("best", {})
    n_epochs  = len(summary.get("epochs", []))
    params_m  = summary.get("model_params", 0) / 1e6

    try:
        model, _, _ = load_model(ckpt_path, device)
    except Exception as e:
        return {"arch": arch, "error": f"load failed: {e}",
                "run_id": run_dir.name}

    # Test PPL on val tokens
    test_ppl = compute_ppl(model, val_tokens, seq_len, device)

    # BLEU-1/2 (greedy generation on val windows)
    bleu1, bleu2 = compute_bleu(model, val_tokens, seq_len, device)

    # Generation for diversity + samples
    prompt = val_tokens[:32]
    try:
        gen_ids = generate(model, prompt, gen_len, device)
        d1 = distinct_n(gen_ids, 1)
        d2 = distinct_n(gen_ids, 2)
    except Exception as e:
        gen_ids, d1, d2 = [], 0.0, 0.0

    return {
        "arch":           arch,
        "arch_label":     ARCH_LABELS[arch],
        "run_id":         run_dir.name,
        "params_m":       round(params_m, 1),
        "n_epochs":       n_epochs,
        "train_best_ppl": round(best_meta.get("val_ppl", 0), 2),
        "test_ppl":       round(test_ppl, 2),
        "bleu_1":         bleu1,
        "bleu_2":         bleu2,
        "distinct_1":     round(d1, 4),
        "distinct_2":     round(d2, 4),
        "gen_ids":        gen_ids,
    }


# ── Pretty print ───────────────────────────────────────────────────────────────

def _ppl_color(ppl: float) -> str:
    if ppl < 50:  return GREEN
    if ppl < 150: return YELLOW
    return RED

def _div_color(d: float) -> str:
    if d > 0.7: return GREEN
    if d > 0.4: return YELLOW
    return RED


def print_table(results: List[Dict]) -> None:
    print(f"\n{BOLD}{CYAN}{'━'*90}{RESET}")
    print(f"{BOLD}{CYAN}  Small-LM Evaluation Summary{RESET}")
    print(f"{BOLD}{CYAN}{'━'*90}{RESET}")

    header = (f"  {'Arch':<14} {'Params(M)':<11} {'Epochs':<8} "
              f"{'Train PPL':<12} {'Test PPL':<12} {'BLEU-1':<9} {'BLEU-2':<9} "
              f"{'Dist-1':<9} {'Dist-2':<9}")
    print(f"\n{BOLD}{header}{RESET}")
    print("  " + "─" * 98)

    for r in results:
        if "error" in r:
            print(f"  {r['arch']:<14} {RED}ERROR: {r['error']}{RESET}")
            continue
        tp   = r["train_best_ppl"]
        ep   = r["test_ppl"]
        b1   = r.get("bleu_1", float("nan"))
        b2   = r.get("bleu_2", float("nan"))
        d1   = r["distinct_1"]
        d2   = r["distinct_2"]
        tc   = _ppl_color(tp)
        ec   = _ppl_color(ep)
        d1c  = _div_color(d1)
        d2c  = _div_color(d2)
        b1s  = f"{b1:.1f}" if not math.isnan(b1) else "N/A"
        b2s  = f"{b2:.1f}" if not math.isnan(b2) else "N/A"
        print(
            f"  {r['arch_label']:<14} {r['params_m']:<11} {r['n_epochs']:<8} "
            f"{tc}{tp:<12.2f}{RESET} {ec}{ep:<12.2f}{RESET} "
            f"{b1s:<9} {b2s:<9} "
            f"{d1c}{d1:<9.4f}{RESET} {d2c}{d2:<9.4f}{RESET}"
        )

    print(f"\n{BOLD}  Legend:{RESET}")
    print(f"    {GREEN}■{RESET} PPL < 50 (good)    {YELLOW}■{RESET} PPL 50–150 (fair)    {RED}■{RESET} PPL > 150 (bad)")
    print(f"    {GREEN}■{RESET} Distinct > 0.7     {YELLOW}■{RESET} 0.4–0.7              {RED}■{RESET} < 0.4 (repetitive)")
    print(f"\n  {DIM}Test PPL   = fresh perplexity on val set with best checkpoint{RESET}")
    print(f"  {DIM}BLEU-1/2   = corpus BLEU (greedy decoding, 256 val windows) — higher is better{RESET}")
    print(f"  {DIM}Distinct   = fraction of unique n-grams in 200 generated tokens{RESET}\n")


def print_samples(results: List[Dict], enc) -> None:
    print(f"\n{BOLD}{CYAN}{'━'*90}{RESET}")
    print(f"{BOLD}{CYAN}  Generated Samples (top-k=50, temp=0.8){RESET}")
    print(f"{BOLD}{CYAN}{'━'*90}{RESET}\n")
    for r in results:
        if "error" in r or not r.get("gen_ids"):
            continue
        try:
            text = enc.decode(r["gen_ids"])
        except Exception:
            text = "[decode error]"
        print(f"  {BOLD}{r['arch_label']}{RESET}:")
        print(f"  {DIM}{text[:300]}{RESET}")
        print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", type=Path,
                   default=ROOT / "artifacts" / "small_lm")
    p.add_argument("--val-text",  type=Path,
                   default=ROOT / "data" / "dialogue" / "val.txt")
    p.add_argument("--seq-len",   type=int, default=256)
    p.add_argument("--gen-len",   type=int, default=200)
    p.add_argument("--out-csv",   type=Path, default=None)
    p.add_argument("--no-samples",action="store_true",
                   help="Skip printing generated text samples")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")

    enc = get_tokenizer()

    # Load val tokens once
    if not args.val_text.exists():
        print(f"{RED}  ERROR: val text not found at {args.val_text}{RESET}")
        sys.exit(1)
    val_raw = args.val_text.read_text(encoding="utf-8")
    val_tokens = enc.encode(val_raw)
    print(f"  Val tokens: {len(val_tokens):,}")

    results = []
    for arch in ARCHS:
        print(f"  Evaluating {ARCH_LABELS[arch]}...", flush=True)
        try:
            r = eval_arch(arch, args.artifacts, val_tokens, device, args.gen_len, args.seq_len)
        except Exception as exc:
            r = {"arch": arch, "arch_label": ARCH_LABELS.get(arch, arch),
                 "error": f"EXCEPTION: {exc}"}
            # Reset CUDA state after a device-side assert
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        results.append(r)
        if "error" not in r:
            _b1 = r.get('bleu_1', float('nan'))
            _b2 = r.get('bleu_2', float('nan'))
            b1s = f"{_b1:.1f}" if not math.isnan(_b1) else "N/A"
            b2s = f"{_b2:.1f}" if not math.isnan(_b2) else "N/A"
            print(f"    train_ppl={r['train_best_ppl']:.2f}  "
                  f"test_ppl={r['test_ppl']:.2f}  "
                  f"bleu1={b1s}  bleu2={b2s}  "
                  f"distinct-1={r['distinct_1']:.4f}  distinct-2={r['distinct_2']:.4f}")
        else:
            print(f"    {RED}{r['error']}{RESET}")

    print_table(results)

    if not args.no_samples:
        print_samples(results, enc)

    # CSV export
    if args.out_csv:
        fields = ["arch", "arch_label", "run_id", "params_m", "n_epochs",
                  "train_best_ppl", "test_ppl", "bleu_1", "bleu_2",
                  "distinct_1", "distinct_2"]
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in results:
                if "error" not in r:
                    w.writerow(r)
        print(f"  Results saved → {args.out_csv}")

    # ── MLflow logging ──────────────────────────────────────────────────────
    _log_to_mlflow(results, args.out_csv)


def _log_to_mlflow(results: List[Dict], csv_path: Optional[Path]) -> None:
    """Log eval results to MLflow for experiment tracking."""
    try:
        sys.path.insert(0, str(ROOT / "src" / "train"))
        from mlflow_tracker import MLflowTracker
        tracker = MLflowTracker(experiment="slm_eval")
        tracker.start_run(run_name=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        for r in results:
            if "error" in r:
                continue
            arch = r.get("arch", "unknown")
            for key in ("train_best_ppl", "test_ppl", "bleu_1", "bleu_2",
                        "distinct_1", "distinct_2", "params_m"):
                if key in r and r[key] is not None:
                    tracker.log_metric(f"{arch}/{key}", float(r[key]))
        if csv_path and csv_path.exists():
            tracker.log_artifact(csv_path)
        tracker.end_run()
    except Exception:
        pass  # MLflow is optional for eval


if __name__ == "__main__":
    from datetime import datetime
    main()
