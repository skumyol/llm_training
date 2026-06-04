#!/usr/bin/env python3
"""
Evaluate multi-turn relational memory vs single-turn baseline.

Compares the RelationalMemoryPredictor (with GRU memory) against the base
LatentStatePredictor on relational-head prediction and routing decisions.

Usage:
    PYTHONPATH=. python scripts/eval_relational_memory.py \
        --config configs/eval.yaml \
        --checkpoint checkpoints/latent_predictor.pt \
        --test-heads data/splits/test_heads.jsonl \
        --output eval_results/relational_memory_eval.json

Requires:
    - src.training.relational_memory.RelationalMemoryPredictor
    - Trained latent predictor checkpoint
"""
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test-heads", required=True)
    p.add_argument("--output", default="eval_results/relational_memory_eval.json")
    p.add_argument("--memory-size", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=8)
    return p.parse_args()


def load_model_and_tokenizer(checkpoint: str, config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    model_name = cfg.get("base_model", "Qwen/Qwen3-4B")
    quantization = cfg.get("quantization", "4bit")
    torch_dtype = cfg.get("torch_dtype", "bfloat16")

    from src.training.model import load_predictor
    predictor, tokenizer = load_predictor(checkpoint, model_name, quantization=quantization, torch_dtype=torch_dtype)
    return predictor, tokenizer, cfg


def eval_baseline(predictor, tokenizer, test_path: str, batch_size: int) -> dict:
    """Evaluate base predictor (no memory) on test set."""
    from src.training.dataset import HeadSupervisionDataset, collate_head_batch
    from torch.utils.data import DataLoader

    ds = HeadSupervisionDataset(test_path, tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_head_batch)

    relational_fields = [
        "trust_level", "respect_level", "affection_level",
        "familiarity_level", "dominance_level", "obligation_level",
    ]

    correct: Dict[str, List[int]] = {f: [] for f in relational_fields}
    all_preds: Dict[str, List[str]] = {f: [] for f in relational_fields}
    all_golds: Dict[str, List[str]] = {f: [] for f in relational_fields}

    predictor.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Baseline eval"):
            input_ids = batch["input_ids"].to(predictor.backbone.device)
            attention_mask = batch["attention_mask"].to(predictor.backbone.device)
            out = predictor(input_ids=input_ids, attention_mask=attention_mask)

            for field in relational_fields:
                label_key = f"label_{field}"
                if label_key not in batch:
                    continue
                gold = batch[label_key]
                if not isinstance(gold, torch.Tensor):
                    continue
                gold = gold.to(predictor.backbone.device)
                valid = gold != -1
                if not valid.any():
                    continue

                logits = out["logits"][field]
                preds = logits.argmax(dim=-1)

                gold_valid = gold[valid]
                pred_valid = preds[valid]

                from src.training.dataset import LABEL_TO_IDX
                for g, p in zip(gold_valid.cpu().tolist(), pred_valid.cpu().tolist()):
                    idx_to_label = {v: k for k, v in LABEL_TO_IDX.get(field, {}).items()}
                    all_golds[field].append(idx_to_label.get(g, str(g)))
                    all_preds[field].append(idx_to_label.get(p, str(p)))
                    correct[field].append(int(g == p))

    metrics = {}
    for field in relational_fields:
        if correct[field]:
            metrics[field] = {
                "accuracy": round(sum(correct[field]) / len(correct[field]), 4),
                "n": len(correct[field]),
            }

    # Overall relational accuracy
    all_correct = sum(sum(correct[f]) for f in relational_fields if correct[f])
    all_total = sum(len(correct[f]) for f in relational_fields if correct[f])
    metrics["relational_macro_acc"] = round(all_correct / max(1, all_total), 4)
    return metrics


def eval_with_memory(predictor, tokenizer, test_path: str, batch_size: int, memory_size: int) -> dict:
    """Evaluate with RelationalMemoryPredictor wrapper."""
    from src.training.relational_memory import RelationalMemoryPredictor
    from src.training.dataset import HeadSupervisionDataset, collate_head_batch
    from torch.utils.data import DataLoader

    device = next(predictor.parameters()).device
    wrapped = RelationalMemoryPredictor(predictor, memory_size=memory_size)
    wrapped.to(device)
    wrapped.eval()

    ds = HeadSupervisionDataset(test_path, tokenizer)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_head_batch)

    relational_fields = [
        "trust_level", "respect_level", "affection_level",
        "familiarity_level", "dominance_level", "obligation_level",
    ]

    correct: Dict[str, List[int]] = {f: [] for f in relational_fields}

    # Group by episode to preserve memory state
    episode_batches: Dict[str, List[Tuple[dict, int]]] = {}
    for idx in range(len(ds)):
        rec = ds.records[idx]
        ep = str(rec.get("episode_id", f"ep_{idx}"))
        turn = rec.get("turn_idx", idx)
        episode_batches.setdefault(ep, []).append((rec, idx))

    # Sort each episode by turn_idx
    for ep in episode_batches:
        episode_batches[ep].sort(key=lambda x: x[0].get("turn_idx", x[1]))

    with torch.no_grad():
        for ep, items in tqdm(episode_batches.items(), desc="Memory eval (episodes)"):
            memory_state = None
            for rec, ds_idx in items:
                batch = collate_head_batch([ds[ds_idx]])
                input_ids = batch["input_ids"].to(predictor.backbone.device)
                attention_mask = batch["attention_mask"].to(predictor.backbone.device)

                out = wrapped(input_ids=input_ids, attention_mask=attention_mask, memory_state=memory_state)
                memory_state = out.get("memory_state")

                for field in relational_fields:
                    label_key = f"label_{field}"
                    if label_key not in batch:
                        continue
                    gold = batch[label_key]
                    if not isinstance(gold, torch.Tensor):
                        continue
                    gold = gold.to(predictor.backbone.device)
                    valid = gold != -1
                    if not valid.any():
                        continue

                    logits = out["logits"][field]
                    preds = logits.argmax(dim=-1)
                    gold_valid = gold[valid]
                    pred_valid = preds[valid]

                    for g, p in zip(gold_valid.cpu().tolist(), pred_valid.cpu().tolist()):
                        correct[field].append(int(g == p))

    metrics = {}
    for field in relational_fields:
        if correct[field]:
            metrics[field] = {
                "accuracy": round(sum(correct[field]) / len(correct[field]), 4),
                "n": len(correct[field]),
            }

    all_correct = sum(sum(correct[f]) for f in relational_fields if correct[f])
    all_total = sum(len(correct[f]) for f in relational_fields if correct[f])
    metrics["relational_macro_acc"] = round(all_correct / max(1, all_total), 4)
    return metrics


def main():
    args = parse_args()

    predictor, tokenizer, cfg = load_model_and_tokenizer(args.checkpoint, args.config)

    print("Evaluating baseline (single-turn, no memory)...")
    baseline = eval_baseline(predictor, tokenizer, args.test_heads, args.batch_size)

    print("Evaluating with relational memory...")
    with_memory = eval_with_memory(predictor, tokenizer, args.test_heads, args.batch_size, args.memory_size)

    # Comparison
    comparison = {}
    for field in baseline:
        if field in with_memory and isinstance(baseline[field], dict):
            b_acc = baseline[field]["accuracy"]
            m_acc = with_memory[field]["accuracy"]
            comparison[field] = {
                "baseline_acc": b_acc,
                "memory_acc": m_acc,
                "delta": round(m_acc - b_acc, 4),
                "n": baseline[field].get("n", 0),
            }

    report = {
        "baseline": baseline,
        "with_memory": with_memory,
        "comparison": comparison,
        "overall_delta": round(
            with_memory.get("relational_macro_acc", 0) - baseline.get("relational_macro_acc", 0), 4
        ),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n=== Relational Memory Evaluation ===")
    print(f"  Baseline relational acc:  {baseline.get('relational_macro_acc', 0):.4f}")
    print(f"  Memory relational acc:    {with_memory.get('relational_macro_acc', 0):.4f}")
    print(f"  Delta:                    {report['overall_delta']:+.4f}")
    for field, comp in comparison.items():
        if field == "relational_macro_acc":
            continue
        print(f"  {field:25s}  base={comp['baseline_acc']:.3f}  mem={comp['memory_acc']:.3f}  Δ={comp['delta']:+.3f}")
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()
