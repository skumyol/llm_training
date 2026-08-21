#!/usr/bin/env python3
"""
Controlled head ablation experiment.

Trains ablated latent predictors with subsets of heads and evaluates routing
performance to produce the ablation curve.

Usage:
    # Experiment A: routing-only minimal state (4 heads)
    PYTHONPATH=. python scripts/run_head_ablation.py \
        --config configs/eval.yaml \
        --heads response_policy reveal_decision value_conflict secrecy_pressure \
        --name exp_a_routing_only \
        --train

    # Experiment B: +affect
    PYTHONPATH=. python scripts/run_head_ablation.py \
        --config configs/eval.yaml \
        --heads response_policy reveal_decision value_conflict secrecy_pressure valence threat control \
        --name exp_b_plus_affect \
        --train

    # Experiment C: +relational
    PYTHONPATH=. python scripts/run_head_ablation.py \
        --config configs/eval.yaml \
        --heads response_policy reveal_decision value_conflict secrecy_pressure trust_level respect_level \
        --name exp_c_plus_relational \
        --train

    # Evaluate a trained ablation without retraining
    PYTHONPATH=. python scripts/run_head_ablation.py \
        --config configs/eval.yaml \
        --name exp_a_routing_only
"""
import argparse
import json
import shutil
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.model import build_latent_predictor, save_predictor
from src.training.dataset import HeadSupervisionDataset, collate_head_batch, LABEL_MAPS
from src.eval.eval_routing import should_route_slow, _gold_slow_path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--heads", nargs="+", required=True,
                   help="Subset of heads to keep (space-separated)")
    p.add_argument("--name", required=True, help="Experiment name / output subdir")
    p.add_argument("--train", action="store_true", help="Train ablated predictor from scratch")
    p.add_argument("--masking-mode", action="store_true",
                   help="Load full model but evaluate with only the specified heads (masking ablation)")
    p.add_argument("--baseline-checkpoint", default=None,
                   help="Full model checkpoint to load for masking ablation (defaults to config latent_predictor_checkpoint)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--train-heads-file", default=None,
                   help="Override train heads file from config")
    p.add_argument("--test-heads-file", default=None,
                   help="Override test heads file from config")
    p.add_argument("--test-trace-file", default=None,
                   help="Override test trace file from config")
    return p.parse_args()


def _filter_head_specs(full_specs: dict, keep_heads: set) -> dict:
    return {k: v for k, v in full_specs.items() if k in keep_heads}


def _train_ablated(cfg: dict, keep_heads: set, output_dir: str, epochs: int, batch_size: int, lr: float):
    from src.training.model import HEAD_SPECS, build_latent_predictor

    ablated_specs = _filter_head_specs(HEAD_SPECS, keep_heads)
    print(f"Training ablated predictor with {len(ablated_specs)} heads: {sorted(ablated_specs.keys())}")

    model_name = cfg.get("base_model", "Qwen/Qwen3-4B")
    quantization = cfg.get("quantization", "4bit")
    torch_dtype = cfg.get("torch_dtype", "bfloat16")

    predictor, tokenizer = build_latent_predictor(
        model_name,
        quantization=quantization,
        lora_config=cfg.get("lora", None),
        torch_dtype=torch_dtype,
    )
    # Replace heads with ablated set
    predictor.head_specs = ablated_specs
    import torch.nn as nn
    from src.training.model import ClassificationHead
    # Rebuild heads
    hidden_size = predictor.hidden_size
    predictor.heads = nn.ModuleDict({
        name: ClassificationHead(hidden_size, spec["n_classes"])
        for name, spec in ablated_specs.items()
    })
    predictor.to(predictor.backbone.device)

    # Defaulting to test_heads_file (as this did) trains the ablated heads on the
    # very split they are then scored on. eval.yaml has no train_heads_file key and
    # no caller in scripts/experiments.sh or scripts/slurm_experiments.sh passes
    # --train-heads-file, so every ablation row produced before this was train-on-eval.
    train_file = args.train_heads_file or cfg["data"].get("train_heads_file")
    if not train_file:
        raise ValueError(
            "No training split for the ablation. Pass --train-heads-file "
            "(e.g. data/splits/train_heads.jsonl) or set data.train_heads_file in the config. "
            "Refusing to fall back to the evaluation split."
        )
    eval_file = cfg["data"].get("test_heads_file")
    if eval_file and Path(train_file).resolve() == Path(eval_file).resolve():
        raise ValueError(
            f"Ablation train file == eval file ({train_file}); results would be train-on-eval."
        )
    train_ds = HeadSupervisionDataset(
        train_file,
        tokenizer,
        max_seq_len=cfg.get("generation", {}).get("max_seq_len", 1024),
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_head_batch, num_workers=0,
    )

    optimizer = torch.optim.AdamW(predictor.parameters(), lr=lr)

    for epoch in range(epochs):
        predictor.train()
        total_loss = 0.0
        n_batches = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = batch["input_ids"].to(predictor.backbone.device)
            attention_mask = batch["attention_mask"].to(predictor.backbone.device)
            out = predictor(input_ids=input_ids, attention_mask=attention_mask)

            losses = []
            for field in ablated_specs:
                label_key = f"label_{field}"
                if label_key not in batch:
                    continue
                gold = batch[label_key].to(predictor.backbone.device)
                logits = out["logits"][field]
                if field == "dialogue_act":
                    valid = gold.sum(dim=1) > 0
                    if not valid.any():
                        continue
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        logits[valid], gold[valid].float()
                    )
                else:
                    valid = gold != -1
                    if not valid.any():
                        continue
                    loss = torch.nn.functional.cross_entropy(
                        logits[valid], gold[valid]
                    )
                losses.append(loss)

            if not losses:
                continue
            loss = sum(losses) / len(losses)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    save_predictor(predictor, output_dir)
    return predictor, tokenizer


def _eval_routing_on_ablated(cfg: dict, keep_heads: set, checkpoint_dir: str, test_trace: str) -> dict:
    from src.training.model import HEAD_SPECS, load_predictor

    ablated_specs = _filter_head_specs(HEAD_SPECS, keep_heads)
    model_name = cfg.get("base_model", "Qwen/Qwen3-4B")
    quantization = cfg.get("quantization", "4bit")
    torch_dtype = cfg.get("torch_dtype", "bfloat16")

    print(f"Loading ablated predictor from {checkpoint_dir}")
    predictor, tokenizer = load_predictor(checkpoint_dir, model_name, quantization=quantization, torch_dtype=torch_dtype)

    # Overwrite head_specs to match what was trained
    predictor.head_specs = ablated_specs

    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    total = 0

    idx_to_label = {field: {i: name for i, name in enumerate(LABEL_MAPS[field])}
                      for field in ablated_specs}

    with open(test_trace) as f:
        for line in tqdm(f, desc="Evaluating routing (ablated)"):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            D_t = record.get("D_t", {})
            N_t = record.get("N_t", {})
            gold_slow = _gold_slow_path(D_t, N_t, record)

            # Encode minimal context for prediction
            context = record.get("player_utterance", "")
            if not context:
                # Fallback: use a dummy context if player utterance missing
                context = "Player speaks to NPC."
            enc = tokenizer(
                context,
                max_length=cfg.get("generation", {}).get("max_seq_len", 1024),
                truncation=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(predictor.backbone.device)
            attention_mask = enc["attention_mask"].to(predictor.backbone.device)

            with torch.no_grad():
                out = predictor(input_ids=input_ids, attention_mask=attention_mask)

            pred_D = {**D_t}
            pred_N = {**N_t}
            for field in ablated_specs:
                if field in out["logits"]:
                    idx = int(out["logits"][field][0].argmax(dim=-1).cpu().item())
                    label = idx_to_label[field].get(idx, "")
                    if field in {"response_policy", "reveal_decision"}:
                        pred_D[field] = label
                    elif field in {"value_conflict", "secrecy_pressure"}:
                        pred_N[field] = label

            pred_slow = should_route_slow(pred_D, pred_N)

            if gold_slow and pred_slow:
                true_positives += 1
            elif not gold_slow and pred_slow:
                false_positives += 1
            elif gold_slow and not pred_slow:
                false_negatives += 1
            else:
                true_negatives += 1
            total += 1

    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    fp_rate = false_positives / max(1, false_positives + true_negatives)
    fnr = false_negatives / max(1, false_negatives + true_positives)
    unsafe_fp = false_negatives / max(1, total)
    slow_rate = (true_positives + false_positives) / max(1, total)
    cost_fn5 = (5 * false_negatives + false_positives) / max(1, total)
    cost_fn10 = (10 * false_negatives + false_positives) / max(1, total)

    return {
        "routing_precision": precision,
        "routing_recall": recall,
        "routing_f1": f1,
        "false_positive_rate": fp_rate,
        "false_negative_rate": fnr,
        "unsafe_fast_path_rate": unsafe_fp,
        "slow_path_precision": precision,
        "slow_path_recall": recall,
        "slow_path_rate": slow_rate,
        "routing_cost_fn5": cost_fn5,
        "routing_cost_fn10": cost_fn10,
        "n_evaluated": total,
        "heads_used": sorted(keep_heads),
        "n_heads": len(keep_heads),
    }


def main():
    global args
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    keep_heads = set(args.heads)
    results_dir = Path(cfg["output"]["results_dir"]) / "ablation" / args.name
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = str(results_dir / "checkpoint")

    if args.train:
        predictor, tokenizer = _train_ablated(
            cfg, keep_heads, checkpoint_dir,
            args.epochs, args.batch_size, args.lr
        )

    # Determine which checkpoint to evaluate from
    if args.masking_mode:
        # Masking ablation: load full baseline model, evaluate with subset of heads
        baseline_ckpt = args.baseline_checkpoint or cfg.get("latent_predictor_checkpoint", checkpoint_dir)
        eval_checkpoint = baseline_ckpt
        print(f"[Masking mode] Loading full model from {eval_checkpoint}")
        print(f"              Evaluating with heads: {sorted(keep_heads)}")
    else:
        eval_checkpoint = checkpoint_dir

    test_trace = args.test_trace_file or cfg["data"].get("test_trace_file", "")
    if not Path(test_trace).exists():
        print(f"[WARN] Test trace not found: {test_trace}; skipping routing eval")
        routing_metrics = {"skipped": True, "reason": "test_trace_not_found"}
    else:
        routing_metrics = _eval_routing_on_ablated(cfg, keep_heads, eval_checkpoint, test_trace)
        if args.masking_mode:
            routing_metrics["ablation_type"] = "masking"
        elif args.train:
            routing_metrics["ablation_type"] = "retraining"
        else:
            routing_metrics["ablation_type"] = "evaluation"

    with open(results_dir / "ablation_metrics.json", "w") as f:
        json.dump(routing_metrics, f, indent=2)

    print(f"\n=== Ablation: {args.name} ===")
    print(f"  Type: {routing_metrics.get('ablation_type', 'evaluation')}")
    print(f"  Heads: {sorted(keep_heads)}")
    if not routing_metrics.get("skipped"):
        print(f"  Routing F1:       {routing_metrics['routing_f1']:.4f}")
        print(f"  Precision:        {routing_metrics['routing_precision']:.4f}")
        print(f"  Recall:           {routing_metrics['routing_recall']:.4f}")
        print(f"  FP rate:          {routing_metrics['false_positive_rate']:.4f}")
        print(f"  Slow path rate:   {routing_metrics['slow_path_rate']:.4f}")
    print(f"  Results dir:      {results_dir}\n")


if __name__ == "__main__":
    main()
