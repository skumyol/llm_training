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
    p.add_argument("--epochs", type=int, default=None,
                   help="Override epochs; defaults to the training config's value")
    p.add_argument("--train-config", default="llm_finetuning/configs/train_latent.yaml",
                   help="Training recipe (lora/lr/epochs/loss) to reuse, so the ablation "
                        "measures the head set rather than a weaker bespoke recipe")
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


def _train_ablated(cfg: dict, keep_heads: set, output_dir: str, epochs: int, batch_size: int, lr: float,
                   train_cfg_path: str | None = None):
    from src.training.model import HEAD_SPECS, build_latent_predictor

    ablated_specs = _filter_head_specs(HEAD_SPECS, keep_heads)
    print(f"Training ablated predictor with {len(ablated_specs)} heads: {sorted(ablated_specs.keys())}")

    model_name = cfg.get("base_model", "Qwen/Qwen3-4B")
    quantization = cfg.get("quantization", "4bit")
    torch_dtype = cfg.get("torch_dtype", "bfloat16")

    # The ablation must use the SAME recipe as the main training run, or it measures
    # the recipe rather than the head set. Pulling lora/lr/epochs/loss settings from
    # the training config fixes four divergences, the first of which was fatal:
    #   1. cfg here is the EVAL config, which has no `lora:` key, so lora_config was
    #      None -> no adapter was ever created and only linear heads trained on
    #      frozen 4-bit features.
    #   2. one lr of 2e-5 for everything vs progressive 2e-4 backbone / 4e-4 heads.
    #   3. plain cross_entropy vs class weights + focal(1.5) + label smoothing(0.1),
    #      which is what keeps imbalanced heads off the majority class.
    #   4. 3 epochs vs 5.
    train_cfg: dict = {}
    if train_cfg_path and Path(train_cfg_path).exists():
        with open(train_cfg_path) as f:
            train_cfg = yaml.safe_load(f) or {}
        print(f"Using training recipe from {train_cfg_path}")

    lora_cfg = train_cfg.get("lora") or cfg.get("lora")
    if lora_cfg is None:
        raise ValueError(
            "No `lora:` section available. The evaluation config has none, so without "
            "--train-config the backbone would be frozen and only the heads would train, "
            "which is what made every previous ablation degenerate."
        )
    tcfg = train_cfg.get("training", {})

    predictor, tokenizer = build_latent_predictor(
        model_name,
        quantization=quantization,
        lora_config=lora_cfg,
        torch_dtype=torch_dtype,
        pooling=train_cfg.get("pooling", cfg.get("pooling", "last")),
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
    # Use the TRAINING context length, not the eval config's. They differ (512 vs
    # 1024), and training the ablation at a different sequence length from the main
    # run is another way to measure the recipe instead of the head set.
    ablation_seq_len = int(tcfg.get("max_seq_len", cfg.get("generation", {}).get("max_seq_len", 1024)))
    print(f"Ablation max_seq_len={ablation_seq_len} batch_size={batch_size}")
    # The packaged train split is 59% counterfactual variants that share a
    # byte-identical context with their original while carrying different labels;
    # training on those forces every affected head to the majority class, which is
    # what the routing heads did in all three previous ablation attempts.
    exclude_cf = train_cfg.get("data", {}).get("exclude_counterfactual", False)
    if exclude_cf:
        print("[ablation] counterfactual records EXCLUDED")
    train_ds = HeadSupervisionDataset(
        train_file,
        tokenizer,
        max_seq_len=ablation_seq_len,
        exclude_counterfactual=exclude_cf,
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_head_batch, num_workers=0,
    )

    # Same loss as the main run: inverse-frequency class weights + focal + label
    # smoothing. Plain cross-entropy let the imbalanced heads collapse to the
    # majority class, which is what produced the always-careful router.
    from src.training.dataset import LABEL_MAPS
    from src.training.loss import MultiHeadLoss, compute_class_weights

    class_weights = compute_class_weights(train_file, LABEL_MAPS,
                                          exclude_counterfactual=exclude_cf)
    loss_fn = MultiHeadLoss(
        train_cfg.get("loss_weights", cfg.get("loss_weights", {})),
        class_weights=class_weights,
        label_smoothing=float(tcfg.get("label_smoothing", 0.0)),
        focal_gamma=float(tcfg.get("focal_gamma", 0.0)),
    )

    # Progressive LR, as in the main run: heads learn faster than the adapter.
    base_lr = float(tcfg.get("lr", lr))
    head_lr = float(tcfg.get("head_lr", base_lr * 2))
    weight_decay = float(tcfg.get("weight_decay", 0.01))
    backbone_params = [p for _, p in predictor.backbone.named_parameters() if p.requires_grad]
    if not backbone_params:
        raise ValueError("No trainable backbone parameters — the LoRA adapter was not attached.")
    optimizer = torch.optim.AdamW(
        [
            {"params": predictor.heads.parameters(), "lr": head_lr, "weight_decay": weight_decay},
            {"params": backbone_params, "lr": base_lr, "weight_decay": weight_decay},
        ],
        lr=base_lr,
    )
    grad_accum = int(tcfg.get("grad_accum", 1))
    max_grad_norm = float(tcfg.get("max_grad_norm", 1.0))
    print(f"Ablation recipe: lr={base_lr} head_lr={head_lr} epochs={epochs} "
          f"grad_accum={grad_accum} focal={tcfg.get('focal_gamma', 0.0)} "
          f"label_smoothing={tcfg.get('label_smoothing', 0.0)} "
          f"trainable_backbone_tensors={len(backbone_params)}")

    for epoch in range(epochs):
        predictor.train()
        total_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            batch = {k: (v.to(predictor.backbone.device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            out = predictor(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])

            # MultiHeadLoss skips any field absent from `logits`, so passing the
            # ablated logit dict restricts the loss to the kept heads automatically.
            loss, _ = loss_fn(out["logits"], batch)
            (loss / grad_accum).backward()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

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
        _tc = {}
        if args.train_config and Path(args.train_config).exists():
            with open(args.train_config) as _f:
                _tc = yaml.safe_load(_f) or {}
        _epochs = args.epochs if args.epochs is not None else int(_tc.get("training", {}).get("epochs", 5))
        predictor, tokenizer = _train_ablated(
            cfg, keep_heads, checkpoint_dir,
            _epochs, args.batch_size, args.lr,
            train_cfg_path=args.train_config,
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
