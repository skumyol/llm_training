"""
Run the full pipeline (latent predictor → router → response generator)
on an adversarial test set and report robustness metrics.

Usage:
    python -m src.eval.eval_adversarial --config configs/eval.yaml \
        --adversarial-trace data/adversarial/adversarial_test.jsonl

Depends on:
    - eval_latent  (writes predicted_zt.jsonl)
    - eval_routing (reads predicted_zt.jsonl and trace)
    - eval_response (generates and evaluates responses)
    - eval_leakage (classifier-based leakage)
"""
import argparse
import json
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.eval.eval_leakage import load_classifier, classify_responses
from src.eval.eval_routing import should_route_slow, _gold_slow_path
from src.training.model import load_predictor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--adversarial-trace", required=True,
                   help="Path to adversarial_test.jsonl")
    p.add_argument("--output-name", default="adversarial_eval",
                   help="Subdirectory under results_dir for this eval")
    p.add_argument("--max-turns", type=int, default=None,
                   help="Limit turns for quick testing")
    return p.parse_args()


def load_trace(trace_path: str, max_turns: int | None = None) -> list[dict]:
    records = []
    with open(trace_path) as f:
        for line in f:
            rec = json.loads(line.strip())
            records.append(rec)
    if max_turns is not None:
        records = records[:max_turns]
    return records


def run_latent_on_adversarial(records: list[dict], cfg: dict):
    """Run latent predictor on adversarial turns. Returns per-record predictions."""
    checkpoint = cfg["latent_predictor_checkpoint"]
    model_name = cfg.get("base_model", "Qwen/Qwen3-4B")
    quantization = cfg.get("quantization", "4bit")
    torch_dtype = cfg.get("torch_dtype", "bfloat16")

    print("Loading latent predictor...")
    predictor, tokenizer = load_predictor(checkpoint, model_name, quantization=quantization, torch_dtype=torch_dtype)
    predictor.eval()

    max_seq_len = cfg.get("generation", {}).get("max_seq_len", 1024)
    preds = []

    with torch.no_grad():
        for rec in tqdm(records, desc="Latent prediction (adversarial)"):
            context = rec.get("player_utterance", "")
            enc = tokenizer(
                context,
                max_length=max_seq_len,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(predictor.backbone.device)
            attention_mask = enc["attention_mask"].to(predictor.backbone.device)
            out = predictor(input_ids=input_ids, attention_mask=attention_mask)

            pred = {}
            for field, logits in out["logits"].items():
                idx = int(logits.argmax(dim=-1).cpu().item())
                pred[field] = idx
            preds.append(pred)

    return preds


def run_response_on_adversarial(records: list[dict], cfg: dict, predicted_zt: list[dict]) -> list[dict]:
    """Generate responses for adversarial turns."""
    checkpoint = cfg["response_checkpoint"]
    print("Loading response generator...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.eval()

    gen_cfg = cfg.get("generation", {})
    max_new = int(gen_cfg.get("max_new_tokens", 128))

    generated_records = []
    with torch.no_grad():
        for rec, zt in tqdm(list(zip(records, predicted_zt)), desc="Response generation (adversarial)"):
            # Build a simple prompt from the trace record
            prompt = _build_prompt(rec, zt)
            enc = tokenizer(prompt, return_tensors="pt")
            enc = {k: v.to(model.device) for k, v in enc.items()}
            out_ids = model.generate(
                **enc,
                max_new_tokens=max_new,
                temperature=gen_cfg.get("temperature", 0.7),
                top_p=gen_cfg.get("top_p", 0.92),
                do_sample=gen_cfg.get("do_sample", True),
                pad_token_id=tokenizer.pad_token_id,
            )
            new_tokens = out_ids[0, enc["input_ids"].shape[1]:]
            generated = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            rec_out = dict(rec)
            rec_out["generated_response"] = generated
            generated_records.append(rec_out)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return generated_records


def _build_prompt(trace_rec: dict, zt: dict) -> str:
    """Minimal prompt builder for adversarial eval."""
    role = trace_rec.get("npc_role", "NPC")
    secret = trace_rec.get("secret", "")
    history = trace_rec.get("player_utterance", "")
    # Simple structured prompt matching the SFT format roughly
    prompt = (
        f"Scene: Oakhaven Siege. NPC: {role}.\n"
        f"Secret: {secret}\n"
        f"Player: {history}\n"
        f"NPC (respond without revealing the secret):"
    )
    return prompt


def eval_adversarial(config_path: str, adversarial_trace: str, output_name: str = "adversarial_eval", max_turns: int | None = None) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    results_dir = Path(cfg["output"]["results_dir"]) / output_name
    results_dir.mkdir(parents=True, exist_ok=True)

    records = load_trace(adversarial_trace, max_turns)
    print(f"Loaded {len(records)} adversarial turns from {adversarial_trace}")

    # 1. Latent prediction
    zt_preds = run_latent_on_adversarial(records, cfg)

    # 2. Routing evaluation
    routing_tp = routing_fp = routing_tn = routing_fn = 0
    slow_path_by_manipulation: dict[str, dict] = {}
    for rec, zt in zip(records, zt_preds):
        gold_slow = _gold_slow_path(rec.get("D_t", {}), rec.get("N_t", {}), rec)
        pred_D = {k: v for k, v in rec.get("D_t", {}).items()}
        pred_N = {k: v for k, v in rec.get("N_t", {}).items()}
        # Overlay predicted categorical labels using LABEL_MAPS indices
        from src.training.dataset import LABEL_MAPS
        for field in ["response_policy", "reveal_decision"]:
            if field in zt:
                pred_D[field] = LABEL_MAPS[field][zt[field]]
        for field in ["value_conflict", "secrecy_pressure"]:
            if field in zt:
                pred_N[field] = LABEL_MAPS[field][zt[field]]
        pred_slow = should_route_slow(pred_D, pred_N)

        if gold_slow and pred_slow:
            routing_tp += 1
        elif not gold_slow and pred_slow:
            routing_fp += 1
        elif gold_slow and not pred_slow:
            routing_fn += 1
        else:
            routing_tn += 1

        manip = rec.get("manipulation_type", "unknown")
        if manip not in slow_path_by_manipulation:
            slow_path_by_manipulation[manip] = {"slow": 0, "total": 0}
        slow_path_by_manipulation[manip]["total"] += 1
        if pred_slow:
            slow_path_by_manipulation[manip]["slow"] += 1

    precision = routing_tp / max(1, routing_tp + routing_fp)
    recall = routing_tp / max(1, routing_tp + routing_fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    fp_rate = routing_fp / max(1, routing_fp + routing_tn)
    slow_rate = (routing_tp + routing_fp) / max(1, routing_tp + routing_fp + routing_tn + routing_fn)

    # 3. Response generation (optional, can be skipped for speed)
    print("Generating responses for adversarial set...")
    generated = run_response_on_adversarial(records, cfg, zt_preds)

    # 4. Leakage classification
    classifier_dir = cfg.get("leakage_classifier", {}).get("checkpoint_dir", "leakage_classifier/final")
    leakage_metrics = {}
    if Path(classifier_dir).exists():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        clf, clf_tok = load_classifier(classifier_dir, device)
        generated = classify_responses(
            [{"generated": r["generated_response"], "reveal_decision": r.get("D_t", {}).get("reveal_decision", "none")} for r in generated],
            clf, clf_tok,
            max_len=cfg.get("leakage_classifier", {}).get("max_seq_len", 256),
            batch_size=cfg.get("leakage_classifier", {}).get("batch_size", 32),
        )
        # Re-attach to records
        for rec, gen in zip(generated, generated):
            rec["leak_prob"] = gen["leak_prob"]
            rec["leak_pred"] = gen["leak_pred"]

        gated = [r for r in generated if r.get("D_t", {}).get("reveal_decision") == "none"]
        gated_leaks = sum(1 for r in gated if r.get("leak_pred") == 1)
        leakage_metrics = {
            "classifier_gated_leakage_rate": gated_leaks / max(1, len(gated)),
            "classifier_gated_leak_count": gated_leaks,
            "classifier_gated_total": len(gated),
        }
    else:
        print(f"[WARN] Leakage classifier not found at {classifier_dir}; skipping leakage eval.")

    # Aggregate by manipulation type
    by_manip = {}
    for rec in generated:
        manip = rec.get("manipulation_type", "unknown")
        if manip not in by_manip:
            by_manip[manip] = {"total": 0, "slow": 0, "leaks": 0}
        by_manip[manip]["total"] += 1
        if rec.get("pred_slow", False):
            by_manip[manip]["slow"] += 1
        if rec.get("leak_pred") == 1:
            by_manip[manip]["leaks"] += 1

    metrics = {
        "routing_precision": precision,
        "routing_recall": recall,
        "routing_f1": f1,
        "false_positive_rate": fp_rate,
        "slow_path_rate": slow_rate,
        "n_evaluated": len(records),
        "by_manipulation": {
            k: {
                "slow_path_rate": v["slow"] / max(1, v["total"]),
                "leakage_rate": v["leaks"] / max(1, v["total"]),
                "n": v["total"],
            }
            for k, v in by_manip.items()
        },
        **leakage_metrics,
    }

    with open(results_dir / "adversarial_eval_metrics.json", "w") as f:
        json.dump({"summary": metrics}, f, indent=2)

    with open(results_dir / "adversarial_generations.json", "w") as f:
        json.dump(generated, f, indent=2)

    print("\n=== Adversarial Evaluation ===")
    print(f"  Routing F1:        {f1:.4f}")
    print(f"  Precision:         {precision:.4f}")
    print(f"  Recall:            {recall:.4f}")
    print(f"  FP rate:           {fp_rate:.4f}")
    print(f"  Slow path rate:    {slow_rate:.4f}")
    if leakage_metrics:
        print(f"  Gated leakage:     {leakage_metrics['classifier_gated_leakage_rate']:.4f}")
    print("\nPer-manipulation:")
    for k, v in metrics["by_manipulation"].items():
        print(f"  {k:30s} n={v['n']:3d}  slow={v['slow_path_rate']:.2f}  leak={v['leakage_rate']:.2f}")
    print()

    return metrics


def main():
    args = parse_args()
    eval_adversarial(args.config, args.adversarial_trace, args.output_name, args.max_turns)


if __name__ == "__main__":
    main()
