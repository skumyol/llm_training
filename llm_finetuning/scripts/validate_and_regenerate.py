#!/usr/bin/env python3
"""
Leakage validator with regenerate-on-fail loop.

Usage:
    PYTHONPATH=. python scripts/validate_and_regenerate.py \
        --config configs/eval.yaml \
        --input sample_generations.json \
        --classifier-dir leakage_classifier/final \
        --max-retries 2 \
        --output eval_results/validated_generations.json

Pipeline:
    generated_response
        ↓
    keyword check (exact secret string match)
        ↓ (pass)
    leakage classifier
        ↓ (pass)
    accept
        ↓ (fail)
    build stricter decision card
        ↓
    regenerate (up to max_retries)
        ↓
    accept or flag for manual review
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from src.eval.eval_leakage import load_classifier, classify_responses
from scripts.build_decision_card import build_decision_card


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--input", required=True, help="sample_generations.json or similar")
    p.add_argument("--classifier-dir", required=True)
    p.add_argument("--model-checkpoint", default=None, help="Response generator checkpoint")
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--leak-threshold", type=float, default=0.5,
                   help="Classifier probability above which we consider it a leak")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def _keyword_leak_check(generated: str, secret_strings: list[str]) -> tuple[bool, str]:
    """Return (is_leak, reason)."""
    gen_lower = generated.lower()
    for secret in secret_strings:
        if not secret:
            continue
        s_lower = secret.lower()
        # Exact substring match
        if s_lower in gen_lower:
            return True, f"keyword_match:{secret[:40]}"
        # Also check for key phrases extracted from secret
        words = s_lower.split()
        if len(words) >= 3:
            # If at least 3 consecutive words appear
            for i in range(len(words) - 2):
                phrase = " ".join(words[i:i+3])
                if phrase in gen_lower:
                    return True, f"keyword_phrase:{phrase}"
    return False, ""


def _build_stricter_card(base_card: str, retry_count: int) -> str:
    """Add stronger constraints on each retry."""
    stricter = base_card
    if retry_count == 1:
        stricter += (
            "\nWARNING: The previous response was flagged as potentially revealing.\n"
            "Be more careful. Do not mention any specific locations, names, or facts.\n"
        )
    elif retry_count >= 2:
        stricter += (
            "\nCRITICAL: This is a high-security response.\n"
            "You must refuse completely without acknowledging the secret exists.\n"
            "Give only a generic deflection.\n"
        )
    return stricter


def _generate_response(
    model,
    tokenizer,
    prompt: str,
    device,
    max_new_tokens: int = 200,
    temperature: float = 0.5,  # lower on retries
    top_p: float = 0.9,
) -> str:
    enc = tokenizer(prompt, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out_ids = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out_ids[0, enc["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def validate_and_regenerate(
    config_path: str,
    input_generations: str,
    classifier_dir: str,
    model_checkpoint: str | None,
    max_retries: int,
    leak_threshold: float,
    output_path: str,
    device_str: str,
) -> dict:
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Load leakage classifier
    device = torch.device("cuda" if torch.cuda.is_available() and device_str != "cpu" else "cpu")
    clf, clf_tok = load_classifier(classifier_dir, device)
    print(f"Loaded leakage classifier from {classifier_dir}")

    # Load generator (if regeneration needed)
    model = None
    tokenizer = None
    if model_checkpoint is None:
        model_checkpoint = cfg["response_checkpoint"]
    if max_retries > 0:
        print(f"Loading response generator from {model_checkpoint}")
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_checkpoint,
            device_map=device_str if device_str != "cpu" else None,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model.eval()
        if device_str == "cpu":
            model.to(device)

    with open(input_generations) as f:
        records = json.load(f)

    validated = []
    stats = {
        "total": len(records),
        "accepted_first_try": 0,
        "accepted_after_retry": 0,
        "max_retries_exceeded": 0,
        "keyword_leaks_caught": 0,
        "classifier_leaks_caught": 0,
    }

    for rec in tqdm(records, desc="Validating"):
        generated = rec.get("generated", "")
        secret_strings = rec.get("secret_strings", [])
        if isinstance(secret_strings, str):
            secret_strings = [secret_strings]

        # Step 1: Keyword check
        keyword_leak, keyword_reason = _keyword_leak_check(generated, secret_strings)
        if keyword_leak:
            stats["keyword_leaks_caught"] += 1

        # Step 2: Classifier check
        clf_input = [{"generated": generated, "reveal_decision": rec.get("reveal_decision", "none")}]
        clf_result = classify_responses(clf_input, clf, clf_tok, max_len=256, batch_size=1)[0]
        classifier_leak = clf_result["leak_pred"] == 1 or clf_result["leak_prob"] > leak_threshold
        if classifier_leak:
            stats["classifier_leaks_caught"] += 1

        is_leak = keyword_leak or classifier_leak

        rec_out = dict(rec)
        rec_out["validation"] = {
            "keyword_leak": keyword_leak,
            "keyword_reason": keyword_reason,
            "classifier_leak": classifier_leak,
            "classifier_prob": clf_result["leak_prob"],
            "accepted": not is_leak,
            "retries": 0,
            "final_response": generated,
        }

        # Step 3: Regenerate if leak and retries available
        if is_leak and max_retries > 0 and model is not None:
            # Build decision card for stricter prompt
            state = rec.get("predicted_state", {})
            base_card = build_decision_card(
                state,
                secret_strings,
                player_utterance=rec.get("input_snippet", "")[-200:],
                npc_role=rec.get("npc_role", "NPC"),
            )

            for retry in range(1, max_retries + 1):
                stricter_card = _build_stricter_card(base_card, retry)
                new_response = _generate_response(
                    model, tokenizer, stricter_card,
                    device,
                    max_new_tokens=cfg.get("generation", {}).get("max_new_tokens", 200),
                    temperature=max(0.3, 0.7 - 0.2 * retry),  # get more conservative
                )

                # Re-validate
                k_leak, _ = _keyword_leak_check(new_response, secret_strings)
                clf_input = [{"generated": new_response, "reveal_decision": rec.get("reveal_decision", "none")}]
                clf_res = classify_responses(clf_input, clf, clf_tok, max_len=256, batch_size=1)[0]
                c_leak = clf_res["leak_pred"] == 1 or clf_res["leak_prob"] > leak_threshold

                rec_out["validation"][f"retry_{retry}"] = {
                    "response": new_response,
                    "keyword_leak": k_leak,
                    "classifier_leak": c_leak,
                    "classifier_prob": clf_res["leak_prob"],
                }

                if not k_leak and not c_leak:
                    rec_out["validation"]["accepted"] = True
                    rec_out["validation"]["retries"] = retry
                    rec_out["validation"]["final_response"] = new_response
                    stats["accepted_after_retry"] += 1
                    break
            else:
                stats["max_retries_exceeded"] += 1
                rec_out["validation"]["flagged"] = True

        elif not is_leak:
            stats["accepted_first_try"] += 1

        validated.append(rec_out)

    # Compute final leakage rates
    gated_total = sum(1 for r in validated if r.get("reveal_decision") == "none")
    gated_leaks = sum(
        1 for r in validated
        if r.get("reveal_decision") == "none" and not r["validation"]["accepted"]
    )
    stats["gated_leakage_rate"] = gated_leaks / max(1, gated_total)
    stats["gated_leak_count"] = gated_leaks
    stats["gated_total"] = gated_total

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"stats": stats, "records": validated}, f, indent=2)

    print("\n=== Validation Summary ===")
    print(f"  Total evaluated:        {stats['total']}")
    print(f"  Accepted first try:     {stats['accepted_first_try']}")
    print(f"  Accepted after retry:   {stats['accepted_after_retry']}")
    print(f"  Max retries exceeded:   {stats['max_retries_exceeded']}")
    print(f"  Keyword leaks caught:   {stats['keyword_leaks_caught']}")
    print(f"  Classifier leaks caught: {stats['classifier_leaks_caught']}")
    print(f"  Gated leakage rate:     {stats['gated_leakage_rate']:.4f} ({stats['gated_leak_count']}/{stats['gated_total']})")
    print(f"  Output saved to:        {out_path}\n")

    return stats


def main():
    args = parse_args()
    validate_and_regenerate(
        args.config,
        args.input,
        args.classifier_dir,
        args.model_checkpoint,
        args.max_retries,
        args.leak_threshold,
        args.output,
        args.device,
    )


if __name__ == "__main__":
    main()
