import json
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.training.dataset import SFTDataset, collate_sft_batch
from src.metrics_report import log_metrics_to_mlflow, write_metrics_bundle


SECRECY_KEYWORDS = [
    "chalice", "vault location", "patrol schedule", "ledger",
    "affair", "supply theft", "succession", "smuggling", "contraband",
    "heresy", "poison", "bribe", "corruption",
]


def eval_response(config_path: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    import mlflow
    from src.mlflow_utils import setup_mlflow

    setup_mlflow(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    checkpoint = cfg["response_checkpoint"]
    gen_cfg = cfg.get("generation", {})

    print(f"Loading response model from {checkpoint}")
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

    test_ds = SFTDataset(
        cfg["data"]["test_sft_file"],
        tokenizer,
        max_seq_len=cfg.get("generation", {}).get("max_seq_len", 2048),
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        collate_fn=collate_sft_batch, num_workers=0,
    )

    results_dir = Path(cfg["output"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    n_samples = cfg["output"].get("n_sample_generations", 100)
    generations: list[dict] = []
    secret_leakage = 0
    total_secret_turns = 0
    contradiction_flags = 0

    rouge_l_scores: list[float] = []
    generated_texts: list[str] = []
    generated_lengths: list[int] = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Generating responses")):
            input_ids      = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)

            # SFTDataset now returns prompt_len which separates context from target
            if "prompt_len" in batch:
                prompt_len = batch["prompt_len"][0].item()
                # Use only the prompt part for generation input
                generation_input_ids = input_ids[:, :prompt_len]
                attention_mask = attention_mask[:, :prompt_len]
            else:
                # Fallback if prompt_len is missing (though it shouldn't be)
                generation_input_ids = input_ids
            
            input_text = tokenizer.decode(generation_input_ids[0], skip_special_tokens=True)
            target_ids  = batch["labels"][0]
            target_mask = target_ids != -100
            gold_text   = tokenizer.decode(target_ids[target_mask], skip_special_tokens=True).strip()

            output_ids = model.generate(
                input_ids=generation_input_ids,
                attention_mask=attention_mask,
                max_new_tokens=gen_cfg.get("max_new_tokens", 200),
                temperature=gen_cfg.get("temperature", 0.7),
                top_p=gen_cfg.get("top_p", 0.9),
                do_sample=gen_cfg.get("do_sample", True),
                pad_token_id=tokenizer.pad_token_id,
            )

            new_tokens = output_ids[0, generation_input_ids.shape[1]:]
            generated = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            rouge_l = _rouge_l(gold_text, generated)
            rouge_l_scores.append(rouge_l)
            generated_texts.append(generated)
            generated_lengths.append(len(generated.split()))

            leaks = _check_secret_leakage(input_text, generated)
            if "reveal_decision=none" in input_text and leaks:
                secret_leakage += 1
            if "reveal_decision" in input_text:
                total_secret_turns += 1

            if _check_contradiction(input_text, generated):
                contradiction_flags += 1

            if len(generations) < n_samples:
                generations.append({
                    "input_snippet": input_text[-300:],
                    "gold": gold_text,
                    "generated": generated,
                    "rouge_l": rouge_l,
                    "secret_leak": leaks,
                })

    metrics = {
        "rouge_l": sum(rouge_l_scores) / max(1, len(rouge_l_scores)),
        "secret_leakage_rate": secret_leakage / max(1, total_secret_turns),
        "contradiction_rate": contradiction_flags / max(1, len(rouge_l_scores)),
        "distinct_1": _distinct_n(generated_texts, 1),
        "distinct_2": _distinct_n(generated_texts, 2),
        "avg_len": sum(generated_lengths) / max(1, len(generated_lengths)),
        "n_evaluated": len(rouge_l_scores),
    }

    bundle = {"summary": metrics}

    with open(results_dir / "response_eval_metrics.json", "w") as f:
        json.dump(bundle, f, indent=2)
    write_metrics_bundle(results_dir, "response_eval_report", bundle, title="Response Evaluation Report")

    if cfg["output"].get("save_sample_generations", True):
        with open(results_dir / "sample_generations.json", "w") as f:
            json.dump(generations, f, indent=2)

    with mlflow.start_run(run_name="response_eval"):
        log_metrics_to_mlflow(bundle["summary"], prefix="eval")
        mlflow.log_artifact(str(results_dir / "response_eval_metrics.json"))
        if (results_dir / "sample_generations.json").exists():
            mlflow.log_artifact(str(results_dir / "sample_generations.json"))
        mlflow.log_artifact(str(results_dir / "response_eval_report.md"))

    _print_summary(metrics, cfg["thresholds"])
    return metrics


def _rouge_l(reference: str, hypothesis: str) -> float:
    if not reference or not hypothesis:
        return 0.0
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    lcs = _lcs_length(ref_tokens, hyp_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(hyp_tokens)
    recall    = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list, b: list) -> int:
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev = curr
    return prev[n]


def _check_secret_leakage(input_text: str, generated: str) -> bool:
    generated_lower = generated.lower()
    for kw in SECRECY_KEYWORDS:
        if kw in input_text.lower() and kw in generated_lower:
            return True
    return False


def _check_contradiction(input_text: str, generated: str) -> bool:
    contradiction_patterns = [
        ("i know nothing", "i saw"),
        ("i was not there", "i watched"),
        ("there is no secret", "the secret"),
        ("i never met", "i have known"),
    ]
    gen_lower = generated.lower()
    for pat_a, pat_b in contradiction_patterns:
        if pat_a in input_text.lower() and pat_b in gen_lower:
            return True
    return False


def _print_summary(metrics: dict, thresholds: dict) -> None:
    print("\n=== Response Generation Evaluation Summary ===")
    print(f"  ROUGE-L:             {metrics.get('rouge_l', 0):.4f}")
    print(f"  Secret Leakage:      {metrics.get('secret_leakage_rate', 0):.4f}  (threshold ≤ {thresholds.get('secret_leakage_rate', 0.05)})")
    print(f"  Contradiction Rate:  {metrics.get('contradiction_rate', 0):.4f}  (threshold ≤ {thresholds.get('contradiction_rate', 0.08)})")
    print()


def _distinct_n(texts: list[str], n: int) -> float:
    ngrams = []
    for text in texts:
        tokens = text.lower().split()
        if len(tokens) < n:
            continue
        ngrams.extend(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)
