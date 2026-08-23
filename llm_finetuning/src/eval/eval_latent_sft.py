"""Score the generative latent predictor against the head-based one.

Generates the state block greedily, parses it, and routes the result through the
same `compute_latent_metrics` the head model uses, so macro-F1 and accuracy mean
the same thing in both tables.

Unparseable or missing fields are scored as errors (an index no gold vector
contains) rather than skipped — dropping them would quietly reward a model that
declines to answer. `parse_rate` and `field_coverage` are reported alongside.

    python -m src.eval.eval_latent_sft --config configs/eval_S1_genstate.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from tqdm import tqdm

from src.metrics_report import compute_latent_metrics
from src.training.dataset import LABEL_MAPS, LABEL_TO_IDX
from src.training.latent_sft import (BEGIN, END, FIELD_ORDER, LatentSFTDataset,
                                     parse, to_indices)
from src.training.model import load_backbone


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))

    model, tokenizer, _ = load_backbone(
        cfg.get("base_model", "Qwen/Qwen3-4B"),
        quantization=cfg.get("quantization", "4bit"),
        lora_config=None,
        torch_dtype=cfg.get("torch_dtype", "bfloat16"),
    )
    model = PeftModel.from_pretrained(model, cfg["adapter"])
    model.eval()

    ds = LatentSFTDataset(cfg["data"]["test_heads_file"], tokenizer,
                          cfg.get("max_seq_len", 1024),
                          cfg["data"].get("exclude_counterfactual", False))
    records = ds.records[: a.limit] if a.limit else ds.records
    print(f"scoring {len(records)} records from {cfg['data']['test_heads_file']}")

    device = next(model.parameters()).device
    preds: dict[str, list] = {}
    golds: dict[str, list] = {}
    n_parsed, field_hits, raw_samples = 0, 0, []

    for r in tqdm(records, desc="generating"):
        prompt = ds.prompt_for(r)
        ids = tokenizer(prompt, return_tensors="pt", truncation=True,
                        max_length=cfg.get("max_seq_len", 1024)).to(device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=cfg.get("max_new_tokens", 320),
                                 do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        text = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        got = parse(BEGIN + "\n" + text)
        if got:
            n_parsed += 1
        field_hits += len(got)
        if len(raw_samples) < 3:
            raw_samples.append(text[:400])

        idx = to_indices(got)
        for f in FIELD_ORDER:
            if f == "dialogue_act":
                continue
            gv = r["labels"].get(f)
            if isinstance(gv, list):
                gv = gv[0] if gv else None
            g = LABEL_TO_IDX[f].get(str(gv), -1)
            if g == -1:
                continue
            preds.setdefault(f, []).append(idx[f])
            golds.setdefault(f, []).append(g)

    n_fields = len(FIELD_ORDER) - 1
    metrics = compute_latent_metrics(preds, golds, extra_summary={
        "parse_rate": n_parsed / max(len(records), 1),
        "field_coverage": field_hits / max(len(records) * len(FIELD_ORDER), 1),
        "n_records": len(records),
    })
    s = metrics.get("summary", {})
    print(f"\nparse_rate      {s.get('parse_rate', 0):.4f}")
    print(f"field_coverage  {s.get('field_coverage', 0):.4f}")
    print(f"mean_macro_f1   {s.get('mean_macro_f1', 0):.4f}")
    print(f"mean_accuracy   {s.get('mean_accuracy', 0):.4f}")
    print(f"response_policy {metrics.get('fields', {}).get('response_policy', {}).get('macro_f1', 0):.4f}")

    out_dir = Path(cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "latent_eval_metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "samples.txt").write_text("\n\n---\n\n".join(raw_samples))
    print(f"wrote {out_dir}/latent_eval_metrics.json")


if __name__ == "__main__":
    main()
