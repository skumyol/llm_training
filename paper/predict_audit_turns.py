#!/usr/bin/env python3
"""
predict_audit_turns.py — emit predictor labels for the human-annotated audit turns.

Closes the third leg of the agreement triangle. The study so far reports
human-teacher, human-human and AI-validator-teacher agreement, but never
predictor-human, because `eval_latent.py` writes `predicted_zt.jsonl` with only
the four routing fields and five of the eight annotated fields were therefore
unavailable.

This writes ALL 29 head predictions for the audit turns, so any annotated field
can be compared.

The audit turns are joined to the packaged split records by
(episode_id, turn_number) -> (episode_id, turn_idx), and the predictor is run on
the packaged `context` string, i.e. exactly the input formatting used in training
and evaluation. Building an input from the raw audit fields instead would risk
measuring a formatting difference rather than the predictor.

Usage:
    python paper/predict_audit_turns.py \
        --config llm_finetuning/configs/eval_test.yaml \
        --audit paper/audit_input_clean.jsonl \
        --heads-file data/splits/test_heads.jsonl \
        --out paper/audit_results/audit_predictor.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.dataset import HeadSupervisionDataset, collate_head_batch, LABEL_MAPS
from src.training.model import load_predictor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="llm_finetuning/configs/eval_test.yaml")
    ap.add_argument("--audit", default="paper/audit_input_clean.jsonl")
    ap.add_argument("--heads-file", default=None, help="defaults to config data.test_heads_file")
    ap.add_argument("--out", default="paper/audit_results/audit_predictor.jsonl")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    heads_file = args.heads_file or cfg["data"]["test_heads_file"]

    audit = [json.loads(l) for l in open(args.audit) if l.strip()]
    wanted = {(str(a["episode_id"]), int(a["turn_number"])): a["turn_id"] for a in audit}
    print(f"audit turns requested: {len(wanted)}")

    records = [json.loads(l) for l in open(heads_file) if l.strip()]

    # The packaged splits carry up to 4 records per turn: one original plus
    # counterfactual variants that share an IDENTICAL context but carry different
    # labels. Keeping all of them would emit several rows per turn_id, all with
    # the same prediction (the input is identical), and collide on join.
    # Keep exactly one record per turn, preferring the non-counterfactual original.
    chosen: dict[tuple[str, int], int] = {}
    for i, r in enumerate(records):
        key = (str(r.get("episode_id")), int(r.get("turn_idx", r.get("turn", -1))))
        if key not in wanted:
            continue
        prev = chosen.get(key)
        if prev is None or (records[prev].get("counterfactual") and not r.get("counterfactual")):
            chosen[key] = i

    keep_idx = [chosen[k] for k in sorted(chosen)]
    keep_meta = [{"episode_id": k[0], "turn_idx": k[1], "turn_id": wanted[k]} for k in sorted(chosen)]
    n_cf = sum(1 for i in keep_idx if records[i].get("counterfactual"))

    print(f"matched {len(keep_idx)} unique turns / {len(wanted)} requested "
          f"(from {len(records)} records in {heads_file})")
    if n_cf:
        print(f"  note: {n_cf} turns had no non-counterfactual record available")
    if not keep_idx:
        raise SystemExit("No audit turns matched the split; check the join keys.")
    missing = len(wanted) - len(keep_idx)
    if missing > 0:
        print(f"  WARNING: {missing} audit turns not present in this split "
              f"(they may belong to a different split)")

    predictor, tokenizer = load_predictor(
        cfg["latent_predictor_checkpoint"],
        cfg.get("base_model", "Qwen/Qwen3-4B"),
        quantization=cfg.get("quantization", "4bit"),
        torch_dtype=cfg.get("torch_dtype", "bfloat16"),
    )
    predictor.eval()
    print(f"pooling: {predictor.pooling}")

    ds = HeadSupervisionDataset(heads_file, tokenizer,
                                max_seq_len=cfg.get("generation", {}).get("max_seq_len", 1024))
    subset = torch.utils.data.Subset(ds, keep_idx)
    loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_head_batch, num_workers=0)

    idx_to_label = {f: {i: n for i, n in enumerate(v)} for f, v in LABEL_MAPS.items()}
    out_rows: list[dict] = []
    device = predictor.backbone.device

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting audit turns"):
            out = predictor(input_ids=batch["input_ids"].to(device),
                            attention_mask=batch["attention_mask"].to(device))
            bsz = batch["input_ids"].size(0)
            base = len(out_rows)
            for i in range(bsz):
                meta = keep_meta[base + i]
                row = {**meta, "labels": {}}
                for field, logits in out["logits"].items():
                    if field == "dialogue_act":
                        # multi-label: emit every class above threshold
                        on = (logits[i].sigmoid() >= 0.5).nonzero().flatten().tolist()
                        row["labels"][field] = [idx_to_label[field][j] for j in on]
                    else:
                        j = int(logits[i].argmax(-1).item())
                        row["labels"][field] = idx_to_label.get(field, {}).get(j, "")
                out_rows.append(row)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(out_rows)} predictor rows -> {args.out}")
    print(f"fields per row: {len(out_rows[0]['labels'])}")


if __name__ == "__main__":
    main()
