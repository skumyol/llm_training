#!/usr/bin/env python3
"""
Train a binary leakage classifier on generated NPC responses.

Usage:
    python scripts/train_leakage_classifier.py \
        --heads-file data/splits/train_heads.jsonl \
        --sft-file data/splits/train_sft.jsonl \
        --output-dir leakage_classifier \
        --hard-negatives 0.3

The label rule is:
    leak = 1 if reveal_decision in {partial, full}
    leak = 0 if reveal_decision == none

Hard negatives are synthetic: for a subset of no-leak responses, we keep the
text but ensure it contains secret-related keywords without actually disclosing
the protected fact.  This prevents the classifier from learning
"secret-keyword = leak".
"""
import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


SECRECY_KEYWORDS = [
    "chalice", "goblet", "vault location", "artifact", "relic", "crown",
    "scepter", "gem", "jewel", "treasure", "hoard",
    "patrol schedule", "patrol route", "guard rotation", "watch change",
    "passphrase", "password", "weak point", "defense", "tunnel",
    "ledger", "accounts", "smuggling", "contraband", "shipment",
    "cargo", "black market", "counterfeit", "bribe", "embezzlement",
    "affair", "scandal", "illegitimate", "bastard", "succession",
    "inheritance", "will", "testament", "alliance", "betrayal",
    "supply theft", "stolen", "thief", "heist", "cache", "stash",
    "heresy", "blasphemy", "sacrilege", "prophecy", "omen",
    "poison", "toxin", "venom", "assassin", "murder", "conspiracy",
    "corruption", "graft", "extortion", "blackmail",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--heads-file", required=True, help="Head supervision JSONL with labels")
    p.add_argument("--sft-file", required=True, help="SFT JSONL with input/target text")
    p.add_argument("--output-dir", default="leakage_classifier")
    p.add_argument("--model-name", default="distilbert-base-uncased")
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--train-epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--hard-negatives", type=float, default=0.3,
                   help="Fraction of no_leak examples to perturb with secret keywords")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.15)
    return p.parse_args()


def load_data(heads_path: str, sft_path: str) -> list[dict]:
    """Merge head labels with SFT targets by (episode_id, turn_idx)."""
    sft_by_key: dict[tuple[str, int], dict] = {}
    with open(sft_path) as f:
        for line in f:
            rec = json.loads(line.strip())
            ep = str(rec.get("episode_id", ""))
            turn = rec.get("turn_idx", rec.get("turn", None))
            if ep and turn is not None:
                sft_by_key[(ep, int(turn))] = rec

    records = []
    with open(heads_path) as f:
        for line in f:
            rec = json.loads(line.strip())
            ep = str(rec.get("episode_id", ""))
            turn = rec.get("turn_idx", rec.get("turn", None))
            if ep is None or turn is None:
                continue
            sft = sft_by_key.get((ep, int(turn)))
            if not sft:
                continue
            labels = rec.get("labels", {})
            reveal = labels.get("reveal_decision", "")
            if reveal in {"partial", "full"}:
                label = 1
            elif reveal == "none":
                label = 0
            else:
                continue
            records.append({
                "episode_id": ep,
                "turn_idx": int(turn),
                "response": sft.get("target", "").strip(),
                "reveal_decision": reveal,
                "label": label,
            })
    return records


def _make_hard_negative(text: str, rng: random.Random) -> str:
    """Inject a secret keyword into a non-leaking response without changing semantics."""
    # Simple strategy: prepend a clause that mentions a secret topic but denies knowledge
    prefixes = [
        "I cannot speak of the {kw}, but ",
        "The {kw} is not my concern; ",
        "Rumors of the {kw} matter little to me. ",
        "I know nothing of the {kw}. ",
    ]
    kw = rng.choice(SECRECY_KEYWORDS)
    prefix = rng.choice(prefixes).format(kw=kw)
    return prefix + text


def build_dataset(records: list[dict], hard_negative_fraction: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for rec in records:
        if rec["label"] == 0 and rng.random() < hard_negative_fraction:
            rec = dict(rec)
            rec["response"] = _make_hard_negative(rec["response"], rng)
            rec["hard_negative"] = True
        out.append(rec)
    return out


def tokenize_batch(examples, tokenizer, max_seq_len: int):
    return tokenizer(
        examples["response"],
        truncation=True,
        padding="max_length",
        max_length=max_seq_len,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    probs = torch.softmax(torch.from_numpy(logits), dim=-1)[:, 1].numpy()

    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = 0.0

    # Precision at 90% recall
    precisions, recalls, _ = precision_recall_curve(labels, probs)
    p_at_90 = 0.0
    for p, r in zip(precisions, recalls):
        if r >= 0.90:
            p_at_90 = max(p_at_90, p)

    return {
        "accuracy": acc,
        "f1": f1,
        "auc": auc,
        "precision_at_90_recall": p_at_90,
    }


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading data from {args.heads_file} + {args.sft_file}")
    records = load_data(args.heads_file, args.sft_file)
    print(f"  Loaded {len(records)} records")

    leak_count = sum(1 for r in records if r["label"] == 1)
    no_leak_count = sum(1 for r in records if r["label"] == 0)
    print(f"  leak={leak_count}, no_leak={no_leak_count}")

    records = build_dataset(records, args.hard_negatives, args.seed)
    print(f"  After hard negatives: {len(records)} records")

    # Split train/val/test
    train_val, test = train_test_split(
        records, test_size=args.test_size, random_state=args.seed, stratify=[r["label"] for r in records]
    )
    train, val = train_test_split(
        train_val, test_size=0.15, random_state=args.seed, stratify=[r["label"] for r in train_val]
    )
    print(f"  Train={len(train)}, Val={len(val)}, Test={len(test)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def _make_hf(split_records):
        ds = Dataset.from_list([
            {"response": r["response"], "labels": r["label"]} for r in split_records
        ])
        ds = ds.map(
            lambda ex: tokenize_batch(ex, tokenizer, args.max_seq_len),
            batched=True,
            remove_columns=["response"],
        )
        ds.set_format("torch")
        return ds

    train_ds = _make_hf(train)
    val_ds = _make_hf(val)
    test_ds = _make_hf(test)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        num_train_epochs=args.train_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_strategy="epoch",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print("\n=== Test-set evaluation ===")
    test_results = trainer.evaluate(test_ds)
    for k, v in test_results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save
    out_dir = Path(args.output_dir) / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"Saved classifier to {out_dir}")

    # Write label mapping
    (out_dir / "label_map.json").write_text(json.dumps({"no_leak": 0, "leak": 1}, indent=2))
    print("Done.")


if __name__ == "__main__":
    main()
