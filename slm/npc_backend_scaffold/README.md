# NPC Backend Scaffold

Modular training and inference stack for personality-conditioned, affect-aware NPC dialogue.
Covers the full pipeline from raw data through encoder training, personality caching, and
conditional dialogue fine-tuning — plus a from-scratch small-LM benchmark suite for A/B comparison.

---

## Architecture overview

```
                  ┌─────────────────────┐
  NPC profile ──► │ Personality Encoder │ ──► p_vec (5-dim OCEAN)  ──► cached
                  │  (DistilBERT OCEAN) │                               in FAISS
                  └─────────────────────┘

  Conversation  ► │  Affect Encoder     │ ──► a_vec (3-dim VAD)    ──► live
  context         │  (DistilBERT VAD)   │
                  └─────────────────────┘

  FAISS memory  ► │  Episodic Memory    │ ──► retrieved memories
                  │  (Sentence-BERT)    │
                  └─────────────────────┘

  p_vec + a_vec ► │  ConditionalSoft-   │ ──► prefix tokens
                  │  Prefix (MLP)       │
                  └─────────────────────┘
                           │
                           ▼
                  ┌─────────────────────┐
  Prompt ───────► │  Causal LM + LoRA   │ ──► NPC response
                  │  (TinyLlama default)│
                  └─────────────────────┘
```

The soft-prefix module prepends learned conditioning embeddings to the token stream.
Upgrade path to true per-layer KV-prefix injection is clean — only `src/models/dialogue.py` changes.

---

## Quick start (smoke test)

```bash
cd slm/npc_backend_scaffold
bash smoke_test.sh
```

Runs the full pipeline end-to-end with tiny synthetic data and lightweight models
(DistilBERT + DistilGPT-2). Verifies data loading, training, caching, and inference.

---

## Data preparation

### Option A — Public datasets (PersonaChat, CRD3, EmpathyDialogues)

```bash
# 1. Download raw data
python -m src.data.datasets \
    --datasets personachat crd3 empathetic_dialogues dailydialog

# 2. Convert to training formats
python -m src.data.prepare_dialogue_data
```

Produces:
- `data/dialogue/train.jsonl` + `val.jsonl`  → ConditionalDialogueModel
- `data/dialogue/train.txt`  + `val.txt`     → small LM architectures
- `data/affect/train.csv`    + `val.csv`     → affect encoder (VAD approximated from emotion labels)
- `data/npc_profiles.csv`                   → personality cache builder

### Option B — Upper-level generated data (recommended for domain quality)

The upper-level repo generates structured NPC dialogue via `run_data_gen.py` using
scenario templates (`data/scenario_bank/`) and world contexts (`data/world_contexts/`).
This data has **real VAD affect labels** (not approximations) and rich NPC personas.

```bash
# From repo root — generate episodes:
python run_data_gen.py --config configs/data_gen.yaml

# From scaffold — convert to training formats:
python -m src.data.convert_generated_data \
    --source-dir ../../data \
    --out-dir    data
```

Produces `from_gen_*` variants of all training files. Affect labels come directly from
`A_t.valence / A_t.arousal / A_t.control` in the validated turns.

Label mapping:
| Label value | valence | arousal / dominance |
|-------------|---------|---------------------|
| negative / low | 0.1 | 0.2 |
| neutral / medium | 0.5 | 0.5 |
| positive / high | 0.9 | 0.8 |

Update `configs/affect.yaml` to point `train_path` at `data/affect/from_gen_train.csv`
to use these higher-quality labels.

### Option C — Merge both sources

```python
import pandas as pd, json
# Affect: simply concatenate CSVs
pd.concat([
    pd.read_csv("data/affect/train.csv"),
    pd.read_csv("data/affect/from_gen_train.csv"),
]).to_csv("data/affect/merged_train.csv", index=False)

# Dialogue: cat JSONL files
import subprocess
subprocess.run("cat data/dialogue/train.jsonl data/dialogue/from_gen_train.jsonl "
               "> data/dialogue/merged_train.jsonl", shell=True)
```

---

## Training pipeline

### Automated (recommended)

```bash
./train_all.sh                        # auto hardware detection, all stages
./train_all.sh --run-id exp_01        # tag all artifacts
./train_all.sh --sequential           # force sequential (low-RAM machines)
./train_all.sh --skip-stage1          # restart from cache + dialogue only
```

**Stage ordering:**
```
Stage 1 [parallel] ─┬─ personality encoder
                    └─ affect encoder
Stage 2             ─── build personality cache
Stage 3             ─── dialogue model (LoRA + prefix)
```

Hardware auto-profile sets batch sizes based on detected VRAM (8/16/24+ GB CUDA, MPS, CPU).

### Individual scripts (ablation / debug)

```bash
# Personality encoder
python -m src.train.run_personality \
    --config configs/personality.yaml \
    --run-id ablation_roberta \
    --model-name roberta-base

# Affect encoder
python -m src.train.run_affect \
    --config configs/affect.yaml \
    --run-id ablation_lr_1e5 \
    --lr 1e-5 --epochs 5

# Personality cache
python -m src.data.build_caches \
    --profiles-path data/npc_profiles.csv \
    --encoder-dir   artifacts/personality_encoder/my_run/best_model \
    --out-path      artifacts/personality_cache.jsonl

# Dialogue model
python -m src.train.run_dialogue \
    --config configs/dialogue.yaml \
    --run-id ablation_lora_r32 \
    --lora-r 32
```

Every run produces under `artifacts/<model>/<run_id>/`:
- `run.log` — timestamped log
- `step_metrics.csv` — per-step loss / lr / grad_norm
- `epoch_metrics.csv` — val MSE / MAE / R² (encoders) or val_loss / ppl (dialogue)
- `predictions_epoch{N}.csv` — predictions vs ground truth per dimension (encoders)
- `run_summary.json` — full hyperparams + results for ablation table

---

## Small-LM A/B benchmark

Six from-scratch architectures for comparison against the fine-tuned LLM path:

| Architecture | Class | Params (m1_small) | Key feature |
|---|---|---|---|
| `gru` | `SmallGRULM` | ~8M | fast, low memory |
| `awdlstm` | `AWDLSTMLM` | ~8M | DropConnect + variational dropout |
| `gpt` | `TinyGPTLM` | ~5M | causal transformer |
| `prefix_gpt` | `PrefixTinyGPTLM` | ~5M | GPT + cond_vec prefix — same interface as ConditionalDialogueModel |
| `moe` | `TinyMoELM` | ~10M | sparse mixture-of-experts FFN |
| `mamba_like` | `MambaLikeLM` | ~5M | selective SSM, pure PyTorch |

```bash
# Train all six on the same data split
for arch in gru awdlstm gpt prefix_gpt moe mamba_like; do
  python -m src.train.run_small_lm \
    --config configs/small_lm.yaml \
    --arch $arch \
    --run-id bench_v1_$arch
done

# Compare results
python -c "
import json, glob
for f in sorted(glob.glob('artifacts/small_lm/bench_v1_*/run_summary.json')):
    s = json.load(open(f))
    print(f\"{s['arch']:12s}  {s['model_params']/1e6:.1f}M params  val_ppl={s['best']['val_ppl']:.2f}\")
"
```

The `val_ppl` metric uses tiktoken (GPT-2 BPE) and is comparable across all architectures
and to the ConditionalDialogueModel's reported perplexity.

Hardware profiles (`m1_small` / `rtx4070_small`) in `src/train/small_lm_architectures.py`
control model width and depth. Pass `--hardware-profile rtx4070_small` for larger models.

---

## Inference

```python
from src.common.config import InferenceConfig
from src.infer.service import NPCInferenceService

svc = NPCInferenceService(InferenceConfig())
svc.register_npc("commander_vance", "A stoic, exhausted commander who values duty above all.")

reply = svc.chat("commander_vance", "Have you caught the spy yet?")
print(reply)
```

Or run the interactive demo:
```bash
python src/infer/run_demo.py
```

---

## Ablation study guide

All `run_summary.json` files share a common schema for easy aggregation:

```python
import json, glob, pandas as pd

rows = []
for f in glob.glob("artifacts/**/run_summary.json", recursive=True):
    s = json.load(open(f))
    rows.append({
        "run_id":  s["run_id"],
        "model":   s.get("arch") or s.get("backbone") or s.get("model"),
        "task":    s["task"],
        "val_mse": s["best"].get("val_mse"),
        "val_ppl": s["best"].get("val_ppl"),
        "params":  s.get("model_params") or s.get("model_stats", {}).get("trainable_params"),
    })
pd.DataFrame(rows).sort_values("val_ppl").to_csv("ablation_table.csv", index=False)
```

---

## File layout

```
slm/npc_backend_scaffold/
├── configs/
│   ├── personality.yaml          encoder hyperparams
│   ├── affect.yaml               encoder hyperparams
│   ├── dialogue.yaml             dialogue model hyperparams
│   └── small_lm.yaml             small-LM benchmark hyperparams
├── src/
│   ├── common/
│   │   └── config.py             dataclasses for all configs
│   ├── data/
│   │   ├── datasets.py           dataset loaders + DataDownloader
│   │   ├── build_caches.py       personality cache builder
│   │   ├── prepare_dialogue_data.py   public dataset converter
│   │   └── convert_generated_data.py  upper-level generated data converter
│   ├── models/
│   │   ├── personality.py        DistilBertRegressor (OCEAN)
│   │   ├── affect.py             DistilBertRegressor (VAD)
│   │   └── dialogue.py           ConditionalDialogueModel (LoRA + soft-prefix)
│   ├── infer/
│   │   ├── memory_store.py       EpisodicMemoryStore (FAISS + SentenceBERT)
│   │   ├── service.py            NPCInferenceService
│   │   └── run_demo.py           interactive demo
│   └── train/
│       ├── run_personality.py    personality encoder runner (logging + ablation)
│       ├── run_affect.py         affect encoder runner
│       ├── run_dialogue.py       dialogue model runner
│       ├── run_small_lm.py       small-LM benchmark runner
│       ├── small_lm_architectures.py  GRU / AWD-LSTM / GPT / PrefixGPT / MoE / Mamba-like
│       ├── train_personality.py  (original low-level train function)
│       ├── train_affect.py       (original low-level train function)
│       └── train_dialogue.py     (original low-level train function + DialogueCollator)
├── train_all.sh                  mega orchestration script
├── smoke_test.py                 end-to-end smoke test
├── smoke_test.sh                 smoke test with venv setup
└── requirements.txt
```

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install tiktoken          # optional, for BPE tokenization in small LM benchmark
```
