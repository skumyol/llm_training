#!/usr/bin/env python3
"""
Small-LM Training Runner
=========================
Trains one from-scratch small language model for NPC dialogue.
Produces the same logging/artifact format as run_dialogue.py for direct A/B comparison.

Supported architectures:
  gru | awdlstm | gpt | prefix_gpt | moe | mamba_like

Usage:
  python -m src.train.run_small_lm --arch gpt
  python -m src.train.run_small_lm --arch prefix_gpt --config configs/small_lm.yaml
  python -m src.train.run_small_lm --arch awdlstm --run-id ablation_awdlstm_01 \\
      --train-text data/dialogue/train.txt --val-text data/dialogue/val.txt

  # Benchmark all architectures sequentially on the same data split:
  for arch in gru awdlstm gpt prefix_gpt moe mamba_like; do
    python -m src.train.run_small_lm --arch $arch --run-id bench_$(date +%s)_$arch
  done

A/B comparison note:
  val_ppl in run_summary.json is on your NPC dialogue corpus using tiktoken (GPT-2 BPE).
  This is comparable across all six architectures here, and to ConditionalDialogueModel
  (which uses the same tokenizer family). Use val_ppl as the primary ablation metric.

Artifacts under output_dir/run_id/:
  best_model.pt          checkpoint (state_dict + config)
  run.log                structured log
  step_metrics.csv       per-step train_loss, lr, grad_norm
  epoch_metrics.csv      per-epoch val_loss, val_ppl
  run_summary.json       hyperparams + results (ablation row)
  run_summary.md         human-readable summary report
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src" / "train"))  # for small_lm_architectures
sys.path.insert(0, str(ROOT))

from small_lm_architectures import (
    LMOutput,
    PrefixTinyGPTLM,
    RECOMMENDED_CONFIGS,
    build_model,
    select_device,
)
from conditioning import build_condition_vector, load_partial_state_dict
from metrics_report import log_metrics_to_mlflow, write_metrics_bundle
from src.data.small_lm_dialogue import DialogueRecordDataset, DialogueTokenizer

try:
    import tiktoken
    _TIKTOKEN_OK = True
except ImportError:
    _TIKTOKEN_OK = False

# Embedding model support for semantic conditioning (A/B testing)
try:
    from transformers import AutoTokenizer as _AutoTok, AutoModel as _AutoModel
    _TRANSFORMERS_OK = True
except ImportError:
    _TRANSFORMERS_OK = False


# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULTS: Dict[str, Any] = {
    "arch":              "gpt",
    "hardware_profile":  "m1_small",
    "train_text":        "data/dialogue/train.txt",
    "val_text":          "data/dialogue/val.txt",
    "train_jsonl":       None,
    "val_jsonl":         None,
    "tokenizer_path":    None,
    "seq_len":           256,
    "batch_size":        16,
    "grad_accum":        4,
    "lr":                3e-4,
    "weight_decay":      0.1,
    "epochs":            3,
    "log_every":         20,
    "eval_every_steps":  200,
    "seed":              42,
    "output_dir":        "artifacts/small_lm",
    "cond_dim":          8,    # OCEAN(5)+VAD(3); used by prefix_gpt only
    "condition_mode":    "ocean_vad",  # ocean_vad | social_state | zero
    "condition_dropout": 0.1,
    "profile_max_tokens": 48,
    "max_turns":         4,
    "deduplicate":       True,
    "source_weights":    {},
    "curriculum_ratio":  0.1,
    "curriculum_max_target_tokens": 48,
    "curriculum_max_turns": 2,
    "init_from":         None,
    "use_amp":           True,
    "device":            "auto",  # auto | cuda | mps | cpu
    "num_workers":       None,    # None selects a backend-safe default
    "pin_memory":        None,    # None enables it only for CUDA
    "prefetch_factor":   2,
    "persistent_workers": True,
    "allow_tf32":        True,
    "cudnn_benchmark":   True,
    "fused_optimizer":   True,
    # Embedding model for semantic conditioning (A/B testing)
    "embedding_model":   None,  # e.g., "Qwen/Qwen3-Embedding-4B" or "sentence-transformers/all-MiniLM-L6-v2"
    "embedding_cache":   True,  # Cache extracted embeddings to disk
    # Scheduler: cosine warm restarts to escape local minima
    "scheduler":         "warmup_cosine",  # warmup_cosine | cosine_warm_restarts | none
    "warmup_ratio":      0.02,
    "min_lr_ratio":      0.1,
    "adam_betas":        [0.9, 0.95],
    "max_steps":         None,
    "early_stop_patience": 5,
    "early_stop_min_delta": 0.005,
    "T_0":               5,       # restart period in epochs
    "T_mult":            2,       # multiply period after each restart
    "eta_min":           1e-6,    # minimum LR
    # MLflow tracking
    "mlflow_experiment": "small_lm",
    "mlflow_enabled":    True,
}


def resolve_device_type(
    requested: str,
    cuda_available: bool,
    mps_available: bool,
) -> str:
    """Resolve an explicit/automatic device request without touching hardware."""
    requested = str(requested).lower()
    if requested not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(
            f"Unknown device {requested!r}; expected auto, cuda, mps, or cpu"
        )
    if requested == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "mps" and not mps_available:
        raise RuntimeError("MPS was requested but is not available")
    if requested != "auto":
        return requested
    if cuda_available:
        return "cuda"
    if mps_available:
        return "mps"
    return "cpu"


def runtime_policy(
    device_type: str,
    *,
    use_amp: bool,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    fused_optimizer: bool = True,
    amp_dtype: str = "float16",
) -> Dict[str, Any]:
    """Return conservative fast defaults for CUDA, Apple MPS, and CPU."""
    if device_type not in {"cuda", "mps", "cpu"}:
        raise ValueError(f"Unsupported device type: {device_type}")
    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 1) if device_type == "cuda" else 0
    if num_workers < 0:
        raise ValueError("num_workers must be >= 0")
    if pin_memory is None:
        pin_memory = device_type == "cuda"
    amp_enabled = bool(use_amp and device_type in {"cuda", "mps"})
    return {
        "amp_enabled": amp_enabled,
        "amp_dtype": (amp_dtype if amp_enabled else "float32"),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "non_blocking": bool(pin_memory and device_type == "cuda"),
        "fused_optimizer": bool(fused_optimizer and device_type == "cuda"),
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple Metal Performance Shaders"
    return platform.processor() or platform.machine() or "CPU"


def _peak_memory_mb(device: torch.device) -> Optional[float]:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    if device.type == "mps" and hasattr(torch, "mps"):
        return torch.mps.current_allocated_memory() / (1024 ** 2)
    return None


def warmup_cosine_multiplier(
    step: int,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    """Linear warm-up followed by one cosine decay."""
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, min(int(warmup_steps), total_steps))
    step = max(0, min(int(step), total_steps))
    min_lr_ratio = float(min_lr_ratio)
    if warmup_steps and step < warmup_steps:
        return step / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps)
    progress = (step - warmup_steps) / decay_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def load_config(config_path: Optional[str], overrides: Dict[str, Any]) -> Dict[str, Any]:
    cfg = dict(DEFAULTS)
    if config_path:
        with open(config_path) as f:
            cfg.update(yaml.safe_load(f))
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logger(log_dir: Path, name: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch  = logging.StreamHandler(sys.stdout); ch.setLevel(logging.INFO); ch.setFormatter(fmt)
    fh  = logging.FileHandler(log_dir / "run.log", mode="w"); fh.setLevel(logging.DEBUG); fh.setFormatter(fmt)
    logger.addHandler(ch); logger.addHandler(fh)
    return logger


class MetricsWriter:
    def __init__(self, path: Path, fieldnames: List[str]) -> None:
        self.path, self.fieldnames = path, fieldnames
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    def write(self, row: Dict) -> None:
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(
                {k: row.get(k, "") for k in self.fieldnames}
            )


# ── Embedding Extractor (for semantic conditioning A/B testing) ───────────────

class EmbeddingExtractor:
    """Extracts sentence embeddings from a pre-trained model for conditioning."""

    def __init__(self, model_name: str, device: torch.device, cache_dir: Optional[Path] = None):
        if not _TRANSFORMERS_OK:
            raise RuntimeError("transformers library required for embedding extraction")
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.cache: Dict[str, torch.Tensor] = {}
        self._disk_cache: Dict[str, torch.Tensor] = {}

        # Load model
        self.tokenizer = _AutoTok.from_pretrained(model_name, trust_remote_code=True)
        self.model = _AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(device)
        self.model.eval()

        # Determine embedding dimension
        with torch.no_grad():
            dummy = self.tokenizer("test", return_tensors="pt").to(device)
            out = self.model(**dummy)
            if hasattr(out, 'last_hidden_state'):
                self.dim = out.last_hidden_state.shape[-1]
            elif hasattr(out, 'pooler_output') and out.pooler_output is not None:
                self.dim = out.pooler_output.shape[-1]
            else:
                self.dim = out[0].shape[-1]

    def _cache_key(self, text: str) -> str:
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()[:16]

    def _load_from_cache(self, key: str) -> Optional[torch.Tensor]:
        if not self.cache_dir or key not in self._disk_cache:
            return None
        return self._disk_cache[key]

    def _save_to_cache(self, key: str, tensor: torch.Tensor) -> None:
        if self.cache_dir:
            self._disk_cache[key] = tensor

    def encode(self, texts: List[str], max_length: int = 512) -> torch.Tensor:
        """Return [batch, dim] sentence embeddings (mean pooled)."""
        if not texts:
            return torch.zeros(0, self.dim, device=self.device)

        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=max_length,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            if hasattr(outputs, 'last_hidden_state'):
                hidden = outputs.last_hidden_state
                mask = inputs['attention_mask'].unsqueeze(-1).expand(hidden.size()).float()
                sum_emb = (hidden * mask).sum(dim=1)
                mean_emb = sum_emb / mask.sum(dim=1).clamp(min=1e-9)
                return mean_emb
            elif hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                return outputs.pooler_output
            else:
                return outputs[0][:, 0]

    def project_to_dim(self, embeddings: torch.Tensor, target_dim: int) -> torch.Tensor:
        """Project embeddings to target dimension (simple truncation/padding)."""
        if embeddings.shape[-1] == target_dim:
            return embeddings
        if embeddings.shape[-1] > target_dim:
            return embeddings[:, :target_dim]  # Truncate
        # Pad with zeros
        pad = torch.zeros(embeddings.shape[0], target_dim - embeddings.shape[-1],
                         device=embeddings.device, dtype=embeddings.dtype)
        return torch.cat([embeddings, pad], dim=-1)


# ── Tokenizer ─────────────────────────────────────────────────────────────────

class CharTokenizer:
    def __init__(self, text: str) -> None:
        vocab = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(vocab)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(vocab)
        self.name = "char"

    def encode(self, text: str) -> List[int]:
        return [self.stoi[c] for c in text if c in self.stoi]


def build_tokenizer(text: str):
    if _TIKTOKEN_OK:
        enc = tiktoken.get_encoding("gpt2")
        enc.name = "tiktoken:gpt2"  # type: ignore[attr-defined]
        enc.vocab_size = enc.n_vocab  # type: ignore[attr-defined]
        return enc
    return CharTokenizer(text)


# ── Dataset ───────────────────────────────────────────────────────────────────

class TokenDataset(Dataset):
    """Chunks a token stream into (x, y) next-token pairs.

    stride < seq_len yields overlapping windows: the same corpus produces
    seq_len/stride times more training examples, and every token is seen at
    many context positions instead of exactly one. Use stride == seq_len
    (the default) for validation so each token is scored exactly once.
    """

    def __init__(self, ids: List[int], seq_len: int, stride: Optional[int] = None) -> None:
        self.t      = torch.tensor(ids, dtype=torch.long)
        self.seq    = seq_len
        self.stride = stride or seq_len

    def __len__(self) -> int:
        return max(0, (len(self.t) - 1 - self.seq) // self.stride + 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = idx * self.stride
        x = self.t[s:s + self.seq]
        y = self.t[s + 1:s + self.seq + 1]
        if len(x) < self.seq:
            x = torch.cat([x, torch.zeros(self.seq - len(x), dtype=torch.long)])
        if len(y) < self.seq:
            y = torch.cat([y, torch.full((self.seq - len(y),), -100, dtype=torch.long)])
        return x, y


# ── AMP context ───────────────────────────────────────────────────────────────

def amp_ctx(device: torch.device, enabled: bool, dtype: str = "float16"):
    if enabled and device.type in {"cuda", "mps"}:
        # bfloat16 has fp32's exponent range, so it needs no GradScaler and does not
        # overflow the way fp16 does. mps only supports fp16.
        want = torch.bfloat16 if dtype == "bfloat16" and device.type == "cuda" else torch.float16
        return torch.autocast(device_type=device.type, dtype=want)
    return torch.autocast(device_type="cpu", enabled=False)


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device,
    cond_dim: int, use_amp: bool, max_batches: int = 0,
    amp_dtype: str = "float16",
    condition_mode: str = "ocean_vad",
    extractor: Optional[EmbeddingExtractor] = None,
    tokenizer: Optional[Any] = None,
) -> Dict[str, float]:
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    total_bytes = 0
    source_nll: Dict[str, float] = {}
    source_tokens: Dict[str, int] = {}
    for bi, batch in enumerate(loader):
        # max_batches=0 means "no cap". The previous default of 200 silently
        # scored only the first 200 batches — always the same prefix, since val
        # is unshuffled — and that truncated number drove checkpoint selection.
        if max_batches and bi >= max_batches:
            break
        if isinstance(batch, dict):
            x, y = batch["input_ids"], batch["labels"]
            aligned_cond = batch["condition"]
            condition_mask = batch["condition_mask"]
            batch_sources = list(batch["source"])
            total_bytes += int(batch["target_bytes"].sum().item())
        else:
            x, y = batch
            aligned_cond = condition_mask = None
            batch_sources = ["legacy_text"] * x.size(0)
        x = x.to(device, non_blocking=device.type == "cuda")
        y = y.to(device, non_blocking=device.type == "cuda")
        with amp_ctx(device, use_amp, amp_dtype):
            if isinstance(model, PrefixTinyGPTLM):
                if aligned_cond is not None:
                    cond = aligned_cond.to(device, non_blocking=device.type == "cuda")
                    mask = condition_mask.to(device, non_blocking=device.type == "cuda")
                    cond = cond * mask
                else:
                    batch_texts = [tokenizer.decode(x[i].tolist()) for i in range(x.size(0))] if tokenizer is not None else [""] * x.size(0)
                    cond = build_condition_vector(
                        batch_texts,
                        condition_mode,
                        cond_dim,
                        extractor=extractor,
                        tokenizer=tokenizer,
                        device=device,
                    )
                    mask = None
                out  = model(x, cond, y, cond_mask=mask)
            else:
                out  = model(x, y)
        # .float() is load-bearing: this call sits OUTSIDE amp_ctx, so under
        # float16 AMP the logits arrive as fp16 and the per-token losses come back
        # as fp16 too. Summing ~12k of them at ~5 nats overflows fp16's 65504 max,
        # so val_loss silently becomes inf for any run above ~5.3 nats — which is
        # every run early in training, and every larger validation set.
        token_losses = F.cross_entropy(
            out.logits.reshape(-1, out.logits.size(-1)).float(),
            y.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(y)
        valid = y != -100
        total_nll += float(token_losses[valid].sum().item())
        total_tokens += int(valid.sum().item())
        for row_index, source in enumerate(batch_sources):
            row_valid = valid[row_index]
            row_tokens = int(row_valid.sum().item())
            if not row_tokens:
                continue
            source_nll[source] = source_nll.get(source, 0.0) + float(
                token_losses[row_index][row_valid].sum().item()
            )
            source_tokens[source] = source_tokens.get(source, 0) + row_tokens
    mean = total_nll / max(total_tokens, 1)
    metrics: Dict[str, float] = {
        "val_loss": mean,
        "val_ppl": math.exp(min(mean, 20)),
        "val_tokens": float(total_tokens),
    }
    if total_bytes:
        metrics["val_bits_per_byte"] = total_nll / (math.log(2) * total_bytes)
    for source, nll in sorted(source_nll.items()):
        metrics[f"source_ppl/{source}"] = math.exp(
            min(nll / max(source_tokens[source], 1), 20)
        )
    return metrics


@torch.no_grad()
def evaluate_generation(
    model: nn.Module,
    dataset: DialogueRecordDataset,
    tokenizer: DialogueTokenizer,
    device: torch.device,
    *,
    max_examples: int = 32,
    max_new_tokens: int = 48,
) -> Dict[str, Any]:
    """Greedy response generation metrics for the record-aware validation set."""
    model.eval()
    generated_texts: List[str] = []
    samples: List[Dict[str, str]] = []
    for record in dataset.records[:max_examples]:
        encoded = dataset.encode_record(record)
        ids = list(encoded["prompt_ids"])
        cond = encoded["condition"].unsqueeze(0).to(device)
        cond_mask = encoded["condition_mask"].unsqueeze(0).to(device)
        cond = cond * cond_mask
        generated: List[int] = []
        for _ in range(max_new_tokens):
            context = ids[-dataset.seq_len :]
            x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)
            if isinstance(model, PrefixTinyGPTLM):
                out = model(x, cond, cond_mask=cond_mask)
            else:
                out = model(x)
            next_id = int(out.logits[0, -1].argmax().item())
            if next_id in (tokenizer.eot_id, tokenizer.eos_id):
                break
            ids.append(next_id)
            generated.append(next_id)
        text = tokenizer.decode(generated).strip()
        generated_texts.append(text)
        if len(samples) < 8:
            samples.append(
                {
                    "prompt": tokenizer.decode(encoded["prompt_ids"]).strip(),
                    "reference": str(record.get("target_response", "")).strip(),
                    "generated": text,
                }
            )

    tokens = [word for text in generated_texts for word in text.lower().split()]

    def distinct(n: int) -> float:
        ngrams = list(zip(*(tokens[offset:] for offset in range(n))))
        return len(set(ngrams)) / len(ngrams) if ngrams else 0.0

    def repetition(n: int) -> float:
        ngrams = list(zip(*(tokens[offset:] for offset in range(n))))
        return 1.0 - len(set(ngrams)) / len(ngrams) if ngrams else 0.0

    return {
        "num_examples": len(generated_texts),
        "distinct_1": distinct(1),
        "distinct_2": distinct(2),
        "repetition_3": repetition(3),
        "repetition_4": repetition(4),
        "empty_response_rate": (
            sum(not text for text in generated_texts) / max(len(generated_texts), 1)
        ),
        "samples": samples,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def train(cfg: Dict[str, Any]) -> Dict[str, Any]:
    random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])

    arch   = cfg["arch"].lower()
    run_id = cfg.get("run_id") or f"{arch}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(cfg["output_dir"]) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    log = setup_logger(out_dir, run_id)
    log.info("=" * 60)
    log.info(f"RUN      : {run_id}")
    log.info(f"ARCH     : {arch}")
    log.info(f"PROFILE  : {cfg['hardware_profile']}")
    if arch == "prefix_gpt":
        log.info(f"COND     : {cfg.get('condition_mode', 'ocean_vad')}")
        if cfg.get("init_from"):
            log.info(f"INIT     : {cfg.get('init_from')}")
    log.info("=" * 60)
    log.debug(f"Config:\n{json.dumps(cfg, indent=2)}")

    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    requested_device = str(cfg.get("device", "auto")).lower()
    resolved_device = resolve_device_type(
        requested_device,
        torch.cuda.is_available(),
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
    )
    device = select_device(resolved_device)
    policy = runtime_policy(
        device.type,
        use_amp=bool(cfg.get("use_amp", True)),
        num_workers=cfg.get("num_workers"),
        pin_memory=cfg.get("pin_memory"),
        fused_optimizer=bool(cfg.get("fused_optimizer", True)),
        amp_dtype=str(cfg.get("amp_dtype", "float16")),
    )
    if device.type == "cuda":
        torch.set_float32_matmul_precision(
            "high" if bool(cfg.get("allow_tf32", True)) else "highest"
        )
        torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", True))
        torch.cuda.reset_peak_memory_stats(device)

    # ── Data ──────────────────────────────────────────────────────────────────
    use_records = bool(cfg.get("train_jsonl") or cfg.get("val_jsonl"))
    if use_records:
        for key in ("train_jsonl", "val_jsonl", "tokenizer_path"):
            value = cfg.get(key)
            if not value or not Path(str(value)).exists():
                raise FileNotFoundError(
                    f"Missing required JSONL-pipeline path: {key}={value!r}"
                )
        tokenizer = DialogueTokenizer.from_file(cfg["tokenizer_path"])
        train_ds = DialogueRecordDataset(
            cfg["train_jsonl"],
            tokenizer,
            cfg["seq_len"],
            profile_max_tokens=cfg.get("profile_max_tokens", 48),
            max_turns=cfg.get("max_turns", 4),
            deduplicate=cfg.get("deduplicate", True),
        )
        val_ds = DialogueRecordDataset(
            cfg["val_jsonl"],
            tokenizer,
            cfg["seq_len"],
            profile_max_tokens=cfg.get("profile_max_tokens", 48),
            max_turns=cfg.get("max_turns", 4),
            deduplicate=cfg.get("deduplicate", True),
        )
        train_ids: List[int] = []
        val_ids: List[int] = []
        train_token_count = train_ds.target_token_count
        val_token_count = val_ds.target_token_count
        log.info(
            "Records — train: %s (%s target tokens)  val: %s (%s target tokens)",
            f"{len(train_ds):,}",
            f"{train_token_count:,}",
            f"{len(val_ds):,}",
            f"{val_token_count:,}",
        )
    else:
        for key in ("train_text", "val_text"):
            p = Path(cfg[key])
            if not p.exists():
                log.error(f"Missing text file: {p}")
                log.error("Run prepare_dialogue_data.py first to produce .txt splits.")
                raise FileNotFoundError(str(p))
        train_text = Path(cfg["train_text"]).read_text(encoding="utf-8")
        val_text   = Path(cfg["val_text"]).read_text(encoding="utf-8")
        tokenizer  = build_tokenizer(train_text)
        train_ids  = tokenizer.encode(train_text)
        val_ids    = tokenizer.encode(val_text)
        train_token_count = len(train_ids)
        val_token_count = len(val_ids)
        # Overlapping windows for training (stride < seq_len multiplies the number
        # of examples); non-overlapping for val so every token is scored once.
        train_stride = int(cfg.get("train_stride") or cfg["seq_len"])
        train_ds   = TokenDataset(train_ids, cfg["seq_len"], stride=train_stride)
        val_ds     = TokenDataset(val_ids,   cfg["seq_len"])
        log.info(f"Windows — train: {len(train_ds):,} (stride={train_stride})  val: {len(val_ds):,}")
        log.info(f"Tokens — train: {len(train_ids):,}  val: {len(val_ids):,}")
    log.info(f"Tokenizer: {tokenizer.name}  vocab={tokenizer.vocab_size:,}")
    num_workers = int(policy["num_workers"])
    pin_memory = bool(policy["pin_memory"])
    loader_runtime: Dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": bool(
            cfg.get("persistent_workers", True) and num_workers > 0
        ),
    }
    if num_workers > 0:
        loader_runtime["prefetch_factor"] = int(cfg.get("prefetch_factor", 2))
    source_sampler = None
    if use_records and cfg.get("source_weights"):
        source_weights = {
            str(key): float(value)
            for key, value in cfg["source_weights"].items()
        }
        sample_weights = [
            source_weights.get(
                str(record.get("metadata", {}).get("source", "unknown")),
                1.0,
            )
            for record in train_ds.records
        ]
        source_sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )
        log.info("Source sampling weights: %s", source_weights)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=source_sampler is None,
        sampler=source_sampler,
        **loader_runtime,
    )
    val_loader = DataLoader(
        val_ds,   batch_size=cfg["batch_size"], shuffle=False,
        **loader_runtime,
    )
    curriculum_loader = None
    curriculum_epochs = 0
    if use_records and float(cfg.get("curriculum_ratio", 0.0)) > 0:
        easy_indices = train_ds.easy_indices(
            max_target_tokens=int(cfg.get("curriculum_max_target_tokens", 48)),
            max_turns=int(cfg.get("curriculum_max_turns", 2)),
        )
        if easy_indices:
            curriculum_epochs = max(
                1,
                math.ceil(int(cfg["epochs"]) * float(cfg["curriculum_ratio"])),
            )
            curriculum_loader = DataLoader(
                Subset(train_ds, easy_indices),
                batch_size=cfg["batch_size"],
                shuffle=True,
                **loader_runtime,
            )
            log.info(
                "Curriculum warm-up: %d epochs on %d short examples",
                curriculum_epochs,
                len(easy_indices),
            )

    # ── Embedding Extractor (for semantic conditioning A/B testing) ────────────
    extractor: Optional[EmbeddingExtractor] = None
    if cfg.get("embedding_model") and _TRANSFORMERS_OK:
        try:
            cache_dir = out_dir / "embedding_cache" if cfg.get("embedding_cache") else None
            extractor = EmbeddingExtractor(cfg["embedding_model"], device, cache_dir)
            log.info(f"Embedding model loaded: {extractor.model_name} (dim={extractor.dim})")
        except Exception as e:
            log.warning(f"Failed to load embedding model: {e}. Using zero conditioning.")

    # ── Model ─────────────────────────────────────────────────────────────────
    profile = RECOMMENDED_CONFIGS.get(cfg["hardware_profile"], {})
    params  = dict(profile.get(arch, {}))
    params["vocab_size"] = tokenizer.vocab_size
    if "max_seq_len" in params:
        params["max_seq_len"] = cfg["seq_len"]
    if arch == "prefix_gpt":
        params["cond_dim"] = cfg["cond_dim"]
        params["condition_mode"] = cfg.get("condition_mode", "ocean_vad")
    # Allow YAML (e.g. Optuna trials) to override any arch-specific param
    for k, v in cfg.get("arch_params", {}).items():
        params[k] = v

    model   = build_model(arch, params).to(device)
    total   = sum(p.numel() for p in model.parameters())
    log.info(
        "Device: %s (%s) | precision=%s | workers=%d | pin_memory=%s | "
        "Parameters: %s (%.1f M)",
        device,
        _device_name(device),
        policy["amp_dtype"],
        num_workers,
        pin_memory,
        f"{total:,}",
        total / 1e6,
    )

    init_from = cfg.get("init_from")
    if init_from and Path(str(init_from)).exists():
        loaded, skipped = load_partial_state_dict(model, init_from, map_location=device)
        log.info(
            "Warm-start loaded from %s: %d tensors loaded, %d skipped",
            init_from,
            len(loaded),
            len(skipped),
        )
        if skipped:
            log.info("Skipped tensors include: %s", ", ".join(skipped[:8]))

    # Ensure numeric types (YAML may load as strings)
    lr = float(cfg["lr"])
    weight_decay = float(cfg["weight_decay"])
    decay_params, no_decay_params = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith(".bias") or "ln" in name.lower() or "norm" in name.lower():
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)
    optimizer_kwargs: Dict[str, Any] = {
        "lr": lr,
        "betas": tuple(float(x) for x in cfg.get("adam_betas", [0.9, 0.95])),
    }
    if policy["fused_optimizer"]:
        optimizer_kwargs["fused"] = True
    if str(cfg.get("optimizer", "adamw")).lower() == "muon":
        # torch.optim.Muon (torch>=2.11) orthogonalises the update, which only makes
        # sense for the hidden weight matrices — it rejects 1-D tensors outright, and
        # embeddings/tied LM head are excluded by convention because their rows get
        # sparse gradients. Those stay on AdamW, so two optimizers are unavoidable.
        muon_params, adamw_decay = [], []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or parameter.ndim < 2:
                continue
            if any(k in name.lower() for k in ("emb", "wte", "wpe", "lm_head", "head.weight")):
                adamw_decay.append(parameter)
            else:
                muon_params.append(parameter)
        muon_lr = float(cfg.get("muon_lr", 0.02))
        optimizers = [
            torch.optim.Muon(muon_params, lr=muon_lr, weight_decay=weight_decay,
                             momentum=float(cfg.get("muon_momentum", 0.95)),
                             # "original" leaves the update magnitude shape-independent,
                             # which mis-scales wide matrices (our FFN is 512->2048);
                             # "match_rms_adamw" rescales per parameter to AdamW's RMS.
                             adjust_lr_fn=cfg.get("muon_adjust_lr") or None),
            torch.optim.AdamW(
                [
                    {"params": adamw_decay, "weight_decay": weight_decay},
                    {"params": no_decay_params, "weight_decay": 0.0},
                ],
                **optimizer_kwargs,
            ),
        ]
        log.info("Optimizer: Muon(%d tensors, lr=%.4g, adjust_lr=%s) + AdamW(%d tensors, lr=%.4g)",
                 len(muon_params), muon_lr, cfg.get("muon_adjust_lr") or "original",
                 len(adamw_decay) + len(no_decay_params), lr)
    else:
        optimizers = [torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            **optimizer_kwargs,
        )]
    optimizer = optimizers[0]  # for logging the primary LR
    scaler = torch.amp.GradScaler(
        device.type,
        # bf16 carries fp32's exponent range, so loss scaling is unnecessary; leaving
        # the scaler on would add its inf/nan checks and skipped steps for nothing.
        enabled=bool(policy["amp_enabled"]) and policy["amp_dtype"] != "bfloat16",
    )

    updates_per_epoch = max(1, math.ceil(len(train_loader) / int(cfg["grad_accum"])))
    planned_steps = updates_per_epoch * int(cfg["epochs"])
    if cfg.get("max_steps") is not None:
        planned_steps = min(planned_steps, int(cfg["max_steps"]))

    # ── LR Scheduler ────────────────────────────────────────────────────────
    schedulers: List[Any] = []
    sched_name = cfg.get("scheduler", "none")
    if sched_name == "warmup_cosine":
        warmup_steps = int(planned_steps * float(cfg.get("warmup_ratio", 0.02)))
        min_lr_ratio = float(cfg.get("min_lr_ratio", 0.1))
        schedulers = [
            LambdaLR(
                opt,
                lr_lambda=lambda step: warmup_cosine_multiplier(
                    step, planned_steps, warmup_steps, min_lr_ratio
                ),
            )
            for opt in optimizers
        ]
        log.info(
            "Scheduler: warmup_cosine (steps=%d, warmup=%d, min_ratio=%.3f)",
            planned_steps,
            warmup_steps,
            min_lr_ratio,
        )
    elif sched_name == "cosine_warm_restarts":
        T_0 = int(cfg.get("T_0", 5))
        T_mult = int(cfg.get("T_mult", 2))
        eta_min = float(cfg.get("eta_min", 1e-6))
        schedulers = [
            CosineAnnealingWarmRestarts(opt, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
            for opt in optimizers
        ]
        log.info(f"Scheduler: CosineAnnealingWarmRestarts (T_0={T_0}, T_mult={T_mult})")
    else:
        log.info("Scheduler: none (constant LR)")

    # ── MLflow tracking ────────────────────────────────────────────────────────
    from mlflow_tracker import MLflowTracker
    tracker = MLflowTracker(
        experiment=cfg.get("mlflow_experiment", "small_lm"),
        enabled=cfg.get("mlflow_enabled", True),
    )
    tracker.start_run(run_name=run_id, tags={
        "arch": arch,
        "seed": str(cfg["seed"]),
        "task": "dialogue_lm",
        "embedding_model": str(cfg.get("embedding_model", "none")),
        "device": device.type,
        "precision": str(policy["amp_dtype"]),
    })
    tracker.log_params(cfg)

    # ── Metric writers ────────────────────────────────────────────────────────
    step_writer  = MetricsWriter(out_dir / "step_metrics.csv",  ["epoch", "global_step", "train_loss", "grad_norm"])
    epoch_writer = MetricsWriter(
        out_dir / "epoch_metrics.csv",
        ["epoch", "global_step", "val_loss", "val_ppl", "val_bits_per_byte"],
    )

    best_val  = math.inf
    best_path = out_dir / "best_model.pt"
    summary: Dict[str, Any] = {
        "run_id":      run_id,
        "arch":        arch,
        "task":        "dialogue_lm_from_scratch",
        "hyperparams": {k: v for k, v in cfg.items() if k not in ("run_id",)},
        "model_params": total,
        "tokenizer":   tokenizer.name,
        "runtime": {
            "requested_device": requested_device,
            "device_type": device.type,
            "device_name": _device_name(device),
            "torch_version": torch.__version__,
            "precision": policy["amp_dtype"],
            "amp_enabled": policy["amp_enabled"],
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "non_blocking_transfers": policy["non_blocking"],
            "fused_optimizer": policy["fused_optimizer"],
            "allow_tf32": bool(cfg.get("allow_tf32", True))
            if device.type == "cuda"
            else False,
            "cudnn_benchmark": bool(cfg.get("cudnn_benchmark", True))
            if device.type == "cuda"
            else False,
        },
        "data":        {
            "format": "jsonl_records" if use_records else "plain_text_stream",
            "train_examples": len(train_ds),
            "val_examples": len(val_ds),
            "train_tokens": train_token_count,
            "val_tokens": val_token_count,
            "condition_coverage": train_ds.condition_coverage if use_records else {},
        },
        "embedding":   {
            "model": cfg.get("embedding_model"),
            "dim":   extractor.dim if extractor else None,
            "cond_dim": cfg["cond_dim"],
            "condition_mode": cfg.get("condition_mode", "ocean_vad"),
            "enabled": extractor is not None,
        },
        "init_from": cfg.get("init_from"),
        "epochs":      [],
        "generation":  {},
        "best":        {},
    }

    global_step  = 0
    running_loss = 0.0
    running_n    = 0
    epochs_without_improvement = 0
    stop_training = False
    train_loop_seconds = 0.0
    train_examples_seen = 0
    train_target_tokens_seen = 0

    for epoch in range(1, cfg["epochs"] + 1):
        log.info(f"── Epoch {epoch}/{cfg['epochs']} ──────────────────────────────────")
        model.train()
        for _opt in optimizers:
            _opt.zero_grad(set_to_none=True)
        active_train_loader = (
            curriculum_loader
            if curriculum_loader is not None and epoch <= curriculum_epochs
            else train_loader
        )
        _synchronize(device)
        epoch_train_started = time.perf_counter()
        epoch_eval_seconds = 0.0

        for bi, batch in enumerate(active_train_loader, start=1):
            if isinstance(batch, dict):
                x, y = batch["input_ids"], batch["labels"]
                aligned_cond = batch["condition"]
                condition_mask = batch["condition_mask"]
            else:
                x, y = batch
                aligned_cond = condition_mask = None
            train_examples_seen += int(x.size(0))
            train_target_tokens_seen += int((y != -100).sum().item())
            x = x.to(device, non_blocking=bool(policy["non_blocking"]))
            y = y.to(device, non_blocking=bool(policy["non_blocking"]))

            with amp_ctx(device, bool(policy["amp_enabled"]), policy["amp_dtype"]):
                if isinstance(model, PrefixTinyGPTLM):
                    if aligned_cond is not None:
                        mask = condition_mask.to(
                            device, non_blocking=bool(policy["non_blocking"])
                        )
                        cond = aligned_cond.to(
                            device, non_blocking=bool(policy["non_blocking"])
                        ) * mask
                        if cfg.get("condition_mode") == "zero":
                            cond = torch.zeros_like(cond)
                            mask = torch.ones_like(mask)
                        elif model.training and float(cfg.get("condition_dropout", 0.0)) > 0:
                            keep = (
                                torch.rand(cond.size(0), 1, device=device)
                                >= float(cfg["condition_dropout"])
                            ).to(cond.dtype)
                            mask = mask * keep
                    else:
                        texts = [tokenizer.decode(x[i].tolist()) for i in range(x.size(0))]
                        cond = build_condition_vector(
                            texts,
                            cfg.get("condition_mode", "ocean_vad"),
                            cfg["cond_dim"],
                            extractor=extractor,
                            tokenizer=tokenizer,
                            device=device,
                        )
                        mask = None
                    out  = model(x, cond, y, cond_mask=mask)
                else:
                    out  = model(x, y)
                group_first = ((bi - 1) // int(cfg["grad_accum"])) * int(cfg["grad_accum"]) + 1
                group_size = min(
                    int(cfg["grad_accum"]),
                    len(active_train_loader) - group_first + 1,
                )
                loss = out.loss / group_size

            if policy["amp_enabled"]:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss += out.loss.item()
            running_n    += 1

            if bi % cfg["grad_accum"] == 0 or bi == len(active_train_loader):
                if policy["amp_enabled"]:
                    for _opt in optimizers:
                        scaler.unscale_(_opt)
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
                    for _opt in optimizers:
                        scaler.step(_opt)
                    scaler.update()
                else:
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
                    for _opt in optimizers:
                        _opt.step()
                for _sched in schedulers:
                    if sched_name == "cosine_warm_restarts":
                        _sched.step(epoch - 1 + bi / len(active_train_loader))
                    else:
                        _sched.step()
                for _opt in optimizers:
                    _opt.zero_grad(set_to_none=True)
                global_step += 1

                # A diverged run is unrecoverable: every later step trains on NaN and
                # the job burns its full wall-clock producing a garbage checkpoint.
                # Under fp16 the GradScaler legitimately produces inf/nan grad norms on
                # steps it then skips, so only the loss is trusted as a divergence signal.
                if not math.isfinite(running_loss):
                    raise RuntimeError(
                        f"training diverged: non-finite loss at step {global_step} "
                        f"(epoch {epoch}, grad_norm={gn:.4g}). Lower the learning rate "
                        f"or switch amp_dtype to bfloat16."
                    )

                if global_step % cfg["log_every"] == 0:
                    avg  = running_loss / running_n
                    ppl  = math.exp(min(avg, 20))
                    log.info(f"  step {global_step:5d} | loss={avg:.4f}  ppl={ppl:.2f}  grad_norm={gn:.4f}")
                    step_writer.write({"epoch": epoch, "global_step": global_step, "train_loss": avg, "grad_norm": gn})
                    tracker.log_metrics({"train_loss": avg, "train_ppl": ppl, "grad_norm": gn,
                                         "lr": optimizer.param_groups[0]["lr"]}, step=global_step)
                    running_loss = 0.0; running_n = 0

                if global_step % cfg["eval_every_steps"] == 0:
                    _synchronize(device)
                    eval_started = time.perf_counter()
                    vm = evaluate(
                        model,
                        val_loader,
                        device,
                        cfg["cond_dim"],
                        bool(policy["amp_enabled"]),
                        amp_dtype=policy["amp_dtype"],
                        condition_mode=cfg.get("condition_mode", "ocean_vad"),
                        extractor=extractor,
                        tokenizer=tokenizer,
                    )
                    _synchronize(device)
                    epoch_eval_seconds += time.perf_counter() - eval_started
                    log.info(f"  [eval] val_loss={vm['val_loss']:.4f}  val_ppl={vm['val_ppl']:.2f}")
                    model.train()

                if cfg.get("max_steps") is not None and global_step >= int(cfg["max_steps"]):
                    stop_training = True
                    break

        _synchronize(device)
        train_loop_seconds += max(
            0.0, time.perf_counter() - epoch_train_started - epoch_eval_seconds
        )
        # ── End-of-epoch validation ───────────────────────────────────────────
        vm = evaluate(
            model,
            val_loader,
            device,
            cfg["cond_dim"],
            bool(policy["amp_enabled"]),
            amp_dtype=policy["amp_dtype"],
            condition_mode=cfg.get("condition_mode", "ocean_vad"),
            extractor=extractor,
            tokenizer=tokenizer,
        )
        log.info(f"  epoch {epoch} end → val_loss={vm['val_loss']:.4f}  val_ppl={vm['val_ppl']:.2f}")
        epoch_writer.write({"epoch": epoch, "global_step": global_step, **vm})
        summary["epochs"].append({"epoch": epoch, "global_step": global_step, **vm})
        tracker.log_metrics(vm, step=global_step)

        improvement = best_val - vm["val_loss"]
        if vm["val_loss"] < best_val:
            best_val = vm["val_loss"]
            _state = model.state_dict()
            if arch == "awdlstm":
                # WeightDropLSTM stores both weight_hh_raw_l{i} (persistent)
                # and weight_hh_l{i} (derived, injected into _parameters at
                # runtime). Save only the raw keys to avoid load key mismatch.
                _state = {k: v for k, v in _state.items()
                          if not ("weight_hh_l" in k and "_raw" not in k)}
            torch.save({
                "arch":   arch, "params": params,
                "state":  _state,
                "epoch":  epoch, "val_loss": best_val,
                "tokenizer_path": cfg.get("tokenizer_path"),
                "data_format": "jsonl_records" if use_records else "plain_text_stream",
            }, best_path)
            log.info(f"  ✓ Best checkpoint saved  (val_loss={best_val:.4f})")

        if improvement > float(cfg.get("early_stop_min_delta", 0.005)):
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        patience = int(cfg.get("early_stop_patience", 0))
        if patience and epochs_without_improvement >= patience:
            log.info("Early stopping after %d epochs without material improvement", patience)
            stop_training = True
        if stop_training:
            break

    best_ep = min(summary["epochs"], key=lambda e: e["val_loss"])
    summary["best"] = {"epoch": best_ep["epoch"], "val_loss": best_val, "val_ppl": best_ep["val_ppl"]}
    if "val_bits_per_byte" in best_ep:
        summary["best"]["val_bits_per_byte"] = best_ep["val_bits_per_byte"]
    summary["best"]["global_step"] = best_ep.get("global_step")
    if use_records and best_path.exists():
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state"], strict=False)
        summary["generation"] = evaluate_generation(
            model,
            val_ds,
            tokenizer,
            device,
            max_examples=int(cfg.get("generation_eval_examples", 32)),
            max_new_tokens=int(cfg.get("generation_max_new_tokens", 48)),
        )

    summary["runtime"].update(
        {
            "train_loop_seconds": train_loop_seconds,
            "train_examples_seen": train_examples_seen,
            "train_target_tokens_seen": train_target_tokens_seen,
            "train_examples_per_second": (
                train_examples_seen / train_loop_seconds
                if train_loop_seconds > 0
                else None
            ),
            "train_target_tokens_per_second": (
                train_target_tokens_seen / train_loop_seconds
                if train_loop_seconds > 0
                else None
            ),
            "peak_memory_mb": _peak_memory_mb(device),
        }
    )

    with open(out_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    write_metrics_bundle(out_dir, "run_summary", summary, title="Small LM Run Summary")

    # ── MLflow: log final metrics and artifacts ──────────────────────────────────
    final_tracking_metrics = {
        "best_val_loss": best_val,
        "best_val_ppl":  best_ep["val_ppl"],
        "best_epoch":    best_ep["epoch"],
        "num_params":    total,
        "train_target_tokens_per_second": summary["runtime"][
            "train_target_tokens_per_second"
        ],
    }
    if summary["runtime"]["peak_memory_mb"] is not None:
        final_tracking_metrics["peak_memory_mb"] = summary["runtime"]["peak_memory_mb"]
    log_metrics_to_mlflow(tracker, final_tracking_metrics)
    tracker.log_artifact(out_dir / "run_summary.json")
    tracker.log_artifact(out_dir / "run_summary.md")
    tracker.log_artifact(out_dir / "epoch_metrics.csv")
    if best_path.exists():
        tracker.log_artifact(best_path)
    tracker.end_run()

    log.info("=" * 60)
    log.info(f"DONE  arch={arch}  best val_ppl={best_ep['val_ppl']:.2f}  (epoch {best_ep['epoch']})")
    log.info(f"Artifacts → {out_dir}")
    log.info("=" * 60)
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a small LM for NPC dialogue")
    p.add_argument("--config",           type=str)
    p.add_argument("--arch",             type=str, choices=["gru", "awdlstm", "gpt", "prefix_gpt", "moe", "mamba_like"])
    p.add_argument("--run-id",           type=str,   dest="run_id")
    p.add_argument("--hardware-profile", type=str,   dest="hardware_profile",
                   choices=["paper_16m", "m1_small", "rtx4070_small"])
    p.add_argument("--train-text",       type=str,   dest="train_text")
    p.add_argument("--val-text",         type=str,   dest="val_text")
    p.add_argument("--train-jsonl",      type=str,   dest="train_jsonl")
    p.add_argument("--val-jsonl",        type=str,   dest="val_jsonl")
    p.add_argument("--tokenizer-path",   type=str,   dest="tokenizer_path")
    p.add_argument("--seq-len",          type=int,   dest="seq_len")
    p.add_argument("--batch-size",       type=int,   dest="batch_size")
    p.add_argument("--grad-accum",       type=int,   dest="grad_accum")
    p.add_argument("--lr",               type=float)
    p.add_argument("--epochs",           type=int)
    p.add_argument("--output-dir",       type=str,   dest="output_dir")
    p.add_argument("--log-every",        type=int,   dest="log_every")
    p.add_argument("--eval-every-steps", type=int,   dest="eval_every_steps")
    p.add_argument("--seed",             type=int)
    p.add_argument("--embedding-model",  type=str,   dest="embedding_model",
                   help="Pre-trained model for semantic conditioning (e.g., Qwen/Qwen3-Embedding-4B)")
    p.add_argument("--condition-mode",   type=str,   dest="condition_mode",
                   choices=["ocean_vad", "aligned", "social_state", "zero"])
    p.add_argument("--init-from",        type=str,   dest="init_from",
                   help="Checkpoint path to warm-start from; compatible tensors are loaded.")
    p.add_argument("--max-steps",         type=int,   dest="max_steps")
    p.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Training device. 'auto' prefers CUDA, then Apple MPS, then CPU.",
    )
    p.add_argument("--num-workers", type=int, dest="num_workers")
    p.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="pin_memory",
    )
    p.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="use_amp",
    )
    p.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="allow_tf32",
    )
    p.add_argument(
        "--fused-optimizer",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="fused_optimizer",
    )
    return p.parse_args()


if __name__ == "__main__":
    args      = parse_args()
    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    cfg = load_config(args.config, overrides)
    train(cfg)
