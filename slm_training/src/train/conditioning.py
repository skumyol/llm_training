from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch


_SOCIAL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "secrecy": ("secret", "hide", "conceal", "classified", "private", "vault", "keep", "never tell"),
    "trust": ("trust", "confide", "rely", "believe", "faith", "honest"),
    "repair": ("sorry", "apolog", "repair", "make up", "forgive", "clarify", "soothe"),
    "reveal": ("reveal", "tell", "share", "explain", "admit", "confess"),
    "challenge": ("challenge", "threat", "force", "demand", "pressure", "test"),
    "affect": ("angry", "sad", "fear", "calm", "happy", "warm", "cold", "tense"),
    "negation": (" not ", " never ", " no ", " none "),
}


def _count_keywords(text: str, keywords: Iterable[str]) -> float:
    lower = f" {text.lower()} "
    count = 0.0
    for kw in keywords:
        if kw in lower:
            count += 1.0
    return count


def build_social_proxy_vector(text: str, cond_dim: int) -> torch.Tensor:
    """Build a lightweight proxy social-state vector from text.

    The current scratch SLM pipeline is text-stream based, so there is no aligned
    explicit social-state annotation to load. This proxy keeps the conditioning
    path usable now and can be replaced with explicit vectors later without
    changing the model interface.
    """
    words = re.findall(r"\w+", text.lower())
    token_count = max(len(words), 1)
    sent_count = max(text.count(".") + text.count("!") + text.count("?"), 1)

    features = [
        _count_keywords(text, _SOCIAL_KEYWORDS["secrecy"]) / token_count,
        _count_keywords(text, _SOCIAL_KEYWORDS["trust"]) / token_count,
        _count_keywords(text, _SOCIAL_KEYWORDS["repair"]) / token_count,
        _count_keywords(text, _SOCIAL_KEYWORDS["reveal"]) / token_count,
        _count_keywords(text, _SOCIAL_KEYWORDS["challenge"]) / token_count,
        _count_keywords(text, _SOCIAL_KEYWORDS["affect"]) / token_count,
        _count_keywords(text, _SOCIAL_KEYWORDS["negation"]) / token_count,
        min(token_count / 64.0, 1.0),
        min(sent_count / 8.0, 1.0),
        1.0 if "?" in text else 0.0,
        1.0 if "!" in text else 0.0,
        1.0 if any(ch.isupper() for ch in text[: min(len(text), 64)]) else 0.0,
    ]

    vec = torch.tensor(features, dtype=torch.float32)
    if cond_dim <= vec.numel():
        return vec[:cond_dim]
    pad = torch.zeros(cond_dim - vec.numel(), dtype=torch.float32)
    return torch.cat([vec, pad], dim=0)


def build_condition_vector(
    texts: List[str],
    mode: str,
    cond_dim: int,
    *,
    extractor: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Construct conditioning vectors for a batch of texts.

    Modes:
    - ``ocean_vad``: current baseline; uses an embedding extractor when provided,
      otherwise zeros.
    - ``social_state``: text-derived proxy features that approximate social-state
      signals in the absence of aligned explicit annotations.
    - ``zero``: always zeros.
    """
    batch_size = len(texts)
    if batch_size == 0:
        return torch.zeros(0, cond_dim, device=device or torch.device("cpu"))

    mode = (mode or "zero").lower()
    device = device or torch.device("cpu")

    if mode == "social_state":
        vecs = [build_social_proxy_vector(text, cond_dim) for text in texts]
        return torch.stack(vecs, dim=0).to(device)

    if extractor is not None and tokenizer is not None:
        embeddings = extractor.encode(texts)
        projected = extractor.project_to_dim(embeddings, cond_dim)
        return projected.to(device)

    return torch.zeros(batch_size, cond_dim, device=device)


def load_checkpoint_payload(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(ckpt, dict):
        return {"state_dict": ckpt}
    return ckpt


def extract_state_dict(ckpt: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ("state", "model_state", "model_state_dict"):
        state = ckpt.get(key)
        if isinstance(state, dict):
            return state
    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt  # plain state_dict
    raise ValueError("checkpoint does not contain a recognizable state_dict")


def load_partial_state_dict(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Tuple[List[str], List[str]]:
    """Load only tensors with matching names and shapes.

    Returns:
        (loaded_keys, skipped_keys)
    """
    ckpt = load_checkpoint_payload(checkpoint_path, map_location=map_location)
    state_dict = extract_state_dict(ckpt)
    current = model.state_dict()

    loaded: List[str] = []
    skipped: List[str] = []
    for key, tensor in state_dict.items():
        if key in current and current[key].shape == tensor.shape:
            current[key] = tensor
            loaded.append(key)
        else:
            skipped.append(key)

    model.load_state_dict(current)
    return loaded, skipped


def _count_blocks(state_dict: dict[str, torch.Tensor]) -> int:
    indices = set()
    for key in state_dict:
        match = re.match(r"blocks\.(\d+)\.", key)
        if match:
            indices.add(int(match.group(1)))
    return (max(indices) + 1) if indices else 0


def infer_gpt_like_config(state_dict: dict[str, torch.Tensor], *, prefix: bool = False) -> dict[str, Any]:
    tok = state_dict.get("tok_emb.weight")
    if tok is None:
        tok = state_dict.get("embed.weight")
    if tok is None:
        raise ValueError("checkpoint state_dict does not contain a token embedding weight")

    vocab_size = int(tok.shape[0])
    n_embd = int(tok.shape[1])
    n_layer = _count_blocks(state_dict)
    pos = state_dict.get("pos_emb.weight")
    max_seq_len = int(pos.shape[0]) if pos is not None else 256
    n_head = max(1, n_embd // 64)

    cfg: dict[str, Any] = {
        "vocab_size": vocab_size,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "dropout": 0.1,
        "max_seq_len": max_seq_len,
        "tie_weights": True,
    }

    if prefix:
        prefix_w = state_dict.get("prefix_proj.0.weight")
        prefix_out = state_dict.get("prefix_proj.2.weight")
        if prefix_w is None or prefix_out is None:
            raise ValueError("prefix checkpoint is missing prefix_proj weights")
        cond_dim = int(prefix_w.shape[1])
        prefix_length = int(prefix_out.shape[0] // n_embd)
        cfg.update({
            "prefix_length": prefix_length,
            "cond_dim": cond_dim,
            "condition_mode": "ocean_vad",
            "max_seq_len": max_seq_len - prefix_length if max_seq_len > prefix_length else max_seq_len,
        })

    return cfg


def infer_arch_config_from_checkpoint(checkpoint_path: str | Path) -> tuple[str, dict[str, Any], dict[str, Any]]:
    ckpt = load_checkpoint_payload(checkpoint_path, map_location="cpu")
    state = extract_state_dict(ckpt)
    arch = str(ckpt.get("arch", "")).lower()
    if not arch:
        arch = "prefix_gpt" if "prefix_proj.0.weight" in state else "gpt"
    if arch == "prefix_gpt":
        cfg = infer_gpt_like_config(state, prefix=True)
    elif arch == "gpt":
        cfg = infer_gpt_like_config(state, prefix=False)
    else:
        cfg = ckpt.get("params") or {}
    return arch, cfg, ckpt
