#!/usr/bin/env python3
"""
gen_model_registry.py — Auto-generate model_registry.md from configs + code
============================================================================
Parses all YAML configs and Python model files, instantiates architectures,
computes exact parameter counts, and writes a complete model registry.

Usage:
  python scripts/gen_model_registry.py                    # Print to stdout
  python scripts/gen_model_registry.py --output docs/source/model_registry.md  # Write file
  python scripts/gen_model_registry.py --check            # Exit 1 if stale

Never manually update model_registry.md again — this script is the source of truth.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "llm_finetuning"))
sys.path.insert(0, str(ROOT / "slm_training"))
sys.path.insert(0, str(ROOT / "slm_training" / "src" / "train"))


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: parameter counting
# ═══════════════════════════════════════════════════════════════════════════════

def count_params(model) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def fmt_params(n: int) -> str:
    if n >= 1e9:
        return f"{n/1e9:.1f}B"
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


# ═══════════════════════════════════════════════════════════════════════════════
# LLM config parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_llm_configs() -> Dict[str, Any]:
    """Parse all LLM training configs."""
    configs = {}
    config_dir = ROOT / "llm_finetuning" / "configs"
    for path in sorted(config_dir.glob("*.yaml")):
        if "cpu" in path.name:
            continue
        with open(path) as f:
            cfg = yaml.safe_load(f)
        name = path.stem
        configs[name] = cfg
    return configs


def parse_llm_model_code() -> Dict[str, Any]:
    """Extract HEAD_SPECS from model.py at runtime (handles loop-added keys)."""
    sys.path.insert(0, str(ROOT / "llm_finetuning"))
    try:
        from src.training.model import HEAD_SPECS
        return dict(HEAD_SPECS)
    except ImportError as e:
        print(f"  [WARN] Could not import HEAD_SPECS: {e}", file=sys.stderr)
        return {}


# ═══════════════════════════════════════════════════════════════════════════════
# SLM config parsing
# ═══════════════════════════════════════════════════════════════════════════════

def parse_slm_configs() -> Dict[str, Any]:
    """Parse all SLM configs."""
    configs = {}
    config_dir = ROOT / "slm_training" / "configs"
    for path in sorted(config_dir.glob("*.yaml")):
        with open(path) as f:
            cfg = yaml.safe_load(f)
        name = path.stem
        configs[name] = cfg
    return configs


def parse_slm_architectures() -> Dict[str, Any]:
    """Extract RECOMMENDED_CONFIGS from small_lm_architectures.py."""
    try:
        from small_lm_architectures import RECOMMENDED_CONFIGS
        return dict(RECOMMENDED_CONFIGS)
    except ImportError:
        return {}


def get_slm_param_counts() -> List[Dict]:
    """Instantiate all 6 architectures × 2 profiles and count params."""
    try:
        from small_lm_architectures import (
            SmallGRULM, GRUConfig,
            AWDLSTMLM, AWDLSTMConfig,
            TinyGPTLM, GPTConfig,
            PrefixTinyGPTLM, PrefixGPTConfig,
            TinyMoELM, MoEConfig,
            MambaLikeLM, MambaLikeConfig,
            RECOMMENDED_CONFIGS,
        )
    except ImportError as e:
        print(f"  [WARN] Could not import SLM architectures: {e}")
        return []

    results = []
    arch_map = {
        "gru": (SmallGRULM, GRUConfig),
        "awdlstm": (AWDLSTMLM, AWDLSTMConfig),
        "gpt": (TinyGPTLM, GPTConfig),
        "prefix_gpt": (PrefixTinyGPTLM, PrefixGPTConfig),
        "moe": (TinyMoELM, MoEConfig),
        "mamba_like": (MambaLikeLM, MambaLikeConfig),
    }

    for profile in ["m1_small", "rtx4070_small"]:
        for arch_name in ["gru", "awdlstm", "gpt", "prefix_gpt", "moe", "mamba_like"]:
            cfg_dict = RECOMMENDED_CONFIGS.get(profile, {}).get(arch_name, {})
            if not cfg_dict:
                continue
            model_cls, cfg_cls = arch_map[arch_name]
            cfg = cfg_cls(**cfg_dict)
            try:
                model = model_cls(cfg)
                total, _ = count_params(model)
                results.append({
                    "arch": arch_name,
                    "profile": profile,
                    "params": total,
                    "cfg": cfg_dict,
                })
            except Exception as e:
                print(f"  [WARN] Could not instantiate {arch_name}/{profile}: {e}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Data stats
# ═══════════════════════════════════════════════════════════════════════════════

def get_data_stats() -> Dict[str, Any]:
    """Count records in all data files."""
    stats = {}

    # LLM data
    for name in ["head_supervision", "full_trace", "sft"]:
        path = ROOT / "data" / "packaged" / f"{name}.jsonl"
        if path.exists():
            stats[f"packaged_{name}"] = sum(1 for _ in open(path))

    for split in ["train", "val", "test"]:
        for kind in ["heads", "sft", "trace"]:
            path = ROOT / "data" / "splits" / f"{split}_{kind}.jsonl"
            if path.exists():
                stats[f"{split}_{kind}"] = sum(1 for _ in open(path))

    # Counterfactuals
    trace = ROOT / "data" / "packaged" / "full_trace.jsonl"
    if trace.exists():
        cf_count = 0
        for line in open(trace):
            if '"counterfactual": true' in line:
                cf_count += 1
        stats["counterfactual_turns"] = cf_count

    # Scenario bank
    bank = ROOT / "data" / "scenario_bank"
    if bank.exists():
        stats["scenario_templates"] = len(list(bank.glob("*.yaml")))

    # SLM data
    for fname, key in [
        ("slm_training/data/dialogue/train.jsonl", "slm_dialogue_train"),
        ("slm_training/data/dialogue/val.jsonl", "slm_dialogue_val"),
        ("slm_training/data/personality/train.csv", "slm_personality_train"),
        ("slm_training/data/personality/val.csv", "slm_personality_val"),
        ("slm_training/data/affect/train.csv", "slm_affect_train"),
        ("slm_training/data/affect/val.csv", "slm_affect_val"),
    ]:
        path = ROOT / fname
        if path.exists():
            stats[key] = sum(1 for _ in open(path))

    # Plain text token estimates
    for fname, key in [
        ("slm_training/data/dialogue/train.txt", "slm_dialogue_text_tokens"),
        ("slm_training/data/dialogue/val.txt", "slm_dialogue_text_val_tokens"),
    ]:
        path = ROOT / fname
        if path.exists():
            stats[key] = path.stat().st_size // 4  # rough BPE token estimate

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# Markdown generation
# ═══════════════════════════════════════════════════════════════════════════════

def _h(n: int, title: str) -> str:
    return f"\n{'#' * n} {title}\n"


def _table(headers: List[str], rows: List[List[str]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out) + "\n"


def _kv_table(data: Dict[str, Any], key_width: int = 30) -> str:
    out = []
    for k, v in data.items():
        out.append(f"| **{k}** | {v} |")
    return "\n".join(out) + "\n"


def generate_markdown(
    llm_configs: Dict,
    llm_heads: Dict,
    slm_configs: Dict,
    slm_params: List[Dict],
    data_stats: Dict,
) -> str:
    out = []
    out.append(f"# Complete Model Registry: Architectures, Data & Hyperparameters\n")
    out.append(f"> Auto-generated {datetime.now().strftime('%Y-%m-%d %H:%M')} from configs + code.\n")

    # ── LLM Backbones ─────────────────────────────────────────────────────────
    out.append(_h(2, "Part 1: LLM Fine-Tuning"))
    out.append(_h(3, "1.1 Backbone Models"))
    out.append("| Model | Layers | Hidden | Heads | Head Dim | FFN | Vocab |")
    out.append("|-------|--------|--------|-------|----------|-----|-------|")
    backbones = [
        ("Qwen3-0.6B", 28, 896, 14, 64, 2432),
        ("Qwen3-1.7B", 28, 2048, 16, 128, 5504),
        ("Qwen3-4B", 36, 2560, 32, 80, 6912),
    ]
    for name, layers, hidden, heads, hdim, ffn in backbones:
        out.append(f"| **{name}** | {layers} | {hidden} | {heads} | {hdim} | {ffn} | 151,936 |")
    out.append("")

    # ── 29 Heads ──────────────────────────────────────────────────────────────
    out.append(_h(3, "1.2 Classification Heads (29 targets)"))
    groups = {
        "C_t — Contextual": ["dialogue_act", "tone", "risk_type"],
        "A_t — Affect": ["valence", "arousal", "threat", "control"],
        "M_t — Mental Model": ["player_intent", "player_knowledge", "player_credibility"],
        "R_t — Stance": [f"{d}_{a}" for d in ["affection","respect","dominance","familiarity","trust","obligation"] for a in ["level","delta"]],
        "N_t — Norms": ["duty_pressure", "secrecy_pressure", "face_pressure", "value_conflict"],
        "D_t — Policy": ["response_policy", "reveal_decision", "repair_strategy"],
    }
    rows = []
    for group, fields in groups.items():
        for f in fields:
            spec = llm_heads.get(f, {})
            n = spec.get("n_classes", "?")
            ml = "✅" if spec.get("multi_label") else "❌"
            rows.append([group, f"`{f}`", str(n), ml])
    out.append(_table(["Group", "Field", "Classes", "Multi-label"], rows))

    # ── LLM Training configs ──────────────────────────────────────────────────
    out.append(_h(3, "1.3 Training Configurations"))

    stages = {
        "Stage 1 (Latent Predictor)": "train_latent",
        "Stage 2 (Response SFT)": "train_response",
        "Stage 3 (Joint)": "train_joint",
    }
    for stage_name, config_key in stages.items():
        cfg = llm_configs.get(config_key, {})
        training = cfg.get("training", {})
        lora = cfg.get("lora", {})
        out.append(_h(4, stage_name))
        out.append(f"- **Base model:** `{cfg.get('base_model', '?')}`")
        out.append(f"- **Quantization:** {cfg.get('quantization', '?')}")
        if lora:
            out.append(f"- **LoRA:** r={lora.get('r','?')}, alpha={lora.get('alpha','?')}, dropout={lora.get('dropout','?')}")
        out.append(f"- **Learning rate:** {training.get('lr','?')}")
        out.append(f"- **Epochs:** {training.get('epochs','?')}")
        out.append(f"- **Max seq len:** {training.get('max_seq_len','?')}")
        out.append(f"- **Batch size:** {training.get('batch_size','?')} × {training.get('grad_accum','?')} grad_accum = **{training.get('batch_size',1) * training.get('grad_accum',1)}** effective")
        out.append(f"- **Weight decay:** {training.get('weight_decay','?')}")
        out.append(f"- **Max grad norm:** {training.get('max_grad_norm','?')}")
        if cfg.get("loss_weights"):
            out.append(f"- **Loss weights:** {json.dumps(cfg['loss_weights'])}")
        out.append("")

    # ── LLM Data ──────────────────────────────────────────────────────────────
    out.append(_h(3, "1.4 Data"))
    out.append(f"- Packaged turns: {data_stats.get('packaged_head_supervision', '?')}")
    out.append(f"- Train: {data_stats.get('train_heads', '?')} turns")
    out.append(f"- Val: {data_stats.get('val_heads', '?')} turns")
    out.append(f"- Test: {data_stats.get('test_heads', '?')} turns")
    out.append(f"- Counterfactual variants: {data_stats.get('counterfactual_turns', '?')}")
    out.append(f"- Scenario templates: {data_stats.get('scenario_templates', '?')}")
    out.append("")

    # ── SLM Architectures ─────────────────────────────────────────────────────
    out.append(_h(2, "Part 2: SLM Training from Scratch"))
    out.append(_h(3, "2.1 Small Language Model Architectures"))
    out.append("| Architecture | m1_small params | rtx4070 params |")
    out.append("|-------------|----------------|----------------|")
    by_arch = defaultdict(dict)
    for r in slm_params:
        by_arch[r["arch"]][r["profile"]] = r["params"]
    for arch in ["gru", "awdlstm", "gpt", "prefix_gpt", "moe", "mamba_like"]:
        small = fmt_params(by_arch.get(arch, {}).get("m1_small", 0))
        large = fmt_params(by_arch.get(arch, {}).get("rtx4070_small", 0))
        out.append(f"| `{arch}` | {small} | {large} |")
    out.append("")

    # ── SLM Training configs ──────────────────────────────────────────────────
    out.append(_h(3, "2.2 Training Configurations"))
    slm_train_keys = {
        "Personality Encoder": "personality",
        "Affect Encoder": "affect",
        "Small LM": "small_lm",
        "Dialogue Model": "dialogue",
    }
    for name, key in slm_train_keys.items():
        cfg = slm_configs.get(key, {})
        if not cfg:
            continue
        out.append(_h(4, name))
        for k, v in cfg.items():
            if k in ("target_columns", "target_modules"):
                out.append(f"- **{k}:** {v}")
            elif isinstance(v, (int, float, str, bool)):
                out.append(f"- **{k}:** {v}")
        out.append("")

    # ── SLM Data ──────────────────────────────────────────────────────────────
    out.append(_h(3, "2.3 Data"))
    slm_data = [
        ("Dialogue (train)", data_stats.get("slm_dialogue_train"), "jsonl examples"),
        ("Dialogue (val)", data_stats.get("slm_dialogue_val"), "jsonl examples"),
        ("Dialogue text (train)", data_stats.get("slm_dialogue_text_tokens"), "tokens (~BPE)"),
        ("Dialogue text (val)", data_stats.get("slm_dialogue_text_val_tokens"), "tokens (~BPE)"),
        ("Personality (train)", data_stats.get("slm_personality_train"), "CSV rows"),
        ("Personality (val)", data_stats.get("slm_personality_val"), "CSV rows"),
        ("Affect (train)", data_stats.get("slm_affect_train"), "CSV rows"),
        ("Affect (val)", data_stats.get("slm_affect_val"), "CSV rows"),
    ]
    for name, count, unit in slm_data:
        if count:
            out.append(f"- **{name}:** {count:,} {unit}")
    out.append("")

    # ── Evaluation ────────────────────────────────────────────────────────────
    out.append(_h(2, "Part 3: Evaluation"))
    eval_cfg = llm_configs.get("eval", {})
    if eval_cfg:
        out.append("### LLM Evaluation Thresholds")
        thresholds = eval_cfg.get("thresholds", {})
        for k, v in thresholds.items():
            out.append(f"- **{k}:** {v}")
        out.append("")
    out.append("### SLM Evaluation Metrics")
    out.append("- val_ppl, test_ppl, BLEU-1, BLEU-2, Distinct-1, Distinct-2")
    out.append("- Personality: MSE, R² per OCEAN trait")
    out.append("- Affect: CCC, MSE, MAE, R² per VAD dimension")
    out.append("")

    # ── Footer ────────────────────────────────────────────────────────────────
    out.append("---\n")
    out.append(f"*Regenerated {datetime.now().isoformat()} by `scripts/gen_model_registry.py`*")
    out.append("*Do not edit manually — run the script to update.*\n")

    return "\n".join(out)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Auto-generate model registry from configs + code")
    p.add_argument("--output", "-o", type=Path,
                   default=ROOT / "docs" / "source" / "model_registry.md",
                   help="Output markdown file")
    p.add_argument("--check", action="store_true",
                   help="Exit 1 if registry is stale (for CI)")
    p.add_argument("--stdout", action="store_true",
                   help="Print to stdout instead of writing file")
    args = p.parse_args()

    # Parse everything
    llm_configs = parse_llm_configs()
    llm_heads = parse_llm_model_code()
    slm_configs = parse_slm_configs()
    slm_params = get_slm_param_counts()
    data_stats = get_data_stats()

    # Generate
    md = generate_markdown(llm_configs, llm_heads, slm_configs, slm_params, data_stats)

    if args.stdout:
        print(md)
        return

    # Add auto-gen header
    header = (
        "<!-- AUTO-GENERATED by scripts/gen_model_registry.py — DO NOT EDIT MANUALLY -->\n"
        "<!-- Run: python scripts/gen_model_registry.py to regenerate -->\n\n"
    )
    content = header + md

    if args.check:
        if args.output.exists():
            existing = args.output.read_text()
            # Strip variable parts (timestamps)
            existing_clean = "\n".join(
                l for l in existing.split("\n")
                if not l.startswith("> Auto-generated") and not l.startswith("*Regenerated")
            )
            new_clean = "\n".join(
                l for l in md.split("\n")
                if not l.startswith("> Auto-generated") and not l.startswith("*Regenerated")
            )
            if existing_clean.strip() == new_clean.strip():
                print("✅ Registry is up to date.")
                sys.exit(0)
            else:
                print("❌ Registry is stale. Run: python scripts/gen_model_registry.py")
                sys.exit(1)
        else:
            print("❌ Registry file not found. Run: python scripts/gen_model_registry.py")
            sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content)
    print(f"✅ Model registry written to {args.output}")
    print(f"   Lines: {len(md.split(chr(10)))}")
    print(f"   Size:  {len(content)} bytes")


if __name__ == "__main__":
    main()
