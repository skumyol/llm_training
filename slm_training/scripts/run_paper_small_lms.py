#!/usr/bin/env python3
"""Train every scratch SLM with one shared protocol and build paper-ready reports.

The command intentionally owns orchestration and reporting only.  Individual
training remains in ``src.train.run_small_lm`` so old entry points keep working.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT.parent / ".venv" / "bin" / "python"
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

sys.path.insert(0, str(ROOT))
from src.data.small_lm_dialogue import train_dialogue_tokenizer


ARCHS = ["gru", "awdlstm", "gpt", "prefix_gpt", "moe", "mamba_like"]
ARCH_LABELS = {
    "gru": "GRU-LM",
    "awdlstm": "AWD-LSTM",
    "gpt": "TinyGPT",
    "prefix_gpt": "PrefixGPT",
    "moe": "TinyMoE",
    "mamba_like": "Mamba-like",
}

SMOKE_ARCH_PARAMS = {
    "gru": {"embed_dim": 32, "hidden_size": 32, "num_layers": 1, "dropout": 0.0},
    "awdlstm": {
        "embed_dim": 32,
        "hidden_size": 32,
        "num_layers": 1,
        "wdrop": 0.0,
        "dropout": 0.0,
        "dropouth": 0.0,
        "dropouti": 0.0,
    },
    "gpt": {"n_embd": 32, "n_head": 4, "n_layer": 1, "dropout": 0.0},
    "prefix_gpt": {
        "n_embd": 32,
        "n_head": 4,
        "n_layer": 1,
        "dropout": 0.0,
        "prefix_length": 2,
    },
    "moe": {
        "n_embd": 32,
        "n_head": 4,
        "n_layer": 1,
        "num_experts": 2,
        "top_k": 1,
        "dropout": 0.0,
    },
    "mamba_like": {
        "n_embd": 32,
        "n_layer": 1,
        "d_state": 4,
        "d_conv": 2,
        "expand": 1,
        "dropout": 0.0,
    },
}


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _mean_std(values: Iterable[float]) -> tuple[float, float]:
    clean = [value for value in values if math.isfinite(value)]
    if not clean:
        return float("nan"), float("nan")
    return statistics.mean(clean), statistics.stdev(clean) if len(clean) > 1 else 0.0


def _latex_escape(value: str) -> str:
    return value.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def _format_mean_std(
    row: dict[str, Any],
    metric: str,
    *,
    decimals: int = 0,
    thousands: bool = False,
) -> str:
    mean = _as_float(row.get(f"{metric}_mean"))
    std = _as_float(row.get(f"{metric}_std"))
    if not math.isfinite(mean):
        return "—"
    comma = "," if thousands else ""
    return f"{mean:{comma}.{decimals}f} ± {std:{comma}.{decimals}f}"


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats with JSON null for strict, portable reports."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def build_report(
    summaries: list[dict[str, Any]],
    output_dir: Path,
    *,
    experiment_id: str,
) -> dict[str, Path]:
    """Create per-run JSON and aggregate CSV/Markdown/LaTeX/learning curves."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries:
        best = summary.get("best", {})
        generation = summary.get("generation", {})
        runtime = summary.get("runtime", {})
        row = {
            "experiment_id": experiment_id,
            "run_id": summary.get("run_id"),
            "arch": summary.get("arch"),
            "arch_label": ARCH_LABELS.get(summary.get("arch"), summary.get("arch")),
            "seed": summary.get("hyperparams", {}).get("seed"),
            "params_m": _as_float(summary.get("model_params")) / 1e6,
            "val_loss": _as_float(best.get("val_loss")),
            "val_ppl": _as_float(best.get("val_ppl")),
            "bits_per_byte": _as_float(best.get("val_bits_per_byte")),
            "best_epoch": best.get("epoch"),
            "global_step": best.get("global_step"),
            "distinct_1": _as_float(generation.get("distinct_1")),
            "distinct_2": _as_float(generation.get("distinct_2")),
            "repetition_3": _as_float(generation.get("repetition_3")),
            "empty_response_rate": _as_float(generation.get("empty_response_rate")),
            "device_type": runtime.get("device_type"),
            "device_name": runtime.get("device_name"),
            "precision": runtime.get("precision"),
            "train_target_tokens_per_second": _as_float(
                runtime.get("train_target_tokens_per_second")
            ),
            "peak_memory_mb": _as_float(runtime.get("peak_memory_mb")),
            "tokenizer": summary.get("tokenizer"),
        }
        rows.append(row)
        grouped[str(row["arch"])].append(row)

    aggregates: list[dict[str, Any]] = []
    for arch in ARCHS:
        arch_rows = grouped.get(arch, [])
        if not arch_rows:
            continue
        aggregate: dict[str, Any] = {
            "arch": arch,
            "arch_label": ARCH_LABELS[arch],
            "n_seeds": len(arch_rows),
        }
        for metric in (
            "params_m",
            "val_ppl",
            "bits_per_byte",
            "distinct_1",
            "distinct_2",
            "repetition_3",
            "empty_response_rate",
            "train_target_tokens_per_second",
            "peak_memory_mb",
        ):
            mean, std = _mean_std(_as_float(row.get(metric)) for row in arch_rows)
            aggregate[f"{metric}_mean"] = mean
            aggregate[f"{metric}_std"] = std
        aggregates.append(aggregate)
    aggregates.sort(key=lambda row: row["val_ppl_mean"])

    json_path = output_dir / "paper_results.json"
    json_path.write_text(
        json.dumps(
            _json_safe(
                {
                    "experiment_id": experiment_id,
                    "runs": rows,
                    "aggregate": aggregates,
                }
            ),
            indent=2,
            allow_nan=False,
        )
    )

    csv_path = output_dir / "paper_results.csv"
    if aggregates:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(aggregates[0]))
            writer.writeheader()
            writer.writerows(_json_safe(aggregates))
    else:
        csv_path.write_text("")

    md_path = output_dir / "paper_results.md"
    md_lines = [
        f"# Scratch-SLM results: {experiment_id}",
        "",
        "All models use the same tokenizer, record-aware NPC-only objective, data split, and optimization protocol.",
        "",
        "| Architecture | Device | Params (M) | Seeds | Target PPL ↓ | Bits/byte ↓ | Distinct-2 ↑ | Repetition-3 ↓ | Train target tok/s ↑ | Peak memory (MB) ↓ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregates:
        devices = sorted(
            {
                str(run.get("device_type") or "unknown")
                for run in grouped.get(row["arch"], [])
            }
        )
        md_lines.append(
            (
                "| {arch_label} | " + "/".join(devices) + " | "
                "{params_m_mean:.2f} | {n_seeds} | "
            "{val_ppl_mean:.2f} ± {val_ppl_std:.2f} | "
            "{bits_per_byte_mean:.3f} ± {bits_per_byte_std:.3f} | "
            "{distinct_2_mean:.3f} ± {distinct_2_std:.3f} | "
            "{repetition_3_mean:.3f} ± {repetition_3_std:.3f} | "
            ).format(**row)
            + _format_mean_std(
                row, "train_target_tokens_per_second", thousands=True
            )
            + " | "
            + _format_mean_std(row, "peak_memory_mb", thousands=True)
            + " |"
        )
    md_path.write_text("\n".join(md_lines) + "\n")

    tex_path = output_dir / "paper_results.tex"
    tex_lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Model & Params (M) & Seeds & PPL $\\downarrow$ & bits/byte $\\downarrow$ & Dist.-2 $\\uparrow$ & Rep.-3 $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for row in aggregates:
        tex_lines.append(
            f"{_latex_escape(row['arch_label'])} & {row['params_m_mean']:.2f} & "
            f"{row['n_seeds']} & {row['val_ppl_mean']:.2f} $\\pm$ {row['val_ppl_std']:.2f} & "
            f"{row['bits_per_byte_mean']:.3f} & {row['distinct_2_mean']:.3f} & "
            f"{row['repetition_3_mean']:.3f} \\\\"
        )
    tex_lines.extend(["\\bottomrule", "\\end{tabular}"])
    tex_path.write_text("\n".join(tex_lines) + "\n")

    outputs = {"json": json_path, "csv": csv_path, "markdown": md_path, "latex": tex_path}
    try:
        os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mpl-cache"))
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for summary in summaries:
            epochs = summary.get("epochs", [])
            if not epochs:
                continue
            ax.plot(
                [row.get("global_step", row.get("epoch")) for row in epochs],
                [row.get("val_ppl") for row in epochs],
                marker="o",
                alpha=0.75,
                label=f"{ARCH_LABELS.get(summary.get('arch'), summary.get('arch'))} s{summary.get('hyperparams', {}).get('seed')}",
            )
        ax.set_xlabel("Optimizer step")
        ax.set_ylabel("Target validation perplexity")
        ax.set_yscale("log")
        ax.grid(alpha=0.25)
        if summaries:
            ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        png_path = output_dir / "learning_curves.png"
        pdf_path = output_dir / "learning_curves.pdf"
        fig.savefig(png_path, dpi=200)
        fig.savefig(pdf_path)
        plt.close(fig)
        outputs.update({"figure_png": png_path, "figure_pdf": pdf_path})
    except ImportError:
        pass
    return outputs


def _load_summary(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arch", choices=["all", *ARCHS], default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--train-jsonl", type=Path, default=ROOT / "data/dialogue/from_gen_train.jsonl")
    parser.add_argument("--val-jsonl", type=Path, default=ROOT / "data/dialogue/from_gen_val.jsonl")
    parser.add_argument("--tokenizer-path", type=Path, default=ROOT / "artifacts/tokenizers/dialogue_8k.json")
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Training backend; auto prefers CUDA, then Apple MPS, then CPU.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        help="DataLoader workers; default is 4 on CUDA and 0 on MPS/CPU.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use float16 autocast and gradient scaling on CUDA/MPS.",
    )
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow TF32 matrix math on supported NVIDIA GPUs.",
    )
    parser.add_argument(
        "--fused-optimizer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use fused AdamW on CUDA; ignored on MPS/CPU.",
    )
    parser.add_argument(
        "--source-weight",
        action="append",
        default=[],
        metavar="SOURCE=WEIGHT",
        help="Optional record-sampling weight; repeat for multiple sources.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/paper_small_lm")
    parser.add_argument("--experiment-id")
    parser.add_argument("--force-tokenizer", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    source_weights: dict[str, float] = {}
    for item in args.source_weight:
        if "=" not in item:
            parser.error(f"Invalid --source-weight {item!r}; expected SOURCE=WEIGHT")
        source, weight = item.split("=", 1)
        source_weights[source] = float(weight)

    experiment_id = args.experiment_id or (
        ("smoke_" if args.smoke else "paper_") + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    experiment_dir = args.output_dir / experiment_id
    configs_dir = experiment_dir / "configs"
    runs_dir = experiment_dir / "runs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    vocab_size = 320 if args.smoke else args.vocab_size
    tokenizer_path = (
        experiment_dir / "smoke_tokenizer.json" if args.smoke else args.tokenizer_path
    )
    if args.force_tokenizer or not tokenizer_path.exists():
        print(f"[tokenizer] training {vocab_size}-token byte BPE → {tokenizer_path}")
        train_dialogue_tokenizer(
            [args.train_jsonl],
            tokenizer_path,
            vocab_size=vocab_size,
            min_frequency=1 if args.smoke else 2,
        )

    archs = ARCHS if args.arch == "all" else [args.arch]
    summaries: list[dict[str, Any]] = []
    failures: list[str] = []
    for arch in archs:
        for seed in args.seeds:
            run_id = f"{experiment_id}_{arch}_s{seed}"
            summary_path = runs_dir / run_id / "run_summary.json"
            if args.skip_existing and summary_path.exists():
                summaries.append(_load_summary(summary_path))
                continue

            cfg: dict[str, Any] = {
                "arch": arch,
                "hardware_profile": "paper_16m",
                "train_jsonl": str(args.train_jsonl.resolve()),
                "val_jsonl": str(args.val_jsonl.resolve()),
                "tokenizer_path": str(tokenizer_path.resolve()),
                "seq_len": 32 if args.smoke else args.seq_len,
                "batch_size": 2 if args.smoke else args.batch_size,
                "grad_accum": 1 if args.smoke else args.grad_accum,
                "lr": 1e-3 if args.smoke else args.lr,
                "weight_decay": 0.1,
                "epochs": 1 if args.smoke else args.epochs,
                "max_steps": 1 if args.smoke else args.max_steps,
                "scheduler": "warmup_cosine",
                "warmup_ratio": 0.02,
                "min_lr_ratio": 0.1,
                "adam_betas": [0.9, 0.95],
                "condition_mode": "aligned",
                "condition_dropout": 0.1,
                "cond_dim": 8,
                "profile_max_tokens": 16 if args.smoke else 48,
                "max_turns": 2 if args.smoke else 4,
                "deduplicate": True,
                "source_weights": source_weights,
                "curriculum_ratio": 0.1,
                "curriculum_max_target_tokens": 48,
                "curriculum_max_turns": 2,
                "log_every": 1 if args.smoke else 20,
                "eval_every_steps": 1 if args.smoke else 50,
                "early_stop_patience": 0 if args.smoke else 5,
                "early_stop_min_delta": 0.005,
                "generation_eval_examples": 2 if args.smoke else 32,
                "generation_max_new_tokens": 4 if args.smoke else 48,
                "device": args.device,
                "num_workers": args.num_workers,
                "use_amp": bool(args.amp and not args.smoke),
                "allow_tf32": args.tf32,
                "fused_optimizer": args.fused_optimizer,
                "mlflow_enabled": False if args.smoke else True,
                "mlflow_experiment": "paper_small_lm",
                "output_dir": str(runs_dir.resolve()),
            }
            if args.smoke:
                cfg["arch_params"] = SMOKE_ARCH_PARAMS[arch]
            cfg_path = configs_dir / f"{run_id}.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=True))
            command = [
                str(PYTHON),
                "-m",
                "src.train.run_small_lm",
                "--config",
                str(cfg_path),
                "--run-id",
                run_id,
                "--arch",
                arch,
            ]
            print(f"[train] {arch} seed={seed}")
            result = subprocess.run(command, cwd=ROOT)
            if result.returncode != 0 or not summary_path.exists():
                failures.append(run_id)
                continue
            summaries.append(_load_summary(summary_path))

    outputs = build_report(summaries, experiment_dir, experiment_id=experiment_id)
    print(f"[report] {outputs['markdown']}")
    if failures:
        print("[failed] " + ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
