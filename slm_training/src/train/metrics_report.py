from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def flatten_metrics(metrics: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        name = f"{prefix}{key}" if not prefix else f"{prefix}/{key}"
        if _is_numeric(value):
            flat[name] = float(value)
        elif isinstance(value, Mapping):
            flat.update(flatten_metrics(value, name))
    return flat


def render_metrics_markdown(title: str, metrics: Mapping[str, Any]) -> str:
    lines = [f"# {title}", ""]

    def render_section(name: str, payload: Mapping[str, Any]) -> None:
        lines.append(f"## {name}")
        lines.append("| Key | Value |")
        lines.append("|---|---:|")
        for key in sorted(payload):
            value = payload[key]
            if isinstance(value, Mapping):
                continue
            if _is_numeric(value):
                lines.append(f"| `{key}` | {float(value):.4f} |")
            else:
                lines.append(f"| `{key}` | {value} |")
        lines.append("")

    for section in ("summary", "best", "data", "hyperparams", "model_stats", "embedding"):
        payload = metrics.get(section)
        if isinstance(payload, Mapping) and payload:
            render_section(section.capitalize(), payload)

    epochs = metrics.get("epochs")
    if isinstance(epochs, list) and epochs:
        lines.append("## Epochs")
        lines.append("| epoch | val_loss | val_ppl |")
        lines.append("|---:|---:|---:|")
        for row in epochs:
            lines.append(
                f"| {row.get('epoch', '')} | {row.get('val_loss', '')} | {row.get('val_ppl', '')} |"
            )
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def write_metrics_bundle(out_dir: Path, stem: str, metrics: Mapping[str, Any], title: str) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    md_path.write_text(render_metrics_markdown(title, metrics))
    return {"json": json_path, "md": md_path}


def log_metrics_to_mlflow(tracker, metrics: Mapping[str, Any], prefix: str = "", step: int | None = None) -> None:
    flat = flatten_metrics(metrics, prefix=prefix)
    if flat:
        tracker.log_metrics(flat, step=step)
