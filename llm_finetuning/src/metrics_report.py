from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    hamming_loss,
    matthews_corrcoef,
)

from src.training.loss import GROUP_FIELDS


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def flatten_metrics(metrics: Mapping[str, Any], prefix: str = "") -> Dict[str, float]:
    flat: Dict[str, float] = {}
    for key, value in metrics.items():
        name = f"{prefix}{key}" if not prefix else f"{prefix}/{key}"
        if _is_numeric(value):
            flat[name] = float(value)
        elif isinstance(value, Mapping):
            flat.update(flatten_metrics(value, name))
    return flat


def compute_latent_metrics(
    all_preds: Mapping[str, list],
    all_golds: Mapping[str, list],
    secret_leakage_rate: float | None = None,
    extra_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    fields: dict[str, dict[str, float]] = {}

    for field, golds in all_golds.items():
        preds = all_preds.get(field, [])
        if not golds or not preds:
            continue

        if field == "dialogue_act":
            gold_bin = [list(map(int, row)) for row in golds]
            pred_bin = [list(map(int, row)) for row in preds]
            subset_acc = float(sum(g == p for g, p in zip(gold_bin, pred_bin)) / max(1, len(gold_bin)))
            fields[field] = {
                "micro_f1": float(f1_score(gold_bin, pred_bin, average="micro", zero_division=0)),
                "macro_f1": float(f1_score(gold_bin, pred_bin, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(gold_bin, pred_bin, average="weighted", zero_division=0)),
                "subset_accuracy": subset_acc,
                "accuracy": subset_acc,
                "hamming_loss": float(hamming_loss(gold_bin, pred_bin)),
                "support": float(len(gold_bin)),
            }
            continue

        fields[field] = {
            "accuracy": float(accuracy_score(golds, preds)),
            "balanced_accuracy": float(balanced_accuracy_score(golds, preds)),
            "cohen_kappa": float(cohen_kappa_score(golds, preds)),
            "macro_f1": float(f1_score(golds, preds, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(golds, preds, average="weighted", zero_division=0)),
            "mcc": float(matthews_corrcoef(golds, preds)),
            "support": float(len(golds)),
        }

    groups: dict[str, dict[str, float]] = {}
    for group, group_fields in GROUP_FIELDS.items():
        group_metrics = [fields.get(field) for field in group_fields if field in fields]
        if not group_metrics:
            continue

        acc_vals = [m["accuracy"] for m in group_metrics if "accuracy" in m]
        bal_acc_vals = [m["balanced_accuracy"] for m in group_metrics if "balanced_accuracy" in m]
        kappa_vals = [m["cohen_kappa"] for m in group_metrics if "cohen_kappa" in m]
        macro_f1_vals = [m["macro_f1"] for m in group_metrics if "macro_f1" in m]
        weighted_f1_vals = [m["weighted_f1"] for m in group_metrics if "weighted_f1" in m]
        mcc_vals = [m["mcc"] for m in group_metrics if "mcc" in m]
        groups[group] = {
            "mean_accuracy": float(sum(acc_vals) / len(acc_vals)) if acc_vals else 0.0,
            "mean_balanced_accuracy": float(sum(bal_acc_vals) / len(bal_acc_vals)) if bal_acc_vals else 0.0,
            "mean_cohen_kappa": float(sum(kappa_vals) / len(kappa_vals)) if kappa_vals else 0.0,
            "mean_macro_f1": float(sum(macro_f1_vals) / len(macro_f1_vals)) if macro_f1_vals else 0.0,
            "mean_weighted_f1": float(sum(weighted_f1_vals) / len(weighted_f1_vals)) if weighted_f1_vals else 0.0,
            "mean_mcc": float(sum(mcc_vals) / len(mcc_vals)) if mcc_vals else 0.0,
        }

    summary: dict[str, Any] = {}
    accuracy_vals = [m["accuracy"] for m in fields.values() if "accuracy" in m]
    balanced_accuracy_vals = [m["balanced_accuracy"] for m in fields.values() if "balanced_accuracy" in m]
    kappa_vals = [m["cohen_kappa"] for m in fields.values() if "cohen_kappa" in m]
    macro_f1_vals = [m["macro_f1"] for m in fields.values() if "macro_f1" in m]
    weighted_f1_vals = [m["weighted_f1"] for m in fields.values() if "weighted_f1" in m]
    mcc_vals = [m["mcc"] for m in fields.values() if "mcc" in m]
    summary["mean_accuracy"] = float(sum(accuracy_vals) / len(accuracy_vals)) if accuracy_vals else 0.0
    summary["mean_balanced_accuracy"] = (
        float(sum(balanced_accuracy_vals) / len(balanced_accuracy_vals)) if balanced_accuracy_vals else 0.0
    )
    summary["mean_cohen_kappa"] = float(sum(kappa_vals) / len(kappa_vals)) if kappa_vals else 0.0
    summary["mean_macro_f1"] = float(sum(macro_f1_vals) / len(macro_f1_vals)) if macro_f1_vals else 0.0
    summary["mean_weighted_f1"] = float(sum(weighted_f1_vals) / len(weighted_f1_vals)) if weighted_f1_vals else 0.0
    summary["mean_mcc"] = float(sum(mcc_vals) / len(mcc_vals)) if mcc_vals else 0.0

    if "response_policy" in fields:
        summary["response_policy_f1"] = fields["response_policy"].get("macro_f1", 0.0)
        summary["response_policy_accuracy"] = fields["response_policy"].get("accuracy", 0.0)
    if "reveal_decision" in fields:
        summary["reveal_decision_f1"] = fields["reveal_decision"].get("macro_f1", 0.0)
        summary["reveal_decision_accuracy"] = fields["reveal_decision"].get("accuracy", 0.0)

    stance_delta_f1_vals = [
        fields[field]["macro_f1"]
        for field in fields
        if field.endswith("_delta") and "macro_f1" in fields[field]
    ]
    stance_delta_acc_vals = [
        fields[field]["accuracy"]
        for field in fields
        if field.endswith("_delta") and "accuracy" in fields[field]
    ]
    if stance_delta_f1_vals:
        summary["stance_delta_f1"] = float(sum(stance_delta_f1_vals) / len(stance_delta_f1_vals))
    if stance_delta_acc_vals:
        summary["stance_delta_accuracy"] = float(sum(stance_delta_acc_vals) / len(stance_delta_acc_vals))
    if "trust_delta" in fields:
        summary["trust_delta_f1"] = fields["trust_delta"].get("macro_f1", 0.0)
        summary["trust_delta_accuracy"] = fields["trust_delta"].get("accuracy", 0.0)

    if secret_leakage_rate is not None:
        summary["secret_leakage_rate"] = float(secret_leakage_rate)

    if extra_summary:
        for key, value in extra_summary.items():
            if _is_numeric(value):
                summary[key] = float(value)
            else:
                summary[key] = value

    return {
        "fields": fields,
        "groups": groups,
        "summary": summary,
    }


def render_metrics_markdown(title: str, metrics: Mapping[str, Any]) -> str:
    lines: list[str] = [f"# {title}", ""]
    summary = metrics.get("summary", {})
    if isinstance(summary, Mapping) and summary:
        lines.append("## Summary")
        lines.append("| Metric | Value |")
        lines.append("|---|---:|")
        for key in sorted(summary):
            value = summary[key]
            if _is_numeric(value):
                lines.append(f"| `{key}` | {float(value):.4f} |")
            else:
                lines.append(f"| `{key}` | {value} |")
        lines.append("")

    groups = metrics.get("groups", {})
    if isinstance(groups, Mapping) and groups:
        lines.append("## Groups")
        lines.append("| Group | Mean Acc | Mean Macro F1 | Mean Weighted F1 |")
        lines.append("|---|---:|---:|---:|")
        for group in sorted(groups):
            row = groups[group]
            lines.append(
                f"| `{group}` | {row.get('mean_accuracy', 0.0):.4f} | "
                f"{row.get('mean_macro_f1', 0.0):.4f} | {row.get('mean_weighted_f1', 0.0):.4f} |"
            )
        lines.append("")

    fields = metrics.get("fields", {})
    if isinstance(fields, Mapping) and fields:
        lines.append("## Fields")
        lines.append("| Field | Metric | Value |")
        lines.append("|---|---|---:|")
        for field in sorted(fields):
            row = fields[field]
            for metric_name, value in row.items():
                if metric_name == "support":
                    continue
                if _is_numeric(value):
                    lines.append(f"| `{field}` | `{metric_name}` | {float(value):.4f} |")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def write_metrics_bundle(out_dir: Path, stem: str, metrics: Mapping[str, Any], title: str) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    md_path.write_text(render_metrics_markdown(title, metrics))
    return {"json": json_path, "md": md_path}


def log_metrics_to_mlflow(metrics: Mapping[str, Any], prefix: str = "", step: int | None = None) -> None:
    import mlflow

    flat = flatten_metrics(metrics, prefix=prefix)
    if flat:
        mlflow.log_metrics(flat, step=step)
