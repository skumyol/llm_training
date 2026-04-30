#!/usr/bin/env python3
"""
query_results.py — Query remote MLflow and generate comparison tables
=====================================================================
Connects to the MLflow tracking server (local or remote) and produces
comparison tables across experiments and runs.

Usage:
  # List all experiments and their latest metrics
  python scripts/query_results.py --list

  # Compare all runs in an experiment (table format)
  python scripts/query_results.py --experiment small_lm

  # Compare SLM architectures by best PPL
  python scripts/query_results.py --experiment small_lm --metric val_ppl --mode min

  # Compare LLM evaluation results
  python scripts/query_results.py --experiment routing_and_policy_eval --csv results.csv

  # Show full run history for a specific run
  python scripts/query_results.py --run-id abc123def --history

  # Show best checkpoint paths for download
  python scripts/query_results.py --experiment small_lm --artifacts
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BOLD = "\033[1m"
DIM  = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED  = "\033[31m"
RESET = "\033[0m"


def _get_client():
    """Connect to MLflow tracking server (local or remote)."""
    import mlflow
    from mlflow.tracking import MlflowClient

    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    return MlflowClient()


def list_experiments(client) -> List[Dict]:
    """List all experiments with run counts."""
    rows = []
    for exp in client.search_experiments():
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        rows.append({
            "id": exp.experiment_id,
            "name": exp.name,
            "runs": len(runs),
            "lifecycle": exp.lifecycle_stage,
        })
    return rows


def get_runs(client, experiment_name: str, filter_str: str = "") -> List[Any]:
    """Get all runs for an experiment."""
    exp = client.get_experiment_by_name(experiment_name)
    if not exp:
        print(f"{RED}Experiment '{experiment_name}' not found.{RESET}")
        return []
    return client.search_runs(experiment_ids=[exp.experiment_id], filter_string=filter_str)


def compare_runs(client, experiment_name: str, metric: str,
                 mode: str = "min", top_n: int = 10) -> List[Dict]:
    """Compare runs by a specific metric (e.g., val_ppl, val_ccc)."""
    runs = get_runs(client, experiment_name)

    rows = []
    for run in runs:
        if run.info.status != "FINISHED":
            continue
        metrics = run.data.metrics
        params = run.data.params
        tags = run.data.tags

        # Find the target metric (use min or max depending on mode)
        candidate_keys = [k for k in metrics if metric in k]
        if not candidate_keys:
            continue

        if mode == "min":
            best_val = min(metrics[k] for k in candidate_keys)
        else:
            best_val = max(metrics[k] for k in candidate_keys)

        rows.append({
            "run_id": run.info.run_id[:8],
            "run_name": run.data.tags.get("mlflow.runName", "?"),
            "status": run.info.status,
            "metric_name": metric,
            "metric_value": round(best_val, 4),
            "seed": tags.get("seed", "?"),
            "arch": tags.get("arch", "?"),
            "lr": params.get("lr", "?"),
            "epochs": params.get("epochs", "?"),
            "started": datetime.fromtimestamp(run.info.start_time / 1000).strftime("%Y-%m-%d %H:%M"),
            "artifact_uri": run.info.artifact_uri,
        })

    # Sort
    reverse = mode == "max"
    rows.sort(key=lambda r: r["metric_value"], reverse=reverse)

    return rows[:top_n]


def show_run_history(client, run_id: str) -> None:
    """Show metric history for a specific run."""
    from mlflow.entities import ViewType

    # Search by run_id prefix
    runs = client.search_runs(
        experiment_ids=[e.experiment_id for e in client.search_experiments()],
        filter_string=f"run_id LIKE '{run_id}%'",
        run_view_type=ViewType.ALL,
    )
    if not runs:
        print(f"{RED}Run '{run_id}' not found.{RESET}")
        return

    run = runs[0]
    print(f"\n{BOLD}{CYAN}Run: {run.data.tags.get('mlflow.runName', '?')}{RESET}")
    print(f"  ID:       {run.info.run_id}")
    print(f"  Status:   {run.info.status}")
    print(f"  Start:    {datetime.fromtimestamp(run.info.start_time / 1000)}")
    if run.info.end_time:
        print(f"  End:      {datetime.fromtimestamp(run.info.end_time / 1000)}")
    print(f"\n{BOLD}Parameters:{RESET}")
    for k, v in sorted(run.data.params.items()):
        print(f"  {k:30s} = {v}")

    print(f"\n{BOLD}Metrics:{RESET}")
    for k, v in sorted(run.data.metrics.items()):
        print(f"  {k:30s} = {v}")

    print(f"\n{BOLD}Tags:{RESET}")
    for k, v in sorted(run.data.tags.items()):
        if not k.startswith("mlflow."):
            print(f"  {k:30s} = {v}")

    print(f"\n{BOLD}Artifacts:{RESET}  {run.info.artifact_uri}")


def show_artifacts(client, experiment_name: str) -> None:
    """List artifact URIs for all runs in an experiment."""
    runs = get_runs(client, experiment_name)
    print(f"\n{BOLD}{CYAN}Artifacts for {experiment_name}:{RESET}\n")
    for run in runs:
        if run.info.status != "FINISHED":
            continue
        name = run.data.tags.get("mlflow.runName", "?")
        best_metric = ""
        for k, v in run.data.metrics.items():
            if "best" in k:
                best_metric += f" {k}={v}"
        print(f"  {run.info.run_id[:8]}  {DIM}{name:40s}{RESET}  {best_metric}")
        print(f"    {DIM}{run.info.artifact_uri}{RESET}")


def print_table(rows: List[Dict], title: str = "") -> None:
    """Print a formatted comparison table."""
    if not rows:
        print(f"{YELLOW}No finished runs found.{RESET}")
        return

    keys = list(rows[0].keys())
    col_widths = {k: max(len(k), max(len(str(r.get(k, ""))) for r in rows)) for k in keys}
    total_width = sum(col_widths.values()) + len(keys) * 3 + 1

    if title:
        print(f"\n{BOLD}{CYAN}{title}{RESET}")

    # Header
    header = " │ ".join(f"{k:{col_widths[k]}}" for k in keys)
    print(f"  {BOLD}{header}{RESET}")
    print(f"  {'─' * (total_width - 2)}")

    # Rows
    for r in rows:
        line = " │ ".join(f"{str(r.get(k, '')):{col_widths[k]}}" for k in keys)
        # Highlight best
        if r.get("metric_value") and rows and r["metric_value"] == rows[0]["metric_value"]:
            line = f"{GREEN}{line}{RESET}"
        print(f"  {line}")

    print(f"\n  {DIM}{len(rows)} runs{RESET}")


def export_csv(rows: List[Dict], path: str) -> None:
    """Export results to CSV."""
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{GREEN}Exported {len(rows)} rows → {path}{RESET}")


def main():
    p = argparse.ArgumentParser(description="Query MLflow experiments and compare results")
    p.add_argument("--list", action="store_true", help="List all experiments")
    p.add_argument("--experiment", type=str, help="Experiment name to query")
    p.add_argument("--metric", type=str, default="val_loss",
                   help="Metric to compare (e.g., val_ppl, val_ccc, val_f1)")
    p.add_argument("--mode", choices=["min", "max"], default="min",
                   help="Optimization direction (min for loss, max for accuracy)")
    p.add_argument("--top", type=int, default=10, help="Number of top runs to show")
    p.add_argument("--run-id", type=str, help="Show detailed history for a run")
    p.add_argument("--history", action="store_true", help="Show metric history")
    p.add_argument("--artifacts", action="store_true", help="List artifact URIs")
    p.add_argument("--csv", type=str, help="Export results to CSV")
    p.add_argument("--tracking-uri", type=str,
                   help="MLflow tracking URI (overrides MLFLOW_TRACKING_URI env)")
    args = p.parse_args()

    if args.tracking_uri:
        os.environ["MLFLOW_TRACKING_URI"] = args.tracking_uri

    client = _get_client()

    if args.list:
        rows = list_experiments(client)
        print_table(rows, "All Experiments")
        return

    if args.run_id:
        show_run_history(client, args.run_id)
        return

    if args.artifacts and args.experiment:
        show_artifacts(client, args.experiment)
        return

    if args.experiment:
        rows = compare_runs(client, args.experiment, args.metric, args.mode, args.top)
        title = f"Experiment: {args.experiment}  (sorted by {args.metric}, {args.mode})"
        print_table(rows, title)

        if args.csv:
            export_csv(rows, args.csv)
        return

    p.print_help()


if __name__ == "__main__":
    main()
