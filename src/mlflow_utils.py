import json
import os
from pathlib import Path

import mlflow


def setup_mlflow(tracking_uri: str = "mlruns") -> None:
    mlflow.set_tracking_uri(tracking_uri)


def log_config(cfg: dict) -> None:
    flat = _flatten(cfg)
    safe = {k: str(v) for k, v in flat.items() if v is not None}
    mlflow.log_params(safe)


def log_dataset_manifest(manifest: dict, artifact_name: str = "data_manifest.json") -> None:
    mlflow.log_dict(manifest, artifact_name)


def log_schema(schema_path: str) -> None:
    with open(schema_path) as f:
        schema = json.load(f)
    mlflow.log_dict(schema, "schema.json")


def _flatten(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items: dict = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten(v, new_key, sep))
        elif isinstance(v, list):
            items[new_key] = json.dumps(v)
        else:
            items[new_key] = v
    return items
