"""
mlflow_tracker.py
=================
Shared MLflow experiment-tracking helpers for all NPC training scripts.

Usage in any training script::

    from mlflow_tracker import MLflowTracker

    tracker = MLflowTracker(experiment="personality_encoder")
    tracker.start_run(run_name="report_p_s42", tags={"seed": "42"})
    tracker.log_params(cfg)
    ...
    tracker.log_metrics({"val_f1": 0.68, "val_mse": 0.05}, step=epoch)
    tracker.log_artifact(out_dir / "run_summary.json")
    tracker.end_run()

If MLflow is not installed, all calls are silent no-ops (safe to keep in code).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

log = logging.getLogger(__name__)

try:
    import mlflow
    import mlflow.pytorch
    HAS_MLFLOW = True
except ImportError:
    mlflow = None  # type: ignore[assignment]
    HAS_MLFLOW = False


# Default tracking URI: local folder in project root
_DEFAULT_TRACKING_URI = "file://" + str(
    Path(__file__).resolve().parent.parent.parent / "mlruns"
)


class MLflowTracker:
    """Thin wrapper around MLflow that degrades to no-ops if MLflow is missing."""

    def __init__(
        self,
        experiment: str = "npc_training",
        tracking_uri: Optional[str] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled and HAS_MLFLOW
        self.experiment_name = experiment
        self._run = None

        if not self.enabled:
            if not HAS_MLFLOW:
                log.info("MLflow not installed – tracking disabled.")
            return

        uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", _DEFAULT_TRACKING_URI)
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        log.info(f"MLflow tracking: experiment={experiment!r}  uri={uri}")

    # ── Run lifecycle ──────────────────────────────────────────────────────────

    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        nested: bool = False,
    ) -> None:
        if not self.enabled:
            return
        self._run = mlflow.start_run(run_name=run_name, nested=nested)
        if tags:
            mlflow.set_tags(tags)
        log.info(f"MLflow run started: {run_name}  (id={self._run.info.run_id[:8]})")

    def end_run(self, status: str = "FINISHED") -> None:
        if not self.enabled or self._run is None:
            return
        mlflow.end_run(status)
        log.info(f"MLflow run ended: {status}")
        self._run = None

    # ── Logging ────────────────────────────────────────────────────────────────

    def log_params(self, params: Dict[str, Any], prefix: str = "") -> None:
        """Log hyperparameters. Handles nested dicts and long values."""
        if not self.enabled:
            return
        flat = {}
        for k, v in params.items():
            key = f"{prefix}{k}" if prefix else k
            if isinstance(v, dict):
                # Flatten nested dicts
                for k2, v2 in v.items():
                    flat[f"{key}.{k2}"] = str(v2)[:250]
            elif isinstance(v, (list, tuple)):
                flat[key] = str(v)[:250]
            else:
                flat[key] = str(v)[:250]
        # MLflow has a 100-param batch limit
        items = list(flat.items())
        for i in range(0, len(items), 90):
            batch = dict(items[i:i + 90])
            try:
                mlflow.log_params(batch)
            except Exception as e:
                log.warning(f"MLflow log_params error: {e}")

    def log_metrics(
        self,
        metrics: Dict[str, Union[float, int]],
        step: Optional[int] = None,
    ) -> None:
        """Log metrics at a given step (epoch or global_step)."""
        if not self.enabled:
            return
        clean = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and v == v:  # skip NaN
                clean[k] = v
        try:
            mlflow.log_metrics(clean, step=step)
        except Exception as e:
            log.warning(f"MLflow log_metrics error: {e}")

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        if not self.enabled:
            return
        try:
            mlflow.log_metric(key, value, step=step)
        except Exception as e:
            log.warning(f"MLflow log_metric error: {e}")

    def log_artifact(self, path: Union[str, Path]) -> None:
        """Log a file artifact (model checkpoint, config, summary, etc.)."""
        if not self.enabled:
            return
        path = Path(path)
        if not path.exists():
            return
        try:
            mlflow.log_artifact(str(path))
        except Exception as e:
            log.warning(f"MLflow log_artifact error: {e}")

    def log_artifacts(self, directory: Union[str, Path]) -> None:
        """Log all files in a directory as artifacts."""
        if not self.enabled:
            return
        directory = Path(directory)
        if not directory.is_dir():
            return
        try:
            mlflow.log_artifacts(str(directory))
        except Exception as e:
            log.warning(f"MLflow log_artifacts error: {e}")

    def set_tag(self, key: str, value: str) -> None:
        if not self.enabled:
            return
        try:
            mlflow.set_tag(key, value)
        except Exception:
            pass

    # ── Model logging ──────────────────────────────────────────────────────────

    def log_model(self, model, artifact_path: str = "model") -> None:
        """Log a PyTorch model."""
        if not self.enabled:
            return
        try:
            mlflow.pytorch.log_model(model, artifact_path)
        except Exception as e:
            log.warning(f"MLflow log_model error: {e}")

    # ── Context manager ────────────────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "FINISHED" if exc_type is None else "FAILED"
        self.end_run(status)
        return False
