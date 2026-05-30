"""
Confidence-aware selective router.

Replaces the deterministic should_route_slow() with a calibrated,
threshold-based version that defaults to slow path when confidence is low.

Usage:
    from src.eval.selective_router import SelectiveRouter, should_route_slow_confident

    router = SelectiveRouter.from_config(cfg)
    pred_slow = router.should_route_slow(pred_D, pred_N, confidences)
"""
from pathlib import Path

import numpy as np
import torch


# Hard routing rules that are ALWAYS slow regardless of confidence
_HARD_TRIGGERS = {
    "value_conflict": {"strong"},
    "response_policy": {"threaten", "negotiate"},
}


class SelectiveRouter:
    """Router that uses confidence thresholds to decide when to abstain (slow path)."""

    DEFAULT_THRESHOLDS = {
        "response_policy": 0.65,
        "reveal_decision": 0.70,
        "value_conflict": 0.75,
        "secrecy_pressure": 0.75,
    }

    def __init__(self, thresholds: dict[str, float], calibration_dir: str | None = None):
        self.thresholds = {**self.DEFAULT_THRESHOLDS, **thresholds}
        self._calibrator = None
        if calibration_dir and Path(calibration_dir).exists():
            self._load_calibration(calibration_dir)

    @classmethod
    def from_config(cls, cfg: dict):
        router_cfg = cfg.get("selective_router", {})
        thresholds = router_cfg.get("thresholds", {})
        calib_dir = cfg.get("calibration", {}).get("output_dir", None)
        return cls(thresholds, calib_dir)

    def _load_calibration(self, calibration_dir: str):
        """Load temperature scaling parameters or isotonic models."""
        wrapper = Path(calibration_dir) / "apply_calibration.py"
        if wrapper.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("calib_wrapper", wrapper)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self._calibrator = mod

    def _calibrate_confidence(self, field: str, logits: torch.Tensor | np.ndarray) -> float:
        """Apply calibration if available, then return max softmax probability."""
        if isinstance(logits, torch.Tensor):
            logits = logits.detach().cpu().numpy()
        if self._calibrator is not None and hasattr(self._calibrator, "apply_calibration"):
            try:
                calibrated_logits = self._calibrator.apply_calibration(field, logits)
                if isinstance(calibrated_logits, torch.Tensor):
                    calibrated_logits = calibrated_logits.detach().cpu().numpy()
                logits = calibrated_logits
            except Exception:
                pass  # fall back to raw logits
        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()
        return float(np.max(probs))

    def should_route_slow(
        self,
        D_t: dict[str, str],
        N_t: dict[str, str],
        confidences: dict[str, float] | None = None,
        logits: dict[str, np.ndarray] | None = None,
    ) -> tuple[bool, dict]:
        """
        Returns (pred_slow, routing_log) where routing_log explains why.

        Priority order:
        1. Hard triggers (value_conflict=strong, response_policy in {threaten,negotiate})
        2. Predicted reveal != none  → always slow
        3. Low confidence on any routing head → slow (selective classification)
        4. Otherwise fast
        """
        log = {"rules_triggered": [], "confidences": {}}

        # 1. Hard triggers (deterministic, no confidence needed)
        if N_t.get("value_conflict") in _HARD_TRIGGERS["value_conflict"]:
            log["rules_triggered"].append("hard:value_conflict=strong")
            return True, log
        if D_t.get("response_policy") in _HARD_TRIGGERS["response_policy"]:
            log["rules_triggered"].append("hard:response_policy=threaten|negotiate")
            return True, log

        # 2. Predicted disclosure
        reveal = D_t.get("reveal_decision", "")
        if reveal in {"hint", "partial", "full"}:
            log["rules_triggered"].append(f"reveal={reveal}")
            return True, log

        # 3. Selective classification: low confidence → slow
        for field in ["response_policy", "reveal_decision", "value_conflict", "secrecy_pressure"]:
            val = D_t.get(field) or N_t.get(field)
            if val is None:
                continue

            conf = None
            if confidences and field in confidences:
                conf = confidences[field]
            elif logits and field in logits:
                conf = self._calibrate_confidence(field, logits[field])
            else:
                # No confidence available — be conservative
                conf = 1.0  # assume confident if we can't measure

            log["confidences"][field] = conf

            tau = self.thresholds.get(field, 0.5)
            if conf is not None and conf < tau:
                log["rules_triggered"].append(f"low_conf:{field}={conf:.3f}<{tau}")
                return True, log

        # 4. Fast path
        log["rules_triggered"].append("fast_path")
        return False, log


def should_route_slow_confident(
    D_t: dict[str, str],
    N_t: dict[str, str],
    confidences: dict[str, float] | None = None,
    logits: dict[str, np.ndarray] | None = None,
    thresholds: dict[str, float] | None = None,
) -> tuple[bool, dict]:
    """Convenience wrapper with default thresholds."""
    router = SelectiveRouter(thresholds or {})
    return router.should_route_slow(D_t, N_t, confidences, logits)
