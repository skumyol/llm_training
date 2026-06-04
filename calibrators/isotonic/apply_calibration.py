# Auto-generated calibration wrapper (isotonic regression)
import pickle
from pathlib import Path
import numpy as np

def _softmax(logits):
    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)

_models = {}
def load_isotonic_model(field: str, calib_dir: str = "calibrators/isotonic"):
    if field in _models:
        return _models[field]
    pkl_path = Path(calib_dir) / f"{field}_isotonic.pkl"
    if not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as f:
        model = pickle.load(f)
    _models[field] = model
    return model

def apply_calibration(field: str, logits: np.ndarray):
    probs = _softmax(logits)
    conf = probs.max(axis=-1)
    model = load_isotonic_model(field)
    if model is None:
        return logits
    calibrated_conf = model.predict(conf)
    # Rescale: keep argmax, adjust confidences proportionally
    pred_class = probs.argmax(axis=-1)
    new_probs = np.zeros_like(probs)
    for i in range(len(probs)):
        new_probs[i, pred_class[i]] = calibrated_conf[i]
        # Distribute remainder uniformly over other classes
        remainder = 1.0 - calibrated_conf[i]
        other = [j for j in range(probs.shape[1]) if j != pred_class[i]]
        if other:
            new_probs[i, other] = remainder / len(other)
    # Convert back to logits (log odds)
    new_logits = np.log(np.clip(new_probs, 1e-12, 1.0))
    return new_logits
