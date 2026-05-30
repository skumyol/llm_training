#!/usr/bin/env python3
"""
Calibrate classifier heads using temperature scaling or isotonic regression.

Usage:
    # Temperature scaling (single parameter per head)
    PYTHONPATH=. python scripts/calibrate_head.py \
        --config configs/eval.yaml \
        --method temperature \
        --calib-heads-file data/splits/train_heads.jsonl \
        --output-dir calibrators/temperature

    # Isotonic regression (non-parametric, more flexible)
    PYTHONPATH=. python scripts/calibrate_head.py \
        --config configs/eval.yaml \
        --method isotonic \
        --calib-heads-file data/splits/train_heads.jsonl \
        --output-dir calibrators/isotonic
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.isotonic import IsotonicRegression
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.model import load_predictor
from src.training.dataset import HeadSupervisionDataset, collate_head_batch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/eval.yaml")
    p.add_argument("--method", choices=["temperature", "isotonic"], default="temperature",
                   help="Calibration method")
    p.add_argument("--calib-heads-file", required=True,
                   help="Held-out calibration split (NOT test)")
    p.add_argument("--output-dir", default="calibrators")
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args()


def _collect_logits_and_labels(predictor, tokenizer, data_path: str, max_seq_len: int, batch_size: int, device):
    """Run predictor on calibration split and return per-field (logits, labels)."""
    ds = HeadSupervisionDataset(data_path, tokenizer, max_seq_len=max_seq_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_head_batch, num_workers=0)

    per_field: dict[str, dict] = {}
    predictor.eval()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Collecting calibration samples"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = predictor(input_ids=input_ids, attention_mask=attention_mask)

            for field, logits in out["logits"].items():
                if field not in per_field:
                    per_field[field] = {"logits": [], "labels": []}
                label_key = f"label_{field}"
                if label_key not in batch:
                    continue
                gold = batch[label_key]
                if field == "dialogue_act":
                    continue  # skip multi-label for calibration simplicity
                if not isinstance(gold, torch.Tensor):
                    continue
                gold = gold.to(device)
                valid = gold != -1
                if not valid.any():
                    continue
                per_field[field]["logits"].append(logits[valid].cpu().numpy())
                per_field[field]["labels"].append(gold[valid].cpu().numpy())

    # Concatenate
    for field in per_field:
        if per_field[field]["logits"]:
            per_field[field]["logits"] = np.concatenate(per_field[field]["logits"], axis=0)
            per_field[field]["labels"] = np.concatenate(per_field[field]["labels"], axis=0)
        else:
            per_field[field]["logits"] = np.array([])
            per_field[field]["labels"] = np.array([])

    return per_field


def _temperature_scale(logits: np.ndarray, labels: np.ndarray, max_iter: int = 50) -> float:
    """Learn a single temperature parameter via gradient-free search."""
    if len(labels) == 0:
        return 1.0

    def nll(T: float) -> float:
        scaled = logits / T
        probs = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        log_probs = np.log(probs[np.arange(len(labels)), labels] + 1e-12)
        return -log_probs.mean()

    # Golden section search on T in [0.1, 10.0]
    lo, hi = 0.1, 10.0
    gr = (np.sqrt(5) + 1) / 2
    a = hi - (hi - lo) / gr
    b = lo + (hi - lo) / gr
    for _ in range(max_iter):
        if nll(a) < nll(b):
            hi = b
        else:
            lo = a
        a = hi - (hi - lo) / gr
        b = lo + (hi - lo) / gr
    return (lo + hi) / 2


def _fit_isotonic(confidences: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    """Fit isotonic regression mapping uncalibrated confidence to calibrated probability."""
    if len(labels) == 0:
        return None
    # Confidences and labels must be 1D
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(confidences, labels)
    return ir


def calibrate(config_path: str, method: str, calib_heads_file: str, output_dir: str, max_seq_len: int, batch_size: int) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    checkpoint = cfg["latent_predictor_checkpoint"]
    model_name = cfg.get("base_model", "Qwen/Qwen3-4B")
    quantization = cfg.get("quantization", "4bit")
    torch_dtype = cfg.get("torch_dtype", "bfloat16")

    print(f"Loading predictor from {checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor, tokenizer = load_predictor(checkpoint, model_name, quantization=quantization, torch_dtype=torch_dtype)
    predictor.to(device)

    print(f"Collecting calibration samples from {calib_heads_file}")
    per_field = _collect_logits_and_labels(predictor, tokenizer, calib_heads_file, max_seq_len, batch_size, device)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    calibrated = {}
    for field, data in per_field.items():
        logits = data["logits"]
        labels = data["labels"]
        if len(labels) == 0:
            print(f"[SKIP] {field}: no valid labels")
            continue

        if method == "temperature":
            T = _temperature_scale(logits, labels)
            # Verify ECE before/after
            conf_before = np.max(np.exp(logits - np.max(logits, axis=1, keepdims=True)) /
                           np.sum(np.exp(logits - np.max(logits, axis=1, keepdims=True)), axis=1), axis=1)
            calibrated_logits = logits / T
            conf_after = np.max(np.exp(calibrated_logits - np.max(calibrated_logits, axis=1, keepdims=True)) /
                          np.sum(np.exp(calibrated_logits - np.max(calibrated_logits, axis=1, keepdims=True)), axis=1), axis=1)
            ece_before = _ece(conf_before, labels)
            ece_after = _ece(conf_after, labels)
            calibrated[field] = {
                "method": "temperature",
                "temperature": float(T),
                "ece_before": float(ece_before),
                "ece_after": float(ece_after),
                "n_samples": int(len(labels)),
            }
            print(f"  {field:25s} T={T:.3f}  ECE: {ece_before:.3f} → {ece_after:.3f}")

        elif method == "isotonic":
            confidences = np.max(np.exp(logits - np.max(logits, axis=1, keepdims=True)) /
                                 np.sum(np.exp(logits - np.max(logits, axis=1, keepdims=True)), axis=1), axis=1)
            ir = _fit_isotonic(confidences, labels)
            if ir is None:
                continue
            calibrated_probs = ir.predict(confidences)
            ece_before = _ece(confidences, labels)
            ece_after = _ece(calibrated_probs, labels)
            calibrated[field] = {
                "method": "isotonic",
                "ece_before": float(ece_before),
                "ece_after": float(ece_after),
                "n_samples": int(len(labels)),
            }
            # Save sklearn model
            with open(out_dir / f"{field}_isotonic.pkl", "wb") as f:
                pickle.dump(ir, f)
            print(f"  {field:25s}  ECE: {ece_before:.3f} → {ece_after:.3f}")

    summary = {
        "method": method,
        "fields": calibrated,
        "output_dir": str(out_dir),
    }
    with open(out_dir / "calibration_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Write a lightweight calibration wrapper module for inference
    wrapper_path = out_dir / "apply_calibration.py"
    if method == "temperature":
        _write_temperature_wrapper(wrapper_path, calibrated)
    else:
        _write_isotonic_wrapper(wrapper_path, out_dir, calibrated)

    print(f"\nSaved calibration artifacts to {out_dir}")
    return summary


def _ece(confidences: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lower, upper = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if mask.sum() == 0:
            continue
        acc = labels[mask].mean()
        conf = confidences[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def _write_temperature_wrapper(path: Path, calibrated: dict):
    lines = [
        "# Auto-generated calibration wrapper (temperature scaling)\n",
        "import numpy as np\n",
        "import torch\n",
        "\n",
        "TEMPERATURES = {\n",
    ]
    for field, m in calibrated.items():
        lines.append(f'    "{field}": {m["temperature"]:.6f},\n')
    lines.append("}\n\n")
    lines.append("def apply_calibration(field: str, logits: np.ndarray | torch.Tensor):\n")
    lines.append('    T = TEMPERATURES.get(field, 1.0)\n')
    lines.append("    if isinstance(logits, torch.Tensor):\n")
    lines.append("        return logits / T\n")
    lines.append("    return logits / T\n")
    with open(path, "w") as f:
        f.writelines(lines)


def _write_isotonic_wrapper(path: Path, out_dir: Path, calibrated: dict):
    lines = [
        "# Auto-generated calibration wrapper (isotonic regression)\n",
        "import pickle\n",
        "from pathlib import Path\n",
        "import numpy as np\n",
        "\n",
        "def _softmax(logits):\n",
        "    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))\n",
        "    return exp / exp.sum(axis=-1, keepdims=True)\n",
        "\n",
        "_models = {}\n",
        'def load_isotonic_model(field: str, calib_dir: str = "calibrators/isotonic"):\n',
        "    if field in _models:\n",
        "        return _models[field]\n",
        '    pkl_path = Path(calib_dir) / f"{field}_isotonic.pkl"\n',
        "    if not pkl_path.exists():\n",
        "        return None\n",
        '    with open(pkl_path, "rb") as f:\n',
        "        model = pickle.load(f)\n",
        "    _models[field] = model\n",
        "    return model\n",
        "\n",
        "def apply_calibration(field: str, logits: np.ndarray):\n",
        "    probs = _softmax(logits)\n",
        "    conf = probs.max(axis=-1)\n",
        "    model = load_isotonic_model(field)\n",
        "    if model is None:\n",
        "        return logits\n",
        "    calibrated_conf = model.predict(conf)\n",
        "    # Rescale: keep argmax, adjust confidences proportionally\n",
        "    pred_class = probs.argmax(axis=-1)\n",
        "    new_probs = np.zeros_like(probs)\n",
        "    for i in range(len(probs)):\n",
        "        new_probs[i, pred_class[i]] = calibrated_conf[i]\n",
        "        # Distribute remainder uniformly over other classes\n",
        "        remainder = 1.0 - calibrated_conf[i]\n",
        "        other = [j for j in range(probs.shape[1]) if j != pred_class[i]]\n",
        "        if other:\n",
        "            new_probs[i, other] = remainder / len(other)\n",
        "    # Convert back to logits (log odds)\n",
        "    new_logits = np.log(np.clip(new_probs, 1e-12, 1.0))\n",
        "    return new_logits\n",
    ]
    with open(path, "w") as f:
        f.writelines(lines)


def main():
    args = parse_args()
    calibrate(args.config, args.method, args.calib_heads_file, args.output_dir, args.max_seq_len, args.batch_size)


if __name__ == "__main__":
    main()
