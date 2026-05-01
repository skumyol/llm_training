"""
Evaluation pipeline entry point.

Usage:
    python run_eval.py --stage latent   --config configs/eval.yaml
    python run_eval.py --stage response --config configs/eval.yaml
    python run_eval.py --stage routing  --config configs/eval.yaml
    python run_eval.py --stage all      --config configs/eval.yaml
"""
import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["latent", "response", "routing", "all"], default="all")
    p.add_argument("--config", default="configs/eval.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    results: dict[str, dict] = {}

    if args.stage in ("latent", "all"):
        from src.eval.eval_latent import eval_latent
        results["latent"] = eval_latent(args.config)

    if args.stage in ("response", "all"):
        from src.eval.eval_response import eval_response
        results["response"] = eval_response(args.config)

    if args.stage in ("routing", "all"):
        from src.eval.eval_routing import eval_routing
        results["routing"] = eval_routing(args.config)

    if results:
        with open(args.config) as f:
            import yaml

            cfg = yaml.safe_load(f)
        results_dir = Path(cfg["output"]["results_dir"])
        results_dir.mkdir(parents=True, exist_ok=True)
        summary_path = results_dir / "evaluation_summary.json"
        summary_path.write_text(json.dumps(results, indent=2, sort_keys=True))
        print(f"Saved aggregated evaluation summary to {summary_path}")


if __name__ == "__main__":
    main()
