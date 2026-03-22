"""
Evaluation pipeline entry point.

Usage:
    python run_eval.py --stage latent   --config configs/eval.yaml
    python run_eval.py --stage response --config configs/eval.yaml
    python run_eval.py --stage routing  --config configs/eval.yaml
    python run_eval.py --stage all      --config configs/eval.yaml
"""
import argparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["latent", "response", "routing", "all"], default="all")
    p.add_argument("--config", default="configs/eval.yaml")
    return p.parse_args()


def main():
    args = parse_args()

    if args.stage in ("latent", "all"):
        from src.eval.eval_latent import eval_latent
        eval_latent(args.config)

    if args.stage in ("response", "all"):
        from src.eval.eval_response import eval_response
        eval_response(args.config)

    if args.stage in ("routing", "all"):
        from src.eval.eval_routing import eval_routing
        eval_routing(args.config)


if __name__ == "__main__":
    main()
