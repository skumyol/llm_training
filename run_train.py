"""
Training pipeline entry point.

Usage:
    python run_train.py --stage latent   --config configs/train_latent.yaml
    python run_train.py --stage response --config configs/train_response.yaml
    python run_train.py --stage joint    --config configs/train_joint.yaml
"""
import argparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["latent", "response", "joint"], required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--debug", action="store_true",
                   help="Use debug_model (smaller/faster) from config")
    return p.parse_args()


def main():
    args = parse_args()

    if args.stage == "latent":
        from src.training.train_latent import train_latent
        train_latent(args.config, debug=args.debug)

    elif args.stage == "response":
        from src.training.train_response import train_response
        train_response(args.config, debug=args.debug)

    elif args.stage == "joint":
        from src.training.train_joint import train_joint
        train_joint(args.config, debug=args.debug)


if __name__ == "__main__":
    main()
