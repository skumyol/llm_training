# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "fastapi>=0.100",
#     "gradio>=5.0",
#     "scikit-learn>=1.3",
#     "uvicorn>=0.23",
# ]
# ///
"""
Dual-mode launcher: serves production app at '/' and test-mode app at '/test'.

Usage (local):
    uv run launcher.py --data audit_input_clean.jsonl --output ./audit_results --port 8765

Then open:
    http://127.0.0.1:8765/      ← production (timer enforced, selections required)
    http://127.0.0.1:8765/test  ← test mode (no timer, selections optional)

Docker:
    The Dockerfile CMD targets launcher.py instead of human_audit_app.py.
"""

import argparse
import sys

from fastapi import FastAPI
import gradio as gr
import uvicorn

from human_audit_app import (
    _DEFAULT_DATA_PATH,
    SAMPLE_SIZE,
    build_interface,
)


def main():
    parser = argparse.ArgumentParser(description="Dual-mode Gradio launcher")
    parser.add_argument("--data", default=None, help="Path to test_heads.jsonl")
    parser.add_argument("--output", default="./audit_results", help="Directory to save annotations")
    parser.add_argument("--host", default="0.0.0.0", help="Server bind address")
    parser.add_argument("--port", type=int, default=8765, help="Server port")
    parser.add_argument("--sample-size", type=int, default=150, help="Number of turns")
    args = parser.parse_args()

    global _DEFAULT_DATA_PATH, SAMPLE_SIZE
    _DEFAULT_DATA_PATH = args.data
    SAMPLE_SIZE = args.sample_size

    # Build production app
    production_demo = build_interface(args.data, args.output, test_mode=False)

    # Build test-mode app
    test_demo = build_interface(args.data, args.output, test_mode=True)

    app = FastAPI()
    app = gr.mount_gradio_app(app, test_demo, path="/test", root_path="/test")
    app = gr.mount_gradio_app(app, production_demo, path="/", root_path="")

    print(f"Starting dual-mode server on http://{args.host}:{args.port}/")
    print(f"  Production: http://{args.host}:{args.port}/")
    print(f"  Test mode:  http://{args.host}:{args.port}/test")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        sys.exit(0)
