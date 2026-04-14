#!/usr/bin/env python3
"""
chat_gemma.py — Terminal chat with Gemma Unsloth models
======================================================
Simple REPL for chatting with fine-tuned Gemma models.

Usage:
  python -m src.infer.chat_gemma --model-dir artifacts/gemma_unsloth/<run_id>/best_model
  python -m src.infer.chat_gemma --base-model models/unsloth_gemma-3-4b-it  # Base model
"""
from __future__ import annotations

import argparse
import readline
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Chat with Gemma model")
    p.add_argument("--model-dir", type=str, default=None,
                   help="Path to fine-tuned model (auto-discovers latest if omitted)")
    p.add_argument("--base-model", type=str, default="unsloth/gemma-3-4b-it",
                   help="Base model name/path if no fine-tuned model found")
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=256)
    args = p.parse_args()

    # Auto-discover latest gemma model
    model_path = args.model_dir
    if model_path is None:
        candidates = sorted(Path("artifacts/gemma_unsloth").glob("*/best_model"))
        if candidates:
            model_path = str(candidates[-1])
            print(f"Using model: {model_path}")
        else:
            model_path = args.base_model
            print(f"No fine-tuned model found, using base: {model_path}")

    print("Loading model... (this may take a moment)")

    # Import here to avoid slow load on --help
    import torch
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        load_in_4bit=True,
        device_map=device,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")

    print("\n" + "="*60)
    print("  Gemma Chat Ready!")
    print("="*60)
    print(f"Model: {model_path}")
    print("Commands: /system <prompt>, /temp <float>, /clear, /quit")
    print("-"*60 + "\n")

    # Chat state
    system_prompt = "You are a helpful AI assistant."
    messages = []
    temperature = args.temp
    max_tokens = args.max_tokens

    while True:
        try:
            user_input = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Commands
        if user_input.startswith("/"):
            parts = user_input.split(None, 1)
            cmd = parts[0].lower()

            if cmd in ("/quit", "/exit"):
                print("Goodbye!")
                break
            elif cmd == "/clear":
                messages = []
                print("Conversation cleared.")
                continue
            elif cmd == "/system":
                system_prompt = parts[1] if len(parts) > 1 else "You are a helpful AI assistant."
                print(f"System prompt updated: {system_prompt[:60]}...")
                continue
            elif cmd == "/temp":
                try:
                    temperature = float(parts[1])
                    print(f"Temperature set to {temperature}")
                except (ValueError, IndexError):
                    print(f"Current temperature: {temperature}")
                continue
            elif cmd == "/help":
                print("Commands: /system <prompt>, /temp <float>, /clear, /quit")
                continue
            else:
                print(f"Unknown command: {cmd}")
                continue

        # Build messages
        msgs = [{"role": "system", "content": system_prompt}]
        msgs.extend(messages)
        msgs.append({"role": "user", "content": user_input})

        # Generate
        inputs = tokenizer.apply_chat_template(
            msgs,
            tokenize=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
            )

        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        print(f"\ngemma › {response}\n")

        # Update history
        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": response})

        # Trim history to prevent OOM
        if len(messages) > 20:
            messages = messages[-20:]


if __name__ == "__main__":
    main()
