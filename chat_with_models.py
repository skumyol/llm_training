#!/usr/bin/env python3
"""
Chat interface for SLM and LLM models
Usage: python chat_with_models.py
"""
import torch
import json
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "slm_training" / "src"))

from train.small_lm_architectures import build_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import tiktoken

def list_available_models():
    """List all trained models."""
    print("\n" + "="*60)
    print("AVAILABLE MODELS")
    print("="*60)
    
    # SLM models
    slm_path = Path("slm_training/artifacts/small_lm")
    if slm_path.exists():
        print("\n📦 SMALL LMs (Custom Architectures):")
        for model_dir in sorted(slm_path.glob("final_*")):
            if (model_dir / "best_model.pt").exists():
                summary = model_dir / "run_summary.json"
                if summary.exists():
                    data = json.load(open(summary))
                    arch = data.get('arch', '?')
                    seed = data.get('hyperparams', {}).get('seed', data.get('seed', '?'))
                    best = data.get('best', {}).get('val_ppl', '?')
                    epochs = len(data.get('epochs', []))
                    print(f"  {arch}_s{seed}: {epochs}ep, PPL={best:.2f} [{model_dir.name}]")
    
    # LLM adapters
    llm_path = Path("artifacts")
    if llm_path.exists():
        print("\n🤖 LLMs (Gemma 4 Fine-tuned):")
        for adapter_dir in llm_path.rglob("adapter_model.safetensors"):
            parent = adapter_dir.parent
            config = parent / "adapter_config.json"
            if config.exists():
                cfg = json.load(open(config))
                print(f"  {parent.parent.name}: {cfg.get('base_model_name_or_path', 'Gemma')}")
    
    print("\n" + "="*60)

def load_slm(model_dir: Path, device="cpu"):
    """Load a Small LM model."""
    ckpt = torch.load(model_dir / "best_model.pt", map_location=device)
    config = json.load(open(model_dir / "config.json"))
    
    model = build_model(ckpt["arch"], ckpt["params"])
    model.load_state_dict(ckpt["state"])
    model.to(device)
    model.eval()
    
    return model, config, ckpt["arch"]

def generate_slm(model, prompt: str, max_tokens: int = 50, temperature: float = 0.8, device="cpu"):
    """Generate text with SLM."""
    # Use tiktoken for encoding
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(prompt)
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Get last seq_len tokens
            x = torch.tensor([tokens[-256:]], dtype=torch.long, device=device)
            logits = model(x)
            logits = logits[0, -1] / temperature
            
            # Sample
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            tokens.append(next_token)
            
            # Stop at end token
            if next_token == enc.eot_token:
                break
    
    return enc.decode(tokens)

def load_llm(adapter_path: Path, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Load a fine-tuned LLM with LoRA adapter."""
    config = json.load(open(adapter_path / "adapter_config.json"))
    base_model = config["base_model_name_or_path"]
    
    print(f"Loading base model: {base_model}")
    print(f"Loading adapter from: {adapter_path}")
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None
    )
    model = PeftModel.from_pretrained(model, str(adapter_path))
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    return model, tokenizer

def generate_llm(model, tokenizer, prompt: str, max_tokens: int = 100, temperature: float = 0.7):
    """Generate with LLM."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def chat_slm(model_dir: str):
    """Interactive chat with SLM."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading SLM from: {model_dir}")
    print(f"Device: {device}")
    
    model, config, arch = load_slm(Path(model_dir), device)
    print(f"Architecture: {arch}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    
    print("\n" + "="*60)
    print(f"Chatting with {arch.upper()} SLM")
    print("Commands: /quit, /temp <val>, /len <tokens>")
    print("="*60 + "\n")
    
    temperature = 0.8
    max_len = 50
    
    while True:
        try:
            prompt = input("You: ").strip()
            
            if prompt == "/quit":
                break
            elif prompt.startswith("/temp "):
                temperature = float(prompt.split()[1])
                print(f"Temperature set to {temperature}")
                continue
            elif prompt.startswith("/len "):
                max_len = int(prompt.split()[1])
                print(f"Max length set to {max_len}")
                continue
            elif not prompt:
                continue
            
            print("Bot: ", end="", flush=True)
            response = generate_slm(model, prompt, max_len, temperature, device)
            # Only show generated part
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            print(response)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nGoodbye!")

def chat_llm(adapter_path: str):
    """Interactive chat with LLM."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading LLM adapter from: {adapter_path}")
    print(f"Device: {device}")
    
    model, tokenizer = load_llm(Path(adapter_path), device)
    print(f"Model loaded successfully!")
    
    print("\n" + "="*60)
    print("Chatting with Fine-tuned Gemma 4")
    print("Commands: /quit, /temp <val>, /len <tokens>")
    print("="*60 + "\n")
    
    temperature = 0.7
    max_len = 100
    
    while True:
        try:
            prompt = input("You: ").strip()
            
            if prompt == "/quit":
                break
            elif prompt.startswith("/temp "):
                temperature = float(prompt.split()[1])
                print(f"Temperature set to {temperature}")
                continue
            elif prompt.startswith("/len "):
                max_len = int(prompt.split()[1])
                print(f"Max length set to {max_len}")
                continue
            elif not prompt:
                continue
            
            print("Bot: ", end="", flush=True)
            response = generate_llm(model, tokenizer, prompt, max_len, temperature)
            # Remove prompt from response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            print(response)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
    
    print("\nGoodbye!")

def main():
    parser = argparse.ArgumentParser(description="Chat with trained models")
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--slm", type=str, help="Path to SLM model directory")
    parser.add_argument("--llm", type=str, help="Path to LLM adapter directory")
    args = parser.parse_args()
    
    if args.list or (not args.slm and not args.llm):
        list_available_models()
        print("\nUsage:")
        print("  python chat_with_models.py --slm slm_training/artifacts/small_lm/final_gpt_s44_140428")
        print("  python chat_with_models.py --llm artifacts/gemma_unsloth/gemma_unsloth_20260411_103834/best_model")
        return
    
    if args.slm:
        chat_slm(args.slm)
    elif args.llm:
        chat_llm(args.llm)

if __name__ == "__main__":
    main()
