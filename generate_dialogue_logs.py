#!/usr/bin/env python3
"""Generate 10-turn dialogue logs for each model for presentation."""
import torch, json, sys, time, os
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "slm_training" / "src"))

# ─── Common RPG scenario ──────────────────────────────────────────────────────

SCENE = """<scene>
Setting: tavern
NPC Role: innkeeper
Goals: gather_information, earn_coins, protect_secret
Values: hospitality, discretion, loyalty
Secrets: harboring_fugitive
Persona: warm, shrewd, cautious
</scene>"""

PLAYER_TURNS = [
    "Hello there. I'm new to this town. What's the latest news?",
    "I've heard rumors the Iron Legion is marching this way. Is that true?",
    "That sounds serious. Do the townspeople know about this threat?",
    "Are there any strangers in town lately? Anyone unusual?",
    "I noticed a hooded figure slipping out your back door last night. Who was that?",
    "I'm not here to cause trouble. I just need to know if I'm safe here.",
    "What if I told you I'm with the town guard? I could help, but only if you're honest with me.",
    "Someone in this tavern is feeding information to the Legion. I know it. Was it the hooded man?",
    "I'm running out of patience. Either you tell me what's going on, or I'll search this place myself.",
    "Last chance. Help me, and I'll protect you. Stay silent, and you're on your own when the gates fall.",
]

# ─── SLM generation ───────────────────────────────────────────────────────────

def generate_slm_dialogue(model_dir, model_name, arch_info):
    """Generate a 10-turn dialogue with an SLM."""
    from train.small_lm_architectures import build_model
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    ckpt = torch.load(model_dir / "best_model.pt", map_location="cpu", weights_only=False)
    model = build_model(ckpt["arch"], ckpt["params"])
    model.load_state_dict(ckpt["state"])
    model.eval()
    params = sum(p.numel() for p in model.parameters())

    # PrefixGPT needs a conditioning vector (cond_dim=8)
    is_prefix = ckpt["arch"] == "prefix_gpt"
    cond_vec = torch.zeros(1, 8) if is_prefix else None

    dialogue = []
    history = "NPC: Welcome to my tavern, traveler. What can I get for you?\n"

    for i, player_line in enumerate(PLAYER_TURNS):
        prompt = f"{history}PLAYER: {player_line}\nNPC:"
        tokens = enc.encode(prompt)
        with torch.no_grad():
            for _ in range(80):
                x = torch.tensor([tokens[-256:]], dtype=torch.long)
                if is_prefix:
                    out = model(x, cond_vec)
                else:
                    out = model(x)
                logits = out.logits[0, -1] / 0.7
                probs = torch.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1).item()
                tokens.append(nxt)
                if nxt == enc.eot_token:
                    break
        response = enc.decode(tokens[len(enc.encode(prompt)):]).strip()
        # Clean up: take until next PLAYER: or NPC:
        for marker in ["PLAYER:", "NPC:", "PLAY", "\n\n"]:
            if marker in response:
                response = response[:response.index(marker)].strip()
        if not response:
            response = "..."

        dialogue.append({"turn": i + 1, "speaker": "Player", "text": player_line})
        dialogue.append({"turn": i + 1, "speaker": "NPC", "text": response})
        history = f"{history}PLAYER: {player_line}\nNPC: {response}\n"

    return {
        "model_name": model_name,
        "arch_info": arch_info,
        "params": f"{params/1e6:.1f}M",
        "scene": "Tavern — innkeeper (warm, shrewd, harboring fugitive secret)",
        "dialogue": dialogue,
    }


# ─── LLM generation ───────────────────────────────────────────────────────────

def generate_llm_dialogue(adapter_path, model_name, arch_info, use_scene=True, base_override=None):
    """Generate a 10-turn dialogue with an LLM + LoRA adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    config = json.load(open(Path(adapter_path) / "adapter_config.json"))
    base_model = base_override or config["base_model_name_or_path"]
    dtype = torch.float16 if device.type == "mps" else torch.float32

    print(f"  Loading {base_model}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=dtype, device_map=str(device))
    model = PeftModel.from_pretrained(model, str(adapter_path))
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model.eval()

    dialogue = []
    history_text = ""

    for i, player_line in enumerate(PLAYER_TURNS):
        if use_scene:
            prompt = f"{SCENE}\n\n<history>\n{history_text}</history>\n\nPlayer: {player_line}\nNPC:"
        else:
            prompt = f"{history_text}Player: {player_line}\nNPC:"

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                top_p=0.9,
            )
        r = tokenizer.decode(out[0], skip_special_tokens=True)
        if r.startswith(prompt):
            r = r[len(prompt):].strip()
        # Clean up
        for marker in ["Player:", "PLAYER:", "\n\n\n"]:
            if marker in r:
                r = r[:r.index(marker)].strip()
        if not r:
            r = "..."

        dialogue.append({"turn": i + 1, "speaker": "Player", "text": player_line})
        dialogue.append({"turn": i + 1, "speaker": "NPC", "text": r})
        history_text += f"[Turn {i+1}] Player: {player_line}\n[Turn {i+1}] NPC: {r}\n"
        print(f"    Turn {i+1} done", flush=True)

    return {
        "model_name": model_name,
        "arch_info": arch_info,
        "params": "1.1B" if "TinyLlama" in str(base_model) else "4B",
        "scene": "Tavern — innkeeper (warm, shrewd, harboring fugitive secret)",
        "dialogue": dialogue,
    }


# ─── Formatting ───────────────────────────────────────────────────────────────

def format_dialogue_md(result):
    """Format a dialogue result as markdown."""
    lines = []
    lines.append(f"# Dialogue Log: {result['model_name']}")
    lines.append(f"")
    lines.append(f"**Architecture:** {result['arch_info']}")
    lines.append(f"**Parameters:** {result['params']}")
    lines.append(f"**Scene:** {result['scene']}")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")

    for msg in result["dialogue"]:
        if msg["speaker"] == "Player":
            lines.append(f"### Turn {msg['turn']} — Player")
            lines.append(f"")
            lines.append(f"> {msg['text']}")
            lines.append(f"")
        else:
            lines.append(f"**NPC:** {msg['text']}")
            lines.append(f"")

    lines.append(f"---")
    lines.append(f"")
    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    output_dir = ROOT / "dialogue_logs"
    output_dir.mkdir(exist_ok=True)

    all_results = []

    # ── SLMs (fast, CPU) ──
    slm_models = [
        # Already done:
        # ("GPT-22M (Best SLM)", "slm_training/artifacts/small_lm/gpt22m_s44_20260521_144444",
        #  "GPT transformer decoder, 6 layers, 512 dim, 8 heads, val_ppl=40.68"),
        # ("MoE", "slm_training/artifacts/small_lm/slurm_816186_slm_small_lm_20260502_170219",
        #  "Mixture-of-Experts GPT, 4 experts top-2, val_ppl=42.07"),
        ("PrefixGPT", "slm_training/artifacts/small_lm/slurm_1141350_slm_small_lm_20260519_225131",
         "GPT with soft-prefix conditioning, val_ppl=43.65"),
        ("Mamba-like", "slm_training/artifacts/small_lm/slurm_1140564_slm_small_lm_20260519_195955",
         "Simplified SSM (pure PyTorch), val_ppl=53.23"),
        ("GPT-16M", "slm_training/artifacts/small_lm/slurm_816183_slm_small_lm_20260502_170210",
         "GPT transformer decoder, 4 layers, 256 dim, 4 heads, val_ppl=45.32"),
    ]

    for name, path, info in slm_models:
        print(f"\n{'='*60}", flush=True)
        print(f"Generating: {name}", flush=True)
        print(f"{'='*60}", flush=True)
        t0 = time.time()
        result = generate_slm_dialogue(ROOT / path, name, info)
        result["generation_time"] = f"{time.time()-t0:.1f}s"
        all_results.append(result)

        fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        (output_dir / f"{fname}.md").write_text(format_dialogue_md(result))
        print(f"  Saved to dialogue_logs/{fname}.md ({time.time()-t0:.1f}s)", flush=True)

    # ── LLMs (need MPS) ──
    llm_models = [
        ("TinyLlama-1.1B + LoRA", "slm_training/artifacts/tinyllama_lora/dialogue_20260502_195558/best_model",
         "TinyLlama-1.1B-Chat-v1.0 + LoRA r=16, val_ppl=3.30", True),
        ("ConditionalDialogue (OCEAN+VAD)", "slm_training/artifacts/dialogue_model/slurm_816490_slm_dialogue_20260502_192607/best_model",
         "TinyLlama-1.1B + LoRA r=16 + OCEAN/VAD prefix encoder, val_ppl=2.90", True),
        ("Qwen3-4B Response Generator", "checkpoints/response_generator_best",
         "Qwen3-4B + LoRA r=32, ROUGE-L=0.145, with latent-state conditioning", True),
    ]

    for name, path, info, use_scene in llm_models:
        print(f"\n{'='*60}", flush=True)
        print(f"Generating: {name}", flush=True)
        print(f"{'='*60}", flush=True)
        t0 = time.time()
        result = generate_llm_dialogue(ROOT / path, name, info, use_scene)
        result["generation_time"] = f"{time.time()-t0:.1f}s"
        all_results.append(result)

        fname = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "").replace("-", "_").replace("__", "_")
        (output_dir / f"{fname}.md").write_text(format_dialogue_md(result))
        print(f"  Saved to dialogue_logs/{fname}.md ({time.time()-t0:.1f}s)", flush=True)

    # ── Combined summary ──
    summary = ["# Dialogue Logs Summary\n"]
    summary.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    summary.append(f"Total models: {len(all_results)}\n")
    summary.append(f"Turns per dialogue: 10 (20 messages)\n")
    summary.append(f"Scene: Tavern — innkeeper (warm, shrewd, harboring fugitive secret)\n\n")
    summary.append("---\n\n")

    for r in all_results:
        summary.append(f"## {r['model_name']}\n")
        summary.append(f"- **Architecture:** {r['arch_info']}\n")
        summary.append(f"- **Parameters:** {r['params']}\n")
        summary.append(f"- **Generation time:** {r['generation_time']}\n")
        summary.append(f"- **File:** `{r['model_name'].lower().replace(' ', '_').replace('(', '').replace(')', '').replace('+', '').replace('-', '_').replace('__', '_')}.md`\n\n")

    (output_dir / "README.md").write_text("\n".join(summary))
    print(f"\n\nAll done! {len(all_results)} dialogue logs saved to dialogue_logs/", flush=True)
    print(f"Summary: dialogue_logs/README.md", flush=True)


if __name__ == "__main__":
    main()
