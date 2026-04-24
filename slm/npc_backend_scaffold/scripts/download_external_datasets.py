#!/usr/bin/env python3
"""
download_external_datasets.py
============================
Download and adapt external datasets for NPC dialogue training.

Datasets processed:
  1. PIPPA (PygmalionAI) — 1M lines, 26K conversations, 1K+ personas
  2. LIGHT (Facebook FAIR) — fantasy text adventure, 160K utterances  
  3. EmpatheticDialogues — 25K emotion-labeled conversations
  4. PersonaChat/ConvAI2 — 164K utterances with persona facts

Output formats:
  data/external/<dataset>/dialogue.jsonl → for dialogue LM training
  data/external/<dataset>/affect.csv → for affect encoder training
  data/external/<dataset>/personality.jsonl → for personality encoder training

Usage:
  python scripts/download_external_datasets.py --all
  python scripts/download_external_datasets.py --datasets pippa light empathetic personachat
"""
from __future__ import annotations

import argparse
import json
import csv
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "external"


def ensure_dirs():
    """Create output directories."""
    for d in ["pippa", "light", "empathetic", "personachat"]:
        (DATA_DIR / d).mkdir(parents=True, exist_ok=True)


# =============================================================================
# PIPPA Dataset Adapter
# =============================================================================

def download_and_process_pippa():
    """Download PIPPA-shareGPT (kingbri/PIPPA-shareGPT) — working parquet alternative."""
    print("\n[PIPPA] Downloading kingbri/PIPPA-shareGPT (shareGPT format)...")
    try:
        from datasets import load_dataset
        ds = load_dataset("kingbri/PIPPA-shareGPT",
                          data_files="pippa_sharegpt.jsonl", split="train")
    except Exception as e:
        print(f"  ERROR: {e}")
        return None, None, None

    print(f"  Loaded {len(ds)} conversations")

    # Filter NSFW by bot categories
    safe_entries = []
    nsfw_count = 0
    for entry in ds:
        bot = entry.get("bot") or {}
        cats = bot.get("categories") or []
        if isinstance(cats, list):
            cat_str = " ".join(str(c) for c in cats).lower()
            if any(w in cat_str for w in ("nsfw", "adult", "sexual", "explicit")):
                nsfw_count += 1
                continue
        safe_entries.append(entry)

    print(f"  Filtered {nsfw_count} NSFW entries, kept {len(safe_entries)} safe")

    dialogue_out = []
    personality_out = []

    for entry in tqdm(safe_entries, desc="Processing PIPPA"):
        bot = entry.get("bot") or {}
        bot_name = bot.get("name", "Unknown")
        bot_desc = bot.get("description", "")

        # Build NPC profile text
        profile_parts = [f"Name: {bot_name}"]
        if bot_desc:
            cleaned = re.sub(r'\*\*[^*]+\*\*', '', bot_desc)
            cleaned = re.sub(r'\([^)]+\)', '', cleaned).strip()
            if cleaned:
                profile_parts.append(cleaned[:400])
        profile_text = " | ".join(profile_parts)

        # conversations: list of {role, content}
        conv = entry.get("conversations") or []
        if len(conv) < 2:
            continue

        roles = entry.get("roles") or ["USER", "CHARACTER"]
        user_role  = roles[0] if roles else "USER"
        char_role  = roles[1] if len(roles) > 1 else "CHARACTER"

        turns = []
        for turn in conv:
            msg  = str(turn.get("value") or turn.get("content") or "").strip()
            role = str(turn.get("from")  or turn.get("role") or "").lower()
            if not msg:
                continue
            speaker = "PLAYER:" if role in ("human", "user", user_role.lower()) else "NPC:"
            turns.append(f"{speaker} {msg}")

        if len(turns) < 2:
            continue

        dialogue_text = "\n".join(turns)
        dialogue_out.append({
            "npc_id":          f"pippa_{entry.get('id', hash(dialogue_text) % 10000000)}",
            "npc_profile":     profile_text,
            "dialogue_context": [],
            "target_response": turns[-1],
            "full_conversation": dialogue_text,
            "source":          "pippa"
        })
        personality_out.append({
            "text":     profile_text,
            "source":   "pippa",
            "npc_name": bot_name
        })
    
    # Save dialogue
    out_path = DATA_DIR / "pippa" / "dialogue.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in dialogue_out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(dialogue_out)} dialogue entries → {out_path}")
    
    # Save personality data
    prof_path = DATA_DIR / "pippa" / "personality.jsonl"
    with open(prof_path, "w", encoding="utf-8") as f:
        for item in personality_out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(personality_out)} personality profiles → {prof_path}")
    
    # Estimate tokens
    total_chars = sum(len(d["full_conversation"]) for d in dialogue_out)
    estimated_tokens = total_chars // 4  # Rough estimate: 4 chars per token
    print(f"  Estimated tokens: ~{estimated_tokens:,}")
    
    return dialogue_out, personality_out, None


# =============================================================================
# LIGHT Dataset Adapter
# =============================================================================

def download_and_process_light():
    """Download LIGHT dataset and convert."""
    print("\n[LIGHT] Downloading from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("npc-engine/light-batch-summarize-dialogue", split="train")
    except Exception as e:
        print(f"  ERROR: {e}")
        return None, None, None
    
    print(f"  Loaded {len(ds)} entries")
    
    dialogue_out = []
    personality_out = []
    
    for entry in tqdm(ds, desc="Processing LIGHT"):
        # Actual schema: dialogue_text (multi-speaker), t0pp_prediction (summary)
        dialogue_text = entry.get("dialogue_text", "").strip()
        if not dialogue_text or len(dialogue_text) < 20:
            continue

        # Extract turns: format is "speaker name: message\n..."
        lines = [l.strip() for l in dialogue_text.split("\n") if l.strip()]
        speakers = []
        turns = []
        for line in lines:
            if ":" in line[:40]:
                sp, _, msg = line.partition(":")
                sp = sp.strip().lower()
                msg = msg.strip()
                if not msg:
                    continue
                if sp not in speakers:
                    speakers.append(sp)
                label = "NPC:" if speakers.index(sp) == 0 else "PLAYER:"
                turns.append(f"{label} {msg}")
            else:
                if turns:
                    turns[-1] += " " + line

        if len(turns) < 2:
            continue

        npc_name = speakers[0].title() if speakers else "Unknown"
        summary = entry.get("t0pp_prediction", "")[:200]
        profile_text = f"Name: {npc_name} | {summary}" if summary else f"Name: {npc_name}"

        full_conv = "\n".join(turns)
        dialogue_out.append({
            "npc_id":            f"light_{hash(full_conv) % 10000000}",
            "npc_profile":       profile_text,
            "dialogue_context":  turns[:-1],
            "target_response":   turns[-1],
            "full_conversation": full_conv,
            "source":            "light"
        })
        personality_out.append({
            "text":     profile_text,
            "source":   "light",
            "npc_name": npc_name
        })
    
    out_path = DATA_DIR / "light" / "dialogue.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in dialogue_out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(dialogue_out)} dialogue entries → {out_path}")
    
    prof_path = DATA_DIR / "light" / "personality.jsonl"
    with open(prof_path, "w", encoding="utf-8") as f:
        for item in personality_out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(personality_out)} personality profiles → {prof_path}")
    
    total_chars = sum(len(d["full_conversation"]) for d in dialogue_out)
    estimated_tokens = total_chars // 4
    print(f"  Estimated tokens: ~{estimated_tokens:,}")
    
    return dialogue_out, personality_out, None


# =============================================================================
# EmpatheticDialogues Adapter
# =============================================================================

def download_and_process_empathetic():
    """Download Estwld/empathetic_dialogues_llm — LLM-formatted working parquet version."""
    print("\n[EmpatheticDialogues] Downloading Estwld/empathetic_dialogues_llm...")
    try:
        from datasets import load_dataset
        ds = load_dataset("Estwld/empathetic_dialogues_llm", split="train")
    except Exception as e:
        print(f"  ERROR: {e}")
        return None, None, None

    print(f"  Loaded {len(ds)} entries")
    
    # Emotion to VAD mapping (simplified - can be refined)
    # Source: Warriner et al. 2013 NRC VAD Lexicon averages
    EMOTION_TO_VAD = {
        "afraid": (0.12, 0.83, 0.25),
        "angry": (0.12, 0.79, 0.58),
        "annoyed": (0.20, 0.65, 0.42),
        "anticipating": (0.65, 0.62, 0.58),
        "anxious": (0.15, 0.72, 0.28),
        "apprehensive": (0.18, 0.58, 0.35),
        "ashamed": (0.12, 0.45, 0.18),
        "bored": (0.28, 0.22, 0.15),
        "confident": (0.75, 0.55, 0.78),
        "content": (0.68, 0.25, 0.58),
        "curious": (0.62, 0.62, 0.65),
        "devastated": (0.08, 0.58, 0.12),
        "disappointed": (0.18, 0.52, 0.28),
        "disgusted": (0.15, 0.68, 0.42),
        "embarrassed": (0.15, 0.55, 0.22),
        "excited": (0.78, 0.82, 0.62),
        "faithful": (0.72, 0.45, 0.68),
        "furious": (0.12, 0.92, 0.52),
        "grateful": (0.75, 0.45, 0.52),
        "guilty": (0.18, 0.62, 0.32),
        "hopeful": (0.68, 0.48, 0.62),
        "impressed": (0.72, 0.62, 0.58),
        "jealous": (0.22, 0.68, 0.48),
        "joyful": (0.88, 0.72, 0.75),
        "lonely": (0.18, 0.48, 0.22),
        "nostalgic": (0.55, 0.32, 0.48),
        "prepared": (0.62, 0.48, 0.75),
        "proud": (0.78, 0.62, 0.72),
        "sad": (0.15, 0.42, 0.18),
        "sentimental": (0.58, 0.42, 0.45),
        "surprised": (0.65, 0.78, 0.55),
        "terrified": (0.08, 0.92, 0.15),
        "trusting": (0.68, 0.38, 0.58),
    }
    
    affect_out = []
    dialogue_out = []

    for entry in tqdm(ds, desc="Processing Empathetic"):
        # New format: conv_id, situation, emotion, conversations=[{role, content}]
        emotion = (entry.get("emotion") or "").lower().strip()
        situation = entry.get("situation") or ""
        conv = entry.get("conversations") or []

        if not emotion or not conv:
            continue

        # Build full dialogue text
        turns = []
        for turn in conv:
            role = (turn.get("role") or "").lower()
            msg  = (turn.get("content") or "").strip()
            if not msg:
                continue
            speaker = "PLAYER:" if role == "user" else "NPC:"
            turns.append(f"{speaker} {msg}")

        if not turns:
            continue

        vad = EMOTION_TO_VAD.get(emotion, (0.5, 0.5, 0.5))
        context_text = f"{situation} " + " ".join(t for t in turns[:2])

        affect_out.append({
            "text":         context_text.strip(),
            "valence":      vad[0],
            "arousal":      vad[1],
            "dominance":    vad[2],
            "emotion_label": emotion,
            "source":       "empathetic"
        })

        full_conv = "\n".join(turns)
        dialogue_out.append({
            "npc_id":          f"empathetic_{hash(full_conv) % 10000000}",
            "npc_profile":     f"Person experiencing {emotion}. Situation: {situation[:200]}",
            "dialogue_context": turns[:-1],
            "target_response": turns[-1] if turns else "",
            "full_conversation": full_conv,
            "source":          "empathetic",
            "emotion":         emotion
        })
    
    # Save affect data
    out_path = DATA_DIR / "empathetic" / "affect.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        if affect_out:
            writer = csv.DictWriter(f, fieldnames=list(affect_out[0].keys()))
            writer.writeheader()
            writer.writerows(affect_out)
    print(f"  Saved {len(affect_out)} affect entries → {out_path}")
    
    # Save dialogue
    dial_path = DATA_DIR / "empathetic" / "dialogue.jsonl"
    with open(dial_path, "w", encoding="utf-8") as f:
        for item in dialogue_out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(dialogue_out)} dialogue entries → {dial_path}")
    
    return dialogue_out, None, affect_out


# =============================================================================
# PersonaChat Adapter
# =============================================================================

def download_and_process_personachat():
    """Download PersonaChat for personality-conditioned dialogue."""
    print("\n[PersonaChat] Downloading from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("AlekseyKorshuk/persona-chat", split="train")
    except Exception as e:
        print(f"  ERROR: {e}")
        return None, None, None
    
    print(f"  Loaded {len(ds)} entries")
    
    dialogue_out = []
    personality_out = []
    
    for entry in tqdm(ds, desc="Processing PersonaChat"):
        personas = entry.get("personality", [])  # List of 5 persona facts
        utterances = entry.get("utterances", [])
        
        if not personas or not utterances:
            continue
        
        # Build profile text from persona facts
        profile_text = " | ".join(personas)
        
        # Process alternating dialogue
        turns = []
        for i, msg in enumerate(utterances):
            speaker = "PLAYER:" if i % 2 == 0 else "NPC:"
            turns.append(f"{speaker} {msg}")
        
        if len(turns) < 2:
            continue
        
        dialogue_text = "\n".join(turns)
        
        dialogue_out.append({
            "npc_id": f"personachat_{hash(dialogue_text) % 10000000}",
            "npc_profile": profile_text,
            "dialogue_context": turns[:-1],
            "target_response": turns[-1],
            "full_conversation": dialogue_text,
            "source": "personachat"
        })
        
        personality_out.append({
            "text": profile_text,
            "source": "personachat",
            "npc_name": f"Persona_{hash(profile_text) % 10000}"
        })
    
    out_path = DATA_DIR / "personachat" / "dialogue.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in dialogue_out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(dialogue_out)} dialogue entries → {out_path}")
    
    prof_path = DATA_DIR / "personachat" / "personality.jsonl"
    with open(prof_path, "w", encoding="utf-8") as f:
        for item in personality_out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Saved {len(personality_out)} personality profiles → {prof_path}")
    
    total_chars = sum(len(d["full_conversation"]) for d in dialogue_out)
    estimated_tokens = total_chars // 4
    print(f"  Estimated tokens: ~{estimated_tokens:,}")
    
    return dialogue_out, personality_out, None


# =============================================================================
# Merge and Create Training Files
# =============================================================================

def merge_and_create_training_files():
    """Merge all external datasets into unified training files."""
    print("\n[Merging datasets into unified training files...]")
    
    # Dialogue merge
    all_dialogue = []
    for dataset in ["pippa", "light", "empathetic", "personachat"]:
        path = DATA_DIR / dataset / "dialogue.jsonl"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        all_dialogue.append(json.loads(line))
                    except:
                        pass
    
    # Write merged dialogue
    merged_path = DATA_DIR / "merged_dialogue.jsonl"
    with open(merged_path, "w", encoding="utf-8") as f:
        for item in all_dialogue:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Merged {len(all_dialogue)} dialogue entries → {merged_path}")
    
    # Create train.txt format (flat text)
    txt_path = DATA_DIR / "merged_dialogue.txt"
    total_chars = 0
    with open(txt_path, "w", encoding="utf-8") as f:
        for item in all_dialogue:
            text = item.get("full_conversation", "")
            if text:
                f.write(text + "\n\n")
                total_chars += len(text)
    
    estimated_tokens = total_chars // 4
    print(f"  Created flat text → {txt_path}")
    print(f"  Estimated total tokens: ~{estimated_tokens:,}")
    
    # Affect merge
    all_affect = []
    path = DATA_DIR / "empathetic" / "affect.csv"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_affect.append(row)
    
    if all_affect:
        affect_path = DATA_DIR / "merged_affect.csv"
        with open(affect_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_affect[0].keys()))
            writer.writeheader()
            writer.writerows(all_affect)
        print(f"  Merged {len(all_affect)} affect entries → {affect_path}")
    
    # Personality merge
    all_personality = []
    for dataset in ["pippa", "light", "personachat"]:
        path = DATA_DIR / dataset / "personality.jsonl"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        all_personality.append(json.loads(line))
                    except:
                        pass
    
    prof_path = DATA_DIR / "merged_personality.jsonl"
    with open(prof_path, "w", encoding="utf-8") as f:
        for item in all_personality:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"  Merged {len(all_personality)} personality profiles → {prof_path}")
    
    return len(all_dialogue), len(all_affect), len(all_personality)


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["pippa", "light", "empathetic", "personachat"],
                   choices=["pippa", "light", "empathetic", "personachat"],
                   help="Datasets to download (default: all)")
    args = p.parse_args()
    
    datasets = args.datasets
    
    ensure_dirs()
    
    print("=" * 70)
    print("Downloading and adapting external datasets")
    print("=" * 70)
    
    if "pippa" in datasets:
        download_and_process_pippa()
    
    if "light" in datasets:
        download_and_process_light()
    
    if "empathetic" in datasets:
        download_and_process_empathetic()
    
    if "personachat" in datasets:
        download_and_process_personachat()
    
    # Merge all
    n_dial, n_aff, n_prof = merge_and_create_training_files()
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Dialogue entries:  {n_dial:,}")
    print(f"  Affect entries:    {n_aff:,}")
    print(f"  Personality texts: {n_prof:,}")
    print(f"\n  Output directory: {DATA_DIR}")
    print("  Use these files:")
    print(f"    - data/external/merged_dialogue.jsonl  → dialogue LM training")
    print(f"    - data/external/merged_dialogue.txt    → small LM training")
    print(f"    - data/external/merged_affect.csv        → affect encoder training")
    print(f"    - data/external/merged_personality.jsonl → personality encoder training")
    print("\n  To use in training, update your configs to point at these paths.")


if __name__ == "__main__":
    main()
