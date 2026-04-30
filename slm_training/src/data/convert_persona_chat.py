#!/usr/bin/env python3
"""Convert Synthetic-Persona-Chat to scaffold dialogue format."""
from pathlib import Path
import json
import csv
import re
from datasets import load_from_disk

def parse_conversation(text: str):
    """Parse 'User 1: ... User 2: ...' format into turns."""
    turns = []
    # Pattern: User 1: text User 2: text User 1: text ...
    pattern = r'(User \d+):\s*([^U\n][^U]*?)(?=User \d+:|$)'
    matches = re.findall(pattern, text, re.DOTALL)
    for speaker, content in matches:
        content = content.strip()
        if content:
            turns.append({"speaker": "player" if speaker == "User 1" else "npc", "text": content})
    return turns

def main():
    src = Path("data/raw/synthetic_persona_chat")
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {src}...")
    ds = load_from_disk(str(src))
    
    # Collect NPC profiles
    npc_profiles = {}
    dialogue_records = []
    
    for split_name, split in ds.items():
        if split_name == "test":
            continue  # Skip test for training
            
        for ex in split:
            # Build profile from user 2 personas (NPC)
            persona_lines = ex.get("user 2 personas", "").strip().split("\n")
            persona_text = " ".join(p.strip() for p in persona_lines if p.strip())
            npc_id = f"npc_{hash(persona_text) % 100000:05d}"
            npc_profiles[npc_id] = persona_text
            
            # Parse conversation
            conv_text = ex.get("Best Generated Conversation", "")
            turns = parse_conversation(conv_text)
            
            if len(turns) < 2:
                continue
                
            # Create dialogue context/target pairs
            for i in range(1, len(turns)):
                if turns[i]["speaker"] != "npc":
                    continue
                    
                context = turns[:i]
                target = turns[i]["text"]
                
                record = {
                    "npc_id": npc_id,
                    "npc_profile": persona_text,
                    "dialogue_context": context,
                    "target_response": target,
                    "metadata": {"source": "synthetic_persona_chat", "split": split_name}
                }
                dialogue_records.append(record)
    
    # Split into train/val
    import random
    random.seed(42)
    random.shuffle(dialogue_records)
    cut = int(len(dialogue_records) * 0.9)
    train_recs = dialogue_records[:cut]
    val_recs = dialogue_records[cut:]
    
    # Write JSONL
    def write_jsonl(path, records):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"  {len(records)} records -> {path}")
    
    write_jsonl(out_dir / "dialogue" / "from_persona_train.jsonl", train_recs)
    write_jsonl(out_dir / "dialogue" / "from_persona_val.jsonl", val_recs)
    
    # Write plain text for small LMs
    def write_txt(path, records):
        lines = []
        for r in records:
            lines.append(f"PROFILE: {r['npc_profile']}")
            for turn in r["dialogue_context"]:
                lines.append(f"{turn['speaker'].upper()}: {turn['text']}")
            lines.append(f"NPC: {r['target_response']}")
            lines.append("")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines))
        print(f"  {len(records)} examples -> {path}")
    
    write_txt(out_dir / "dialogue" / "from_persona_train.txt", train_recs)
    write_txt(out_dir / "dialogue" / "from_persona_val.txt", val_recs)
    
    # Write NPC profiles CSV
    with open(out_dir / "npc_profiles_from_persona.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["npc_id", "profile_text"])
        w.writeheader()
        for npc_id, text in npc_profiles.items():
            w.writerow({"npc_id": npc_id, "profile_text": text})
    print(f"  {len(npc_profiles)} NPC profiles -> {out_dir}/npc_profiles_from_persona.csv")
    
    print(f"\nDone. Total dialogue pairs: {len(dialogue_records)}")

if __name__ == "__main__":
    main()
