import argparse
import yaml
import torch
import textwrap
from pathlib import Path
from typing import Dict, List, Any

from src.training.model import load_predictor, STANCE_DIMS
from src.training.dataset import LABEL_MAPS

# Reverse map for decoding predictions
IDX_TO_LABEL = {
    field: {i: v for i, v in enumerate(values)}
    for field, values in LABEL_MAPS.items()
}

def decode_heads(logits: Dict[str, torch.Tensor]) -> Dict[str, str]:
    """Decode head logits into readable labels."""
    decoded = {}
    for name, logit in logits.items():
        if name not in IDX_TO_LABEL:
            continue
            
        # Handle multi-label for dialogue_act if needed, but for now assuming argmax or threshold
        # The training logic used BCE for multi-label (dialogue_act)? 
        # Checking dataset.py: _encode_labels uses multi-hot for dialogue_act.
        # Checking model.py: ClassificationHead output size is n_classes.
        # Usually we use sigmoid + threshold for multi-label.
        
        if name == "dialogue_act":
            # Multi-label
            probs = torch.sigmoid(logit).squeeze()
            # Get all indices > 0.5
            indices = (probs > 0.5).nonzero(as_tuple=False).squeeze(-1).tolist()
            if isinstance(indices, int):
                indices = [indices]
            labels = [IDX_TO_LABEL[name][i] for i in indices]
            decoded[name] = labels # List of strings
        else:
            # Single-label
            idx = torch.argmax(logit, dim=-1).item()
            decoded[name] = IDX_TO_LABEL[name].get(idx, "unknown")
            
    return decoded

def format_context(record: Dict[str, Any]) -> str:
    """Reconstruct the context string from a record dictionary."""
    # This matches src/packaging/packager.py logic but adapted for inference state
    W = record.get("W", {})
    R_t_prev_parts = []
    
    # Current stance (starts as initial, updates over time)
    current_stance = record.get("current_stance", W.get("initial_stance", {}))
    
    for dim in STANCE_DIMS:
        # Default to Neutral if missing
        val = current_stance.get(dim, "N")
        R_t_prev_parts.append(f"{dim}={val}")

    history_lines = []
    for h in record.get("dialogue_history", []):
        history_lines.append(f"[Turn {h['turn_idx']}] Player: {h['player_utterance']}")
        if h.get("response"):
            history_lines.append(f"[Turn {h['turn_idx']}] NPC: {h['response']}")

    lines = [
        f"<scene>",
        f"Setting: {record.get('setting', '')}",
        f"NPC Role: {W.get('role', '')}",
        f"Goals: {', '.join(W.get('core_goals', []))}",
        f"Values: {', '.join(W.get('values', []))}",
        f"Secrets: {', '.join(s['secret_id'] for s in W.get('secrets', []))}",
        f"Persona: {', '.join(W.get('persona_style', []))}",
        f"</scene>",
        f"",
        f"<prior_stance>",
        "  ".join(R_t_prev_parts),
        f"</prior_stance>",
        f"",
        f"<history>",
    ] + history_lines + [
        f"</history>",
        f"",
        f"Player: {record.get('input', '')}",
    ]
    return "\n".join(lines)

def format_latent_state(decoded_labels: Dict[str, Any]) -> str:
    """Reconstruct the latent state string from decoded labels."""
    # Matches src/packaging/packager.py logic
    
    # Helper to get value safely
    def get(k, default=""):
        val = decoded_labels.get(k, default)
        if isinstance(val, list): # For dialogue_act
            return str(val) # keep as list representation for now, or join?
            # dataset.py says: dialogue_act=["ask", "probe"] in json.
            # _latent_state_str uses: f"C_t: dialogue_act={C_t.get('dialogue_act', [])} ..."
            # So it expects a list.
        return str(val)

    # Reconstruct stance parts
    stance_parts = []
    for dim in STANCE_DIMS:
        l = get(f"{dim}_level", "N")
        d = get(f"{dim}_delta", "0")
        stance_parts.append(f"{dim}={l}({d})")

    lines = [
        "<latent_state>",
        f"C_t: dialogue_act={get('dialogue_act', [])}  tone={get('tone')}  risk={get('risk_type')}",
        f"A_t: valence={get('valence')}  arousal={get('arousal')}  threat={get('threat')}  control={get('control')}",
        f"M_t: player_intent={get('player_intent')}  player_knowledge={get('player_knowledge')}  credibility={get('player_credibility')}",
        f"R_t: {' '.join(stance_parts)}",
        f"N_t: duty={get('duty_pressure')}  secrecy={get('secrecy_pressure')}  face={get('face_pressure')}  conflict={get('value_conflict')}",
        f"D_t: policy={get('response_policy')}  reveal={get('reveal_decision')}  repair={get('repair_strategy')}",
        "</latent_state>",
    ]
    return "\n".join(lines)

def run_interactive(
    checkpoint_dir: str,
    base_model: str,
    npc_id: str,
    scenario_file: str,
    quantization: str = "4bit",
    device: str = "cuda"
):
    print(f"Loading scenario from {scenario_file}...")
    with open(scenario_file, "r") as f:
        scenario = yaml.safe_load(f)
        
    # Find NPC
    npc_data = next((n for n in scenario["npcs"] if n["npc_id"] == npc_id), None)
    if not npc_data:
        raise ValueError(f"NPC {npc_id} not found in scenario.")
        
    print(f"Initializing model (base={base_model}, ckpt={checkpoint_dir}, quant={quantization})...")
    predictor, tokenizer = load_predictor(checkpoint_dir, base_model, quantization=quantization)
    predictor.eval()
    
    # Initialize conversation state
    state = {
        "W": npc_data,
        "setting": scenario["description"],
        "dialogue_history": [],
        "current_stance": {dim: "N" for dim in STANCE_DIMS}, # Default start neutral
        "input": ""
    }
    
    # Use initial stance from scenario if available (needs parsing relationships)
    # For now, let's just stick to default or manually set if needed.
    
    print(f"\n--- Starting chat with {npc_data['name']} ---")
    print(f"Role: {npc_data['role']}")
    print(f"Persona: {npc_data['persona_style']}")
    print("Type 'quit' or 'exit' to end.\n")
    
    turn_idx = 1
    
    while True:
        user_input = input("Player: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
            
        state["input"] = user_input
        state["W"]["turn_idx"] = turn_idx # Hacky way to pass turn index if needed
        
        # 1. Prepare Context
        context_str = format_context(state)
        
        # 2. Tokenize for Heads (Context Only)
        heads_inputs = tokenizer(
            context_str, 
            return_tensors="pt", 
            truncation=True, 
            max_length=1024
        ).to(device)
        
        # 3. Predict Latent State
        with torch.no_grad():
            out_heads = predictor(
                input_ids=heads_inputs.input_ids,
                attention_mask=heads_inputs.attention_mask
            )
            logits = out_heads["logits"]
            decoded = decode_heads(logits)
            
        latent_str = format_latent_state(decoded)
        
        print("\n--- Predicted Latent State ---")
        print(textwrap.indent(latent_str, "  "))
        print("------------------------------\n")
        
        # 4. Generate Response
        # Training data has a newline between input and target
        full_input = context_str + "\n\n" + latent_str + "\n\nGenerate NPC response:\n"
        
        lm_inputs = tokenizer(
            full_input,
            return_tensors="pt",
            truncation=True,
            max_length=2048 # Allow room for generation
        ).to(device)
        
        with torch.no_grad():
            generated_ids = predictor.backbone.generate(
                **lm_inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.92,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode only the new tokens
        input_len = lm_inputs.input_ids.shape[1]
        raw_text = tokenizer.decode(generated_ids[0][input_len:], skip_special_tokens=True)
        # print(f"[DEBUG] Raw output: {repr(raw_text)}")
        
        response_text = raw_text.strip()
        
        # Clean up artifacts
        if "</response>" in response_text:
            response_text = response_text.split("</response>")[0].strip()
        if "<response>" in response_text:
            response_text = response_text.replace("<response>", "").strip()
        if "Human-Style Response:" in response_text:
            response_text = response_text.replace("Human-Style Response:", "").strip()
        
        # Aggressive cleanup of hallucinations/CoT artifacts
        artifacts = [
            "Human needs here", "Questioned about", "The guard needs to", "Wait, there's something wrong",
            "But wait,", "I should check", "Let me rethink", "The NPC is being", "The response should",
            "What does this mean?", "Question:", "Also,"
        ]
        for artifact in artifacts:
             if artifact in response_text:
                 response_text = response_text.split(artifact)[0].strip()

        # Stop at "Player:" or "Human:" if generated
        for stop_word in ["Player:", "Human:", "\nPlayer", "\nHuman"]:
            if stop_word in response_text:
                response_text = response_text.split(stop_word)[0].strip()
        
        # If response starts with a quote, keep only the quoted part and discard the rest (often CoT)
        if response_text.startswith('"') and response_text.count('"') >= 2:
            # Find the second quote
            end_quote_idx = response_text.find('"', 1)
            if end_quote_idx != -1:
                response_text = response_text[1:end_quote_idx].strip()
        elif response_text.startswith('"') and response_text.endswith('"'):
             response_text = response_text[1:-1].strip()
            
        print(f"NPC: {response_text}\n")
        
        # 5. Update State
        state["dialogue_history"].append({
            "turn_idx": turn_idx,
            "player_utterance": user_input,
            "response": response_text
        })
        
        # Update current stance based on predicted deltas
        # (Simplified logic: actually applying the deltas)
        for dim in STANCE_DIMS:
            # This would require logic to map levels/deltas to values and update.
            # For this demo, we can just assume the model sees the 'prior_stance' 
            # and predicts the 'next_stance' implicitly in R_t.
            # But the prompt expects 'prior_stance' to be updated for the NEXT turn.
            # So we should update 'current_stance' based on the R_t predictions of THIS turn.
            
            # Extract level from decoded R_t
            level_key = f"{dim}_level"
            if level_key in decoded:
                state["current_stance"][dim] = decoded[level_key]
                
        turn_idx += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to joint model checkpoint")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-4B", help="Base model name")
    parser.add_argument("--scenario", type=str, required=True, help="Path to scenario yaml")
    parser.add_argument("--npc", type=str, required=True, help="NPC ID to chat with")
    parser.add_argument("--quantization", type=str, default="4bit", choices=["4bit", "8bit", "none"], help="Quantization mode")
    
    args = parser.parse_args()
    
    run_interactive(
        checkpoint_dir=args.checkpoint,
        base_model=args.base_model,
        npc_id=args.npc,
        scenario_file=args.scenario,
        quantization=args.quantization
    )
