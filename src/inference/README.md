# Interactive NPC Inference

This directory contains scripts for interacting with the trained NPC model.

## Usage

To chat with an NPC interactively:

```bash
python src/inference/interactive.py \
    --checkpoint checkpoints/joint_model_best \
    --base_model Qwen/Qwen3-4B \
    --scenario data/world_contexts/oakhaven_siege.yaml \
    --npc commander_vance \
    --quantization 4bit
```

### Arguments

- `--checkpoint`: Path to the trained joint model checkpoint (containing `backbone/` and `heads.pt`).
- `--base_model`: The base HF model used for training (default: `Qwen/Qwen3-4B`).
- `--scenario`: Path to the scenario YAML file defining the world and NPCs.
- `--npc`: The ID of the NPC to converse with (must exist in the scenario file).
- `--quantization`: Quantization mode for loading the model (`4bit`, `8bit`, or `none`). Default is `4bit`.

## How it works

1. **Context Construction**: The script builds a prompt context from the scenario description, NPC profile, and conversation history.
2. **Latent Prediction**: It uses the trained classification heads to predict the NPC's internal state (Latent State) based on the current context.
3. **Response Generation**: It appends the predicted latent state to the context and generates the NPC's response using the fine-tuned backbone.
4. **State Update**: The conversation history and NPC stance are updated for the next turn.
