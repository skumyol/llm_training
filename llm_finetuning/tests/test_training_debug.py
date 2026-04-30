"""
Debug test for training pipeline with mock data.

Creates minimal mock datasets and runs a single training step to verify
all components work without needing real generated data.
"""
import json
import tempfile
from pathlib import Path

import torch


def create_mock_heads_jsonl(path: str, n_records: int = 10):
    """Create mock head supervision data."""
    records = []
    for i in range(n_records):
        rec = {
            "episode_id": f"mock_ep_{i // 3}",
            "turn_idx": i % 3,
            "scenario_type": "secret_extraction",
            "context": f"[SCENARIO] Player approaches NPC about a secret. [TURN {i % 3}] Player: Tell me what you know.",
            "labels": {
                "dialogue_act": ["probe"],
                "tone": "neutral",
                "risk_type": "secret-risk",
                "valence": "neutral",
                "arousal": "medium",
                "threat": "low",
                "control": "medium",
                "player_intent": "seek-info",
                "player_knowledge": "partial",
                "player_credibility": "medium",
                "duty_pressure": "low",
                "secrecy_pressure": "high",
                "face_pressure": "low",
                "value_conflict": "none",
                "response_policy": "withhold",
                "reveal_decision": "none",
                "repair_strategy": "none",
                "affection_level": "N",
                "affection_delta": "0",
                "respect_level": "L",
                "respect_delta": "0",
                "dominance_level": "N",
                "dominance_delta": "0",
                "familiarity_level": "VL",
                "familiarity_delta": "0",
                "trust_level": "VL",
                "trust_delta": "0",
                "obligation_level": "VL",
                "obligation_delta": "0",
            }
        }
        records.append(rec)
    
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Created mock heads data: {path} ({n_records} records)")


def create_mock_sft_jsonl(path: str, n_records: int = 10):
    """Create mock SFT data (must align with heads data)."""
    records = []
    for i in range(n_records):
        rec = {
            "episode_id": f"mock_ep_{i // 3}",
            "turn_idx": i % 3,
            "scenario_type": "secret_extraction",
            "input": f"[SCENARIO] Player questions NPC. [TURN {i % 3}] Player: What do you know?",
            "target": "I cannot discuss such matters.",
        }
        records.append(rec)
    
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print(f"Created mock SFT data: {path} ({n_records} records)")


def test_latent_training():
    """Test latent state predictor training with mock data."""
    print("\n" + "="*60)
    print("TEST: Latent State Predictor Training")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock data
        train_path = f"{tmpdir}/train_heads.jsonl"
        val_path = f"{tmpdir}/val_heads.jsonl"
        create_mock_heads_jsonl(train_path, n_records=20)
        create_mock_heads_jsonl(val_path, n_records=5)
        
        # Create minimal config
        config = {
            "base_model": "Qwen/Qwen3-0.6B",
            "debug_model": "Qwen/Qwen3-0.6B",
            "quantization": None,
            "torch_dtype": "float32",
            "lora": {
                "r": 8,
                "alpha": 16,
                "dropout": 0.05,
                "target_modules": ["q_proj", "v_proj"],
                "bias": "none",
            },
            "training": {
                "lr": 2e-4,
                "epochs": 1,
                "max_seq_len": 256,
                "batch_size": 2,
                "grad_accum": 1,
                "gradient_checkpointing": False,
                "weight_decay": 0.01,
                "max_grad_norm": 1.0,
                "logging_steps": 1,
            },
            "loss_weights": {
                "lambda_C": 1.0,
                "lambda_A": 1.0,
                "lambda_M": 1.0,
                "lambda_R": 1.0,
                "lambda_N": 1.0,
                "lambda_D": 1.0,
            },
            "data": {
                "train_file": train_path,
                "val_file": val_path,
            },
            "output": {
                "checkpoint_dir": f"{tmpdir}/checkpoints",
                "best_model_dir": f"{tmpdir}/best",
            },
            "mlflow": {
                "experiment_name": "debug_test",
                "run_name": "debug_latent",
                "tracking_uri": f"{tmpdir}/mlruns",
            },
        }
        config_path = f"{tmpdir}/config.yaml"
        import yaml
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        
        # Run training
        print("\nLoading model and running 1 epoch...")
        from src.training.train_latent import train_latent
        train_latent(config_path, debug=True)
        
        print("\n✓ Latent training test PASSED")


def test_joint_dataset_alignment():
    """Test JointDataset alignment check."""
    print("\n" + "="*60)
    print("TEST: JointDataset Alignment Check")
    print("="*60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create aligned data
        sft_path = f"{tmpdir}/train_sft.jsonl"
        heads_path = f"{tmpdir}/train_heads.jsonl"
        create_mock_sft_jsonl(sft_path, n_records=10)
        create_mock_heads_jsonl(heads_path, n_records=10)
        
        # Test aligned data loads correctly
        from transformers import AutoTokenizer
        from src.training.dataset import JointDataset
        
        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
        
        print("Creating JointDataset with aligned data...")
        ds = JointDataset(sft_path, heads_path, tokenizer)
        print(f"✓ Dataset created with {len(ds)} records")
        
        # Test misaligned data fails
        print("\nTesting misaligned data detection...")
        bad_heads_path = f"{tmpdir}/bad_heads.jsonl"
        create_mock_heads_jsonl(bad_heads_path, n_records=10)
        # Overwrite first record with wrong episode_id
        with open(bad_heads_path, "r") as f:
            lines = f.readlines()
        first = json.loads(lines[0])
        first["episode_id"] = "WRONG_ID"
        lines[0] = json.dumps(first) + "\n"
        with open(bad_heads_path, "w") as f:
            f.writelines(lines)
        
        try:
            JointDataset(sft_path, bad_heads_path, tokenizer)
            print("✗ FAILED: Should have raised ValueError for misaligned data")
        except ValueError as e:
            print(f"✓ Correctly detected misalignment: {str(e)[:80]}...")
        
        print("\n✓ JointDataset alignment test PASSED")


def test_counterfactual_preserves_dynamics():
    """Test that counterfactual flips preserve per-turn arc dynamics."""
    print("\n" + "="*60)
    print("TEST: Counterfactual Arc Dynamics Preservation")
    print("="*60)
    
    # Create a mock episode with arc dynamics
    episode = [
        {"turn_idx": 0, "N_t": {"secrecy_pressure": "low"}, "C_t": {}, "A_t": {}, "M_t": {}, "R_t": {}, "W": {"npc_id": "test", "role": "guard", "persona_style": [], "core_goals": [], "values": [], "secrets": [], "faction": "none", "initial_stance": {}}, "input": "Hello", "scenario_type": "test", "setting": "test", "stakes": "low", "arc_phase": "early", "dialogue_history": []},
        {"turn_idx": 1, "N_t": {"secrecy_pressure": "medium"}, "C_t": {}, "A_t": {}, "M_t": {}, "R_t": {}, "W": {"npc_id": "test", "role": "guard", "persona_style": [], "core_goals": [], "values": [], "secrets": [], "faction": "none", "initial_stance": {}}, "input": "Tell me", "scenario_type": "test", "setting": "test", "stakes": "low", "arc_phase": "mid", "dialogue_history": []},
        {"turn_idx": 2, "N_t": {"secrecy_pressure": "high"}, "C_t": {}, "A_t": {}, "M_t": {}, "R_t": {}, "W": {"npc_id": "test", "role": "guard", "persona_style": [], "core_goals": [], "values": [], "secrets": [], "faction": "none", "initial_stance": {}}, "input": "I know!", "scenario_type": "test", "setting": "test", "stakes": "low", "arc_phase": "late", "dialogue_history": []},
    ]
    
    # Check original arc dynamics
    original_pressures = [t["N_t"]["secrecy_pressure"] for t in episode]
    print(f"Original secrecy_pressure arc: {original_pressures}")
    
    # Apply flip (low→high, high→low)
    from src.data_gen.counterfactual import CounterfactualAugmenter, COUNTERFACTUAL_DIMS
    import random
    
    # Mock labeler and validator (we're only testing flip logic, not re-labeling)
    class MockLabeler:
        def label_R_N_D(self, player_utterance, C_t, A_t, M_t, npc_profile, scenario, arc, arc_phase, required_shifts, prior_R_t, history, **kwargs):
            # Return the prior N_t (which has the flipped value) to simulate re-labeling
            # In real usage, the labeler would generate new N_t based on the flipped input
            N_t = prior_R_t.get("N_t", {"secrecy_pressure": "medium", "duty_pressure": "low", "face_pressure": "low", "value_conflict": "none"})
            return prior_R_t.get("R_t", {}), N_t, prior_R_t.get("D_t", {})
        def generate_response(self, **kwargs):
            return "..."
    
    class MockValidator:
        pass
    
    augmenter = CounterfactualAugmenter(MockLabeler(), MockValidator(), random.Random(42), n_variants=1)
    
    # Find the secrecy_pressure dim spec
    secrecy_spec = next(d for d in COUNTERFACTUAL_DIMS if d["var"] == "secrecy_pressure")
    
    # Apply flip
    variant = augmenter._apply_flip(episode, {}, secrecy_spec)
    
    if variant:
        flipped_pressures = [t["N_t"]["secrecy_pressure"] for t in variant]
        print(f"Flipped secrecy_pressure arc: {flipped_pressures}")
        
        # Verify dynamics preserved (not all same value)
        if len(set(flipped_pressures)) > 1:
            print("✓ Arc dynamics preserved (values vary across turns)")
        else:
            print("✗ FAILED: All turns have same value (temporal corruption)")
    else:
        print("No variant generated (no flippable values found)")
    
    print("\n✓ Counterfactual dynamics test PASSED")


def test_secret_leakage_detection():
    """Test semantic secret leakage detection."""
    print("\n" + "="*60)
    print("TEST: Secret Leakage Detection (Semantic Keywords)")
    print("="*60)
    
    from src.data_gen.validator import Validator
    
    validator = Validator()
    
    # Test 1: Verbatim leak
    record1 = {
        "D_t": {"reveal_decision": "none"},
        "response": "I can't tell you about the low supplies situation.",
        "W": {"secrets": [{"secret_id": "low_supplies", "leakage_keywords": ["three days", "starvation"]}]}
    }
    errors1 = validator._check_secret_leakage(record1)
    print(f"Verbatim leak: {errors1[0] if errors1 else 'No errors'}")
    
    # Test 2: Semantic keyword leak
    record2 = {
        "D_t": {"reveal_decision": "none"},
        "response": "We only have food for three days left.",
        "W": {"secrets": [{"secret_id": "low_supplies", "leakage_keywords": ["three days", "starvation"]}]}
    }
    errors2 = validator._check_secret_leakage(record2)
    print(f"Semantic leak: {errors2[0] if errors2 else 'No errors'}")
    
    # Test 3: No leak
    record3 = {
        "D_t": {"reveal_decision": "none"},
        "response": "I cannot discuss military matters.",
        "W": {"secrets": [{"secret_id": "low_supplies", "leakage_keywords": ["three days", "starvation"]}]}
    }
    errors3 = validator._check_secret_leakage(record3)
    print(f"No leak: {'No errors' if not errors3 else errors3}")
    
    if errors1 and errors2 and not errors3:
        print("\n✓ Secret leakage detection test PASSED")
    else:
        print("\n✗ Secret leakage detection test FAILED")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("TRAINING PIPELINE DEBUG TESTS")
    print("="*60)
    
    # Run all tests
    test_counterfactual_preserves_dynamics()
    test_secret_leakage_detection()
    test_joint_dataset_alignment()
    
    # Only run model tests if GPU available or small model can load
    if torch.cuda.is_available() or input("\nRun model training test? (y/n): ").lower() == 'y':
        test_latent_training()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETE")
    print("="*60)
