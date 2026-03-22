import re
from unittest.mock import MagicMock, patch
from src.data_gen.labeler import Labeler

def test_cleaning():
    # Mock client
    mock_client = MagicMock()
    
    # Mock _load_prompt to avoid file I/O
    with patch("src.data_gen.labeler._load_prompt", return_value="Dummy Prompt"):
        # Initialize Labeler with dummy config
        labeler = Labeler(client=mock_client, prompt_config={"player_generation": "dummy/path.txt"})
    
    # Test cases: (dirty_output, expected_clean_output, history_context)
    history_base = [{"turn_idx": 1, "player_utterance": "prev input", "response": "I cannot allow that."}]
    
    test_cases = [
        ("PLAYER UTTERANCE: Hello there.", "Hello there.", []),
        ("Player: I disagree.", "I disagree.", []),
        ("Me: What is this?", "What is this?", []),
        ("You: That's fine.", "That's fine.", []),
        ("\"I am ready.\"", "I am ready.", []),
        ("PLAYER UTTERANCE: \"Let's go.\"", "Let's go.", []),
        # Repetition case (User Simulator blindly completing NPC's text)
        ("NPC: I cannot allow that.\nPlayer: But why?", "But why?", history_base),
        # Repetition case with prefix and completion
        ("I cannot allow that. But why?", "But why?", history_base),
        # No repetition (normal case)
        ("But why?", "But why?", history_base),
        # Complex repetition with whitespace
        ("  I cannot allow that.   \nPLAYER UTTERANCE: But why?  ", "But why?", history_base),
        # Case where repetition isn't exact (should not strip if not starting with it)
        ("I allow that. But why?", "I allow that. But why?", history_base),
        # Partial repetition (suffix) case - observed in wild
        # NPC: "... Do you seek to aid or undermine?" -> Player: "Do you seek to aid or undermine?"
        ("I cannot allow that.", "I remain silent, watching carefully.", history_base), 
    ]
    
    print("Running cleaning tests...")
    passed = 0
    
    # 1. Run simple single-shot tests
    print("--- Single Shot Tests ---")
    for dirty, expected, history in test_cases:
        # Setup mock return
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=dirty))]
        mock_client.chat.completions.create.return_value = mock_response
        
        # Call method
        result = labeler.generate_player_utterance(
            scenario={"scenario_type": "test", "world_context": ""},
            arc_phase="test",
            npc_profile={"role": "test", "persona_style": []},
            history=history
        )
        
        # Verify
        if result == expected:
            print(f"[PASS] Input: {dirty!r} -> Output: {result!r}")
            passed += 1
        else:
            print(f"[FAIL] Input: {dirty!r} -> Expected: {expected!r}, Got: {result!r}")

    # 2. Run Retry Test
    print("\n--- Retry Logic Test ---")
    # Scenario: 
    # Attempt 1: "I cannot allow that." (NPC Repeat -> Fail)
    # Attempt 2: "Valid player response." (Success)
    
    resp1 = MagicMock()
    resp1.choices = [MagicMock(message=MagicMock(content="I cannot allow that."))]
    
    resp2 = MagicMock()
    resp2.choices = [MagicMock(message=MagicMock(content="Valid player response."))]
    
    mock_client.chat.completions.create.side_effect = [resp1, resp2]
    
    result = labeler.generate_player_utterance(
        scenario={"scenario_type": "test", "world_context": ""},
        arc_phase="test",
        npc_profile={"role": "test", "persona_style": []},
        history=history_base # contains "I cannot allow that." as last response
    )
    
    if result == "Valid player response.":
        print(f"[PASS] Retry Logic: Got {result!r}")
        passed += 1
    else:
        print(f"[FAIL] Retry Logic: Expected 'Valid player response.', Got {result!r}")

    print(f"\nPassed {passed}/{len(test_cases) + 1} tests.")

if __name__ == "__main__":
    test_cleaning()
