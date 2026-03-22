
import re
from unittest.mock import MagicMock, patch
from src.data_gen.labeler import Labeler

def test_cleaning_response():
    # Mock client
    mock_client = MagicMock()
    
    # Mock _load_prompt to avoid file I/O
    with patch("src.data_gen.labeler._load_prompt", return_value="Dummy Prompt"):
        # Initialize Labeler with dummy config
        labeler = Labeler(client=mock_client, prompt_config={"response_generation": "dummy/path.txt"})
    
    # Test cases: (dirty_output, expected_clean_output)
    test_cases = [
        ("NPC: I cannot do that.", "I cannot do that."),
        ("Response: I agree.", "I agree."),
        ("NPC (whispering): Keep it down.", "Keep it down."),
        ("\"I am ready.\"", "I am ready."),
        ("NPC: \"Let's go.\"", "Let's go."),
        # Normal case
        ("I will help you.", "I will help you."),
    ]
    
    print("Running response cleaning tests...")
    passed = 0
    for dirty, expected in test_cases:
        # Setup mock return
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=dirty))]
        mock_client.chat.completions.create.return_value = mock_response
        
        # Call method
        # We pass dummy args for the required parameters
        result = labeler.generate_response(
            player_utterance="dummy",
            C_t={}, A_t={}, M_t={}, R_t={}, N_t={}, D_t={},
            npc_profile={"role": "test", "persona_style": []},
            history=[]
        )
        
        # Verify
        if result == expected:
            print(f"[PASS] Input: {dirty!r} -> Output: {result!r}")
            passed += 1
        else:
            print(f"[FAIL] Input: {dirty!r} -> Expected: {expected!r}, Got: {result!r}")

    print(f"\nPassed {passed}/{len(test_cases)} tests.")

if __name__ == "__main__":
    test_cleaning_response()
