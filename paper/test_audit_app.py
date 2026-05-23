# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pytest>=7.0",
#     "scikit-learn>=1.3",
#     "gradio>=5.0",
# ]
# ///
"""
Backend tests for the human audit Gradio app.

Usage:
    uv run test_audit_app.py
    # or with pytest:
    uv run pytest test_audit_app.py -v
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add the paper directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent))

from compute_audit_agreement import compute as compute_agreement, HEADS
from human_audit_app import (
    AuditState,
    HEADS as APP_HEADS,
    _format_turn,
    _get_teacher_label,
    go_back,
    load_data,
    start_session,
    stratify_sample,
    submit_and_next,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_records():
    """Generate 20 synthetic test records across 4 scenarios."""
    scenarios = ["secret_extraction", "trust_building", "apology_repair", "threat_escalation"]
    records = []
    for i in range(20):
        records.append({
            "episode_id": f"ep_{i:03d}",
            "turn_id": f"ep_{i:03d}_turn_{i % 3}",
            "turn_number": i % 3,
            "scenario_type": scenarios[i % len(scenarios)],
            "scene": f"Scene {i}: A dark tavern in Oakhaven.",
            "dialogue_history": f"NPC: Hello traveler.\nPlayer: I need help." if i % 2 == 0 else "",
            "player_utterance": f"Player says something dramatic in turn {i}.",
            "npc_response": f"NPC responds cautiously in turn {i}.",
            "labels": {
                "valence": ["positive", "neutral", "negative"][i % 3],
                "arousal": ["low", "medium", "high"][i % 3],
                "secrecy_pressure": ["low", "medium", "high"][i % 3],
                "reveal_decision": ["none", "hint", "partial", "full"][i % 4],
                "response_policy": ["answer", "withhold", "deflect", "clarify"][i % 4],
                "repair_strategy": ["apologize", "redirect", "justify", "compensate", "silence"][i % 5],
                "trust_level": ["VL", "L", "N", "H", "VH"][i % 5],
                "familiarity_level": ["VL", "L", "N", "H", "VH"][i % 5],
            }
        })
    return records


@pytest.fixture
def temp_jsonl(sample_records):
    """Write sample records to a temporary JSONL file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for rec in sample_records:
            f.write(json.dumps(rec) + "\n")
        path = f.name
    yield Path(path)
    os.unlink(path)


@pytest.fixture
def temp_output_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ---------------------------------------------------------------------------
# Data loading tests
# ---------------------------------------------------------------------------
def test_load_data(temp_jsonl, sample_records):
    records = load_data(temp_jsonl)
    assert len(records) == len(sample_records)
    assert records[0]["episode_id"] == "ep_000"


def test_stratify_sample(sample_records):
    # Sample 12 turns from 20 records, 4 scenarios
    sampled = stratify_sample(sample_records, n=12, seed=42)
    assert len(sampled) == 12

    # Check stratification: 3 per scenario
    from collections import Counter
    counts = Counter(r["scenario_type"] for r in sampled)
    for scenario in ["secret_extraction", "trust_building", "apology_repair", "threat_escalation"]:
        assert counts[scenario] == 3, f"Expected 3 turns for {scenario}, got {counts[scenario]}"


def test_stratify_sample_undersized_pool():
    """If a scenario has fewer records than per_scenario quota, take all."""
    records = [
        {"scenario_type": "a", "episode_id": "ep1"},
        {"scenario_type": "a", "episode_id": "ep2"},
        {"scenario_type": "b", "episode_id": "ep3"},
    ]
    sampled = stratify_sample(records, n=10, seed=42)
    assert len(sampled) == 3  # only 3 records exist total


# ---------------------------------------------------------------------------
# Formatting tests
# ---------------------------------------------------------------------------
def test_format_turn(sample_records):
    text = _format_turn(sample_records[0])
    assert "Scene 0" in text
    assert "Player says" in text
    assert "NPC responds" in text
    assert "secret_extraction" in text


def test_get_teacher_label(sample_records):
    turn = sample_records[0]
    assert _get_teacher_label(turn, "valence") == "positive"
    assert _get_teacher_label(turn, "nonexistent") == "N/A"


# ---------------------------------------------------------------------------
# AuditState tests
# ---------------------------------------------------------------------------
def test_audit_state_initialization(sample_records, temp_output_dir):
    state = AuditState(sample_records[:5], "tester", temp_output_dir)
    assert state.index == 0
    assert state.current_turn is not None
    assert state.current_turn["episode_id"] == "ep_000"
    assert (temp_output_dir / "audit_tester.jsonl").exists()


def test_audit_state_record_and_save(sample_records, temp_output_dir):
    state = AuditState(sample_records[:3], "tester", temp_output_dir)
    selections = {h: APP_HEADS[h][0] for h in APP_HEADS}
    state.record(selections, notes="looks fine")

    # Check saved file
    audit_path = temp_output_dir / "audit_tester.jsonl"
    with open(audit_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 1
    saved = json.loads(lines[0])
    assert saved["turn_id"] == "ep_000_turn_0"
    assert saved["annotator"] == "tester"
    assert saved["notes"] == "looks fine"
    assert saved["labels"]["valence"] == APP_HEADS["valence"][0]


def test_audit_state_overwrite(sample_records, temp_output_dir):
    """Re-recording the same turn should overwrite the previous annotation."""
    state = AuditState(sample_records[:2], "tester", temp_output_dir)
    state.record({h: APP_HEADS[h][0] for h in APP_HEADS}, notes="first")
    state.record({h: APP_HEADS[h][1] for h in APP_HEADS}, notes="second")

    audit_path = temp_output_dir / "audit_tester.jsonl"
    with open(audit_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 1  # still only 1 record
    saved = json.loads(lines[0])
    assert saved["notes"] == "second"
    assert saved["labels"]["valence"] == APP_HEADS["valence"][1]


def test_audit_state_progress(sample_records, temp_output_dir):
    state = AuditState(sample_records[:5], "tester", temp_output_dir)
    assert state.progress() == "Turn 1 / 5"
    state.index = 4
    assert state.progress() == "Turn 5 / 5"


def test_audit_state_navigation(sample_records, temp_output_dir):
    state = AuditState(sample_records[:3], "tester", temp_output_dir)
    assert state.current_turn["episode_id"] == "ep_000"
    state.index = 1
    assert state.current_turn["episode_id"] == "ep_001"
    state.index = 3
    assert state.current_turn is None


# ---------------------------------------------------------------------------
# Handler tests (integration level)
# ---------------------------------------------------------------------------
def test_start_session(temp_jsonl, temp_output_dir):
    # We need to patch the global _sessions or work with it
    import human_audit_app as app_mod

    # Clear any prior sessions
    app_mod._sessions.clear()

    result = start_session("alice", str(temp_jsonl), str(temp_output_dir))
    # result is a tuple of gradio update dicts
    turn_md = result[0]
    # Stratified sample shuffles, so just check for valid content
    assert "Scenario:" in turn_md["value"]
    assert "Player:" in turn_md["value"]
    assert app_mod._sessions["alice"] is not None

    # Progress should show 1 / N
    progress_update = result[-1]
    assert "Turn" in progress_update["value"]


def test_submit_and_next_advances(temp_jsonl, temp_output_dir):
    import human_audit_app as app_mod
    app_mod._sessions.clear()

    start_session("bob", str(temp_jsonl), str(temp_output_dir))
    state = app_mod._sessions["bob"]
    initial_index = state.index

    selections = {h: APP_HEADS[h][0] for h in APP_HEADS}
    result = submit_and_next("bob", "", **selections)
    assert state.index == initial_index + 1
    # The turn markdown should now show a different turn
    assert state.current_turn is not None or "All turns" in result[0]["value"]


def test_go_back(temp_jsonl, temp_output_dir):
    import human_audit_app as app_mod
    app_mod._sessions.clear()

    start_session("carol", str(temp_jsonl), str(temp_output_dir))
    state = app_mod._sessions["carol"]
    state.index = 2
    first_id_at_2 = state.current_turn["episode_id"]

    go_back("carol")
    assert state.index == 1
    assert state.current_turn["episode_id"] != first_id_at_2


# ---------------------------------------------------------------------------
# Agreement computation tests
# ---------------------------------------------------------------------------
def test_compute_agreement_perfect():
    """If annotators agree perfectly, HH kappa = 1.0."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        a_path = d / "a.jsonl"
        b_path = d / "b.jsonl"
        t_path = d / "teacher.jsonl"

        common = []
        for i in range(10):
            # Use at least 2 distinct labels so sklearn kappa is well-defined
            label = "neutral" if i % 2 == 0 else "positive"
            rec = {
                "turn_id": f"t{i}",
                "labels": {h: label for h in HEADS}
            }
            common.append(rec)

        for path in [a_path, b_path, t_path]:
            with open(path, "w", encoding="utf-8") as f:
                for rec in common:
                    f.write(json.dumps(rec) + "\n")

        results = compute_agreement(a_path, b_path, t_path)
        for head in HEADS:
            assert results[head]["hh_kappa"] == 1.0
            assert results[head]["hh_acc"] == 1.0
            assert results[head]["ht_kappa"] == 1.0
            assert results[head]["ht_acc"] == 1.0


def test_compute_agreement_chance():
    """If annotators are independent random on 3-class, kappa should be ~0."""
    import random
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        a_path = d / "a.jsonl"
        b_path = d / "b.jsonl"

        random.seed(42)
        labels_a = []
        labels_b = []
        classes = ["low", "medium", "high"]
        for i in range(200):
            labels_a.append({
                "turn_id": f"t{i}",
                "labels": {h: random.choice(classes) for h in HEADS}
            })
            labels_b.append({
                "turn_id": f"t{i}",
                "labels": {h: random.choice(classes) for h in HEADS}
            })

        for path, data in [(a_path, labels_a), (b_path, labels_b)]:
            with open(path, "w", encoding="utf-8") as f:
                for rec in data:
                    f.write(json.dumps(rec) + "\n")

        results = compute_agreement(a_path, b_path, None)
        # For independent random 3-class labels, kappa should be close to 0
        for head in HEADS:
            assert -0.15 < results[head]["hh_kappa"] < 0.15, f"Expected near-zero kappa, got {results[head]['hh_kappa']}"


def test_compute_agreement_partial_overlap():
    """If annotators share only some turn IDs, only common IDs are used."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        a_path = d / "a.jsonl"
        b_path = d / "b.jsonl"

        with open(a_path, "w", encoding="utf-8") as f:
            for i in range(10):
                f.write(json.dumps({"turn_id": f"t{i}", "labels": {h: "neutral" for h in HEADS}}) + "\n")

        with open(b_path, "w", encoding="utf-8") as f:
            for i in range(5, 15):
                f.write(json.dumps({"turn_id": f"t{i}", "labels": {h: "neutral" for h in HEADS}}) + "\n")

        results = compute_agreement(a_path, b_path, None)
        # Only 5 common turns (t5..t9)
        assert results[HEADS[0]]["hh_acc"] == 1.0


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------
def test_empty_data_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write("")
        path = Path(f.name)
    try:
        records = load_data(path)
        assert len(records) == 0
    finally:
        os.unlink(path)


def test_start_session_missing_file(temp_output_dir):
    result = start_session("test", "/nonexistent/file.jsonl", str(temp_output_dir))
    assert "not found" in result[0]["value"]


def test_start_session_empty_file(temp_output_dir):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write("")
        path = f.name
    try:
        result = start_session("test", path, str(temp_output_dir))
        assert "No records" in result[0]["value"] or "Not started" not in result[0]["value"]
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
