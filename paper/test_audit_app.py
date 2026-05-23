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
    uv run pytest test_audit_app.py -v
"""

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from compute_audit_agreement import compute as compute_agreement, HEADS
from human_audit_app import (
    PLACEHOLDER,
    AuditState,
    HEADS as APP_HEADS,
    _all_selected,
    _format_turn,
    _get_teacher_label,
    _make_default_selections,
    _sanitize_name,
    _tick_timer,
    back_handler,
    end_session,
    init_session,
    load_data,
    stratify_sample,
    submit_handler,
    toggle_session,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sample_records():
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
    sampled = stratify_sample(sample_records, n=12, seed=42)
    assert len(sampled) == 12
    from collections import Counter
    counts = Counter(r["scenario_type"] for r in sampled)
    for scenario in ["secret_extraction", "trust_building", "apology_repair", "threat_escalation"]:
        assert counts[scenario] == 3


def test_stratify_sample_undersized_pool():
    records = [
        {"scenario_type": "a", "episode_id": "ep1"},
        {"scenario_type": "a", "episode_id": "ep2"},
        {"scenario_type": "b", "episode_id": "ep3"},
    ]
    sampled = stratify_sample(records, n=10, seed=42)
    assert len(sampled) == 3


def test_stratify_sample_deterministic_across_annotators(sample_records):
    """Every annotator must see the same turns in the same order for IAA."""
    a = stratify_sample(sample_records, n=150, seed=42)
    b = stratify_sample(sample_records, n=150, seed=42)
    assert len(a) == len(b)
    assert [r["turn_id"] for r in a] == [r["turn_id"] for r in b]


# ---------------------------------------------------------------------------
# Formatting tests
# ---------------------------------------------------------------------------
def test_format_turn(sample_records):
    text = _format_turn(sample_records[0])
    assert "Scenario:" in text
    assert "Player:" in text
    assert "NPC:" in text


def test_get_teacher_label(sample_records):
    turn = sample_records[0]
    assert _get_teacher_label(turn, "valence") == "positive"
    assert _get_teacher_label(turn, "nonexistent") == "N/A"


# ---------------------------------------------------------------------------
# Selection helpers
# ---------------------------------------------------------------------------
def test_all_selected_true():
    selections = ["positive", "low", "medium", "none", "answer", "apologize", "VL", "L"]
    assert _all_selected(selections) is True


def test_all_selected_false_with_placeholder():
    selections = [PLACEHOLDER, "low", "medium", "none", "answer", "apologize", "VL", "L"]
    assert _all_selected(selections) is False


def test_all_selected_bypass_in_test_mode():
    import human_audit_app as app_mod
    original = app_mod._TEST_MODE
    app_mod._TEST_MODE = True
    try:
        # Even with all placeholders, should pass in test mode
        assert _all_selected([PLACEHOLDER] * 8) is True
    finally:
        app_mod._TEST_MODE = original


def test_make_default_selections():
    defaults = _make_default_selections()
    assert len(defaults) == len(APP_HEADS)
    for h in APP_HEADS:
        assert defaults[h] == PLACEHOLDER


# ---------------------------------------------------------------------------
# AuditState tests
# ---------------------------------------------------------------------------
def test_audit_state_initialization(sample_records, temp_output_dir):
    state = AuditState(sample_records[:5], "tester", temp_output_dir)
    assert state.index == 0
    assert state.current_turn is not None
    assert state.current_turn["episode_id"] == "ep_000"
    assert (temp_output_dir / "audit_tester.jsonl").exists()


def test_audit_state_begin_and_end(sample_records, temp_output_dir):
    state = AuditState(sample_records[:3], "tester", temp_output_dir)
    assert state.start_time is None
    state.begin()
    assert state.start_time is not None
    assert isinstance(state.start_time, datetime)
    state.end()
    assert state.end_time is not None
    assert (temp_output_dir / "audit_tester_meta.json").exists()


def test_audit_state_timer(sample_records, temp_output_dir):
    state = AuditState(sample_records[:3], "tester", temp_output_dir)
    # Before start_turn, timer should require full duration
    assert state.time_remaining() == 50
    assert state.can_submit() is False

    state.start_turn()
    # Immediately after starting, should not be able to submit
    assert state.can_submit() is False
    assert state.time_remaining() > 0

    # Simulate time passing
    state.turn_start_time = datetime.now() - timedelta(seconds=60)
    assert state.can_submit() is True
    assert state.time_remaining() == 0


def test_audit_state_timer_bypass_in_test_mode(sample_records, temp_output_dir):
    import human_audit_app as app_mod
    original = app_mod._TEST_MODE
    app_mod._TEST_MODE = True
    try:
        state = AuditState(sample_records[:3], "tester", temp_output_dir)
        state.start_turn()
        # In test mode, should be able to submit immediately
        assert state.can_submit() is True
        assert state.time_remaining() <= 1
    finally:
        app_mod._TEST_MODE = original


def test_audit_state_record_and_save(sample_records, temp_output_dir):
    import time
    state = AuditState(sample_records[:3], "tester", temp_output_dir)
    state.start_time = datetime.now() - timedelta(seconds=300)
    state.start_turn()
    time.sleep(0.1)
    # Use real label values, not PLACEHOLDER
    selections = {h: APP_HEADS[h][1] for h in APP_HEADS}
    state.record(selections, notes="looks fine")
    audit_path = temp_output_dir / "audit_tester.jsonl"
    with open(audit_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 1
    saved = json.loads(lines[0])
    assert saved["turn_id"] == "ep_000_turn_0"
    assert saved["annotator"] == "tester"
    assert saved["notes"] == "looks fine"
    assert len(saved["labels"]) == len(APP_HEADS)
    assert "recorded_at" in saved
    assert "turn_elapsed_seconds" in saved
    assert "session_elapsed_seconds" in saved
    assert saved["session_elapsed_seconds"] >= 300
    assert saved["turn_elapsed_seconds"] >= 0


def test_audit_state_record_filters_placeholders(sample_records, temp_output_dir):
    state = AuditState(sample_records[:3], "tester", temp_output_dir)
    selections = {h: PLACEHOLDER for h in APP_HEADS}
    state.record(selections, notes="empty")
    audit_path = temp_output_dir / "audit_tester.jsonl"
    with open(audit_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 1
    saved = json.loads(lines[0])
    assert saved["labels"] == {}


def test_audit_state_overwrite(sample_records, temp_output_dir):
    state = AuditState(sample_records[:2], "tester", temp_output_dir)
    state.record({h: APP_HEADS[h][1] for h in APP_HEADS}, notes="first")
    state.record({h: APP_HEADS[h][2] for h in APP_HEADS}, notes="second")
    audit_path = temp_output_dir / "audit_tester.jsonl"
    with open(audit_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    assert len(lines) == 1
    saved = json.loads(lines[0])
    assert saved["notes"] == "second"


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


def test_audit_state_get_previous_annotation(sample_records, temp_output_dir):
    state = AuditState(sample_records[:3], "tester", temp_output_dir)
    assert state.get_previous_annotation() is None
    selections = {h: APP_HEADS[h][1] for h in APP_HEADS}
    state.record(selections, notes="test")
    retrieved = state.get_previous_annotation()
    assert retrieved is not None
    assert retrieved["valence"] == APP_HEADS["valence"][1]


# ---------------------------------------------------------------------------
# Handler tests
# ---------------------------------------------------------------------------
def test_init_session_success(temp_jsonl, temp_output_dir):
    import human_audit_app as app_mod
    app_mod._sessions.clear()
    app_mod._DEFAULT_DATA_PATH = str(temp_jsonl)

    result = init_session("alice", str(temp_output_dir), None)
    assert "Scenario:" in result[0]["value"]
    assert app_mod._sessions[_sanitize_name("alice")] is not None
    assert "Turn" in result[3]["value"]
    # Verify timer was started
    state = app_mod._sessions[_sanitize_name("alice")]
    assert state.turn_start_time is not None


def test_init_session_no_name(temp_jsonl, temp_output_dir):
    result = init_session("", str(temp_output_dir), None)
    assert "color:red" in result[0]["value"]


def test_init_session_missing_file(temp_output_dir):
    import human_audit_app as app_mod
    app_mod._DEFAULT_DATA_PATH = None
    result = init_session("test", str(temp_output_dir), None)
    assert "not found" in result[0]["value"]


def test_init_session_empty_file(temp_output_dir):
    import human_audit_app as app_mod
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write("")
        path = f.name
    try:
        app_mod._DEFAULT_DATA_PATH = path
        result = init_session("test", str(temp_output_dir), None)
        assert "No records" in result[0]["value"]
    finally:
        os.unlink(path)


def test_submit_and_next_advances(temp_jsonl, temp_output_dir):
    import human_audit_app as app_mod
    app_mod._sessions.clear()
    app_mod._DEFAULT_DATA_PATH = str(temp_jsonl)

    init_session("bob", str(temp_output_dir), None)
    state = app_mod._sessions[_sanitize_name("bob")]
    initial_index = state.index

    # Bypass timer by setting turn_start_time far in the past
    state.turn_start_time = datetime.now() - timedelta(seconds=100)

    # Use real label values (index 1, skipping PLACEHOLDER at index 0)
    selections = [APP_HEADS[h][1] for h in APP_HEADS]
    result = submit_handler("bob", "", *selections)
    assert state.index == initial_index + 1
    # Check warning is cleared on successful submit
    assert result[4]["value"] == ""


def test_submit_blocks_on_timer(temp_jsonl, temp_output_dir):
    import human_audit_app as app_mod
    app_mod._sessions.clear()
    app_mod._DEFAULT_DATA_PATH = str(temp_jsonl)

    init_session("timer_test", str(temp_output_dir), None)
    state = app_mod._sessions[_sanitize_name("timer_test")]
    # turn_start_time was set by init_session to now, so timer should block

    selections = [APP_HEADS[h][1] for h in APP_HEADS]
    result = submit_handler("timer_test", "", *selections)
    # Index should NOT advance
    assert state.index == 0
    # Warning message should mention waiting
    assert "wait" in result[4]["value"].lower()


def test_submit_blocks_on_missing_selections(temp_jsonl, temp_output_dir):
    import human_audit_app as app_mod
    app_mod._sessions.clear()
    app_mod._DEFAULT_DATA_PATH = str(temp_jsonl)

    init_session("missing_test", str(temp_output_dir), None)
    state = app_mod._sessions[_sanitize_name("missing_test")]
    state.turn_start_time = datetime.now() - timedelta(seconds=100)

    # Submit with some PLACEHOLDER values
    selections = [APP_HEADS[h][1] for h in list(APP_HEADS.keys())[:4]]
    selections += [PLACEHOLDER] * 4
    result = submit_handler("missing_test", "", *selections)
    # Index should NOT advance
    assert state.index == 0
    # Warning should mention selecting all heads
    assert "all 8 heads" in result[4]["value"].lower()


def test_back_handler(temp_jsonl, temp_output_dir):
    import human_audit_app as app_mod
    app_mod._sessions.clear()
    app_mod._DEFAULT_DATA_PATH = str(temp_jsonl)

    init_session("carol", str(temp_output_dir), None)
    state = app_mod._sessions[_sanitize_name("carol")]
    state.index = 2
    first_id_at_2 = state.current_turn["episode_id"]

    back_handler("carol")
    assert state.index == 1
    assert state.current_turn["episode_id"] != first_id_at_2


def test_toggle_session_begin(temp_jsonl, temp_output_dir):
    import human_audit_app as app_mod
    app_mod._sessions.clear()
    app_mod._DEFAULT_DATA_PATH = str(temp_jsonl)

    # is_active=False means we should begin
    result = toggle_session(False, "toggle_alice", str(temp_output_dir), None)
    # First output is session_active state (should be True)
    assert result[0]["value"] is True
    # Second output is turn_display
    assert "Scenario:" in result[1]["value"]
    assert app_mod._sessions[_sanitize_name("toggle_alice")] is not None


def test_toggle_session_end(temp_jsonl, temp_output_dir):
    import human_audit_app as app_mod
    app_mod._sessions.clear()
    app_mod._DEFAULT_DATA_PATH = str(temp_jsonl)

    init_session("toggle_bob", str(temp_output_dir), None)
    state = app_mod._sessions[_sanitize_name("toggle_bob")]
    state.begin()

    # is_active=True means we should end
    result = toggle_session(True, "toggle_bob", str(temp_output_dir), None)
    # First output is session_active state (should be False)
    assert result[0]["value"] is False
    # Second output is turn_display with "Audit ended"
    assert "ended" in result[1]["value"].lower()
    assert (temp_output_dir / "audit_toggle_bob_meta.json").exists()


def test_end_session_not_found(temp_output_dir):
    result = end_session("nonexistent")
    assert "not found" in result[0]["value"].lower()


# ---------------------------------------------------------------------------
# Agreement computation tests
# ---------------------------------------------------------------------------
def test_compute_agreement_perfect():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        a_path = d / "a.jsonl"
        b_path = d / "b.jsonl"
        t_path = d / "teacher.jsonl"

        common = []
        for i in range(10):
            label = "neutral" if i % 2 == 0 else "positive"
            rec = {"turn_id": f"t{i}", "labels": {h: label for h in HEADS}}
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
    import random
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        a_path = d / "a.jsonl"
        b_path = d / "b.jsonl"

        random.seed(42)
        classes = ["low", "medium", "high"]
        for i in range(200):
            for path, src in [(a_path, [random.choice(classes) for _ in HEADS]),
                              (b_path, [random.choice(classes) for _ in HEADS])]:
                label_dict = {h: src[j] for j, h in enumerate(HEADS)}
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"turn_id": f"t{i}", "labels": label_dict}) + "\n")

        results = compute_agreement(a_path, b_path, None)
        for head in HEADS:
            assert -0.15 < results[head]["hh_kappa"] < 0.15


def test_compute_agreement_partial_overlap():
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
        assert results[HEADS[0]]["hh_acc"] == 1.0


# ---------------------------------------------------------------------------
# Sanitization tests
# ---------------------------------------------------------------------------
def test_sanitize_name_normal():
    assert _sanitize_name("alice") == "alice"
    assert _sanitize_name("Alice_Smith-99") == "Alice_Smith-99"


def test_sanitize_name_special_chars():
    assert _sanitize_name("alice@prolific.co") == "alice_prolific_co"
    assert _sanitize_name("  spaces  ") == "spaces"
    assert _sanitize_name("你好/世界") == "annotator"
    assert _sanitize_name("!!!") == "annotator"


def test_sanitize_name_truncation():
    long_name = "a" * 100
    assert len(_sanitize_name(long_name)) == 64


# ---------------------------------------------------------------------------
# Resume / reload tests
# ---------------------------------------------------------------------------
def test_audit_state_loads_existing_annotations(sample_records, temp_output_dir):
    """AuditState should resume from existing JSONL file."""
    # Pre-populate an annotation file
    pre_path = temp_output_dir / "audit_resume_user.jsonl"
    selections = {h: APP_HEADS[h][1] for h in APP_HEADS}
    with open(pre_path, "w", encoding="utf-8") as f:
        for rec in sample_records[:5]:
            tid = rec.get("turn_id") or f"{rec.get('episode_id')}_{rec.get('turn_number')}"
            f.write(json.dumps({
                "turn_id": tid,
                "labels": selections,
            }) + "\n")

    state = AuditState(sample_records[:10], "resume_user", temp_output_dir)
    # Should have loaded 5 annotations and advanced index to turn 6
    assert len(state.annotations) == 5
    assert state.index == 5
    assert state.progress() == "Turn 6 / 10"


def test_audit_state_resume_all_done(sample_records, temp_output_dir):
    """If all turns are already annotated, index should be at len(turns)."""
    pre_path = temp_output_dir / "audit_done_user.jsonl"
    selections = {h: APP_HEADS[h][1] for h in APP_HEADS}
    with open(pre_path, "w", encoding="utf-8") as f:
        for rec in sample_records[:5]:
            tid = rec.get("turn_id") or f"{rec.get('episode_id')}_{rec.get('turn_number')}"
            f.write(json.dumps({
                "turn_id": tid,
                "labels": selections,
            }) + "\n")

    state = AuditState(sample_records[:5], "done_user", temp_output_dir)
    assert len(state.annotations) == 5
    assert state.index == 5
    assert state.is_done()


# ---------------------------------------------------------------------------
# Duplicate session test
# ---------------------------------------------------------------------------
def test_init_session_blocks_duplicate(temp_jsonl, temp_output_dir):
    import human_audit_app as app_mod
    app_mod._sessions.clear()
    app_mod._DEFAULT_DATA_PATH = str(temp_jsonl)

    init_session("dave", str(temp_output_dir), None)
    assert _sanitize_name("dave") in app_mod._sessions

    # Try to start again with same name
    result = init_session("dave", str(temp_output_dir), None)
    assert "already active" in result[0]["value"].lower()


# ---------------------------------------------------------------------------
# Timer tick tests
# ---------------------------------------------------------------------------
def test_tick_timer_no_session():
    assert _tick_timer("nobody") == ""


def test_tick_timer_active_session(sample_records, temp_output_dir):
    state = AuditState(sample_records[:3], "timer_user", temp_output_dir)
    state.begin()
    state.start_turn()

    import human_audit_app as app_mod
    app_mod._sessions["timer_user"] = state

    label = _tick_timer("timer_user")
    assert "Turn time:" in label
    assert "Session time:" in label

    # Clean up
    del app_mod._sessions["timer_user"]


# ---------------------------------------------------------------------------
# Completion code tests
# ---------------------------------------------------------------------------
def test_completion_code_deterministic():
    from human_audit_app import _completion_code
    assert _completion_code("alice") == _completion_code("alice")
    assert _completion_code("alice") != _completion_code("bob")
    assert len(_completion_code("alice")) == 8


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
