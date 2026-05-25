"""
Pure-Python audit engine (no Gradio).
Extracted from human_audit_app.py for use with the Next.js frontend.
"""

import html
import hashlib
import json
import random
import re
import sys
from collections import defaultdict, OrderedDict
from datetime import datetime
from threading import Lock
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PLACEHOLDER = "-- select --"

HEADS = {
    "valence": [PLACEHOLDER, "positive", "neutral", "negative"],
    "arousal": [PLACEHOLDER, "low", "medium", "high"],
    "secrecy_pressure": [PLACEHOLDER, "low", "medium", "high"],
    "reveal_decision": [PLACEHOLDER, "none", "hint", "partial", "full"],
    "response_policy": [
        PLACEHOLDER, "answer", "withhold", "deflect", "clarify", "soothe",
        "challenge", "threaten", "negotiate", "test", "partial",
    ],
    "repair_strategy": [PLACEHOLDER, "none", "soften", "apologize", "clarify", "redirect"],
    "trust_level": [PLACEHOLDER, "VL", "L", "N", "H", "VH"],
    "familiarity_level": [PLACEHOLDER, "VL", "L", "N", "H", "VH"],
}

SAMPLE_SIZE = 150
MIN_TIME_PER_TURN = 30

# Module-level caches
_DATA_CACHE: dict[str, list[dict]] = {}
_STRATIFIED_CACHE: dict[str, list[dict]] = {}
_MAX_SESSIONS = 50

_sessions: OrderedDict[str, "AuditState"] = OrderedDict()
_sessions_lock = Lock()


def _effective_name(annotator_name: str | None, prolific_pid: str | None) -> str:
    return (prolific_pid or "").strip() or (annotator_name or "").strip()


def _sanitize_name(name: str | None) -> str:
    cleaned = re.sub(r"[^A-Za-z0_9_\-]", "_", (name or "").strip())
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned.strip("_")[:64] or "annotator"


# ---------------------------------------------------------------------------
# Data loading & stratification
# ---------------------------------------------------------------------------
def load_data(path: Path) -> list[dict]:
    key = str(path)
    if key in _DATA_CACHE:
        return _DATA_CACHE[key]
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    _DATA_CACHE[key] = records
    return records


def stratify_sample(records: list[dict], n: int = SAMPLE_SIZE, seed: int = 42) -> list[dict]:
    cache_key = f"{id(records)}_{n}_{seed}"
    if cache_key in _STRATIFIED_CACHE:
        return _STRATIFIED_CACHE[cache_key]

    by_scenario: dict[str, list[dict]] = defaultdict(list)
    seen_in_scenario: dict[str, set[str]] = defaultdict(set)
    for r in records:
        scenario = r.get("scenario_type", "unknown")
        tid = r.get("turn_id") or f"{r.get('episode_id')}_{r.get('turn_number', -1)}"
        if tid not in seen_in_scenario[scenario]:
            seen_in_scenario[scenario].add(tid)
            by_scenario[scenario].append(r)

    scenarios = sorted(by_scenario.keys())
    per_scenario = n // len(scenarios)
    remainder = n % len(scenarios)

    random.seed(seed)
    sampled = []
    for i, scenario in enumerate(scenarios):
        pool = by_scenario[scenario]
        k = per_scenario + (1 if i < remainder else 0)
        if len(pool) <= k:
            sampled.extend(pool)
        else:
            sampled.extend(random.sample(pool, k))

    random.shuffle(sampled)
    _STRATIFIED_CACHE[cache_key] = sampled
    return sampled


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------
class AuditState:
    def __init__(
        self,
        turns: list[dict],
        annotator: str,
        output_dir: Path,
        test_mode: bool = False,
        prolific_meta: dict | None = None,
    ):
        self.turns = turns
        self.annotator = annotator
        self.output_dir = output_dir
        self.test_mode = test_mode
        self.prolific_meta = prolific_meta or {}
        self.index = 0
        self.annotations: list[dict] = []
        self.start_time: datetime | None = None
        self.end_time: datetime | None = None
        self.turn_start_time: datetime | None = None
        self._ensure_dir()
        self._load_existing()

    def _load_existing(self):
        out_path = self.output_dir / f"audit_{self.annotator}.jsonl"
        if not out_path.exists():
            return
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.annotations.append(json.loads(line))
        except (json.JSONDecodeError, OSError):
            self.annotations = []
        if self.annotations:
            annotated_ids = {a["turn_id"] for a in self.annotations}
            for i, turn in enumerate(self.turns):
                tid = turn.get("turn_id") or f"{turn.get('episode_id')}_{turn.get('turn_number', i)}"
                if tid not in annotated_ids:
                    self.index = i
                    return
            self.index = len(self.turns)

    def begin(self):
        self.start_time = datetime.now()

    def start_turn(self):
        self.turn_start_time = datetime.now()

    def _effective_min_time(self) -> int:
        return 0 if self.test_mode else MIN_TIME_PER_TURN

    def time_remaining(self) -> int:
        min_time = self._effective_min_time()
        if self.turn_start_time is None:
            return min_time
        elapsed = (datetime.now() - self.turn_start_time).total_seconds()
        return max(0, int(min_time - elapsed) + 1)

    def can_submit(self) -> bool:
        if self.turn_start_time is None:
            return False
        return (datetime.now() - self.turn_start_time).total_seconds() >= self._effective_min_time()

    def elapsed_this_turn(self) -> int:
        if self.turn_start_time is None:
            return 0
        return int((datetime.now() - self.turn_start_time).total_seconds())

    def total_elapsed(self) -> int:
        if self.start_time is None:
            return 0
        return int((datetime.now() - self.start_time).total_seconds())

    def end(self):
        self.end_time = datetime.now()
        self._save_metadata()

    def _save_metadata(self):
        meta = {
            "annotator": self.annotator,
            "total_turns": len(self.turns),
            "annotated_count": len(self.annotations),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": (self.end_time - self.start_time).total_seconds()
            if self.start_time and self.end_time
            else None,
        }
        meta_path = self.output_dir / f"audit_{self.annotator}_meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def _ensure_dir(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / f"audit_{self.annotator}.jsonl"
        out_path.touch(exist_ok=True)

    @property
    def current_turn(self) -> dict | None:
        if 0 <= self.index < len(self.turns):
            return self.turns[self.index]
        return None

    def save(self):
        out_path = self.output_dir / f"audit_{self.annotator}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for ann in self.annotations:
                f.write(json.dumps(ann, ensure_ascii=False) + "\n")

    def record(self, values: dict[str, str], notes: str = ""):
        turn = self.current_turn
        if turn is None:
            return
        now = datetime.now()
        turn_elapsed = self.elapsed_this_turn()
        session_elapsed = self.total_elapsed()
        record = {
            "turn_id": turn.get("turn_id") or f"{turn.get('episode_id')}_{turn.get('turn_number', self.index)}",
            "episode_id": turn.get("episode_id"),
            "scenario_type": turn.get("scenario_type"),
            "annotator": self.annotator,
            "labels": {k: values[k] for k in HEADS if values.get(k, PLACEHOLDER) != PLACEHOLDER},
            "notes": notes,
            "recorded_at": now.isoformat(),
            "turn_elapsed_seconds": turn_elapsed,
            "session_elapsed_seconds": session_elapsed,
            **self.prolific_meta,
        }
        self.annotations = [a for a in self.annotations if a["turn_id"] != record["turn_id"]]
        self.annotations.append(record)
        self.save()

    def get_previous_annotation(self) -> dict | None:
        turn = self.current_turn
        if turn is None:
            return None
        tid = turn.get("turn_id") or f"{turn.get('episode_id')}_{turn.get('turn_number', self.index)}"
        for a in self.annotations:
            if a["turn_id"] == tid:
                return a
        return None

    def progress(self) -> str:
        return f"Turn {self.index + 1} / {len(self.turns)}"

    def is_done(self) -> bool:
        return self.index >= len(self.turns)

    def to_dict(self) -> dict:
        """Serialize current state for the frontend."""
        turn = self.current_turn
        prev = self.get_previous_annotation()
        return {
            "annotator": self.annotator,
            "index": self.index,
            "total_turns": len(self.turns),
            "progress": self.progress(),
            "is_done": self.is_done(),
            "test_mode": self.test_mode,
            "turn": turn,
            "previous_labels": prev.get("labels", {}) if prev else {},
            "previous_notes": prev.get("notes", "") if prev else "",
            "time_remaining": self.time_remaining(),
            "can_submit": self.can_submit(),
            "elapsed_this_turn": self.elapsed_this_turn(),
            "total_elapsed": self.total_elapsed(),
            "annotated_count": len(self.annotations),
        }


# ---------------------------------------------------------------------------
# Formatting helpers (return plain dicts for React to render)
# ---------------------------------------------------------------------------
def format_turn(turn: dict | None) -> dict[str, Any]:
    if turn is None:
        return {"scenario": "", "episode": "", "turn_number": "", "scene": "", "history": [], "player": "", "npc": ""}
    return {
        "scenario": turn.get("scenario_type", "unknown"),
        "episode": turn.get("episode_id", "unknown"),
        "turn_number": turn.get("turn_number", "?"),
        "scene": turn.get("scene", ""),
        "history": _parse_history(turn.get("dialogue_history", "")),
        "player": turn.get("player_utterance", ""),
        "npc": turn.get("npc_response", ""),
    }


def _parse_history(history: str) -> list[dict]:
    messages = []
    for line in history.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("Player:"):
            messages.append({"speaker": "Player", "text": line[len("Player:"):].strip()})
        elif line.startswith("NPC:"):
            messages.append({"speaker": "NPC", "text": line[len("NPC:"):].strip()})
        else:
            messages.append({"speaker": "System", "text": line})
    return messages


def get_teacher_label(turn: dict, head: str) -> str:
    labels = turn.get("labels", {})
    return labels.get(head, "N/A")


def make_default_selections() -> dict[str, str]:
    return {h: PLACEHOLDER for h in HEADS}


def all_selected(selections: dict[str, str], test_mode: bool = False) -> bool:
    if test_mode:
        return True
    return all(sel != PLACEHOLDER for sel in selections.values())


def completion_code(annotator: str) -> str:
    return "C1E0GRFO"


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------
def get_or_create_session(
    annotator_name: str,
    output_dir: str,
    data_path: str | None,
    test_mode: bool = False,
    prolific_meta: dict | None = None,
) -> AuditState:
    name = _sanitize_name(annotator_name)
    with _sessions_lock:
        if name in _sessions:
            state = _sessions[name]
            _sessions.move_to_end(name)
            return state

        path = None
        if data_path and Path(data_path).exists():
            path = Path(data_path)

        if path is None:
            raise FileNotFoundError("Data file not found.")

        records = load_data(path)
        if len(records) == 0:
            raise ValueError("No records found in data file.")

        while len(_sessions) >= _MAX_SESSIONS:
            old_name, old_state = _sessions.popitem(last=False)
            old_state.end()

        turns = stratify_sample(records)
        state = AuditState(turns, name, Path(output_dir), test_mode=test_mode, prolific_meta=prolific_meta)
        state.begin()
        state.start_turn()
        _sessions[name] = state
        return state


def get_session(name: str) -> AuditState | None:
    with _sessions_lock:
        return _sessions.get(_sanitize_name(name))


def submit_turn(
    annotator_name: str,
    labels: dict[str, str],
    notes: str,
) -> AuditState:
    name = _sanitize_name(annotator_name)
    with _sessions_lock:
        state = _sessions.get(name)
    if state is None:
        raise ValueError("Session not found.")
    if not all_selected(labels, state.test_mode):
        raise ValueError("Please select a label for all 8 heads before submitting.")
    if not state.can_submit():
        raise ValueError(f"Please wait {state.time_remaining()} more seconds before submitting.")

    state.record(labels, notes)
    state.index += 1

    if state.is_done():
        state.end()
    else:
        state.start_turn()

    with _sessions_lock:
        _sessions.move_to_end(name)
    return state


def go_back(annotator_name: str) -> AuditState:
    name = _sanitize_name(annotator_name)
    with _sessions_lock:
        state = _sessions.get(name)
    if state is None or state.index <= 0:
        raise ValueError("Cannot go back. You are at the first turn.")
    state.index -= 1
    state.start_turn()
    with _sessions_lock:
        _sessions.move_to_end(name)
    return state


def end_session_now(annotator_name: str) -> AuditState:
    name = _sanitize_name(annotator_name)
    with _sessions_lock:
        state = _sessions.get(name)
    if state is None:
        raise ValueError("Session not found.")
    state.end()
    return state
