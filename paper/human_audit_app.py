# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "gradio>=5.0",
#     "scikit-learn>=1.3",
# ]
# ///
"""
Human Audit Gradio Interface

Usage:
    uv run human_audit_app.py --data path/to/test_heads.jsonl --output ./audit_results

Expected input format (test_heads.jsonl):
    Each line is a JSON object with at least these fields:
    {
        "episode_id": "...",
        "scenario_type": "secret_extraction",
        "scene": "...",
        "dialogue_history": "...",
        "player_utterance": "...",
        "npc_response": "...",
        "labels": {
            "valence": "negative",
            "arousal": "high",
            "secrecy_pressure": "high",
            "reveal_decision": "none",
            "response_policy": "deflect",
            "repair_strategy": "redirect",
            "trust_level": "VL",
            "familiarity_level": "N",
            ... other heads ...
        }
    }

The app stratifies 150 turns (~21 per scenario type), presents them one by one,
and saves annotations as JSONL with one record per turn per annotator.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import gradio as gr

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HEADS = {
    "valence":           ["positive", "neutral", "negative"],
    "arousal":           ["low", "medium", "high"],
    "secrecy_pressure":  ["low", "medium", "high"],
    "reveal_decision":   ["none", "hint", "partial", "full"],
    "response_policy":   [
        "answer", "withhold", "deflect", "clarify", "soothe",
        "challenge", "threaten", "negotiate", "redirect", "partial"
    ],
    "repair_strategy":   ["apologize", "redirect", "justify", "compensate", "silence"],
    "trust_level":       ["VL", "L", "N", "H", "VH"],
    "familiarity_level": ["VL", "L", "N", "H", "VH"],
}

SAMPLE_SIZE = 150


# ---------------------------------------------------------------------------
# Data loading & stratification
# ---------------------------------------------------------------------------
def load_data(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def stratify_sample(records: list[dict], n: int = SAMPLE_SIZE, seed: int = 42) -> list[dict]:
    """Stratified random sample: ~n / scenario_type turns."""
    by_scenario: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        scenario = r.get("scenario_type", "unknown")
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
    return sampled


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------
class AuditState:
    def __init__(self, turns: list[dict], annotator: str, output_dir: Path):
        self.turns = turns
        self.annotator = annotator
        self.output_dir = output_dir
        self.index = 0
        self.annotations: list[dict] = []
        self._ensure_dir()

    def _ensure_dir(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Touch the output file so it exists even before the first annotation
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
        record = {
            "turn_id": turn.get("turn_id") or f"{turn.get('episode_id')}_{turn.get('turn_number', self.index)}",
            "episode_id": turn.get("episode_id"),
            "scenario_type": turn.get("scenario_type"),
            "annotator": self.annotator,
            "labels": {k: values[k] for k in HEADS},
            "notes": notes,
        }
        # Overwrite if we revisit
        self.annotations = [a for a in self.annotations if a["turn_id"] != record["turn_id"]]
        self.annotations.append(record)
        self.save()

    def progress(self) -> str:
        return f"Turn {self.index + 1} / {len(self.turns)}"


# Global registry of active sessions
_sessions: dict[str, AuditState] = {}


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def _format_turn(turn: dict) -> str:
    scene = turn.get("scene", "")
    history = turn.get("dialogue_history", "")
    player = turn.get("player_utterance", "")
    npc = turn.get("npc_response", "")
    scenario = turn.get("scenario_type", "unknown")
    episode = turn.get("episode_id", "unknown")

    parts = [
        f"**Scenario:** {scenario}  |  **Episode:** {episode}",
        "---",
        f"**Scene:** {scene}" if scene else "",
        f"**Dialogue History:**\n{history}" if history else "",
        f"**Player:** {player}",
        f"**NPC Response:** {npc}",
    ]
    return "\n\n".join(p for p in parts if p)


def _get_teacher_label(turn: dict, head: str) -> str:
    labels = turn.get("labels", {})
    return labels.get(head, "N/A")


# ---------------------------------------------------------------------------
# Gradio handlers
# ---------------------------------------------------------------------------
def start_session(annotator_name: str, data_path: str, output_dir: str) -> tuple:
    path = Path(data_path)
    if not path.exists():
        return (
            gr.update(value=f"File not found: {data_path}"),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    records = load_data(path)
    if len(records) == 0:
        return (
            gr.update(value="No records found in file."),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    turns = stratify_sample(records)
    state = AuditState(turns, annotator_name, Path(output_dir))
    _sessions[annotator_name] = state

    turn = state.current_turn
    text = _format_turn(turn)
    teacher_labels = {f"teacher_{h}": _get_teacher_label(turn, h) for h in HEADS}

    return (
        gr.update(value=text),
        gr.update(visible=True),
        gr.update(visible=False),
        *[gr.update(value=v) for v in teacher_labels.values()],
        gr.update(value=state.progress()),
    )


def submit_and_next(annotator_name: str, notes: str, **selections) -> tuple:
    state = _sessions.get(annotator_name)
    if state is None:
        return (gr.update(value="Session not found. Restart."),) + (gr.update(),) * (len(HEADS) + 3)

    state.record(selections, notes)
    state.index += 1

    if state.index >= len(state.turns):
        return (
            gr.update(value="All turns annotated. Thank you!"),
            gr.update(visible=False),
            gr.update(visible=True, value=f"Saved to: {state.output_dir / f'audit_{annotator_name}.jsonl'}"),
            *[gr.update(value="") for _ in HEADS],
            gr.update(value="Done"),
        )

    turn = state.current_turn
    text = _format_turn(turn)
    teacher_labels = {f"teacher_{h}": _get_teacher_label(turn, h) for h in HEADS}

    return (
        gr.update(value=text),
        gr.update(visible=True),
        gr.update(visible=False),
        *[gr.update(value=v) for v in teacher_labels.values()],
        gr.update(value=state.progress()),
    )


def go_back(annotator_name: str) -> tuple:
    state = _sessions.get(annotator_name)
    if state is None or state.index <= 0:
        return (gr.update(),) * (len(HEADS) + 3)

    state.index -= 1
    turn = state.current_turn
    text = _format_turn(turn)
    teacher_labels = {f"teacher_{h}": _get_teacher_label(turn, h) for h in HEADS}

    # Pre-fill annotator's previous choices if any
    prev = next((a for a in state.annotations if a["turn_id"] == (turn.get("turn_id") or f"{turn.get('episode_id')}_{turn.get('turn_number', state.index)}")), None)
    if prev:
        user_values = [prev["labels"].get(h, HEADS[h][0]) for h in HEADS]
    else:
        user_values = [HEADS[h][0] for h in HEADS]

    return (
        gr.update(value=text),
        gr.update(visible=True),
        gr.update(visible=False),
        *[gr.update(value=v) for v in teacher_labels.values()],
        gr.update(value=state.progress()),
        *[gr.update(value=v) for v in user_values],
    )


# ---------------------------------------------------------------------------
# Build interface
# ---------------------------------------------------------------------------
def build_interface(data_path: str | None, output_dir: str) -> gr.Blocks:
    with gr.Blocks(title="NPC Social-State Human Audit") as demo:
        gr.Markdown("# NPC Social-State Human Audit")
        gr.Markdown(
            "Evaluate whether synthetic social-state labels match human judgment. "
            "You will see 150 stratified turns. Select the label that best describes each dimension."
        )

        with gr.Row():
            annotator_name = gr.Textbox(label="Annotator Name", placeholder="e.g. alice")
            data_path_box = gr.Textbox(
                label="Path to test_heads.jsonl",
                value=data_path or "",
                placeholder="/path/to/test_heads.jsonl",
            )
            output_dir_box = gr.Textbox(label="Output Directory", value=output_dir)

        start_btn = gr.Button("Start / Load Session", variant="primary")

        progress_label = gr.Textbox(label="Progress", value="Not started", interactive=False)

        turn_display = gr.Markdown("### Waiting to start...")

        # Teacher labels (read-only, for reference)
        with gr.Accordion("Teacher labels (reference only -- do not peek before judging!)", open=False):
            teacher_boxes = {}
            for head in HEADS:
                teacher_boxes[head] = gr.Textbox(label=head, interactive=False)

        # User selections
        user_choices = {}
        with gr.Row():
            with gr.Column():
                for head in HEADS:
                    user_choices[head] = gr.Radio(
                        label=head,
                        choices=HEADS[head],
                        value=HEADS[head][0],
                    )

        notes_box = gr.Textbox(label="Notes (optional)", placeholder="Ambiguity, disagreements, etc.")

        with gr.Row():
            back_btn = gr.Button("Previous Turn")
            submit_btn = gr.Button("Submit & Next Turn", variant="primary")

        done_message = gr.Textbox(
            label="Status",
            value="",
            visible=False,
            interactive=False,
        )

        # Event wiring
        start_outputs = [
            turn_display,
            submit_btn,
            done_message,
            *teacher_boxes.values(),
            progress_label,
        ]
        start_btn.click(
            fn=start_session,
            inputs=[annotator_name, data_path_box, output_dir_box],
            outputs=start_outputs,
        )

        submit_outputs = [
            turn_display,
            submit_btn,
            done_message,
            *teacher_boxes.values(),
            progress_label,
        ]
        submit_inputs = [annotator_name, notes_box, *user_choices.values()]
        submit_btn.click(
            fn=submit_and_next,
            inputs=submit_inputs,
            outputs=submit_outputs,
        )

        back_outputs = [
            turn_display,
            submit_btn,
            done_message,
            *teacher_boxes.values(),
            progress_label,
            *user_choices.values(),
        ]
        back_btn.click(
            fn=go_back,
            inputs=[annotator_name],
            outputs=back_outputs,
        )

    return demo


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Human audit Gradio app")
    parser.add_argument("--data", default=None, help="Path to test_heads.jsonl")
    parser.add_argument("--output", default="./audit_results", help="Directory to save annotations")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    args = parser.parse_args()

    demo = build_interface(args.data, args.output)
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
