"""
FastAPI backend for the human audit app (replaces Gradio).
"""

import argparse
import json
import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Allow importing audit_core from the same directory
sys.path.insert(0, str(Path(__file__).parent))
from audit_core import (
    HEADS,
    PLACEHOLDER,
    SAMPLE_SIZE,
    format_turn,
    get_or_create_session,
    get_session,
    submit_turn,
    go_back,
    end_session_now,
    completion_code,
    _sanitize_name,
)
from analysis import (
    qc_annotator,
    compute_agreement,
    evaluate_all,
    generate_final_results,
    load_jsonl,
    index_by_turn_id,
)

app = FastAPI(title="NPC Social-State Human Audit API")

# CORS: allow the Next.js dev server and any deployed origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Config (set at startup)
# ---------------------------------------------------------------------------
_DEFAULT_DATA_PATH: str | None = None
_OUTPUT_DIR: str = "./audit_results"
_TEST_MODE: bool = False


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class BeginRequest(BaseModel):
    annotator_name: str
    prolific_pid: str = ""
    output_dir: str | None = None
    data_path: str | None = None
    test_mode: bool | None = None
    sample_size: int | None = None


class SubmitRequest(BaseModel):
    annotator_name: str
    labels: dict[str, str]
    notes: str = ""


class BackRequest(BaseModel):
    annotator_name: str


class EndRequest(BaseModel):
    annotator_name: str


class TimerResponse(BaseModel):
    time_remaining: int
    can_submit: bool
    elapsed_this_turn: int
    total_elapsed: int
    progress: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_state_response(state):
    turn_data = format_turn(state.current_turn)
    prev = state.get_previous_annotation()
    done = state.is_done()
    code = completion_code(state.annotator) if done else ""
    return {
        "annotator": state.annotator,
        "index": state.index,
        "total_turns": len(state.turns),
        "progress": state.progress(),
        "is_done": done,
        "test_mode": state.test_mode,
        "turn": turn_data,
        "previous_labels": prev.get("labels", {}) if prev else {},
        "previous_notes": prev.get("notes", "") if prev else "",
        "time_remaining": state.time_remaining(),
        "can_submit": state.can_submit(),
        "elapsed_this_turn": state.elapsed_this_turn(),
        "total_elapsed": state.total_elapsed(),
        "annotated_count": len(state.annotations),
        "completion_code": code,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/heads")
def get_heads():
    """Return the label heads and their choices."""
    return {"heads": HEADS, "placeholder": PLACEHOLDER}


@app.post("/api/session")
def begin_session(req: BeginRequest):
    output_dir = req.output_dir or _OUTPUT_DIR
    data_path = req.data_path or _DEFAULT_DATA_PATH
    test_mode = req.test_mode if req.test_mode is not None else _TEST_MODE

    effective_name = req.prolific_pid.strip() or req.annotator_name.strip()
    if not effective_name:
        raise HTTPException(status_code=400, detail="Annotator name or Prolific PID is required.")

    pmeta = {}
    if req.prolific_pid.strip():
        pmeta["prolific_pid"] = req.prolific_pid.strip()

    try:
        state = get_or_create_session(
            effective_name,
            output_dir,
            data_path,
            test_mode=test_mode,
            prolific_meta=pmeta,
            sample_size=req.sample_size,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return _build_state_response(state)


@app.get("/api/session/{annotator_name}")
def read_session(annotator_name: str):
    state = get_session(annotator_name)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return _build_state_response(state)


@app.post("/api/session/{annotator_name}/submit")
def submit(annotator_name: str, req: SubmitRequest):
    if req.annotator_name != annotator_name:
        raise HTTPException(status_code=400, detail="Annotator name mismatch.")
    try:
        state = submit_turn(annotator_name, req.labels, req.notes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _build_state_response(state)


@app.post("/api/session/{annotator_name}/back")
def back(annotator_name: str, req: BackRequest):
    if req.annotator_name != annotator_name:
        raise HTTPException(status_code=400, detail="Annotator name mismatch.")
    try:
        state = go_back(annotator_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _build_state_response(state)


@app.post("/api/session/{annotator_name}/end")
def end(annotator_name: str, req: EndRequest):
    if req.annotator_name != annotator_name:
        raise HTTPException(status_code=400, detail="Annotator name mismatch.")
    try:
        state = end_session_now(annotator_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return _build_state_response(state)


@app.get("/api/session/{annotator_name}/timer")
def timer(annotator_name: str):
    state = get_session(annotator_name)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {
        "time_remaining": state.time_remaining(),
        "can_submit": state.can_submit(),
        "elapsed_this_turn": state.elapsed_this_turn(),
        "total_elapsed": state.total_elapsed(),
        "progress": state.progress(),
    }


# ---------------------------------------------------------------------------
# Guidelines content served as JSON so the React app can render it
# ---------------------------------------------------------------------------
GUIDELINES = [
    {
        "head": "valence",
        "title": "Valence",
        "description": "Emotional valence of the NPC toward the player / situation",
        "choices": [
            {"value": "positive", "label": "Positive", "help": "warm, approving, optimistic"},
            {"value": "neutral", "label": "Neutral", "help": "flat, factual, neither warm nor cold"},
            {"value": "negative", "label": "Negative", "help": "cold, disapproving, hostile"},
        ],
    },
    {
        "head": "arousal",
        "title": "Arousal",
        "description": "Intensity of the NPC's emotional state",
        "choices": [
            {"value": "low", "label": "Low", "help": "calm, relaxed, indifferent"},
            {"value": "medium", "label": "Medium", "help": "engaged, alert, moderately tense"},
            {"value": "high", "label": "High", "help": "agitated, excited, very tense"},
        ],
    },
    {
        "head": "secrecy_pressure",
        "title": "Secrecy Pressure",
        "description": "How much the NPC feels pressured to keep information hidden",
        "choices": [
            {"value": "low", "label": "Low", "help": "no secret to protect, or no pressure"},
            {"value": "medium", "label": "Medium", "help": "some tension around disclosure"},
            {"value": "high", "label": "High", "help": "strong pressure to withhold critical information"},
        ],
    },
    {
        "head": "reveal_decision",
        "title": "Reveal Decision",
        "description": "How much the NPC reveals in this turn",
        "choices": [
            {"value": "none", "label": "None", "help": "gives nothing away"},
            {"value": "hint", "label": "Hint", "help": "implies information without stating it"},
            {"value": "partial", "label": "Partial", "help": "gives some but not all relevant information"},
            {"value": "full", "label": "Full", "help": "fully discloses the secret"},
        ],
    },
    {
        "head": "response_policy",
        "title": "Response Policy",
        "description": "The NPC's conversational strategy",
        "choices": [
            {"value": "answer", "label": "Answer", "help": "directly responds to the player's query"},
            {"value": "withhold", "label": "Withhold", "help": "refuses to provide information"},
            {"value": "deflect", "label": "Deflect", "help": "changes topic or evades"},
            {"value": "clarify", "label": "Clarify", "help": "asks for clarification or rephrases"},
            {"value": "soothe", "label": "Soothe", "help": "calms the player, reassures"},
            {"value": "challenge", "label": "Challenge", "help": "pushes back, questions player's motives"},
            {"value": "threaten", "label": "Threaten", "help": "warns or implies consequences"},
            {"value": "negotiate", "label": "Negotiate", "help": "offers a trade or bargain"},
            {"value": "test", "label": "Test", "help": "probes the player's motives before committing"},
            {"value": "partial", "label": "Partial", "help": "gives an incomplete or hedged answer"},
        ],
    },
    {
        "head": "repair_strategy",
        "title": "Repair Strategy",
        "description": "Strategy to repair social damage (if applicable)",
        "choices": [
            {"value": "none", "label": "None", "help": "no repair move is attempted"},
            {"value": "soften", "label": "Soften", "help": "reduces harshness or tension"},
            {"value": "apologize", "label": "Apologize", "help": "expresses regret"},
            {"value": "clarify", "label": "Clarify", "help": "corrects or explains a misunderstanding"},
            {"value": "redirect", "label": "Redirect", "help": "shifts attention elsewhere"},
        ],
    },
    {
        "head": "trust_level",
        "title": "Trust Level",
        "description": "NPC's current trust toward the player (ordinal)",
        "choices": [
            {"value": "VL", "label": "Very Low", "help": "deeply suspicious"},
            {"value": "L", "label": "Low", "help": "cautious"},
            {"value": "N", "label": "Neutral", "help": "neither trusting nor suspicious"},
            {"value": "H", "label": "High", "help": "generally trusting"},
            {"value": "VH", "label": "Very High", "help": "fully trusting"},
        ],
    },
    {
        "head": "familiarity_level",
        "title": "Familiarity Level",
        "description": "NPC's familiarity with the player (ordinal)",
        "choices": [
            {"value": "VL", "label": "Very Low", "help": "stranger"},
            {"value": "L", "label": "Low", "help": "acquaintance"},
            {"value": "N", "label": "Neutral", "help": "known but not close"},
            {"value": "H", "label": "High", "help": "frequent interaction"},
            {"value": "VH", "label": "Very High", "help": "close companion"},
        ],
    },
]


@app.get("/api/guidelines")
def get_guidelines():
    return {"guidelines": GUIDELINES}


EXPERIMENT_INFO = """## Experiment Overview

You are participating in a **human validation audit** for a research paper on NPC (non-player character) dialogue generation.

### What you will do
You will review 150 stratified dialogue turns from a fantasy-game dataset and label each turn on **8 social-state dimensions**. These labels will be compared against (a) another human annotator and (b) the synthetic labels generated by a teacher LLM.

### Important rules
1. **Minimum time per turn: 30 seconds.** The Submit button will be blocked until you have spent at least 30 seconds on the current turn. This prevents rushed or random labeling.
2. Read the **Scene**, **Dialogue History**, **Player Utterance**, and **NPC Response** carefully before selecting labels.
3. **All 8 heads must be actively selected** before you can submit. There is no default choice.
4. Click **Submit & Next** after each turn. Your progress auto-saves.
5. Use **Previous Turn** to go back and edit if you change your mind.
6. **Relational delta heads (e.g., trust_delta, familiarity_delta) are NOT part of this audit.** Only static levels are validated.

### Time estimate
~2 hours for 150 turns (roughly 45 seconds per turn).

### Contact
If you have questions about a specific label, email skumyol@hotmail.com or consult the **Annotation Guidelines** below.
"""


@app.get("/api/info")
def get_info():
    return {"info": EXPERIMENT_INFO}


# ---------------------------------------------------------------------------
# Analysis API endpoints
# ---------------------------------------------------------------------------
class QCRequest(BaseModel):
    annotator_file: str  # path to audit_{name}.jsonl
    teacher_file: str | None = None


@app.post("/api/qc")
def qc_endpoint(req: QCRequest):
    path = Path(req.annotator_file)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Annotator file not found: {req.annotator_file}")
    records = load_jsonl(path)
    teacher_recs = None
    if req.teacher_file:
        tpath = Path(req.teacher_file)
        if tpath.exists():
            teacher_recs = index_by_turn_id(load_jsonl(tpath))
    return qc_annotator(records, teacher_recs)


class AgreementRequest(BaseModel):
    annotator_a: str
    annotator_b: str
    teacher_file: str | None = None


@app.post("/api/agreement")
def agreement_endpoint(req: AgreementRequest):
    a_path = Path(req.annotator_a)
    b_path = Path(req.annotator_b)
    if not a_path.exists():
        raise HTTPException(status_code=404, detail=f"Annotator A file not found: {req.annotator_a}")
    if not b_path.exists():
        raise HTTPException(status_code=404, detail=f"Annotator B file not found: {req.annotator_b}")
    teacher_path = Path(req.teacher_file) if req.teacher_file else None
    try:
        results = compute_agreement(a_path, b_path, teacher_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"agreement": results}


class EvaluateRequest(BaseModel):
    audit_dir: str = "./audit_results"
    teacher_file: str | None = None


@app.post("/api/evaluate")
def evaluate_endpoint(req: EvaluateRequest):
    audit_dir = Path(req.audit_dir)
    if not audit_dir.exists():
        raise HTTPException(status_code=404, detail=f"Audit directory not found: {req.audit_dir}")
    teacher_path = Path(req.teacher_file) if req.teacher_file else None
    try:
        results = evaluate_all(audit_dir, teacher_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return results


class GenerateResultsRequest(BaseModel):
    human_a: str
    human_b: str
    ai: str | None = None
    teacher: str


@app.post("/api/generate-results")
def generate_results_endpoint(req: GenerateResultsRequest):
    human_a = Path(req.human_a)
    human_b = Path(req.human_b)
    teacher = Path(req.teacher)
    ai = Path(req.ai) if req.ai else None
    for p, label in [(human_a, "human_a"), (human_b, "human_b"), (teacher, "teacher")]:
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"{label} file not found: {p}")
    if ai and not ai.exists():
        raise HTTPException(status_code=404, detail=f"AI file not found: {ai}")
    try:
        results, latex = generate_final_results(human_a, human_b, ai, teacher)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"results": results, "latex": latex}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Human audit FastAPI backend")
    parser.add_argument("--data", default=None, help="Path to test_heads.jsonl")
    parser.add_argument("--output", default="./audit_results", help="Directory to save annotations")
    parser.add_argument("--host", default="0.0.0.0", help="Server bind address")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--test", action="store_true", help="Test mode: no timer, selections optional")
    parser.add_argument("--sample-size", type=int, default=150, help="Number of turns to sample")
    args = parser.parse_args()

    global _DEFAULT_DATA_PATH, _OUTPUT_DIR, _TEST_MODE
    _DEFAULT_DATA_PATH = args.data
    _OUTPUT_DIR = args.output
    _TEST_MODE = args.test

    if _TEST_MODE:
        print("[TEST MODE] Timer disabled. Selections optional. For internal UX testing only.")

    print(f"Starting audit API server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        sys.exit(0)
