import json
import hashlib
from pathlib import Path


STANCE_DIMS = ["affection", "respect", "dominance", "familiarity", "trust", "obligation"]


def _context_str(record: dict) -> str:
    W = record.get("W", {})
    R_t_prev_parts = []
    for dim in STANCE_DIMS:
        init = W.get("initial_stance", {}).get(dim, "N")
        R_t_prev_parts.append(f"{dim}={init}")

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


def _latent_state_str(record: dict) -> str:
    C_t = record.get("C_t", {})
    A_t = record.get("A_t", {})
    M_t = record.get("M_t", {})
    R_t = record.get("R_t", {})
    N_t = record.get("N_t", {})
    D_t = record.get("D_t", {})

    stance_parts = []
    for dim in STANCE_DIMS:
        entry = R_t.get(dim, {})
        l = entry.get("level", "N")
        d = entry.get("delta", "0")
        stance_parts.append(f"{dim}={l}({d})")

    lines = [
        "<latent_state>",
        f"C_t: dialogue_act={C_t.get('dialogue_act', [])}  tone={C_t.get('tone', '')}  risk={C_t.get('risk_type', '')}",
        f"A_t: valence={A_t.get('valence', '')}  arousal={A_t.get('arousal', '')}  threat={A_t.get('threat', '')}  control={A_t.get('control', '')}",
        f"M_t: player_intent={M_t.get('player_intent', '')}  player_knowledge={M_t.get('player_knowledge', '')}  credibility={M_t.get('player_credibility', '')}",
        f"R_t: {' '.join(stance_parts)}",
        f"N_t: duty={N_t.get('duty_pressure', '')}  secrecy={N_t.get('secrecy_pressure', '')}  face={N_t.get('face_pressure', '')}  conflict={N_t.get('value_conflict', '')}",
        f"D_t: policy={D_t.get('response_policy', '')}  reveal={D_t.get('reveal_decision', '')}  repair={D_t.get('repair_strategy', '')}",
        "</latent_state>",
    ]
    return "\n".join(lines)


def _to_head_record(record: dict) -> dict:
    C_t = record.get("C_t", {})
    A_t = record.get("A_t", {})
    M_t = record.get("M_t", {})
    R_t = record.get("R_t", {})
    N_t = record.get("N_t", {})
    D_t = record.get("D_t", {})

    labels: dict = {}
    labels["dialogue_act"] = C_t.get("dialogue_act", [])
    labels["tone"] = C_t.get("tone", "")
    labels["risk_type"] = C_t.get("risk_type", "")
    labels["valence"] = A_t.get("valence", "")
    labels["arousal"] = A_t.get("arousal", "")
    labels["threat"] = A_t.get("threat", "")
    labels["control"] = A_t.get("control", "")
    labels["player_intent"] = M_t.get("player_intent", "")
    labels["player_knowledge"] = M_t.get("player_knowledge", "")
    labels["player_credibility"] = M_t.get("player_credibility", "")

    for dim in STANCE_DIMS:
        entry = R_t.get(dim, {})
        labels[f"{dim}_level"] = entry.get("level", "N")
        labels[f"{dim}_delta"] = entry.get("delta", "0")

    labels["duty_pressure"] = N_t.get("duty_pressure", "")
    labels["secrecy_pressure"] = N_t.get("secrecy_pressure", "")
    labels["face_pressure"] = N_t.get("face_pressure", "")
    labels["value_conflict"] = N_t.get("value_conflict", "")
    labels["response_policy"] = D_t.get("response_policy", "")
    labels["reveal_decision"] = D_t.get("reveal_decision", "")
    labels["repair_strategy"] = D_t.get("repair_strategy", "")

    return {
        "episode_id": record.get("episode_id", ""),
        "turn_idx": record.get("turn_idx", 0),
        "scenario_type": record.get("scenario_type", ""),
        "context": _context_str(record),
        "labels": labels,
        "counterfactual": record.get("counterfactual", False),
    }


def _to_sft_record(record: dict) -> dict:
    context = _context_str(record)
    latent = _latent_state_str(record)
    full_input = context + "\n\n" + latent + "\n\nGenerate NPC response:"

    return {
        "episode_id": record.get("episode_id", ""),
        "turn_idx": record.get("turn_idx", 0),
        "scenario_type": record.get("scenario_type", ""),
        "input": full_input,
        "target": record.get("response", ""),
        "counterfactual": record.get("counterfactual", False),
    }


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


class Packager:
    def __init__(self, validated_turns_dir: str, output_dir: str):
        self.validated_turns_dir = Path(validated_turns_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_all(self) -> dict:
        records = self._load_validated_turns()
        print(f"Loaded {len(records)} validated turn records.")

        trace_path = self.output_dir / "full_trace.jsonl"
        heads_path = self.output_dir / "head_supervision.jsonl"
        sft_path   = self.output_dir / "sft.jsonl"

        with open(trace_path, "w") as ft, \
             open(heads_path, "w") as fh, \
             open(sft_path, "w") as fs:
            for record in records:
                ft.write(json.dumps(record) + "\n")
                fh.write(json.dumps(_to_head_record(record)) + "\n")
                fs.write(json.dumps(_to_sft_record(record)) + "\n")

        manifest = {
            "n_turns": len(records),
            "n_episodes": len({r["episode_id"] for r in records}),
            "scenario_type_counts": self._count_scenario_types(records),
            "counterfactual_count": sum(1 for r in records if r.get("counterfactual")),
            "trace_hash": _file_hash(trace_path),
            "heads_hash": _file_hash(heads_path),
            "sft_hash": _file_hash(sft_path),
            "trace_path": str(trace_path),
            "heads_path": str(heads_path),
            "sft_path": str(sft_path),
        }

        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"Packaged artifacts to {self.output_dir}")
        return manifest

    def _load_validated_turns(self) -> list[dict]:
        records = []
        for jsonl_file in sorted(self.validated_turns_dir.glob("*.jsonl")):
            with open(jsonl_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        return records

    @staticmethod
    def _count_scenario_types(records: list[dict]) -> dict:
        counts: dict = {}
        for r in records:
            t = r.get("scenario_type", "unknown")
            counts[t] = counts.get(t, 0) + 1
        return counts
