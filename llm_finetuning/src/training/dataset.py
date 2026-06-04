import json
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset


DIALOGUE_ACT_LABELS = ["ask", "accuse", "threaten", "flatter", "apologize", "negotiate", "joke", "confess", "probe", "command"]
TONE_LABELS         = ["warm", "neutral", "confrontational", "sarcastic", "fearful", "evasive"]
RISK_LABELS         = ["none", "secret-risk", "face-risk", "status-risk", "conflict-risk"]
VALENCE_LABELS      = ["negative", "neutral", "positive"]
THREE_LEVEL_LABELS  = ["low", "medium", "high"]
PLAYER_INTENT_LABELS = ["seek-info", "trap", "bond", "manipulate", "test", "persuade", "intimidate", "probe", "negotiate"]
PLAYER_KNOWLEDGE_LABELS = ["unaware", "partial", "informed", "knows-secret"]
STANCE_LEVEL_LABELS = ["VL", "L", "N", "H", "VH"]
STANCE_DELTA_LABELS = ["--", "-", "0", "+", "++"]
VALUE_CONFLICT_LABELS = ["none", "mild", "strong", ""]
RESPONSE_POLICY_LABELS = ["answer", "partial", "withhold", "deflect", "challenge", "soothe", "test", "threaten", "negotiate", "clarify"]
REVEAL_LABELS       = ["none", "hint", "partial", "full"]
REPAIR_LABELS       = ["none", "soften", "apologize", "clarify", "redirect"]
STANCE_DIMS         = ["affection", "respect", "dominance", "familiarity", "trust", "obligation"]

LABEL_MAPS: dict[str, list] = {
    "dialogue_act":       DIALOGUE_ACT_LABELS,
    "tone":               TONE_LABELS,
    "risk_type":          RISK_LABELS,
    "valence":            VALENCE_LABELS,
    "arousal":            THREE_LEVEL_LABELS,
    "threat":             THREE_LEVEL_LABELS,
    "control":            THREE_LEVEL_LABELS,
    "player_intent":      PLAYER_INTENT_LABELS,
    "player_knowledge":   PLAYER_KNOWLEDGE_LABELS,
    "player_credibility": THREE_LEVEL_LABELS,
    "duty_pressure":      THREE_LEVEL_LABELS,
    "secrecy_pressure":   THREE_LEVEL_LABELS,
    "face_pressure":      THREE_LEVEL_LABELS,
    "value_conflict":     ["none", "mild", "strong"],
    "response_policy":    RESPONSE_POLICY_LABELS,
    "reveal_decision":    REVEAL_LABELS,
    "repair_strategy":    REPAIR_LABELS,
}

for _dim in STANCE_DIMS:
    LABEL_MAPS[f"{_dim}_level"] = STANCE_LEVEL_LABELS
    LABEL_MAPS[f"{_dim}_delta"] = STANCE_DELTA_LABELS

LABEL_TO_IDX: dict[str, dict] = {
    field: {v: i for i, v in enumerate(values)}
    for field, values in LABEL_MAPS.items()
}


def _encode_labels(labels: dict) -> dict[str, torch.Tensor]:
    encoded: dict[str, torch.Tensor] = {}
    for field, idx_map in LABEL_TO_IDX.items():
        val = labels.get(field, None)
        if val is None:
            encoded[field] = torch.tensor(-1, dtype=torch.long)
            continue

        if field == "dialogue_act":
            if isinstance(val, str):
                val = [val]
            multi_hot = torch.zeros(len(DIALOGUE_ACT_LABELS), dtype=torch.float)
            for v in val:
                if v in idx_map:
                    multi_hot[idx_map[v]] = 1.0
            encoded[field] = multi_hot
        else:
            if isinstance(val, list):
                val = val[0] if val else ""
            idx = idx_map.get(str(val), -1)
            encoded[field] = torch.tensor(idx, dtype=torch.long)

    return encoded


class HeadSupervisionDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        max_seq_len: int = 1024,
        jepa_fields: Optional[list[str]] = None,
        jepa_horizons: Optional[list[int]] = None,
        shuffle_future_labels: bool = False,
        shuffle_seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.jepa_fields = jepa_fields or []
        self.jepa_horizons = jepa_horizons or []
        self.shuffle_future_labels = shuffle_future_labels
        self._shuffle_rng = random.Random(shuffle_seed) if shuffle_future_labels else None
        self.records: list[dict] = []
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        self._future_index: dict[tuple[str, int], dict] = {}
        self._all_label_records: list[dict] = []  # used for shuffling
        if self.jepa_fields and self.jepa_horizons:
            self._build_future_index()
            if self.shuffle_future_labels:
                self._all_label_records = [r for r in self.records if r.get("labels")]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        context = record["context"]

        encoding = self.tokenizer(
            context,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        label_tensors = _encode_labels(record.get("labels", {}))
        future_label_tensors = self._encode_future_labels(record)

        # Store raw labels for sampler weight computation
        raw_labels = record.get("labels", {})

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            **{f"label_{k}": v for k, v in label_tensors.items()},
            **future_label_tensors,
            "episode_id": record.get("episode_id", ""),
            "scenario_type": record.get("scenario_type", ""),
            "_raw_labels": raw_labels,  # for sampler, removed in collate
        }

    def _build_future_index(self) -> None:
        for record in self.records:
            episode_id = record.get("episode_id", "")
            turn_idx = record.get("turn_idx", record.get("turn", None))
            if episode_id == "" or turn_idx is None:
                continue
            self._future_index[(str(episode_id), int(turn_idx))] = record

    def _encode_future_labels(self, record: dict) -> dict[str, torch.Tensor]:
        if not self.jepa_fields or not self.jepa_horizons:
            return {}

        # Shuffled mode: pick a random record's labels for each horizon+field
        if self.shuffle_future_labels and self._shuffle_rng is not None and self._all_label_records:
            encoded: dict[str, torch.Tensor] = {}
            for horizon in self.jepa_horizons:
                shuffled_record = self._shuffle_rng.choice(self._all_label_records)
                future_labels = shuffled_record.get("labels", {})
                for field in self.jepa_fields:
                    idx_map = LABEL_TO_IDX.get(field, {})
                    value = future_labels.get(field)
                    if isinstance(value, list):
                        value = value[0] if value else ""
                    encoded[f"future_{horizon}_{field}"] = torch.tensor(
                        idx_map.get(str(value), -1),
                        dtype=torch.long,
                    )
            return encoded

        # Real future labels: lookup by (episode_id, turn_idx + horizon)
        episode_id = str(record.get("episode_id", ""))
        turn_idx = record.get("turn_idx", record.get("turn", None))
        if episode_id == "" or turn_idx is None:
            return {
                f"future_{k}_{field}": torch.tensor(-1, dtype=torch.long)
                for k in self.jepa_horizons
                for field in self.jepa_fields
            }

        encoded: dict[str, torch.Tensor] = {}
        for horizon in self.jepa_horizons:
            future = self._future_index.get((episode_id, int(turn_idx) + int(horizon)))
            future_labels = future.get("labels", {}) if future else {}
            for field in self.jepa_fields:
                idx_map = LABEL_TO_IDX.get(field, {})
                value = future_labels.get(field)
                if isinstance(value, list):
                    value = value[0] if value else ""
                encoded[f"future_{horizon}_{field}"] = torch.tensor(
                    idx_map.get(str(value), -1),
                    dtype=torch.long,
                )
        return encoded


class SFTDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        tokenizer,
        max_seq_len: int = 2048,
        mask_secret_spans: bool = False,
        secret_strings_file: Optional[str] = None,
        conditioning_mode: str = "gold",
        predicted_state_file: Optional[str] = None,
        state_dropout_prob: float = 0.0,
    ):
        """
        Args:
            mask_secret_spans: If True, set labels to -100 on tokens that overlap
                with secret spans, so the LM loss never trains on producing the
                literal secret text. This is the upstream complement to the
                downstream leakage evaluator.
            secret_strings_file: Optional JSON mapping episode_id -> list[str] of
                secret strings. If absent, only per-record `secret_spans` /
                `secret_strings` fields are used.
            conditioning_mode: "gold" (use gold states from input field) or "predicted"
                (use predicted states from predicted_state_file).
            predicted_state_file: Optional JSON mapping (episode_id, turn_idx) to
                predicted latent state dict. Used when conditioning_mode="predicted".
            state_dropout_prob: Probability of dropping each latent state head during
                training to improve robustness. Only applies when conditioning_mode="predicted".
        """
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mask_secret_spans = bool(mask_secret_spans)
        self.conditioning_mode = conditioning_mode
        self.state_dropout_prob = state_dropout_prob
        self.records: list[dict] = []
        with open(jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

        self._secret_strings_by_episode: dict[str, list[str]] = {}
        if self.mask_secret_spans and secret_strings_file and Path(secret_strings_file).exists():
            with open(secret_strings_file, "r") as f:
                raw = json.load(f)
            # Accept either {episode_id: [strings]} or {episode_id: {"secrets": [...]}}
            for ep, val in raw.items():
                if isinstance(val, dict):
                    val = val.get("secrets") or val.get("strings") or []
                if isinstance(val, list):
                    self._secret_strings_by_episode[str(ep)] = [str(s) for s in val if s]

        # Load predicted states if provided
        self._predicted_states: dict[tuple[str, int], dict] = {}
        if predicted_state_file and Path(predicted_state_file).exists():
            with open(predicted_state_file, "r") as f:
                raw = json.load(f)
            # Expected format: {"episode_id_turn_idx": {...state dict...}}
            for key, state in raw.items():
                # Parse key as "episode_id_turn_idx"
                parts = key.rsplit("_", 1)
                if len(parts) == 2:
                    episode_id, turn_idx = parts
                    try:
                        turn_idx_int = int(turn_idx)
                        self._predicted_states[(str(episode_id), turn_idx_int)] = state
                    except ValueError:
                        # Skip if turn_idx is not an integer
                        pass

    def __len__(self) -> int:
        return len(self.records)

    def _apply_state_dropout(self, state: dict) -> dict:
        """Randomly drop latent state heads to improve robustness.
        
        Args:
            state: Latent state dict with keys like C_t, A_t, M_t, R_t, N_t, D_t
        
        Returns:
            State dict with some heads randomly dropped (set to empty or None)
        """
        if self.state_dropout_prob == 0.0:
            return state
        
        dropped_state = {}
        for key, value in state.items():
            if random.random() < self.state_dropout_prob:
                # Drop this head by setting to empty dict or None
                if isinstance(value, dict):
                    dropped_state[key] = {}
                else:
                    dropped_state[key] = None
            else:
                dropped_state[key] = value
        return dropped_state

    def _state_dict_to_string(self, state: dict) -> str:
        """Convert a latent state dict to string format matching packager output.
        
        Args:
            state: Latent state dict with keys C_t, A_t, M_t, R_t, N_t, D_t
        
        Returns:
            Formatted string representation of the state
        """
        C_t = state.get("C_t", {})
        A_t = state.get("A_t", {})
        M_t = state.get("M_t", {})
        R_t = state.get("R_t", {})
        N_t = state.get("N_t", {})
        D_t = state.get("D_t", {})

        # Format stance dimensions
        stance_dims = ["trust", "intimacy", "power", "conflict"]
        stance_parts = []
        for dim in stance_dims:
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

    def _collect_secret_strings(self, record: dict) -> list[str]:
        ep = str(record.get("episode_id", ""))
        out: list[str] = []
        out.extend(self._secret_strings_by_episode.get(ep, []))
        rec_secrets = record.get("secret_strings") or record.get("secret_spans")
        if isinstance(rec_secrets, list):
            out.extend(str(s) for s in rec_secrets if s)
        elif isinstance(rec_secrets, str):
            out.append(rec_secrets)
        # de-dup while preserving order
        seen: set[str] = set()
        return [s for s in out if (s not in seen and not seen.add(s))]

    def _mask_secret_token_spans(
        self,
        labels: torch.Tensor,
        full_text: str,
        target_start_char: int,
        secret_strings: list[str],
    ) -> int:
        """Set labels[i] = -100 for tokens whose character span overlaps a secret
        substring inside the target portion of full_text. Returns count masked.
        """
        if not secret_strings:
            return 0
        try:
            encoding = self.tokenizer(
                full_text,
                max_length=self.max_seq_len,
                truncation=True,
                return_offsets_mapping=True,
                add_special_tokens=True,
            )
        except (TypeError, ValueError):
            # Tokenizer does not support offset mapping (e.g. slow tokenizer);
            # fall back to no-op masking. The eval-time detector still catches leaks.
            return 0
        offsets = encoding["offset_mapping"]
        text_lower = full_text.lower()
        masked = 0
        for secret in secret_strings:
            s = secret.strip().lower()
            if not s or len(s) < 4:  # avoid masking 1-3 char tokens
                continue
            start = text_lower.find(s, target_start_char)
            while start != -1:
                end = start + len(s)
                for tok_idx, (a, b) in enumerate(offsets):
                    if tok_idx >= labels.size(0):
                        break
                    if a == 0 and b == 0:
                        continue  # special tokens
                    if b <= start or a >= end:
                        continue
                    if labels[tok_idx].item() != -100:
                        labels[tok_idx] = -100
                        masked += 1
                start = text_lower.find(s, end)
        return masked

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        
        # Determine prompt text based on conditioning mode
        if self.conditioning_mode == "predicted":
            episode_id = str(record.get("episode_id", ""))
            turn_idx = record.get("turn_idx", 0)
            state_key = (episode_id, turn_idx)
            
            if state_key in self._predicted_states:
                # Use predicted state
                predicted_state = self._predicted_states[state_key]
                # Apply state dropout for robustness
                if self.state_dropout_prob > 0.0:
                    predicted_state = self._apply_state_dropout(predicted_state)
                
                # Build prompt with predicted state
                # For now, we'll use a simple string representation
                # In production, this should use the same format as the packager
                state_str = self._state_dict_to_string(predicted_state)
                prompt_text = state_str + "\n"
            else:
                # Fall back to gold state if predicted not available
                prompt_text = record["input"] + "\n"
        else:
            # Use gold state (default)
            prompt_text = record["input"] + "\n"
        
        full_text = prompt_text + record["target"]

        encoding = self.tokenizer(
            full_text,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_len = len(self.tokenizer(record["input"]).input_ids)
        labels = encoding["input_ids"].clone().squeeze(0)
        labels[:min(input_len, labels.size(0))] = -100

        if self.mask_secret_spans:
            secrets = self._collect_secret_strings(record)
            if secrets:
                self._mask_secret_token_spans(
                    labels=labels,
                    full_text=full_text,
                    target_start_char=len(prompt_text),
                    secret_strings=secrets,
                )

        # Calculate prompt length for evaluation (input + separator)
        prompt_ids = self.tokenizer(prompt_text, truncation=True, max_length=self.max_seq_len)["input_ids"]
        prompt_len = len(prompt_ids)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels,
            "prompt_len": torch.tensor(prompt_len, dtype=torch.long),
            "episode_id": record.get("episode_id", ""),
            "turn_idx": record.get("turn_idx", 0),
            "scenario_type": record.get("scenario_type", ""),
            "secret_strings": record.get("secret_strings", []),
        }


def collate_head_batch(batch: list[dict]) -> dict:
    keys = [k for k in batch[0] if k not in ("episode_id", "scenario_type", "_raw_labels")]
    result: dict = {}
    for k in keys:
        tensors = [b[k] for b in batch]
        try:
            result[k] = torch.stack(tensors)
        except Exception:
            result[k] = tensors
    result["episode_ids"] = [b.get("episode_id", "") for b in batch]
    return result


class JointDataset(Dataset):
    """Merges SFT and HeadSupervision records by index for joint training.

    Both JSONL files are produced from the same validated turns in the same
    order, so zipping by index is safe. Verifies alignment on init.
    """

    def __init__(
        self,
        sft_jsonl_path: str,
        heads_jsonl_path: str,
        tokenizer,
        sft_max_seq_len: int = 2048,
        heads_max_seq_len: int = 1024,
        alignment_check_n: int = 100,
    ):
        self.sft_ds   = SFTDataset(sft_jsonl_path,   tokenizer, sft_max_seq_len)
        self.heads_ds = HeadSupervisionDataset(heads_jsonl_path, tokenizer, heads_max_seq_len)
        assert len(self.sft_ds) == len(self.heads_ds), (
            f"SFT ({len(self.sft_ds)}) and heads ({len(self.heads_ds)}) record counts must match"
        )
        
        # Cross-check episode_id + turn_idx alignment on first N records
        self._verify_alignment(alignment_check_n)

    def _verify_alignment(self, n: int) -> None:
        """Verify that (episode_id, turn_idx) pairs match between datasets."""
        check_count = min(n, len(self.sft_ds))
        for i in range(check_count):
            sft_rec = self.sft_ds.records[i]
            heads_rec = self.heads_ds.records[i]
            
            sft_ep = sft_rec.get("episode_id", "")
            sft_turn = sft_rec.get("turn_idx", sft_rec.get("turn", -1))
            heads_ep = heads_rec.get("episode_id", "")
            heads_turn = heads_rec.get("turn_idx", heads_rec.get("turn", -1))
            
            if sft_ep != heads_ep or sft_turn != heads_turn:
                raise ValueError(
                    f"Dataset misalignment at index {i}: "
                    f"SFT has (episode_id={sft_ep}, turn_idx={sft_turn}) but "
                    f"heads has (episode_id={heads_ep}, turn_idx={heads_turn}). "
                    f"Re-run data packaging to regenerate aligned datasets."
                )

    def __len__(self) -> int:
        return len(self.sft_ds)

    def __getitem__(self, idx: int) -> dict:
        sft_item   = self.sft_ds[idx]
        heads_item = self.heads_ds[idx]
        
        # Start with SFT item (input_ids, attention_mask, labels, etc.)
        merged = {**sft_item}
        
        # Add heads item with prefixed keys for inputs to avoid collision
        merged["heads_input_ids"] = heads_item["input_ids"]
        merged["heads_attention_mask"] = heads_item["attention_mask"]
        
        # Add all labels from heads item
        for k, v in heads_item.items():
            if k.startswith("label_"):
                merged[k] = v
                
        return merged


def collate_joint_batch(batch: list[dict]) -> dict:
    skip = {"episode_id", "scenario_type", "episode_ids"}
    result: dict = {}
    for k in batch[0]:
        if k in skip:
            continue
        tensors = [b[k] for b in batch]
        try:
            result[k] = torch.stack(tensors)
        except Exception:
            result[k] = tensors
    result["episode_ids"] = [b.get("episode_id", "") for b in batch]
    return result


def collate_sft_batch(batch: list[dict]) -> dict:
    keys = [k for k in batch[0] if k not in ("episode_id", "scenario_type", "secret_strings")]
    result: dict = {}
    for k in keys:
        tensors = [b[k] for b in batch]
        try:
            result[k] = torch.stack(tensors)
        except Exception:
            result[k] = tensors
    result["episode_id"] = [b.get("episode_id", "") for b in batch]
    result["secret_strings"] = [b.get("secret_strings", []) for b in batch]
    return result
