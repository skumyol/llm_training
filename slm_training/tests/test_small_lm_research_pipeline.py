#!/usr/bin/env python3
"""Behavioral tests for the paper-oriented scratch-SLM pipeline."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src" / "train"))


class TestDialogueEncodingSeam(unittest.TestCase):
    def _write_records(self, path: Path) -> None:
        rows = [
            {
                "npc_id": "keeper",
                "npc_profile": "A cautious keeper who values discretion.",
                "dialogue_context": [
                    {"speaker": "player", "text": "What is behind the door?"}
                ],
                "target_response": "That knowledge has a price.",
                "ocean": [0.1, 0.2, 0.3, 0.4, 0.5],
                "vad": [0.6, 0.7, 0.8],
                "metadata": {"source": "fixture"},
            },
            {
                "npc_id": "guard",
                "npc_profile": "A loyal guard.",
                "dialogue_context": [
                    {"speaker": "player", "text": "Let me pass."}
                ],
                "target_response": "Not without the captain's seal.",
                "metadata": {"source": "fixture"},
            },
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_jsonl_becomes_target_masked_conditioned_batch(self) -> None:
        from src.data.small_lm_dialogue import (
            DialogueRecordDataset,
            DialogueTokenizer,
            train_dialogue_tokenizer,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            records = tmpdir / "dialogue.jsonl"
            tokenizer_path = tmpdir / "dialogue-tokenizer.json"
            self._write_records(records)
            train_dialogue_tokenizer(
                [records], tokenizer_path, vocab_size=320, min_frequency=1
            )

            tokenizer = DialogueTokenizer.from_file(tokenizer_path)
            dataset = DialogueRecordDataset(
                records, tokenizer, seq_len=64, profile_max_tokens=16
            )
            item = dataset[0]

            self.assertEqual(tuple(item["input_ids"].shape), (64,))
            self.assertEqual(tuple(item["labels"].shape), (64,))
            first_target = torch.nonzero(item["labels"] != -100)[0].item()
            self.assertGreater(first_target, 0)
            self.assertTrue(torch.all(item["labels"][:first_target] == -100))
            self.assertGreater(item["target_bytes"].item(), 0)
            self.assertEqual(item["source"], "fixture")
            self.assertTrue(
                torch.allclose(
                    item["condition"],
                    torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
                )
            )
            self.assertTrue(torch.all(item["condition_mask"] == 1))
            self.assertEqual(dataset.condition_coverage["all_dimensions"], 0.5)
            self.assertEqual(dataset.easy_indices(max_target_tokens=32, max_turns=2), [0, 1])

    def test_duplicate_targets_can_be_filtered_without_cross_split_state(self) -> None:
        from src.data.small_lm_dialogue import (
            DialogueRecordDataset,
            DialogueTokenizer,
            train_dialogue_tokenizer,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            records = tmpdir / "dialogue.jsonl"
            tokenizer_path = tmpdir / "dialogue-tokenizer.json"
            self._write_records(records)
            duplicate = json.loads(records.read_text().splitlines()[0])
            with records.open("a") as handle:
                handle.write(json.dumps(duplicate) + "\n")
            train_dialogue_tokenizer(
                [records], tokenizer_path, vocab_size=320, min_frequency=1
            )
            dataset = DialogueRecordDataset(
                records,
                DialogueTokenizer.from_file(tokenizer_path),
                seq_len=64,
                deduplicate=True,
            )
            self.assertEqual(len(dataset), 2)


class TestTrainingControlSeam(unittest.TestCase):
    def test_warmup_cosine_multiplier_has_expected_endpoints(self) -> None:
        from src.train.run_small_lm import warmup_cosine_multiplier

        self.assertEqual(warmup_cosine_multiplier(0, 100, 10, 0.1), 0.0)
        self.assertAlmostEqual(
            warmup_cosine_multiplier(10, 100, 10, 0.1), 1.0, places=6
        )
        self.assertAlmostEqual(
            warmup_cosine_multiplier(100, 100, 10, 0.1), 0.1, places=6
        )

    def test_device_policy_is_explicit_and_backend_aware(self) -> None:
        from src.train.run_small_lm import resolve_device_type, runtime_policy

        self.assertEqual(resolve_device_type("auto", True, True), "cuda")
        self.assertEqual(resolve_device_type("auto", False, True), "mps")
        self.assertEqual(resolve_device_type("auto", False, False), "cpu")
        with self.assertRaisesRegex(RuntimeError, "CUDA"):
            resolve_device_type("cuda", False, True)

        cuda = runtime_policy("cuda", use_amp=True)
        self.assertEqual(cuda["amp_dtype"], "float16")
        self.assertTrue(cuda["pin_memory"])
        self.assertTrue(cuda["non_blocking"])
        self.assertGreater(cuda["num_workers"], 0)
        self.assertTrue(cuda["fused_optimizer"])

        mps = runtime_policy("mps", use_amp=True)
        self.assertEqual(mps["amp_dtype"], "float16")
        self.assertFalse(mps["pin_memory"])
        self.assertFalse(mps["non_blocking"])
        self.assertEqual(mps["num_workers"], 0)
        self.assertFalse(mps["fused_optimizer"])

        overridden = runtime_policy(
            "cuda", use_amp=False, num_workers=2, pin_memory=False
        )
        self.assertEqual(overridden["amp_dtype"], "float32")
        self.assertEqual(overridden["num_workers"], 2)
        self.assertFalse(overridden["pin_memory"])


class TestPaperReportingSeam(unittest.TestCase):
    def test_completed_runs_become_csv_markdown_and_latex(self) -> None:
        from scripts.run_paper_small_lms import build_report

        summaries = [
            {
                "run_id": "gpt_s42",
                "arch": "gpt",
                "model_params": 15_500_000,
                "tokenizer": "byte_bpe:dialogue.json",
                "best": {
                    "val_ppl": 20.0,
                    "val_loss": 3.0,
                    "val_bits_per_byte": 1.2,
                    "epoch": 2,
                    "global_step": 10,
                },
                "generation": {
                    "distinct_1": 0.5,
                    "distinct_2": 0.8,
                    "repetition_3": 0.1,
                    "empty_response_rate": 0.0,
                },
                "runtime": {
                    "device_type": "cuda",
                    "device_name": "Fixture GPU",
                    "precision": "float16",
                    "train_target_tokens_per_second": 12345.0,
                    "peak_memory_mb": None,
                },
                "hyperparams": {"seed": 42},
                "epochs": [{"epoch": 1, "val_ppl": 25.0}, {"epoch": 2, "val_ppl": 20.0}],
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            outputs = build_report(summaries, Path(tmp), experiment_id="fixture")
            self.assertTrue(outputs["csv"].exists())
            self.assertIn("TinyGPT", outputs["markdown"].read_text())
            self.assertIn("12,345", outputs["markdown"].read_text())
            self.assertIn("—", outputs["markdown"].read_text())
            self.assertIn("\\begin{tabular}", outputs["latex"].read_text())
            self.assertTrue(outputs["json"].exists())
            self.assertNotIn("NaN", outputs["json"].read_text())


class TestDailyDialogConversionSeam(unittest.TestCase):
    def test_dialogue_is_expanded_without_inventing_speaker_names(self) -> None:
        from src.data.prepare_dialogue_data import records_from_dailydialog_example

        rows = records_from_dailydialog_example(
            {"dialog": ["Hello.", "Welcome.", "Can I ask something?"]},
            split_name="train",
            dialogue_id="d1",
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[-1]["target_response"], "Can I ask something?")
        self.assertEqual(rows[-1]["dialogue_context"][-1]["speaker"], "player")
        self.assertEqual(rows[-1]["metadata"]["source"], "dailydialog")


class TestSyntheticSplitSeam(unittest.TestCase):
    def test_episode_groups_never_cross_train_and_validation(self) -> None:
        from src.data.convert_generated_data import split_dialogue_records_by_episode

        rows = [
            {"metadata": {"episode_id": "a", "turn_idx": 1}},
            {"metadata": {"episode_id": "a", "turn_idx": 2}},
            {"metadata": {"episode_id": "b", "turn_idx": 1}},
            {"metadata": {"episode_id": "c", "turn_idx": 1}},
        ]
        train, val = split_dialogue_records_by_episode(rows, val_frac=0.34, seed=42)
        train_ids = {row["metadata"]["episode_id"] for row in train}
        val_ids = {row["metadata"]["episode_id"] for row in val}
        self.assertFalse(train_ids & val_ids)
        self.assertEqual(len(train) + len(val), len(rows))


if __name__ == "__main__":
    unittest.main()
