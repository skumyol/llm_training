from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _have_cmd(name: str) -> bool:
    return shutil.which(name) is not None


def _run(cmd: List[str], cwd: Optional[Path] = None) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _write_json(path: Path, obj: dict) -> None:
    _ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _download_url(url: str, out_path: Path) -> None:
    _ensure_dir(out_path.parent)
    try:
        import requests
    except ImportError:
        raise RuntimeError("Please install requests: pip install requests")

    print(f"[DOWNLOAD] {url} -> {out_path}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _git_clone(repo_url: str, out_dir: Path) -> None:
    if out_dir.exists():
        print(f"[SKIP] Repo already exists: {out_dir}")
        return
    if not _have_cmd("git"):
        raise RuntimeError("git is required for repo-based downloads")
    _run(["git", "clone", "--depth", "1", repo_url, str(out_dir)])


def _hf_download(dataset_id: str, out_dir: Path, config: Optional[str] = None) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError("Please install datasets: pip install datasets")

    print(f"[HF] Loading {dataset_id} {f'(config={config})' if config else ''}")
    ds = load_dataset(dataset_id, config)
    if hasattr(ds, "save_to_disk"):
        _ensure_dir(out_dir.parent)
        ds.save_to_disk(str(out_dir))
    else:
        _ensure_dir(out_dir)
        _write_json(out_dir / "info.json", {"dataset_id": dataset_id, "config": config})


def _parlai_download(task_name: str, out_dir: Path) -> None:
    if not _have_cmd("python"):
        raise RuntimeError("Python executable not found in PATH")

    env = os.environ.copy()
    env["PARLAI_HOME"] = str(out_dir.resolve())
    print(f"[PARLAI] Triggering task download: {task_name} into {env['PARLAI_HOME']}")
    subprocess.run(
        [sys.executable, "-m", "parlai", "display_data", "-t", task_name, "-ne", "1"],
        check=True,
        env=env,
    )


@dataclass
class DatasetEntry:
    name: str
    description: str
    method: str
    runner: Callable[[Path], None]
    notes: str


@dataclass
class DialogueExample:
    npc_id: str
    npc_profile: str
    dialogue_context: List[Dict[str, str]]
    target_response: str
    metadata: Dict[str, Any]


class RegressionTextDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer_name: str,
        text_column: str,
        target_columns: List[str],
        max_length: int,
    ) -> None:
        self.df = pd.read_csv(path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.text_column = text_column
        self.target_columns = target_columns
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        encoded = self.tokenizer(
            str(row[self.text_column]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(row[self.target_columns].astype(float).values, dtype=torch.float)
        return item


class DialogueJsonlDataset(Dataset):
    """Expected JSONL record shape:

    {
      "npc_id": "npc_001",
      "npc_profile": "Long profile text",
      "dialogue_context": [{"speaker": "player", "text": "..."}, ...],
      "target_response": "NPC answer",
      "metadata": {"source": "LIGHT"}
    }
    """

    def __init__(self, path: str) -> None:
        self.records: List[DialogueExample] = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                raw = json.loads(line)
                self.records.append(
                    DialogueExample(
                        npc_id=raw["npc_id"],
                        npc_profile=raw["npc_profile"],
                        dialogue_context=raw["dialogue_context"],
                        target_response=raw["target_response"],
                        metadata=raw.get("metadata", {}),
                    )
                )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> DialogueExample:
        return self.records[idx]


class PersonalityCache:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.cache: Dict[str, List[float]] = {}
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    self.cache[row["npc_id"]] = row["vector"]

    def get(self, npc_id: str) -> Optional[List[float]]:
        return self.cache.get(npc_id)

    def items(self) -> Iterable[tuple[str, List[float]]]:
        return self.cache.items()

    def save_many(self, rows: List[Dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                self.cache[row["npc_id"]] = row["vector"]


class DataDownloader:
    """
    Unified downloader for narrative/RPG NPC backend datasets.

    Supports:
    - Hugging Face Datasets API for openly mirrored datasets.
    - Direct URL download for archival releases (e.g., PAN15 on Zenodo).
    - Git clone for repos hosting corpus files (e.g., EmoBank, MELD).
    - Manual placeholders for gated datasets (e.g., IEMOCAP).

    Academic references are included inline for each dataset entry.
    """

    DEFAULT_BUNDLE = [
        "essays_big5",
        "pan15",
        "emobank",
        "goemotions",
        "empathetic_dialogues",
        "dailydialog",
        "meld",
        "personachat",
        "crd3",
    ]

    def __init__(self, base_dir: str | Path = "data") -> None:
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        _ensure_dir(self.raw_dir)
        self._registry: Dict[str, DatasetEntry] = {}
        self._build_registry()

    def _build_registry(self) -> None:
        """Build the dataset registry with all supported datasets."""

        # ---------------- Personality / profile ----------------
        # Essays / Big Five convenience mirror.
        # Academic refs:
        # - Pennebaker, J. W., & King, L. A. (1999). Linguistic styles: Language use as an individual difference.
        # - Mairesse, F., et al. (2007). Using linguistic cues for the automatic recognition of personality in conversation and text.
        self._registry["essays_big5"] = DatasetEntry(
            name="essays_big5",
            description="Big Five essay corpus mirror for static personality regression.",
            method="hf",
            runner=lambda out: _hf_download("jingjietan/essays-big5", out),
            notes="Convenience mirror, not the original hosting location.",
        )

        # PAN 2015 Author Profiling.
        # Academic ref:
        # - Rangel, F., et al. (2015). Overview of the 3rd Author Profiling Task at PAN 2015.
        self._registry["pan15"] = DatasetEntry(
            name="pan15",
            description="PAN15 Author Profiling training + test zips from Zenodo.",
            method="url",
            runner=lambda out: self._download_pan15(out),
            notes="Open archival release on Zenodo.",
        )

        # ---------------- Affect / emotion ----------------
        # EmoBank.
        # Academic refs:
        # - Buechel, S., & Hahn, U. (2017). EmoBank: Studying the impact of annotation perspective and representation format on dimensional emotion analysis.
        self._registry["emobank"] = DatasetEntry(
            name="emobank",
            description="Valence-Arousal-Dominance text corpus from JULIELab GitHub.",
            method="git",
            runner=lambda out: _git_clone("https://github.com/JULIELab/emobank.git", out),
            notes="GitHub repo contains corpus + splits.",
        )

        # GoEmotions.
        # Academic ref:
        # - Demszky, D., et al. (2020). GoEmotions: A Dataset of Fine-Grained Emotions.
        self._registry["goemotions"] = DatasetEntry(
            name="goemotions",
            description="58k Reddit comments with 27 emotion labels + Neutral.",
            method="hf",
            runner=lambda out: _hf_download("mrm8488/goemotions", out),
            notes="Good supplement to dimensional affect with categorical emotions.",
        )

        # EmpatheticDialogues.
        # Academic ref:
        # - Rashkin, H., et al. (2019). Towards Empathetic Open-domain Conversation Models: A New Benchmark and Dataset.
        self._registry["empathetic_dialogues"] = DatasetEntry(
            name="empathetic_dialogues",
            description="Emotion-grounded dialogue dataset from Meta's HF mirror.",
            method="hf",
            runner=lambda out: _hf_download("bdotloh/empathetic-dialogues", out),
            notes="Useful for emotionally grounded response behavior.",
        )

        # DailyDialog.
        # Academic ref:
        # - Li, Y., et al. (2017). DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset.
        self._registry["dailydialog"] = DatasetEntry(
            name="dailydialog",
            description="DailyDialog with emotion and dialog act annotations.",
            method="hf",
            runner=lambda out: _hf_download("ConvLab/dailydialog", out),
            notes="Convenient HF-hosted version for quick prototyping.",
        )

        # MELD.
        # Academic ref:
        # - Poria, S., et al. / Ghosal, D., et al. (2019). MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations.
        self._registry["meld"] = DatasetEntry(
            name="meld",
            description="Multi-party conversation emotion dataset (annotations/repo).",
            method="git",
            runner=lambda out: _git_clone("https://github.com/declare-lab/MELD.git", out),
            notes="Repo contains annotations and metadata; media may require extra download.",
        )

        # IEMOCAP.
        # Academic ref:
        # - Busso, C., et al. (2008). IEMOCAP: Interactive emotional dyadic motion capture database.
        self._registry["iemocap"] = DatasetEntry(
            name="iemocap",
            description="Manual-only placeholder for gated USC IEMOCAP release.",
            method="manual",
            runner=lambda out: self._write_manual_iemocap_instructions(out),
            notes="USC requires an electronic release form; no direct public auto-download.",
        )

        # ---------------- Dialogue / roleplay / RPG ----------------
        # PersonaChat.
        # Academic ref:
        # - Zhang, S., et al. (2018). Personalizing Dialogue Agents: I have a dog, do you have pets too?
        self._registry["personachat"] = DatasetEntry(
            name="personachat",
            description="Persona-conditioned chat dataset via HF mirror.",
            method="hf",
            runner=lambda out: _hf_download("bavard/personachat_truecased", out),
            notes="Widely used persona-conditioning benchmark.",
        )

        # CRD3.
        # Academic ref:
        # - Rameshkumar, R., & Bailey, P. (2020). Storytelling with Dialogue: A Critical Role Dungeons and Dragons Dataset.
        self._registry["crd3"] = DatasetEntry(
            name="crd3",
            description="Critical Role D&D dialogue dataset from Hugging Face.",
            method="hf",
            runner=lambda out: _hf_download("microsoft/crd3", out),
            notes="Strong RPG-adjacent multi-turn dialogue source.",
        )

        # LIGHT.
        # Academic refs:
        # - Urbanek, J., et al. (2019). Learning to Speak and Act in a Fantasy Text Adventure Game.
        self._registry["light_dialog"] = DatasetEntry(
            name="light_dialog",
            description="Original LIGHT fantasy dialogue/action data via ParlAI task downloader.",
            method="parlai",
            runner=lambda out: _parlai_download("light_dialog", out),
            notes="Requires ParlAI installed; task download is triggered by ParlAI.",
        )

        # LIGHT WILD.
        self._registry["light_dialog_wild"] = DatasetEntry(
            name="light_dialog_wild",
            description="In-the-wild deployed LIGHT dialogue data via ParlAI task downloader.",
            method="parlai",
            runner=lambda out: _parlai_download("light_dialog_wild", out),
            notes="Requires ParlAI installed; useful for more natural interactive roleplay data.",
        )

        # Synthetic Persona Chat (optional supplement).
        self._registry["synthetic_persona_chat"] = DatasetEntry(
            name="synthetic_persona_chat",
            description="Optional synthetic persona dialogue supplement.",
            method="hf",
            runner=lambda out: _hf_download("google/Synthetic-Persona-Chat", out),
            notes="Supplemental only; do not let it dominate the corpus mix.",
        )

    def _download_pan15(self, out_dir: Path) -> None:
        """Download PAN15 dataset from Zenodo."""
        _ensure_dir(out_dir)
        urls = {
            "pan15-author-profiling-training-dataset-2015-04-23.zip": (
                "https://zenodo.org/records/3745945/files/"
                "pan15-author-profiling-training-dataset-2015-04-23.zip?download=1"
            ),
            "pan15-author-profiling-test-dataset-2015-04-23.zip": (
                "https://zenodo.org/records/3745945/files/"
                "pan15-author-profiling-test-dataset-2015-04-23.zip?download=1"
            ),
        }
        for filename, url in urls.items():
            dst = out_dir / filename
            if dst.exists():
                print(f"[SKIP] Exists: {dst}")
            else:
                _download_url(url, dst)

    def _write_manual_iemocap_instructions(self, out_dir: Path) -> None:
        """Write manual access instructions for IEMOCAP."""
        _ensure_dir(out_dir)
        text = """IEMOCAP manual access required.

Academic reference:
- Busso, C., Bulut, M., Lee, C.-C., Kazemzadeh, A., Mower, E., Kim, S., Chang, J. N., Lee, S., & Narayanan, S. S. (2008).
  IEMOCAP: Interactive emotional dyadic motion capture database.

Official access page:
- https://sail.usc.edu/iemocap/
- Release form: https://sail.usc.edu/iemocap/iemocap_release.htm

What to do:
1. Fill out the USC release form.
2. Download the archive manually after approval.
3. Put the extracted data under this directory.
"""
        (out_dir / "README_MANUAL_ACCESS.txt").write_text(text, encoding="utf-8")
        print(f"[INFO] Wrote manual instructions to {out_dir / 'README_MANUAL_ACCESS.txt'}")

    @property
    def registry(self) -> Dict[str, DatasetEntry]:
        """Get the dataset registry."""
        return self._registry.copy()

    def list_datasets(self) -> List[str]:
        """Return list of available dataset keys."""
        return list(self._registry.keys())

    def get_dataset_info(self, key: str) -> DatasetEntry:
        """Get info for a specific dataset."""
        if key not in self._registry:
            raise KeyError(f"Unknown dataset: {key}. Available: {self.list_datasets()}")
        return self._registry[key]

    def download(self, datasets: Optional[List[str]] = None, all_datasets: bool = False) -> None:
        """
        Download specified datasets.

        Args:
            datasets: List of dataset keys to download. If None, uses DEFAULT_BUNDLE.
            all_datasets: If True, download all registered datasets.
        """
        if all_datasets:
            selected = list(self._registry.keys())
        elif datasets:
            selected = datasets
        else:
            selected = self.DEFAULT_BUNDLE

        unknown = [d for d in selected if d not in self._registry]
        if unknown:
            raise ValueError(f"Unknown dataset keys: {unknown}. Use list_datasets() to see available.")

        for key in selected:
            entry = self._registry[key]
            dataset_out = self.raw_dir / key
            print(f"\n=== Downloading: {key} ===")
            print(f"Description: {entry.description}")
            print(f"Method: {entry.method}")
            print(f"Notes: {entry.notes}")
            entry.runner(dataset_out)

        self._build_manifest()
        print(f"\nDone. Manifest written to: {self.base_dir / 'dataset_manifest.json'}")

    def _build_manifest(self) -> None:
        """Build and save the dataset manifest."""
        manifest = {}
        for key, entry in self._registry.items():
            manifest[key] = {
                "description": entry.description,
                "method": entry.method,
                "notes": entry.notes,
            }
        _write_json(self.base_dir / "dataset_manifest.json", manifest)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download narrative / RPG backend datasets")
    p.add_argument("--out-dir", type=Path, default=Path("data"), help="Root output directory")
    p.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Dataset keys to download. If omitted, uses a sensible default bundle.",
    )
    p.add_argument("--all", action="store_true", help="Download every registered dataset")
    p.add_argument(
        "--print-registry",
        action="store_true",
        help="Print available dataset keys and exit",
    )
    return p.parse_args()


def main() -> None:
    """CLI entry point for downloading datasets."""
    args = _parse_args()
    downloader = DataDownloader(args.out_dir)

    if args.print_registry:
        print("Available datasets:")
        for key, entry in downloader.registry.items():
            print(f"- {key:24s} | {entry.method:7s} | {entry.description}")
        return

    downloader.download(datasets=args.datasets, all_datasets=args.all)


if __name__ == "__main__":
    main()
