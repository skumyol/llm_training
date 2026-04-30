#!/usr/bin/env python3
"""
Download script for narrative / RPG NPC backend datasets.

This script mixes three strategies:
1) Hugging Face Datasets API for openly mirrored datasets.
2) Direct URL download for archival releases (e.g., PAN15 on Zenodo).
3) Git clone for repos that directly host corpus files / annotations (e.g., EmoBank, MELD).
4) Manual placeholders for gated datasets (e.g., IEMOCAP).

Academic references are included inline next to each dataset entry.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional


# -----------------------------
# Helper utilities
# -----------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def have_cmd(name: str) -> bool:
    return shutil.which(name) is not None


def run(cmd: List[str], cwd: Optional[Path] = None) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def write_json(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def download_url(url: str, out_path: Path) -> None:
    ensure_dir(out_path.parent)
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


def git_clone(repo_url: str, out_dir: Path) -> None:
    if out_dir.exists():
        print(f"[SKIP] Repo already exists: {out_dir}")
        return
    if not have_cmd("git"):
        raise RuntimeError("git is required for repo-based downloads")
    run(["git", "clone", "--depth", "1", repo_url, str(out_dir)])


def hf_download(dataset_id: str, out_dir: Path, config: Optional[str] = None) -> None:
    """
    Load a dataset via 🤗 datasets and save it to disk.
    """
    try:
        from datasets import load_dataset, DatasetDict
    except ImportError:
        raise RuntimeError("Please install datasets: pip install datasets")

    print(f"[HF] Loading {dataset_id} {f'(config={config})' if config else ''}")
    ds = load_dataset(dataset_id, config, trust_remote_code=True)
    if hasattr(ds, "save_to_disk"):
        ensure_dir(out_dir.parent)
        ds.save_to_disk(str(out_dir))
    else:
        # Very old datasets edge case.
        ensure_dir(out_dir)
        write_json(out_dir / "info.json", {"dataset_id": dataset_id, "config": config})


def parlai_download(task_name: str, out_dir: Path) -> None:
    """
    Trigger ParlAI task download by asking ParlAI to display one example.
    ParlAI tasks auto-download into PARLAI_HOME / data.
    """
    if not have_cmd("python"):
        raise RuntimeError("Python executable not found in PATH")

    env = os.environ.copy()
    env["PARLAI_HOME"] = str(out_dir.resolve())
    print(f"[PARLAI] Triggering task download: {task_name} into {env['PARLAI_HOME']}")
    subprocess.run(
        [sys.executable, "-m", "parlai", "display_data", "-t", task_name, "-ne", "1"],
        check=True,
        env=env,
    )


# -----------------------------
# Dataset registry
# -----------------------------

@dataclass
class DatasetEntry:
    name: str
    description: str
    method: str
    runner: Callable[[Path], None]
    notes: str


BASE = Path(".")


def make_registry(base_dir: Path) -> Dict[str, DatasetEntry]:
    raw = base_dir / "raw"
    ensure_dir(raw)

    registry: Dict[str, DatasetEntry] = {}

    # ---------------- Personality / profile ----------------
    # Essays / Big Five convenience mirror.
    # Academic refs:
    # - Pennebaker, J. W., & King, L. A. (1999). Linguistic styles: Language use as an individual difference.
    # - Mairesse, F., et al. (2007). Using linguistic cues for the automatic recognition of personality in conversation and text.
    # Mirror note: this uses a Hugging Face convenience mirror of the Essays-style Big Five essay corpus.
    registry["essays_big5"] = DatasetEntry(
        name="essays_big5",
        description="Big Five essay corpus mirror for static personality regression.",
        method="hf",
        runner=lambda out: hf_download("jingjietan/essays-big5", out),
        notes="Convenience mirror, not the original hosting location.",
    )

    # PAN 2015 Author Profiling.
    # Academic ref:
    # - Rangel, F., et al. (2015). Overview of the 3rd Author Profiling Task at PAN 2015.
    # Zenodo archive note: includes personality scores for the author profiling task.
    registry["pan15"] = DatasetEntry(
        name="pan15",
        description="PAN15 Author Profiling training + test zips from Zenodo.",
        method="url",
        runner=lambda out: download_pan15(out),
        notes="Open archival release on Zenodo.",
    )

    # ---------------- Affect / emotion ----------------
    # EmoBank.
    # Academic refs:
    # - Buechel, S., & Hahn, U. (2017). EmoBank: Studying the impact of annotation perspective and representation format on dimensional emotion analysis.
    # - Buechel, S., et al. (2017). Building an Emotion Bank with Clear and Reusable Annotations.
    # Repo note: JULIELab hosts the corpus and split files directly in GitHub.
    registry["emobank"] = DatasetEntry(
        name="emobank",
        description="Valence-Arousal-Dominance text corpus from JULIELab GitHub.",
        method="git",
        runner=lambda out: git_clone("https://github.com/JULIELab/emobank.git", out),
        notes="GitHub repo contains corpus + splits.",
    )

    # GoEmotions.
    # Academic ref:
    # - Demszky, D., et al. (2020). GoEmotions: A Dataset of Fine-Grained Emotions.
    registry["goemotions"] = DatasetEntry(
        name="goemotions",
        description="58k Reddit comments with 27 emotion labels + Neutral.",
        method="hf",
        runner=lambda out: hf_download("mrm8488/goemotions", out),
        notes="Good supplement to dimensional affect with categorical emotions.",
    )

    # EmpatheticDialogues.
    # Academic ref:
    # - Rashkin, H., et al. (2019). Towards Empathetic Open-domain Conversation Models: A New Benchmark and Dataset.
    registry["empathetic_dialogues"] = DatasetEntry(
        name="empathetic_dialogues",
        description="Emotion-grounded dialogue dataset from Meta's HF mirror.",
        method="hf",
        runner=lambda out: hf_download("facebook/empathetic_dialogues", out),
        notes="Useful for emotionally grounded response behavior.",
    )

    # DailyDialog.
    # Academic ref:
    # - Li, Y., et al. (2017). DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset.
    registry["dailydialog"] = DatasetEntry(
        name="dailydialog",
        description="DailyDialog with emotion and dialog act annotations.",
        method="hf",
        runner=lambda out: hf_download("ConvLab/dailydialog", out),
        notes="Convenient HF-hosted version for quick prototyping.",
    )

    # MELD.
    # Academic ref:
    # - Poria, S., et al. / Ghosal, D., et al. (2019). MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations.
    # Repo note: declare-lab/MELD hosts annotations and download instructions.
    registry["meld"] = DatasetEntry(
        name="meld",
        description="Multi-party conversation emotion dataset (annotations/repo).",
        method="git",
        runner=lambda out: git_clone("https://github.com/declare-lab/MELD.git", out),
        notes="Repo contains annotations and metadata; media may require extra download.",
    )

    # IEMOCAP.
    # Academic ref:
    # - Busso, C., et al. (2008). IEMOCAP: Interactive emotional dyadic motion capture database.
    registry["iemocap"] = DatasetEntry(
        name="iemocap",
        description="Manual-only placeholder for gated USC IEMOCAP release.",
        method="manual",
        runner=lambda out: write_manual_iemocap_instructions(out),
        notes="USC requires an electronic release form; no direct public auto-download.",
    )

    # ---------------- Dialogue / roleplay / RPG ----------------
    # PersonaChat.
    # Academic ref:
    # - Zhang, S., et al. (2018). Personalizing Dialogue Agents: I have a dog, do you have pets too?
    registry["personachat"] = DatasetEntry(
        name="personachat",
        description="Persona-conditioned chat dataset via HF mirror.",
        method="hf",
        runner=lambda out: hf_download("bavard/personachat_truecased", out),
        notes="Widely used persona-conditioning benchmark.",
    )

    # CRD3.
    # Academic ref:
    # - Rameshkumar, R., & Bailey, P. (2020). Storytelling with Dialogue: A Critical Role Dungeons and Dragons Dataset.
    registry["crd3"] = DatasetEntry(
        name="crd3",
        description="Critical Role D&D dialogue dataset from Hugging Face.",
        method="hf",
        runner=lambda out: hf_download("microsoft/crd3", out),
        notes="Strong RPG-adjacent multi-turn dialogue source.",
    )

    # LIGHT.
    # Academic refs:
    # - Urbanek, J., et al. (2019). Learning to Speak and Act in a Fantasy Text Adventure Game.
    # - LIGHT docs note 11,000 episodes of character interactions in the original dataset.
    registry["light_dialog"] = DatasetEntry(
        name="light_dialog",
        description="Original LIGHT fantasy dialogue/action data via ParlAI task downloader.",
        method="parlai",
        runner=lambda out: parlai_download("light_dialog", out),
        notes="Requires ParlAI installed; task download is triggered by ParlAI.",
    )

    # LIGHT WILD.
    # Academic / official note:
    # - ParlAI tasks docs list light_dialog_wild with 41,131+ training episodes collected from deployment.
    registry["light_dialog_wild"] = DatasetEntry(
        name="light_dialog_wild",
        description="In-the-wild deployed LIGHT dialogue data via ParlAI task downloader.",
        method="parlai",
        runner=lambda out: parlai_download("light_dialog_wild", out),
        notes="Requires ParlAI installed; useful for more natural interactive roleplay data.",
    )

    # Optional: synthetic persona supplement.
    # Academic / dataset note:
    # - Google Synthetic Persona Chat (2023) extends PersonaChat with synthetic conversations.
    registry["synthetic_persona_chat"] = DatasetEntry(
        name="synthetic_persona_chat",
        description="Optional synthetic persona dialogue supplement.",
        method="hf",
        runner=lambda out: hf_download("google/Synthetic-Persona-Chat", out),
        notes="Supplemental only; do not let it dominate the corpus mix.",
    )

    return registry


# -----------------------------
# Custom runners
# -----------------------------

def download_pan15(out_dir: Path) -> None:
    ensure_dir(out_dir)
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
            download_url(url, dst)


def write_manual_iemocap_instructions(out_dir: Path) -> None:
    ensure_dir(out_dir)
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


# -----------------------------
# Main
# -----------------------------

def build_manifest(registry: Dict[str, DatasetEntry], out_dir: Path) -> None:
    manifest = {}
    for key, entry in registry.items():
        manifest[key] = {
            "description": entry.description,
            "method": entry.method,
            "notes": entry.notes,
        }
    write_json(out_dir / "dataset_manifest.json", manifest)


def parse_args() -> argparse.Namespace:
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
    args = parse_args()
    registry = make_registry(args.out_dir)

    if args.print_registry:
        print("Available datasets:")
        for key, entry in registry.items():
            print(f"- {key:24s} | {entry.method:7s} | {entry.description}")
        return

    build_manifest(registry, args.out_dir)

    default_bundle = [
        "essays_big5",
        "pan15",
        "emobank",
        "goemotions",
        "empathetic_dialogues",
        "dailydialog",
        "meld",
        "personachat",
        "crd3",
        # leave LIGHT and IEMOCAP opt-in because they are heavier / more fragile
    ]

    if args.all:
        selected = list(registry.keys())
    elif args.datasets:
        selected = args.datasets
    else:
        selected = default_bundle

    unknown = [d for d in selected if d not in registry]
    if unknown:
        raise SystemExit(f"Unknown dataset keys: {unknown}. Use --print-registry.")

    for key in selected:
        entry = registry[key]
        dataset_out = args.out_dir / "raw" / key
        print(f"\n=== Downloading: {key} ===")
        print(f"Description: {entry.description}")
        print(f"Method: {entry.method}")
        print(f"Notes: {entry.notes}")
        entry.runner(dataset_out)

    print("\nDone.")
    print(f"Manifest written to: {args.out_dir / 'dataset_manifest.json'}")


if __name__ == "__main__":
    main()
