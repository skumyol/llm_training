#!/usr/bin/env python3
"""
DeepSeek fact-checking script for bibliography entries.
Reads references.bib, calls DeepSeek API for each entry, writes summaries to a file.

Usage:
    python paper/deepseek_factcheck.py                          # all entries
    python paper/deepseek_factcheck.py --limit 5                # first 5 only
    python paper/deepseek_factcheck.py --output factcheck.md    # custom output
"""

import re
import json
import time
import argparse
import urllib.request
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
BIB_PATH = Path(__file__).parent / "references.bib"
OUTPUT_PATH = Path(__file__).parent / "factcheck_deepseek.md"
ENV_PATH = Path(__file__).parent.parent / ".env"

API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"
DELAY = 1.5  # seconds between requests (rate limiting)

# ── Load API key ──────────────────────────────────────────────────────────────
def load_api_key() -> str:
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            if line.startswith("DEEPSEEK_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError(f"DEEPSEEK_API_KEY not found in {ENV_PATH}")

# ── Parse .bib file ──────────────────────────────────────────────────────────
def parse_bib(path: Path) -> list[dict]:
    """Parse a .bib file into a list of {key, type, fields, raw} dicts."""
    text = path.read_text(encoding="utf-8")
    entries = []
    # Match @type{key, ...}
    pattern = re.compile(r'@(\w+)\s*\{\s*(\w+)\s*,\s*(.*?)\}\s*$', re.MULTILINE | re.DOTALL)
    for m in pattern.finditer(text):
        etype, key, body = m.group(1), m.group(2), m.group(3)
        fields = {}
        # Extract field = {value} or field = "value"
        for fm in re.finditer(r'(\w+)\s*=\s*[{"]([^}"]*)[}"]', body):
            fields[fm.group(1)] = fm.group(2)
        entries.append({"key": key, "type": etype, "fields": fields,
                        "raw": m.group(0)})
    return entries


# ── DeepSeek API call ─────────────────────────────────────────────────────────
def call_deepseek(api_key: str, prompt: str, max_tokens: int = 500) -> str:
    """Call DeepSeek chat completions API. Returns response text or error message."""
    payload = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system", "content": (
                "You are a research assistant fact-checking bibliography entries. "
                "Answer concisely and accurately. If you don't know a paper, say so."
            )},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,  # low temp for factual accuracy
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[ERROR] {e}"


# ── Build prompt ──────────────────────────────────────────────────────────────
def build_prompt(entry: dict) -> str:
    """Construct a fact-checking prompt from a bib entry."""
    f = entry["fields"]
    title = f.get("title", "Unknown title")
    authors = f.get("author", "Unknown authors")
    year = f.get("year", "????")
    venue = f.get("journal", f.get("booktitle", "Unknown venue"))

    # Clean up LaTeX in title
    title_clean = re.sub(r"\{|\}|\\", "", title)

    return (
        f"Summarize this paper in 2-3 sentences for fact-checking purposes:\n"
        f"Title: {title_clean}\n"
        f"Authors: {authors[:200]}\n"
        f"Year: {year}\n"
        f"Venue: {venue}\n\n"
        f"In your response, include:\n"
        f"1. What the paper is about (1 sentence)\n"
        f"2. Key findings or contributions (1 sentence)\n"
        f"3. Whether this paper actually exists and matches the citation (yes/no/cannot verify)\n"
        f"4. Any concerns about the citation accuracy\n\n"
        f"Keep it under 150 words."
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="DeepSeek fact-check for bib entries")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N entries (0=all)")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH), help="Output file path")
    parser.add_argument("--start", type=int, default=0, help="Start from entry N")
    args = parser.parse_args()

    api_key = load_api_key()
    entries = parse_bib(BIB_PATH)

    if args.limit > 0:
        entries = entries[args.start : args.start + args.limit]
    elif args.start > 0:
        entries = entries[args.start:]

    total = len(entries)
    print(f"Processing {total} bibliography entries via DeepSeek...")
    print(f"Model: {MODEL}  |  Output: {args.output}")
    print()

    results = []
    for i, entry in enumerate(entries):
        key = entry["key"]
        title = entry["fields"].get("title", "?")[:80]
        print(f"[{i+1}/{total}] {key} — {title}...", end=" ", flush=True)

        prompt = build_prompt(entry)
        response = call_deepseek(api_key, prompt)
        results.append((entry, response))

        # Quick status indicator
        has_error = response.startswith("[ERROR]")
        print("❌" if has_error else "✅")

        if i < total - 1:
            time.sleep(DELAY)

    # ── Write output ──────────────────────────────────────────────────────────
    out = Path(args.output)
    lines = [
        "# DeepSeek Fact-Check Report",
        f"",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}",
        f"**Model:** {MODEL}",
        f"**Entries processed:** {total}",
        f"**Source:** `{BIB_PATH}`",
        f"",
        "---",
        "",
    ]

    for entry, response in results:
        f = entry["fields"]
        title = re.sub(r"\{|\}|\\", "", f.get("title", "?"))
        authors = f.get("author", "?")[:150]
        year = f.get("year", "?")
        venue = f.get("journal", f.get("booktitle", "?"))

        lines.extend([
            f"## {entry['key']}",
            f"",
            f"| Field | Value |",
            f"|-------|-------|",
            f"| Title | {title} |",
            f"| Authors | {authors} |",
            f"| Year | {year} |",
            f"| Venue | {venue} |",
            f"| Type | @{entry['type']} |",
            f"",
            f"### DeepSeek Response",
            f"",
            response,
            f"",
            "---",
            "",
        ])

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✅ Written {total} entries to {out}")


if __name__ == "__main__":
    main()
