#!/usr/bin/env python3
"""
chat.py  —  Interactive NPC terminal chat
==========================================
Ollama/Codex-style REPL for testing trained NPC models from the command line.

Usage:
  python -m src.infer.chat                              # auto-load from artifacts/
  python -m src.infer.chat --npc commander_vance        # start with specific NPC
  python -m src.infer.chat --model-dir artifacts/dialogue_model_persona/best_model
  python -m src.infer.chat --no-model                   # encoder-only mode (test conditioning)

In-session commands (prefix with /):
  /npc <id> [profile text...]   — register & switch to a new NPC
  /load <yaml_path>             — load all NPCs from a world context YAML
  /list                         — show registered NPCs
  /switch <id>                  — switch active NPC
  /profile                      — show active NPC profile
  /state                        — show current personality & affect vectors
  /memory                       — show retrieved episodic memories
  /reset                        — clear conversation history for active NPC
  /temp <float>                 — set generation temperature
  /tokens <int>                 — set max new tokens
  /save <path>                  — save conversation log to file
  /help                         — show this list
  /quit or /exit                — exit

Examples:
  > /npc vance A stoic castle guard who values duty above all. Speaks bluntly.
  > Have you caught the spy yet?
  > /state
  > /npc elara A diplomatic mayor anxious about the siege. She may be hiding something.
  > We need to negotiate a surrender before people starve.
  > /save session_001.json
"""
from __future__ import annotations

import argparse
import json
import os
import readline
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ── ANSI colours ──────────────────────────────────────────────────────────────
BOLD   = "\033[1m"
DIM    = "\033[2m"
ITALIC = "\033[3m"
NPC_C  = "\033[32m"      # green  — NPC speech
PLY_C  = "\033[36m"      # cyan   — player
CMD_C  = "\033[33m"      # yellow — commands / meta
ERR_C  = "\033[31m"      # red    — errors
DIM_C  = "\033[2m"       # dim    — state info
RESET  = "\033[0m"

COMMANDS = [
    "/npc", "/load", "/list", "/switch", "/profile",
    "/state", "/memory", "/reset", "/temp", "/tokens",
    "/save", "/help", "/quit", "/exit",
]


def _c(color: str, text: str) -> str:
    return f"{color}{text}{RESET}"


def _wrap(text: str, prefix: str = "", width: int = 76) -> str:
    lines = textwrap.wrap(text, width=width - len(prefix))
    return ("\n" + " " * len(prefix)).join(lines)


def _print_banner(model_dir: Optional[str]) -> None:
    tag = Path(model_dir).name if model_dir else "encoder-only mode"
    print(f"\n{BOLD}{NPC_C}{'━'*60}{RESET}")
    print(f"{BOLD}{NPC_C}  NPC Chat  {RESET}{DIM_C}({tag}){RESET}")
    print(f"{BOLD}{NPC_C}{'━'*60}{RESET}")
    print(f"  {DIM_C}Type a message to chat, /help for commands, /quit to exit.{RESET}\n")


# ── Tab completion ─────────────────────────────────────────────────────────────

class _Completer:
    def __init__(self, npc_ids: List[str]) -> None:
        self.npc_ids = npc_ids

    def complete(self, text: str, state: int):
        options = [c for c in COMMANDS if c.startswith(text)]
        if text.startswith("/switch ") or text.startswith("/npc "):
            options += [c for c in self.npc_ids if c.startswith(text.split()[-1])]
        return options[state] if state < len(options) else None


# ── State display ──────────────────────────────────────────────────────────────

def _fmt_vec(vec, dims: List[str], decimals: int = 3) -> str:
    if vec is None:
        return "—"
    vals = vec[0].tolist() if hasattr(vec, "tolist") else list(vec)
    return "  ".join(f"{d}={v:.{decimals}f}" for d, v in zip(dims, vals))


def _show_state(service, npc_id: str, last_query: str = "") -> None:
    import torch
    try:
        cfg = service.cfg
        # personality
        pvec = service._personality_vec(npc_id)
        ocean = ["O", "C", "E", "A", "N"]
        print(f"\n  {CMD_C}Personality (OCEAN){RESET}  {_fmt_vec(pvec, ocean)}")

        # affect
        avec = service._affect_vec(last_query or "hello")
        vad = ["valence", "arousal", "dominance"]
        print(f"  {CMD_C}Affect (VAD)       {RESET}  {_fmt_vec(avec, vad)}")
    except Exception as e:
        print(f"  {DIM_C}State unavailable: {e}{RESET}")


# ── Load world context YAML ───────────────────────────────────────────────────

def _load_world_yaml(yaml_path: Path, service) -> List[str]:
    try:
        import yaml
    except ImportError:
        import json as yaml   # fallback (won't work for YAML but gives clear error)

    with open(yaml_path, encoding="utf-8") as f:
        world = yaml.safe_load(f)

    npc_ids: List[str] = []
    for npc in world.get("npcs", []):
        npc_id = npc["npc_id"]
        persona = ", ".join(npc.get("persona_style", []))
        values  = ", ".join(npc.get("values", []))
        goals   = ". ".join(npc.get("core_goals", []))
        role    = npc.get("role", "character")
        profile = (
            f"{npc.get('name', npc_id)}, a {role}. "
            f"Persona: {persona}. Values: {values}. Goals: {goals}."
        )
        service.register_npc(npc_id, profile)
        npc_ids.append(npc_id)
        print(f"  {NPC_C}+{RESET} {npc_id}  {DIM_C}— {profile[:60]}...{RESET}")

    return npc_ids


# ── Conversation log ───────────────────────────────────────────────────────────

class ConversationLog:
    def __init__(self) -> None:
        self.turns: List[Dict] = []
        self.started_at = datetime.now().isoformat()

    def add(self, npc_id: str, player: str, npc: str, elapsed_ms: int) -> None:
        self.turns.append({
            "npc_id": npc_id, "player": player,
            "npc": npc, "elapsed_ms": elapsed_ms,
            "ts": datetime.now().isoformat(),
        })

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"started_at": self.started_at, "turns": self.turns}, f, indent=2)
        print(f"  {CMD_C}Saved{RESET} {len(self.turns)} turns → {path}")


# ── Main REPL ─────────────────────────────────────────────────────────────────

def run_chat(
    model_dir:    Optional[str],
    initial_npc:  Optional[str],
    initial_profile: Optional[str],
    world_yaml:   Optional[Path],
    no_model:     bool,
    temperature:  float,
    max_tokens:   int,
) -> None:

    # ── Load service ──────────────────────────────────────────────────────────
    print(f"  {DIM_C}Loading models...{RESET}", end="", flush=True)
    try:
        from src.common.config import InferenceConfig
        from src.infer.service import NPCInferenceService

        # Auto-discover model dir
        if model_dir is None and not no_model:
            candidates = sorted(Path("artifacts").glob("dialogue_model*/*/best_model"))
            if candidates:
                model_dir = str(candidates[-1])
                print(f"\r  {NPC_C}✓{RESET} Model: {DIM_C}{model_dir}{RESET}         ")
            else:
                print(f"\r  {CMD_C}⚠{RESET} No dialogue model found — encoding only (no text generation)")
                no_model = True

        cfg_kwargs: Dict = {"temperature": temperature, "max_new_tokens": max_tokens}
        if model_dir:
            cfg_kwargs["dialogue_model_dir"] = model_dir

        cfg = InferenceConfig(**cfg_kwargs)
        service = NPCInferenceService(cfg)
        print(f"\r  {NPC_C}✓{RESET} Service ready.                                      ")

    except Exception as e:
        print(f"\r  {ERR_C}✗{RESET} Failed to load service: {e}")
        raise SystemExit(1)

    npc_profiles: Dict[str, str] = {}
    log = ConversationLog()
    active_npc: Optional[str] = None
    last_player_msg = ""

    # ── Register startup NPCs ─────────────────────────────────────────────────
    if world_yaml and world_yaml.exists():
        _h2 = lambda s: print(f"\n  {CMD_C}{s}{RESET}")
        _h2(f"Loading world: {world_yaml}")
        npc_ids = _load_world_yaml(world_yaml, service)
        if npc_ids and not initial_npc:
            active_npc = npc_ids[0]
            print(f"  {DIM_C}Active NPC: {active_npc}{RESET}")
        for nid in npc_ids:
            npc_profiles[nid] = ""

    if initial_npc:
        profile = initial_profile or f"An NPC named {initial_npc}."
        service.register_npc(initial_npc, profile)
        npc_profiles[initial_npc] = profile
        active_npc = initial_npc

    if not active_npc:
        print(f"  {CMD_C}No NPC loaded.{RESET} "
              f"Use {CMD_C}/npc <id> <profile>{RESET} to register one.")

    # ── Tab completion ────────────────────────────────────────────────────────
    completer = _Completer(list(npc_profiles.keys()))
    readline.set_completer(completer.complete)
    readline.parse_and_bind("tab: complete")

    _print_banner(model_dir)

    # ── REPL ──────────────────────────────────────────────────────────────────
    while True:
        # Prompt line
        npc_tag = f"{NPC_C}{active_npc}{RESET} " if active_npc else ""
        prompt = f"{npc_tag}{PLY_C}you{RESET} › "
        try:
            raw = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{DIM_C}Bye.{RESET}")
            break

        if not raw:
            continue

        # ── Commands ──────────────────────────────────────────────────────────
        if raw.startswith("/"):
            parts = raw.split(None, 2)
            cmd = parts[0].lower()

            if cmd in ("/quit", "/exit"):
                print(f"{DIM_C}Bye.{RESET}")
                break

            elif cmd == "/help":
                print(f"\n{CMD_C}Commands:{RESET}")
                for line in __doc__.split("In-session commands")[1].split("\n"):
                    if line.strip().startswith("/") or "—" in line:
                        print(f"  {line}")

            elif cmd == "/npc":
                if len(parts) < 2:
                    print(f"  {ERR_C}Usage: /npc <id> [profile text]{RESET}")
                    continue
                nid = parts[1]
                profile = parts[2] if len(parts) > 2 else f"An NPC named {nid}."
                service.register_npc(nid, profile)
                npc_profiles[nid] = profile
                completer.npc_ids = list(npc_profiles.keys())
                active_npc = nid
                print(f"  {NPC_C}✓{RESET} Registered & switched to {BOLD}{nid}{RESET}")
                print(f"  {DIM_C}{profile}{RESET}")

            elif cmd == "/load":
                if len(parts) < 2:
                    print(f"  {ERR_C}Usage: /load <yaml_path>{RESET}")
                    continue
                ypath = Path(parts[1])
                if not ypath.exists():
                    print(f"  {ERR_C}File not found: {ypath}{RESET}")
                    continue
                ids = _load_world_yaml(ypath, service)
                for nid in ids:
                    npc_profiles[nid] = ""
                completer.npc_ids = list(npc_profiles.keys())
                if ids and not active_npc:
                    active_npc = ids[0]

            elif cmd == "/list":
                if not npc_profiles:
                    print(f"  {DIM_C}No NPCs registered yet.{RESET}")
                for nid, prof in npc_profiles.items():
                    marker = f"{NPC_C}●{RESET}" if nid == active_npc else f"{DIM_C}○{RESET}"
                    print(f"  {marker} {BOLD}{nid}{RESET}  {DIM_C}{prof[:60]}{RESET}")

            elif cmd == "/switch":
                if len(parts) < 2:
                    print(f"  {ERR_C}Usage: /switch <id>{RESET}")
                    continue
                nid = parts[1]
                if nid not in npc_profiles:
                    print(f"  {ERR_C}Unknown NPC: {nid}. Register first with /npc{RESET}")
                    continue
                active_npc = nid
                print(f"  {NPC_C}Switched to {BOLD}{nid}{RESET}")

            elif cmd == "/profile":
                if not active_npc:
                    print(f"  {ERR_C}No active NPC.{RESET}")
                    continue
                print(f"  {BOLD}{active_npc}{RESET}: {npc_profiles.get(active_npc, '?')}")

            elif cmd == "/state":
                if not active_npc:
                    print(f"  {ERR_C}No active NPC.{RESET}")
                    continue
                _show_state(service, active_npc, last_player_msg)

            elif cmd == "/memory":
                print(f"  {DIM_C}Memory retrieval is handled internally per response.{RESET}")

            elif cmd == "/reset":
                if active_npc and hasattr(service, "_states"):
                    service._states.pop(active_npc, None)
                print(f"  {NPC_C}✓{RESET} Conversation history cleared for {active_npc}")

            elif cmd == "/temp":
                if len(parts) < 2:
                    print(f"  Temperature: {temperature}")
                    continue
                try:
                    temperature = float(parts[1])
                    if hasattr(service, "cfg"):
                        service.cfg.temperature = temperature
                    print(f"  {NPC_C}✓{RESET} Temperature → {temperature}")
                except ValueError:
                    print(f"  {ERR_C}Invalid float: {parts[1]}{RESET}")

            elif cmd == "/tokens":
                if len(parts) < 2:
                    print(f"  Max tokens: {max_tokens}")
                    continue
                try:
                    max_tokens = int(parts[1])
                    if hasattr(service, "cfg"):
                        service.cfg.max_new_tokens = max_tokens
                    print(f"  {NPC_C}✓{RESET} Max tokens → {max_tokens}")
                except ValueError:
                    print(f"  {ERR_C}Invalid int: {parts[1]}{RESET}")

            elif cmd == "/save":
                path = Path(parts[1]) if len(parts) > 1 else \
                       Path(f"chat_logs/{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                log.save(path)

            else:
                print(f"  {ERR_C}Unknown command: {cmd}  (type /help){RESET}")

            continue

        # ── Regular dialogue turn ─────────────────────────────────────────────
        if not active_npc:
            print(f"  {CMD_C}Register an NPC first: /npc <id> <profile text>{RESET}")
            continue

        if no_model:
            print(f"  {DIM_C}[No dialogue model loaded — run training to enable responses]{RESET}")
            continue

        last_player_msg = raw
        t0 = time.time()
        try:
            reply = service.respond(active_npc, raw)
            elapsed = int((time.time() - t0) * 1000)
        except Exception as e:
            print(f"  {ERR_C}Error: {e}{RESET}")
            continue

        # Print NPC response
        print(f"\n  {NPC_C}{BOLD}{active_npc}{RESET}  "
              f"{DIM_C}({elapsed}ms){RESET}")
        wrapped = _wrap(reply, prefix="    ")
        print(f"    {NPC_C}{wrapped}{RESET}\n")

        log.add(active_npc, raw, reply, elapsed)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Interactive NPC chat (ollama-style REPL)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model-dir",  type=str,  default=None,
                   help="Path to best_model directory (auto-discovered if omitted)")
    p.add_argument("--npc",        type=str,  default=None, dest="npc_id",
                   help="Register and start with this NPC id")
    p.add_argument("--profile",    type=str,  default=None,
                   help="NPC profile text (used with --npc)")
    p.add_argument("--load-world", type=Path, default=None,
                   help="Load all NPCs from a world context YAML")
    p.add_argument("--no-model",   action="store_true",
                   help="Skip dialogue model (encoder-only, for testing conditioning)")
    p.add_argument("--temp",       type=float, default=0.8, dest="temperature")
    p.add_argument("--tokens",     type=int,   default=120, dest="max_tokens")
    args = p.parse_args()

    run_chat(
        model_dir       = args.model_dir,
        initial_npc     = args.npc_id,
        initial_profile = args.profile,
        world_yaml      = args.load_world,
        no_model        = args.no_model,
        temperature     = args.temperature,
        max_tokens      = args.max_tokens,
    )


if __name__ == "__main__":
    main()
