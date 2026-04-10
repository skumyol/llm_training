#!/usr/bin/env python3
"""
Convert generated episode JSONL files to human-readable dialogue transcripts.

Usage:
    python scripts/visualize_dialogues.py --episodes data/raw_episodes/ep_*.jsonl --output dialogue_inspection.txt
    python scripts/visualize_dialogues.py --all --output all_dialogues.txt
"""

import json
import argparse
from pathlib import Path
from datetime import datetime


def format_turn(turn: dict, turn_num: int, npc_name: str = "NPC") -> str:
    """Format a single turn into readable text."""
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"TURN {turn_num}")
    lines.append(f"{'='*60}")
    
    # Context
    scenario = turn.get('scenario', {})
    lines.append(f"\n📋 SCENARIO: {scenario.get('type', 'unknown')} | Phase: {turn.get('arc_phase', 'unknown')}")
    lines.append(f"   Summary: {scenario.get('summary', 'N/A')[:80]}...")
    
    # NPC State
    npc = turn.get('npc_state', {})
    lines.append(f"\n🎭 NPC STATE:")
    lines.append(f"   Name: {npc.get('name', npc_name)}")
    lines.append(f"   Stance: familiarity={npc.get('familiarity_level', '?')}, trust={npc.get('trust_level', '?')}")
    lines.append(f"   Secret: {npc.get('has_secret', False)} | Pressure: duty={npc.get('duty_pressure', '?')}, secrecy={npc.get('secrecy_pressure', '?')}")
    
    # Player Utterance
    player_text = turn.get('input', '[NO TEXT]')
    lines.append(f"\n👤 PLAYER SAYS:")
    lines.append(f'   "{player_text}"')
    
    # C_t Labels (Contextual Analysis)
    C_t = turn.get('C_t', {})
    lines.append(f"\n📊 LABELS - C_t (Context):")
    lines.append(f"   Dialogue Acts: {', '.join(C_t.get('dialogue_act', []))}")
    lines.append(f"   Tone: {C_t.get('tone', '?')} | Risk Type: {C_t.get('risk_type', '?')}")
    
    # A_t Labels (Affective)
    A_t = turn.get('A_t', {})
    lines.append(f"\n📊 LABELS - A_t (Affective):")
    lines.append(f"   Valence: {A_t.get('valence', '?')} | Arousal: {A_t.get('arousal', '?')}")
    lines.append(f"   Threat: {A_t.get('threat', '?')} | Control: {A_t.get('control', '?')}")
    
    # M_t Labels (Mental Model)
    M_t = turn.get('M_t', {})
    lines.append(f"\n📊 LABELS - M_t (Mental Model):")
    lines.append(f"   Player Intent: {M_t.get('player_intent', '?')}")
    lines.append(f"   Knowledge: {M_t.get('player_knowledge', '?')} | Credibility: {M_t.get('player_credibility', '?')}")
    
    # R_N_D Labels (Response Planning)
    D_t = turn.get('D_t', {})
    prior_R = turn.get('prior_R', {})
    lines.append(f"\n📊 LABELS - R_t/D_t (Response Planning):")
    lines.append(f"   Response Policy: {D_t.get('response_policy', '?')} (Target: {turn.get('target_policy', '?')})")
    lines.append(f"   Reveal Decision: {D_t.get('reveal_decision', '?')}")
    lines.append(f"   Repair Strategy: {D_t.get('repair_strategy', '?')}")
    lines.append(f"   Prior Stance Shift: {prior_R.get('stance_shift', '?')}")
    
    # NPC Response
    response = turn.get('response', '[NO RESPONSE GENERATED]')
    lines.append(f"\n🤖 NPC RESPONSE:")
    lines.append(f'   "{response}"')
    
    # Quality flags
    flags = []
    if not response or len(response) < 20:
        flags.append("⚠️ SHORT_RESPONSE")
    if turn.get('validation_error'):
        flags.append(f"⚠️ VALIDATION_ERROR: {turn.get('validation_error')}")
    if D_t.get('response_policy') == turn.get('target_policy'):
        flags.append("✅ POLICY_MATCH")
    else:
        flags.append(f"⚠️ POLICY_MISMATCH (target={turn.get('target_policy', '?')})")
    
    if flags:
        lines.append(f"\n🏷️  FLAGS: {' | '.join(flags)}")
    
    return '\n'.join(lines)


def process_episode(episode_path: Path) -> str:
    """Process a single episode file."""
    lines = []
    lines.append(f"\n{'#'*70}")
    lines.append(f"# EPISODE: {episode_path.stem}")
    lines.append(f"# File: {episode_path}")
    lines.append(f"# Generated: {datetime.fromtimestamp(episode_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"{'#'*70}")
    
    turns = []
    with open(episode_path) as f:
        for line in f:
            if line.strip():
                turns.append(json.loads(line))
    
    lines.append(f"\nTotal Turns: {len(turns)}")
    
    # Get NPC name from first turn if available
    npc_name = "NPC"
    if turns:
        npc_name = turns[0].get('npc_state', {}).get('name', 'NPC')
    
    for i, turn in enumerate(turns, 1):
        lines.append(format_turn(turn, i, npc_name))
    
    lines.append(f"\n{'#'*70}")
    lines.append(f"# END OF EPISODE: {episode_path.stem}")
    lines.append(f"{'#'*70}\n")
    
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Convert episode JSONL files to human-readable dialogue transcripts'
    )
    parser.add_argument(
        '--episodes', '-e',
        nargs='+',
        help='Specific episode JSONL files to process'
    )
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Process all episodes in data/raw_episodes/'
    )
    parser.add_argument(
        '--output', '-o',
        default='dialogue_inspection.txt',
        help='Output text file (default: dialogue_inspection.txt)'
    )
    parser.add_argument(
        '--latest', '-l',
        type=int,
        metavar='N',
        help='Process only the N most recent episodes'
    )
    
    args = parser.parse_args()
    
    # Collect episode files
    episode_files = []
    
    if args.all:
        episode_files = sorted(Path('data/raw_episodes').glob('*.jsonl'))
    elif args.latest:
        all_files = sorted(
            Path('data/raw_episodes').glob('*.jsonl'),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        episode_files = all_files[:args.latest]
    elif args.episodes:
        episode_files = [Path(e) for e in args.episodes]
    else:
        print("Error: Specify --episodes, --all, or --latest")
        parser.print_help()
        return 1
    
    if not episode_files:
        print("No episode files found!")
        return 1
    
    print(f"Processing {len(episode_files)} episode(s)...")
    
    # Process and write output
    with open(args.output, 'w', encoding='utf-8') as out_f:
        out_f.write(f"DIALOGUE INSPECTION REPORT\n")
        out_f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        out_f.write(f"Episodes: {len(episode_files)}\n")
        out_f.write(f"{'='*70}\n\n")
        
        for ep_path in episode_files:
            print(f"  Processing {ep_path.name}...")
            try:
                content = process_episode(ep_path)
                out_f.write(content)
            except Exception as e:
                print(f"    Error: {e}")
                out_f.write(f"\n# ERROR processing {ep_path}: {e}\n")
    
    print(f"\nOutput written to: {args.output}")
    print(f"Total episodes: {len(episode_files)}")
    
    # Print summary stats
    total_turns = 0
    policy_counts = {}
    for ep_path in episode_files:
        with open(ep_path) as f:
            for line in f:
                if line.strip():
                    turn = json.loads(line)
                    total_turns += 1
                    policy = turn.get('D_t', {}).get('response_policy', 'unknown')
                    policy_counts[policy] = policy_counts.get(policy, 0) + 1
    
    print(f"\nSummary Statistics:")
    print(f"  Total turns: {total_turns}")
    print(f"  Response policies:")
    for policy, count in sorted(policy_counts.items(), key=lambda x: -x[1]):
        pct = count / total_turns * 100
        print(f"    {policy:15s}: {count:4d} ({pct:5.1f}%)")
    
    return 0


if __name__ == '__main__':
    exit(main())
