#!/usr/bin/env python3
"""Deep data quality analysis for generated episodes."""
import json
import os
from pathlib import Path
from collections import Counter, defaultdict
import statistics

DATA_DIR = Path("/Users/skumyol/Documents/GitHub/llm_training/data")

def analyze_episodes():
    """Analyze all generated episodes."""
    raw_dir = DATA_DIR / "raw_episodes"
    validated_dir = DATA_DIR / "validated_turns"
    cf_dir = DATA_DIR / "counterfactuals"
    
    results = {
        "counts": {
            "raw_episodes": len(list(raw_dir.glob("*.jsonl"))),
            "validated_episodes": len(list(validated_dir.glob("*.jsonl"))),
            "counterfactual_files": len(list(cf_dir.glob("*.jsonl"))),
        },
        "episode_stats": defaultdict(list),
        "field_completeness": Counter(),
        "scenario_types": Counter(),
        "stakes_distribution": Counter(),
        "arc_phases": Counter(),
        "label_distributions": {
            "dialogue_act": Counter(),
            "tone": Counter(),
            "risk_type": Counter(),
            "valence": Counter(),
            "arousal": Counter(),
            "threat": Counter(),
            "control": Counter(),
            "player_intent": Counter(),
            "player_knowledge": Counter(),
            "player_credibility": Counter(),
            "duty_pressure": Counter(),
            "secrecy_pressure": Counter(),
            "face_pressure": Counter(),
            "value_conflict": Counter(),
            "response_policy": Counter(),
            "reveal_decision": Counter(),
            "repair_strategy": Counter(),
        },
        "stance_levels": defaultdict(Counter),
        "stance_deltas": defaultdict(Counter),
        "turns_per_episode": [],
        "response_lengths": [],
        "issues": [],
    }
    
    # Analyze raw episodes
    for ep_file in sorted(raw_dir.glob("*.jsonl")):
        try:
            with open(ep_file) as f:
                turns = [json.loads(line) for line in f if line.strip()]
            
            results["turns_per_episode"].append(len(turns))
            
            for turn in turns:
                _analyze_turn(turn, results)
                
        except Exception as e:
            results["issues"].append(f"Error in {ep_file.name}: {e}")
    
    return results


def _analyze_turn(turn, results):
    """Analyze a single turn."""
    # Required fields check
    required_fields = [
        "episode_id", "turn_idx", "scenario_type", "W", "arc_phase",
        "input", "C_t", "A_t", "M_t", "R_t", "N_t", "D_t", "response"
    ]
    
    for field in required_fields:
        if field in turn:
            results["field_completeness"][field] += 1
        else:
            results["issues"].append(f"Missing {field} in turn {turn.get('turn_idx', '?')}")
    
    # Scenario type
    if "scenario_type" in turn:
        results["scenario_types"][turn["scenario_type"]] += 1
    
    # Stakes
    if "stakes" in turn:
        results["stakes_distribution"][turn["stakes"]] += 1
    
    # Arc phase
    if "arc_phase" in turn:
        results["arc_phases"][turn["arc_phase"]] += 1
    
    # C_t labels
    C_t = turn.get("C_t", {})
    for key in ["dialogue_act", "tone", "risk_type"]:
        if key in C_t:
            results["label_distributions"][key][C_t[key]] += 1
    
    # A_t labels
    A_t = turn.get("A_t", {})
    for key in ["valence", "arousal", "threat", "control"]:
        if key in A_t:
            results["label_distributions"][key][A_t[key]] += 1
    
    # M_t labels
    M_t = turn.get("M_t", {})
    for key in ["player_intent", "player_knowledge", "player_credibility"]:
        if key in M_t:
            results["label_distributions"][key][M_t[key]] += 1
    
    # R_t (stance)
    R_t = turn.get("R_t", {})
    for stance_dim, values in R_t.items():
        if isinstance(values, dict):
            if "level" in values:
                results["stance_levels"][stance_dim][values["level"]] += 1
            if "delta" in values:
                results["stance_deltas"][stance_dim][values["delta"]] += 1
    
    # N_t labels
    N_t = turn.get("N_t", {})
    for key in ["duty_pressure", "secrecy_pressure", "face_pressure", "value_conflict"]:
        if key in N_t:
            results["label_distributions"][key][N_t[key]] += 1
    
    # D_t labels
    D_t = turn.get("D_t", {})
    for key in ["response_policy", "reveal_decision", "repair_strategy"]:
        if key in D_t:
            results["label_distributions"][key][D_t[key]] += 1
    
    # Response length
    if "response" in turn:
        results["response_lengths"].append(len(turn["response"].split()))


def print_report(results):
    """Print formatted report."""
    print("=" * 70)
    print("DATA QUALITY ANALYSIS REPORT")
    print("=" * 70)
    
    print("\n📊 VOLUME SUMMARY")
    print("-" * 40)
    for key, val in results["counts"].items():
        print(f"  {key}: {val}")
    
    total_turns = sum(results["turns_per_episode"])
    print(f"  total_turns: {total_turns}")
    
    if results["turns_per_episode"]:
        print(f"  avg_turns_per_episode: {statistics.mean(results['turns_per_episode']):.1f}")
        print(f"  min_turns: {min(results['turns_per_episode'])}")
        print(f"  max_turns: {max(results['turns_per_episode'])}")
    
    print("\n📋 FIELD COMPLETENESS")
    print("-" * 40)
    for field, count in sorted(results["field_completeness"].items(), key=lambda x: -x[1]):
        pct = count / total_turns * 100 if total_turns else 0
        status = "✓" if pct > 95 else "⚠"
        print(f"  {status} {field}: {count}/{total_turns} ({pct:.1f}%)")
    
    print("\n🎭 SCENARIO DISTRIBUTION")
    print("-" * 40)
    for scenario, count in results["scenario_types"].most_common():
        print(f"  {scenario}: {count}")
    
    print("\n⚡ STAKES DISTRIBUTION")
    print("-" * 40)
    for stake, count in results["stakes_distribution"].most_common():
        print(f"  {stake}: {count}")
    
    print("\n🔄 ARC PHASE DISTRIBUTION")
    print("-" * 40)
    for phase, count in results["arc_phases"].most_common():
        print(f"  {phase}: {count}")
    
    print("\n🏷️ LABEL DISTRIBUTIONS")
    print("-" * 40)
    for label_type, dist in results["label_distributions"].items():
        if dist:
            print(f"\n  {label_type}:")
            for val, count in dist.most_common():
                print(f"    - {val}: {count}")
    
    print("\n📈 STANCE LEVELS")
    print("-" * 40)
    for stance_dim, dist in results["stance_levels"].items():
        print(f"\n  {stance_dim}:")
        for val, count in dist.most_common():
            print(f"    - {val}: {count}")
    
    print("\n📉 STANCE DELTAS")
    print("-" * 40)
    for stance_dim, dist in results["stance_deltas"].items():
        print(f"\n  {stance_dim}:")
        for val, count in dist.most_common():
            print(f"    - {val}: {count}")
    
    print("\n✍️ RESPONSE LENGTHS")
    print("-" * 40)
    if results["response_lengths"]:
        print(f"  avg_words: {statistics.mean(results['response_lengths']):.1f}")
        print(f"  median_words: {statistics.median(results['response_lengths']):.1f}")
        print(f"  min_words: {min(results['response_lengths'])}")
        print(f"  max_words: {max(results['response_lengths'])}")
    
    print("\n⚠️ ISSUES FOUND")
    print("-" * 40)
    if results["issues"]:
        for issue in results["issues"][:20]:  # Limit to first 20
            print(f"  • {issue}")
        if len(results["issues"]) > 20:
            print(f"  ... and {len(results['issues']) - 20} more issues")
    else:
        print("  No issues found ✓")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    results = analyze_episodes()
    print_report(results)
    
    # Save to file
    report_path = DATA_DIR / "quality_report.json"
    with open(report_path, 'w') as f:
        # Convert Counters to dicts for JSON serialization
        json_results = {}
        for key, val in results.items():
            if isinstance(val, Counter):
                json_results[key] = dict(val)
            elif isinstance(val, defaultdict) and isinstance(val.default_factory(), Counter):
                json_results[key] = {k: dict(v) for k, v in val.items()}
            else:
                json_results[key] = val
        json.dump(json_results, f, indent=2)
    print(f"\nReport saved to: {report_path}")
