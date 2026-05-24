#!/usr/bin/env python3
"""Check for repeating data in the himaan annotation (audit_input.jsonl)."""

import json
from collections import defaultdict, Counter

filepath = "/root/llm_training/paper/audit_input.jsonl"
rows = []
with open(filepath) as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

print(f"Total rows: {len(rows)}")
print()

# 1. Check for exact duplicate rows (all fields identical)
seen_hashes = defaultdict(list)
for i, row in enumerate(rows):
    h = hash(json.dumps(row, sort_keys=True))
    seen_hashes[h].append(i)

exact_dupes = {h: idxs for h, idxs in seen_hashes.items() if len(idxs) > 1}
if exact_dupes:
    print(f"=== EXACT DUPLICATE ROWS ({len(exact_dupes)} groups) ===")
    for h, idxs in exact_dupes.items():
        print(f"  Lines {[i+1 for i in idxs]}: {rows[idxs[0]]['episode_id']} turn {rows[idxs[0]]['turn_number']} cf={rows[idxs[0]]['counterfactual']}")
    print()
else:
    print("No exact duplicate rows found.")
    print()

# 2. Check for same (episode_id, turn_number) appearing multiple times
turn_groups = defaultdict(list)
for i, row in enumerate(rows):
    key = (row["episode_id"], row["turn_number"])
    turn_groups[key].append(i)

repeated_turns = {k: v for k, v in turn_groups.items() if len(v) > 1}
print(f"=== TURNS WITH MULTIPLE ENTRIES ({len(repeated_turns)} groups) ===")
for (ep, tn), idxs in sorted(repeated_turns.items()):
    entries = []
    for i in idxs:
        r = rows[i]
        entries.append(f"line {i+1} (cf={r['counterfactual']})")
    print(f"  Episode {ep} Turn {tn}: {', '.join(entries)}")
print()

# 3. Check for npc_response == player_utterance (echo)
echo_cases = []
for i, row in enumerate(rows):
    pu = row.get("player_utterance", "")
    nr = row.get("npc_response", "")
    if pu and nr and pu.strip() == nr.strip():
        echo_cases.append((i, row))

if echo_cases:
    print(f"=== NPC ECHOES PLAYER ({len(echo_cases)} cases) ===")
    for i, row in echo_cases:
        print(f"  Line {i+1}: Episode {row['episode_id']} Turn {row['turn_number']} cf={row['counterfactual']}")
        print(f"    Text: {row['player_utterance'][:120]}")
    print()
else:
    print("No cases where npc_response == player_utterance.")
    print()

# 4. Check for same player_utterance across different turns in same episode
print("=== SAME PLAYER UTTERANCE ACROSS DIFFERENT TURNS (same episode) ===")
ep_utterances = defaultdict(lambda: defaultdict(list))
for i, row in enumerate(rows):
    pu = row.get("player_utterance", "")
    ep_utterances[row["episode_id"]][pu].append((row["turn_number"], i))

found = False
for ep_id, utt_map in ep_utterances.items():
    for utt, occurrences in utt_map.items():
        turns = [t for t, _ in occurrences]
        if len(set(turns)) > 1:
            found = True
            locs = [f"Turn {t} (line {i+1})" for t, i in occurrences]
            print(f"  Episode {ep_id}: \"{utt[:80]}\" -> {', '.join(locs)}")

if not found:
    print("  None found.")
print()

# 5. Check for same npc_response across different turns in same episode
print("=== SAME NPC RESPONSE ACROSS DIFFERENT TURNS (same episode) ===")
ep_npc = defaultdict(lambda: defaultdict(list))
for i, row in enumerate(rows):
    nr = row.get("npc_response", "")
    ep_npc[row["episode_id"]][nr].append((row["turn_number"], i))

found = False
for ep_id, npc_map in ep_npc.items():
    for nr, occurrences in npc_map.items():
        turns = [t for t, _ in occurrences]
        if len(set(turns)) > 1:
            found = True
            locs = [f"Turn {t} (line {i+1})" for t, i in occurrences]
            print(f"  Episode {ep_id}: \"{nr[:80]}\" -> {', '.join(locs)}")

if not found:
    print("  None found.")
print()

# 6. Check dialogue_history consistency - does NPC response in dialogue_history
#    match the npc_response from the counterfactual=false row for that turn?
print("=== DIALOGUE HISTORY CONSISTENCY CHECK ===")
# Build lookup: (episode_id, turn_number, cf=False) -> npc_response
canonical_npc = {}
for row in rows:
    if not row["counterfactual"]:
        key = (row["episode_id"], row["turn_number"])
        canonical_npc[key] = row["npc_response"]

mismatches = 0
for row in rows:
    dh = row.get("dialogue_history", "")
    if not dh:
        continue
    # Parse "[Turn N] NPC: ..." patterns
    import re
    for m in re.finditer(r'\[Turn (\d+)\] NPC: (.+?)(?=\n\[Turn|\Z)', dh, re.DOTALL):
        turn_num = int(m.group(1))
        npc_text = m.group(2).strip()
        canon_key = (row["episode_id"], turn_num)
        if canon_key in canonical_npc:
            expected = canonical_npc[canon_key].strip()
            if npc_text != expected:
                mismatches += 1
                if mismatches <= 10:
                    print(f"  Line {rows.index(row)+1}: Episode {row['episode_id']} Turn {row['turn_number']}")
                    print(f"    DH Turn {turn_num} NPC: {npc_text[:100]}")
                    print(f"    Canonical NPC:     {expected[:100]}")

print(f"  Total mismatches: {mismatches}")
print()

# 7. Summary stats
print("=== SUMMARY ===")
print(f"  Total rows: {len(rows)}")
print(f"  Unique episodes: {len(set(r['episode_id'] for r in rows))}")
print(f"  Counterfactual rows: {sum(1 for r in rows if r['counterfactual'])}")
print(f"  Non-counterfactual rows: {sum(1 for r in rows if not r['counterfactual'])}")
print(f"  Unique (episode, turn) pairs: {len(turn_groups)}")
print(f"  Turns with multiple entries: {len(repeated_turns)}")

# Count by episode
ep_counts = Counter(r["episode_id"] for r in rows)
print(f"\n  Rows per episode (showing episodes with >2 rows):")
for ep, count in ep_counts.most_common():
    if count > 2:
        print(f"    {ep}: {count} rows")

# Counterfactual analysis: for turns with cf=True, check if labels differ from cf=False
print("\n=== COUNTERFACTUAL LABEL DIFFERENCES ===")
for (ep, tn), idxs in sorted(repeated_turns.items()):
    cf_false = None
    cf_true = None
    for i in idxs:
        if rows[i]["counterfactual"]:
            cf_true = rows[i]
        else:
            cf_false = rows[i]
    if cf_false and cf_true:
        diff_keys = []
        for k, v in cf_false["labels"].items():
            if cf_true["labels"].get(k) != v:
                diff_keys.append(k)
        if diff_keys:
            print(f"  Episode {ep} Turn {tn}: {len(diff_keys)} label diffs: {', '.join(diff_keys[:10])}")