# Head Utility Report
Mutual information between each head and the routing decision (slow-path vs fast-path).
Higher NMI = more operationally necessary.

> **Caveat:** MI/NMI is used for exploratory ranking only, not as the sole pruning criterion.
> A head may have low individual MI but still be useful in combination with another head.
> The real criterion is **ablation impact on routing and leakage**.

| Rank | Head | Category | MI | NMI | Entropy | Support | Avg Redundancy | Verdict |
|------|------|----------|----|-----|---------|---------|----------------|---------|
| 1 | secrecy_pressure | routing | 0.334 | 0.335 | 1.152 | 4 | -0.054 | Keep (hard gate) |
| 2 | response_policy | routing | 0.208 | 0.209 | 2.512 | 8 | -0.034 | Keep (hard gate) |
| 3 | value_conflict | routing | 0.147 | 0.156 | 0.943 | 3 | 0.002 | Keep (hard gate) |
| 4 | dominance_delta | relational | 0.108 | 0.109 | 1.816 | 5 | N/A | Advisory |
| 5 | dialogue_act | descriptive | 0.108 | 0.109 | 3.155 | 14 | N/A | Advisory |
| 6 | player_intent | meta | 0.085 | 0.085 | 2.030 | 8 | N/A | Advisory |
| 7 | obligation_level | relational | 0.054 | 0.054 | 1.948 | 7 | N/A | Advisory |
| 8 | reveal_decision | routing | 0.054 | 0.054 | 1.284 | 4 | -0.095 | Keep (hard gate) |
| 9 | affection_level | relational | 0.042 | 0.042 | 1.887 | 6 | N/A | Drop |
| 10 | affection_delta | relational | 0.041 | 0.041 | 1.985 | 5 | N/A | Drop |
| 11 | control | affect | 0.039 | 0.039 | 1.504 | 3 | N/A | Drop |
| 12 | obligation_delta | relational | 0.038 | 0.038 | 1.563 | 5 | N/A | Drop |
| 13 | familiarity_level | relational | 0.036 | 0.036 | 2.113 | 7 | N/A | Drop |
| 14 | dominance_level | relational | 0.033 | 0.033 | 1.669 | 5 | N/A | Drop |
| 15 | player_credibility | meta | 0.032 | 0.032 | 0.997 | 3 | N/A | Drop |
| 16 | tone | descriptive | 0.030 | 0.030 | 2.150 | 6 | N/A | Drop |
| 17 | trust_level | relational | 0.024 | 0.024 | 1.992 | 7 | N/A | Drop |
| 18 | repair_strategy | descriptive | 0.023 | 0.023 | 1.962 | 5 | N/A | Drop |
| 19 | familiarity_delta | relational | 0.023 | 0.023 | 2.160 | 6 | N/A | Drop |
| 20 | respect_level | relational | 0.021 | 0.021 | 2.218 | 7 | N/A | Drop |
| 21 | trust_delta | relational | 0.020 | 0.020 | 2.031 | 6 | N/A | Drop |
| 22 | threat | affect | 0.020 | 0.020 | 1.402 | 3 | N/A | Drop |
| 23 | face_pressure | meta | 0.011 | 0.016 | 0.704 | 3 | N/A | Drop |
| 24 | arousal | affect | 0.016 | 0.016 | 1.177 | 3 | N/A | Drop |
| 25 | risk_type | descriptive | 0.010 | 0.015 | 0.692 | 4 | N/A | Drop |
| 26 | valence | affect | 0.014 | 0.014 | 1.395 | 3 | N/A | Drop |
| 27 | respect_delta | relational | 0.014 | 0.014 | 1.797 | 5 | N/A | Drop |
| 28 | duty_pressure | meta | 0.003 | 0.003 | 0.992 | 2 | N/A | Drop |
| 29 | player_knowledge | meta | 0.002 | 0.002 | 1.817 | 4 | N/A | Drop |

## Redundancy Analysis (Routing Heads Only)
Redundancy = MI(head; routing) − avg conditional MI given other routing heads.
Low redundancy = unique contribution. High redundancy = overlapping signal.

### response_policy
  vs reveal_decision: conditional MI=0.261, redundancy=-0.052
  vs secrecy_pressure: conditional MI=0.258, redundancy=-0.050
  vs value_conflict: conditional MI=0.208, redundancy=0.000

### reveal_decision
  vs response_policy: conditional MI=0.096, redundancy=-0.042
  vs secrecy_pressure: conditional MI=0.230, redundancy=-0.176
  vs value_conflict: conditional MI=0.121, redundancy=-0.067

### secrecy_pressure
  vs response_policy: conditional MI=0.380, redundancy=-0.046
  vs reveal_decision: conditional MI=0.510, redundancy=-0.177
  vs value_conflict: conditional MI=0.272, redundancy=0.062

### value_conflict
  vs response_policy: conditional MI=0.137, redundancy=0.010
  vs reveal_decision: conditional MI=0.214, redundancy=-0.067
  vs secrecy_pressure: conditional MI=0.085, redundancy=0.062

