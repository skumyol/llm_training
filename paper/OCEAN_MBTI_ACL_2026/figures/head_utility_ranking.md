
## Table: Head Utility Ranking (by Operational Importance)

Ranked by composite score: Δ routing F1 + 0.5 × (unsafe FP reduction).

| Rank | Head | Routing F1 | Δ F1 | Unsafe FP | Δ Unsafe | Importance |
|------|------|-----------:|-----:|----------:|---------:|------------|
| 1 | secrecy_pressure     | 0.788 | +0.116 | 0.113 | +0.061 | Critical |
| 2 | reveal_decision      | 0.710 | +0.038 | 0.155 | +0.019 | High     |
| 3 | response_policy      | 0.691 | +0.019 | 0.164 | +0.010 | Medium   |
| 4 | value_conflict       | 0.687 | +0.015 | 0.173 | +0.001 | Medium   |

### Summary

- **Baseline**: routing F1 = 0.672, unsafe FP = 0.174
- **Critical heads** (ΔF1 > 0.05): 1
- **High heads** (ΔF1 0.02–0.05): 1
- **Medium heads** (ΔF1 0–0.02): 2
- **Low heads** (ΔF1 ≤ 0): 0

### Recommendation

Retain for routing: **secrecy_pressure, reveal_decision** (2 heads)
Candidates for compression: remaining 27 heads
