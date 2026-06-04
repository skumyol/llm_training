# Head–Leakage Correlation Report

Mutual information between predicted head values and response leakage.
Higher NMI = head is more predictive of whether a leak occurs.

| Rank | Head | MI | NMI | Max Leak Value | Max Leak Rate | Interpretation |
|------|------|----|-----|----------------|---------------|----------------|
| 1 | record_idx | 0.000 | 0.000 | N/A | 0.00% | No significant association |
| 2 | response_policy | 0.000 | 0.000 | N/A | 0.00% | No significant association |
| 3 | reveal_decision | 0.000 | 0.000 | N/A | 0.00% | No significant association |
| 4 | secrecy_pressure | 0.000 | 0.000 | N/A | 0.00% | No significant association |
| 5 | value_conflict | 0.000 | 0.000 | N/A | 0.00% | No significant association |

## Usage for Router Improvement

- Heads with high NMI should be monitored closely in the router.
- If a specific head value (e.g., `reveal_decision=partial`) has high leak rate,
  the router should always slow-path when that value is predicted.
