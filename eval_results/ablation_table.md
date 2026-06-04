# Head Masking Ablation Results

Baseline (no mask): F1=0.672, unsafe_FP=0.174, slow=0.543

## Full Table

| Masked head(s) | Mode | Routing F1 | Unsafe fast-path | Slow-path rate | FPR | FNR | Cost(5:1) |
|----------------|------|-----------:|-----------------:|---------------:|----:|----:|----------:|
| response_policy      | majority | 0.594 | 0.240 | 0.464 | 0.361 | 0.447 | 1.37 |
| response_policy      | random   | 0.613 | 0.212 | 0.523 | 0.427 | 0.395 | 1.26 |
| response_policy      | gold     | 0.691 | 0.164 | 0.543 | 0.367 | 0.305 | 0.99 |
| reveal_decision      | majority | 0.675 | 0.135 | 0.656 | 0.547 | 0.251 | 0.93 |
| reveal_decision      | random   | 0.624 | 0.206 | 0.523 | 0.415 | 0.384 | 1.22 |
| reveal_decision      | gold     | 0.710 | 0.155 | 0.539 | 0.339 | 0.289 | 0.93 |
| secrecy_pressure     | majority | 0.720 | 0.070 | 0.760 | 0.633 | 0.131 | 0.64 |
| secrecy_pressure     | random   | 0.668 | 0.176 | 0.546 | 0.399 | 0.327 | 1.06 |
| secrecy_pressure     | gold     | 0.788 | 0.113 | 0.540 | 0.250 | 0.210 | 0.68 |
| value_conflict       | majority | 0.582 | 0.253 | 0.439 | 0.335 | 0.471 | 1.42 |
| value_conflict       | random   | 0.595 | 0.214 | 0.550 | 0.490 | 0.398 | 1.30 |
| value_conflict       | gold     | 0.687 | 0.173 | 0.524 | 0.345 | 0.322 | 1.02 |
| response_policy, reveal_decision, value_conflict, secrecy_pressure | majority | 0.699 | 0.000 | 1.000 | 1.000 | 0.000 | 0.46 |
| response_policy, reveal_decision, value_conflict, secrecy_pressure | random   | 0.519 | 0.243 | 0.596 | 0.652 | 0.452 | 1.52 |
| response_policy, reveal_decision, value_conflict, secrecy_pressure | gold     | 0.893 | 0.061 | 0.529 | 0.114 | 0.114 | 0.36 |

## Key Findings

- Gold **response_policy**: F1 +0.019, unsafe_FP -0.010
- Gold **reveal_decision**: F1 +0.038, unsafe_FP -0.019
- Gold **secrecy_pressure**: F1 +0.116, unsafe_FP -0.061
- Gold **value_conflict**: F1 +0.015, unsafe_FP -0.001
