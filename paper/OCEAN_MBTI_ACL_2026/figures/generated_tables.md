
## Table 1: Head Utility

| Head Group | Head | Cohen's κ | Macro-F1 | Routing Impact | Operational? |
|------------|------|----------:|---------:|---------------:|---------------|
| Relational | affection_delta      | 0.391 | 0.429 | None   (+0.000) | Drop     |
| Relational | affection_level      | 0.552 | 0.645 | None   (+0.000) | Drop     |
| Affect     | arousal              | 0.692 | 0.725 | None   (+0.000) | Drop     |
| Affect     | control              | 0.561 | 0.689 | None   (+0.000) | Drop     |
| Relational | dominance_delta      | 0.368 | 0.385 | None   (+0.000) | Drop     |
| Relational | dominance_level      | 0.380 | 0.412 | None   (+0.000) | Drop     |
| Normative  | duty_pressure        | 0.629 | 0.814 | None   (+0.000) | Drop     |
| Normative  | face_pressure        | 0.307 | 0.527 | None   (+0.000) | Drop     |
| Relational | familiarity_delta    | 0.296 | 0.428 | None   (+0.000) | Drop     |
| Relational | familiarity_level    | 0.389 | 0.522 | None   (+0.000) | Drop     |
| Relational | obligation_delta     | 0.443 | 0.349 | None   (+0.000) | Drop     |
| Relational | obligation_level     | 0.508 | 0.449 | None   (+0.000) | Drop     |
| Mental     | player_credibility   | 0.376 | 0.579 | None   (+0.000) | Drop     |
| Mental     | player_intent        | 0.596 | 0.373 | None   (+0.000) | Drop     |
| Mental     | player_knowledge     | 0.487 | 0.611 | None   (+0.000) | Drop     |
| Decision   | repair_strategy      | 0.470 | 0.481 | None   (+0.000) | Drop     |
| Relational | respect_delta        | 0.584 | 0.514 | None   (+0.000) | Drop     |
| Relational | respect_level        | 0.594 | 0.558 | None   (+0.000) | Drop     |
| Decision   | response_policy      | 0.632 | 0.622 | High   (+0.019) | Yes      |
| Decision   | reveal_decision      | 0.420 | 0.656 | High   (+0.038) | Yes      |
| Context    | risk_type            | 0.448 | 0.300 | None   (+0.000) | Drop     |
| Normative  | secrecy_pressure     | 0.175 | 0.491 | Medium (+0.116) | Yes      |
| Affect     | threat               | 0.641 | 0.690 | None   (+0.000) | Drop     |
| Context    | tone                 | 0.490 | 0.510 | None   (+0.000) | Drop     |
| Relational | trust_delta          | 0.484 | 0.515 | None   (+0.000) | Drop     |
| Relational | trust_level          | 0.617 | 0.591 | None   (+0.000) | Drop     |
| Affect     | valence              | 0.688 | 0.752 | None   (+0.000) | Drop     |
| Normative  | value_conflict       | 0.344 | 0.557 | Medium (+0.015) | Yes      |

## Table 2: Routing Performance (Retraining Ablations)

| System | Routing F1 | Unsafe fast-path | Slow-path rate | Slow-path recall | Cost (5:1) |
|--------|-----------:|-----------------:|---------------:|-----------------:|-----------:|
| M0: Full 29-head             | 0.574 | 0.187 | 0.682 | 0.651 | 1.269 |
| M1: Routing only (4-head)    | 0.697 | 0.003 | 0.997 | 0.995 | 0.477 |
| M2: +Affect (7-head)         | 0.695 | 0.012 | 0.975 | 0.978 | 0.508 |
| M3: +Relational (6-head)     | 0.631 | 0.105 | 0.832 | 0.804 | 0.927 |

## Table 3: Disclosure Metrics

| System | Over-disclosure | Under-disclosure | Exact match | Policy consistency |
|--------|---------------:|-----------------:|-----------:|-------------------:|
| 29-head baseline | 0.003 | 0.659 | 0.338 | N/A |
| Decision card    | 0.002 | 0.694 | 0.304 | N/A |
