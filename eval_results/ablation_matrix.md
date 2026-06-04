# Head Ablation Matrix

| Experiment | Heads | Routing F1 | Precision | Recall | FP Rate | Slow-Path Rate |
|------------|-------|------------|-----------|--------|---------|----------------|
| exp_a_routing_only | 4 | 0.698 | 0.537 | 0.997 | 0.997 | 0.997 |
| exp_b_plus_affect | 7 | 0.686 | 0.533 | 0.962 | 0.978 | 0.969 |
| exp_c_plus_relational | 6 | 0.666 | 0.545 | 0.856 | 0.829 | 0.843 |
| exp_d_full_29head | 28 | 0.604 | 0.533 | 0.698 | 0.709 | 0.703 |

## Notes
- Routing F1 computed on predicted Z_t against gold routing labels.
- FP Rate = false positives / (false positives + true negatives).
- Slow-Path Rate = fraction of turns routed to slow path.
- Compare experiments side-by-side; drops in F1 > 0.03 suggest the removed heads are operationally necessary.
- All bootstrap CIs must use **episode-level resampling** (not turn-level), because turns within an episode are not independent.
