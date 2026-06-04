# Oracle Upper-Bound Comparison Table

| System | State source | Prompt | Routing F1 | Unsafe fast-path | Slow-path rate | Policy consistency | Over-disclosure |
|--------|-------------|--------|-----------:|-----------------:|---------------:|-------------------:|----------------:|
| Oracle (gold 29D) | gold | full | 1.000 | 0.000 | 0.548 | — | — |
| Current (predicted 29D) | predicted | full | 0.735 | N/A | 0.525 | 0.735 | 0.003 |
| Gold-head oracle | predicted + gold heads | full | 0.893 | 0.061 | 0.529 | — | — |
| Decision card | predicted | card | — | — | — | 0.850 | 0.002 |

## Notes
- **Oracle (gold 29D)**: Deterministic router with perfect state; F1=1.0 by construction.
- **Current**: Real deployment setting with predicted $\hat{Z}_t$ from Qwen3-4B.
- **Gold-head oracle**: Predicted state except routing heads are replaced with gold labels. Shows upper bound if routing heads were perfect.
- **Decision card**: Uses 4-field compressed prompt instead of full 29-head dump.

## Interpretation
- Routing F1 gap between predicted and gold-head oracle: 0.158. This is the **predictor error** (imperfect head prediction).
- Remaining gap to F1=1.0 after gold heads: 0.107. This is the **router aggregation error** (coarse fast/slow decision misses fine distinctions).
- Total gap to oracle: 0.265 = predictor error + router error.
