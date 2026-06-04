# Decision Card A/B Evaluation

**Paired samples:** 683

## Metric Comparison

| Metric | Baseline | Treatment | Delta (T−B) | 95% CI | Interpretation |
|--------|----------|-----------|-------------|--------|----------------|
| rouge_l_mean | 0.1263 | 0.0797 | -0.0466 | [-0.0691, -0.0423] | higher is better |
| secret_leakage_rate | 0.0000 | 0.0000 | +0.0000 | [+0.0000, +0.0000] | lower is better |
| contradiction_rate | 0.0000 | 0.0000 | +0.0000 | [+0.0000, +0.0000] | lower is better |
| mean_policy_consistency | 0.7349 | 0.8497 | +0.1148 | [+0.0057, +0.1013] | higher is better |
| exact_disclosure_match | 0.3382 | 0.3104 | -0.0278 | N/A | higher is better |
| over_disclosure_rate | 0.0029 | 0.0015 | -0.0014 | [-0.0084, +0.0043] | lower is better |
| under_disclosure_rate | 0.6589 | 0.6881 | +0.0292 | [-0.0085, +0.0697] | lower is better |

## Details

- **Baseline**: Full 29-head state dumped into prompt.
- **Treatment**: Compressed decision card (stance + disclosure + risk + tone).
- **Bootstrap**: Episode-level resampling (episodes are independent, turns within episode are correlated).
