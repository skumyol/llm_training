# Predictor agreement per annotator

```
annotator                  kind           n overlap   pred_k  pred_acc  teach_k
----------------------------------------------------------------------------------
654cfad67f990b0393b85132   human        150     150    0.113     0.309    0.096
9_ceef_c_a_d_a0f_c_c       human        150     150    0.113     0.356    0.146
9_caa_b_c_0_c              human        150     150    0.058     0.297    0.079
67c87fc1b3ba111d0e1526a0   human        160     160   -0.008     0.291   -0.018
698e5201784ea100a335f823   human        150     150   -0.011     0.233   -0.052
69839d229a24c16338e6e327   human        150     150   -0.129     0.117   -0.269
69a840923344cb757f028603   human        150     150   -0.137     0.110   -0.280
ai_validator               ai           150     150    0.120     0.402    0.154
d9_ca_efe_fbd_99           partial        2       2    0.048     0.438   -0.278
0_c_da0_b_a_edba9a         partial        4       4    0.008     0.219   -0.060
annotator                  partial        2       2    0.000     0.312   -0.095
c_fc_b_ba_d0e_a0           partial        0       0      n/a       n/a      n/a
synthetic_03               synthetic    150     150    0.115     0.371    0.234
synthetic_01               synthetic    150     150    0.099     0.360    0.198
synthetic_02               synthetic    150     150    0.071     0.340    0.146
synthetic_07               synthetic    150     150   -0.128     0.113   -0.271
synthetic_06               synthetic    150     150   -0.131     0.116   -0.286
synthetic_05               synthetic    150     150   -0.133     0.106   -0.270
synthetic_04               synthetic    150     150   -0.146     0.103   -0.278
----------------------------------------------------------------------------------
HUMAN mean (n=7)                                      -0.000     0.245
HUMAN range                                           -0.137 to 0.113

Per-field predictor kappa, human annotators only:
field               654cfad67f 9_ceef_c_a 9_caa_b_c_ 67c87fc1b3 698e520178 69839d229a 69a8409233
------------------------------------------------------------------------------------------------
valence                  0.230      0.222      0.107      0.040      0.062     -0.221     -0.220
arousal                  0.071     -0.006      0.055      0.004     -0.076     -0.355     -0.294
secrecy_pressure        -0.013      0.168      0.015     -0.034     -0.067     -0.091     -0.044
reveal_decision          0.075      0.152      0.077     -0.013     -0.029     -0.160     -0.167
response_policy          0.313      0.280      0.174      0.012      0.124     -0.033     -0.074
repair_strategy          0.176      0.013     -0.002     -0.040     -0.065     -0.083     -0.173
trust_level              0.038      0.073      0.019     -0.047     -0.005     -0.055     -0.052
familiarity_level        0.014      0.000      0.015      0.018     -0.033     -0.032     -0.068
```

## Reading the table

`pred_k` / `pred_acc` are the predictor's Cohen kappa and raw accuracy against that annotator,
averaged over the eight annotated fields. `teach_k` is the annotator's own agreement with the
teacher labels, shown for reference. Turn overlap with the predictor is 150 for every full
annotator, so the columns are directly comparable.

### The human annotators split into two groups

| group | annotators | pred kappa | pred acc | teacher kappa |
|---|---|---:|---:|---:|
| plausible | `654cfad…`, `9_ceef…`, `9_caa…`, `67c87fc…`, `698e520…` | 0.058 … 0.113 | 0.23 – 0.36 | −0.02 … 0.15 |
| anomalous | `69839d2…`, `69a8409…` | −0.129, −0.137 | 0.11 | −0.27, −0.28 |

Excluding the anomalous pair, mean predictor kappa over the five plausible annotators is **0.053**;
including them it is **−0.000**. The published pair (`654cfad…` = Human A, `67c87fc…` = Human B)
are 0.113 and −0.008, i.e. the two ends of the plausible range — so the headline depends noticeably
on which annotator is used, which is the reason for reporting per annotator rather than pooling.

### The anomalous pair needs provenance checking before use

`69839d2…` and `69a8409…` carry the statistical signature of `synthetic_04`–`07` and of no other
annotator:

- predictor kappa −0.129 / −0.137 vs synthetic_04–07's −0.128 … −0.146
- predictor accuracy 0.117 / 0.110 vs 0.103 … 0.116
- teacher kappa −0.269 / −0.280 vs −0.270 … −0.286

A teacher kappa near −0.27 is not inattentive responding, which produces kappa near 0; it means the
labels are systematically *anti*-correlated with the teacher, which is what a deliberately
disagreeing generator produces. Note that the synthetic files themselves come in two flavours:
`synthetic_01`–`03` behave like plausible annotators (kappa 0.071 – 0.115) while `synthetic_04`–`07`
are the anti-correlated kind.

Pairwise exact label-match rate over all eight fields:

|  | anomalous + synthetic_04–07 | plausible humans |
|---|---:|---:|
| within the anomalous/synthetic cluster | 0.44 – 0.55 | — |
| cluster vs plausible humans | — | 0.29 – 0.38 |
| `654cfad…` vs `67c87fc…` | — | 0.234 |

No file is a literal copy of another, so this is **not** evidence that the two are synthetic. It is
evidence that they are drawn from a different distribution than the other five, and that their
provenance should be confirmed before they enter any human-agreement statistic. If they are in the
pool, they will drag human–human agreement toward zero — which is the audit's headline finding.

### Caveat on the classifier

`synthetic_04`–`07` ship without a `*_meta.json`, so classifying on the metadata flag alone marks
them as human. This script classifies on the filename first. Any other analysis in this repository
that filters synthetics by metadata will silently include four synthetic annotators.
