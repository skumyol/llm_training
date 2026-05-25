# Human and AI Audit Interpretation

Generated from:

- Human A: `audit_654cfad67f990b0393b85132.jsonl`
- Human B: `audit_67c87fc1b3ba111d0e1526a0.jsonl`
- AI validator: `audit_ai_validator.jsonl`
- Teacher packet: `audit_input_clean.jsonl`

## Sample

- Common human turns: 150
- Common AI turns: 150
- Human B completed 160 rows, but only the 150 rows shared with Human A and the AI validator are used for final agreement.

## Final Agreement Table

| Head | Human-Teacher Acc | Human-Teacher Kappa | Human-Human Acc | Human-Human Kappa | AI-Teacher Acc | AI-Teacher Kappa |
|---|---:|---:|---:|---:|---:|---:|
| valence | 0.42 | 0.12 | 0.33 | -0.06 | 0.53 | 0.32 |
| arousal | 0.39 | -0.01 | 0.41 | 0.07 | 0.37 | 0.04 |
| secrecy_pressure | 0.23 | -0.01 | 0.11 | -0.01 | 0.54 | 0.15 |
| response_policy | 0.26 | 0.14 | 0.11 | 0.00 | 0.51 | 0.40 |
| reveal_decision | 0.37 | -0.01 | 0.32 | 0.03 | 0.52 | 0.15 |
| trust_level | 0.21 | 0.04 | 0.32 | 0.05 | 0.21 | 0.02 |
| familiarity_level | 0.19 | -0.02 | 0.16 | 0.01 | 0.37 | 0.02 |
| repair_strategy | 0.27 | 0.04 | 0.11 | -0.01 | 0.41 | 0.14 |

## Interpretation

The human audit does **not** support the current paper placeholder claim that human judges show moderate agreement with the synthetic labels. Human-human agreement is near chance across most heads, with average kappa close to zero. This means the audit should be framed as evidence that the current eight-head annotation task is difficult and that the teacher labels are noisy or underspecified for non-expert annotators.

The strongest teacher-aligned signal comes from the AI validator, especially for:

- `response_policy`: AI-teacher kappa 0.40
- `valence`: AI-teacher kappa 0.32
- `secrecy_pressure`: AI-teacher accuracy 0.54, but low kappa 0.15
- `reveal_decision`: AI-teacher accuracy 0.52, but low kappa 0.15

Human A and Human B used very different label distributions. Examples:

- `secrecy_pressure`: Human A labeled 140/150 turns as `low`; Human B labeled mostly `medium` or `high`.
- `familiarity_level`: Human A labeled 135/150 turns as `N`; Human B labeled most turns as `L`.
- `repair_strategy`: Human A mostly chose `none`; Human B often chose `soften` or `apologize`.

This points to a guideline/calibration problem, not just a model-label problem.

## Recommended Paper Claim

Do not write that the human audit confirms moderate agreement.

Use a more conservative claim:

> A stratified audit of 150 natural test-set turns shows that the audited social-state heads are difficult for uncalibrated annotators: human-human agreement is low across most heads. A model-based third validator aligns more strongly with the teacher on response policy and valence, suggesting that parts of the schema are internally consistent but that the human annotation protocol requires clearer calibration before the labels can be treated as human-grounded supervision.

## Recommended Limitations Text

> Human validation revealed low inter-annotator agreement for the audited heads, especially relational stance and repair strategy. We therefore treat the 29-head labels as teacher-generated supervision signals rather than ground-truth human social judgments. The AI validator's stronger agreement with teacher labels on response policy and valence suggests internal consistency in parts of the schema, but future work should add annotator training, calibration examples, and adjudication before using these labels as human-validated gold annotations.

## Files Generated

- `audit_agreement_final.json`: machine-readable agreement metrics
- `audit_table_rows.tex`: LaTeX rows with Human-Teacher, Human-Human, and AI-Teacher metrics
- `audit_interpretation.md`: this paper-facing interpretation
