# Human Label Audit Protocol

This protocol implements the small human validation study required to close the
paper's biggest methodological gap (synthetic/teacher-generated labels).
You must run this audit and replace the placeholder numbers in
Appendix~\ref{app:human_audit} of `main.tex` with real results before submission.

## Goal
Validate the synthetic 29-dim social-state labels by measuring
(1) Human--Teacher agreement, (2) Human--Human inter-annotator agreement,
and (3) AI-validator agreement on a stratified sample of 150 test-set turns.

## Population
- Source: `audit_input_clean.jsonl` (356 natural, non-counterfactual turns, 78 episodes, 7 scenario types)
- Sample: 150 turns, stratified by scenario type (~21 per type)
- Sampling method: take the first 21 episodes from each scenario type in the test
  split, then sample 1 random turn per episode. If an episode has fewer than
  1 turn, oversample from the next episode of the same type.

## Annotators
- Minimum 2 annotators (you + 1 colleague).
- Each annotator works independently; no model predictions shown during judgement.
- Briefing: read the schema description (Table~\ref{tab:schema}) and the
  class definitions in `configs/schema.yaml` before starting.
- Optional third validator: run `ai_audit.py` on the same cleaned packet to
  produce `audit_ai_validator.jsonl`. Treat this as a model-based validator,
  not as a replacement for human--human agreement.

## Heads to validate (8 heads only)
Do not annotate all 29. Focus on the most actionable / subjective subset:

1. `valence` (3-class: positive, neutral, negative)
2. `arousal` (3-class: low, medium, high)
3. `secrecy_pressure` (3-class: low, medium, high)
4. `reveal_decision` (4-class: none, hint, partial, full)
5. `response_policy` (10-class: answer, withhold, deflect, etc.)
6. `repair_strategy` (5-class: apologize, redirect, etc.)
7. `trust_level` (5-class ordinal: VL, L, N, H, VH)
8. `familiarity_level` (5-class ordinal: VL, L, N, H, VH)

## Annotation interface
A minimal JSONL viewer or spreadsheet with these columns per row:
- `turn_id`
- `scenario_type`
- `scene` (text)
- `dialogue_history` (text)
- `player_utterance` (text)
- `npc_response` (text)
- Dropdowns for each of the 8 heads above
- Optional: `notes` free-text field for ambiguous cases

## Procedure
1. Build the audit packet with `uv run build_human_audit_packet.py --input audit_input.jsonl --output audit_input_clean.jsonl`.
2. Launch the app with `uv run launcher.py --data audit_input_clean.jsonl --output ./audit_results --port 8765`.
3. Annotator A labels all 150 turns independently.
4. Annotator B labels all 150 turns independently.
5. Run the AI validator with `uv run ai_audit.py --data audit_input_clean.jsonl --output ./audit_results --resume`.
6. Resolve conflicts by discussion; note whether the conflict was due to
   ambiguity in the schema, ambiguity in the text, or clear teacher-label error.
7. Compute metrics with:
   `uv run generate_audit_results.py --human-a ./audit_results/audit_ANNOTATOR_A.jsonl --human-b ./audit_results/audit_ANNOTATOR_B.jsonl --ai ./audit_results/audit_ai_validator.jsonl --teacher ./audit_input_clean.jsonl --output ./audit_results/audit_agreement_final.json --latex-output ./audit_results/audit_table_rows.tex`.

## Metrics to compute
For each head:
- **Human--Teacher (HT) Accuracy**: % of turns where annotator label == teacher label.
- **HT Cohen's kappa**: use `sklearn.metrics.cohen_kappa_score` with the
  teacher labels as "ground truth" and annotator labels as predictions.
  For 5-class ordinal heads, use `weights='quadratic'` if desired, but the
  paper currently reports unweighted kappa; be consistent.
- **Human--Human (HH) Accuracy**: % of turns where A == B.
- **HH Cohen's kappa**: `cohen_kappa_score(A, B)`.

Report mean and std across annotators for HT metrics.

## Interpretation thresholds
- kappa > 0.60: strong agreement (synthetic labels are credible for this head)
- kappa 0.40--0.60: moderate agreement (usable but noisy)
- kappa < 0.40: weak agreement (synthetic labels are unreliable; treat as limitation)

## Updating the paper
Replace the placeholder table in `main.tex` Appendix~\ref{app:human_audit}
(`tab:human_audit`) with your real numbers.
Also update the abstract sentence:
> "A stratified human audit of 150 test-set turns confirms moderate agreement
> with synthetic labels for the most actionable heads."

If your results differ materially from the placeholders, also update the
Limitations paragraph to reflect the actual agreement levels.

## Expected effort
- Sampling + interface setup: 30 min
- Annotation (150 turns x 8 heads): ~2.5 hours per annotator
- Computation + write-up: 30 min
- Total: ~3 hours per person
