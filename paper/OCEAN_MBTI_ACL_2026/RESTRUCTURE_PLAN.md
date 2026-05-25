# EMNLP Restructure Plan

Based on reviewer feedback: make paper primarily about **structured latent social state as an inspectable bottleneck for NPC dialogue**, not three papers at once.

---

## Step 1: Abstract & Audit Framing
- Soften audit caveat: replace "fails to validate ... all five ... are flagged" with reviewer-suggested phrasing.
- Keep all result claims but frame audit as "annotation guidelines insufficient" not "labels are bad."

## Step 2: Research Questions & Contributions
- **RQ1**: Which dimensions of NPC social state are recoverable from dialogue context?
- **RQ2**: Can predicted social state support routing and constrained disclosure decisions?
- **RQ3**: Does explicit social-state conditioning improve response generation, and are gains attributable to semantic content rather than prefix capacity?
- Rewrite 4 contributions to match new framing (recoverability, routing, constrained generation, audit as methodological contribution).

## Step 3: Add Running Example
- Insert a concrete dialogue turn in Introduction or Method showing:
  - Player utterance + NPC response
  - Predicted Z_t fields (secrecy_pressure, response_policy, reveal_decision, etc.)
  - Routing decision + allowed/blocked responses
- Makes schema feel concrete, not abstract.

## Step 4: Sharper Operational Definition
- Prominently state: "Z_t is an operational control schema for NPC dialogue, not a claim about human psychology."
- Elevate this from one sentence to a standalone paragraph early in Method.

## Step 5: Move Track A to Appendix
- Move "From-scratch SLM comparison" (15–22M GPT/MoE/Mamba) from main Results to Appendix.
- Keep a 1-paragraph summary in Discussion or Method if needed for completeness.
- This is the biggest structural change.

## Step 6: Restructure Results Section
New order:
1. **Latent State Prediction (RQ1)** — per-head/per-group recoverability (was Track D)
2. **Routing & Constrained Generation (RQ2)** — F1=0.669, zero detected gated leakage
3. **Response Conditioning (RQ3)** — soft-prefix PPL results, placebo ablation
4. Move Track A tables/figures to Appendix

## Step 7: Rewrite Discussion
- Remove "Architecture choice at small scale" as a main discussion point (moved to appendix).
- Add "Audit as methodological contribution" section: insufficient annotation guidelines, not bad labels.
- Emphasize recoverability gradient as core finding.

## Step 8: Update Title (optional)
Reviewer suggestions:
- "Recovering Structured Social State from NPC Dialogue"
- "Inspectable Social State for NPC Dialogue: Recoverability, Routing, and Constrained Generation"
- "What Social State Can Be Recovered from NPC Dialogue?"
- "Structured Social State as an Inspectable Bottleneck for NPC Dialogue"

## Step 9: Compile & Verify
- Ensure all cross-references update correctly after section moves.
- Check for broken Table/Figure references.
- Run latexmk until clean.

---

## Implementation Order
Each step is self-contained; we can stop/review after any step.
