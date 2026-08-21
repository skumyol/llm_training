# Draft edits for `ch5_signals.tex`

Copy-paste-ready LaTeX. Status markers:

- **[FINAL]** — number is measured, multi-seed or otherwise defensible; safe to insert now.
- **[SINGLE-SEED]** — measured but n=1; insert only with the stated hedge, or wait for seeds.
- **[PENDING]** — job running; do not insert yet.

Line numbers refer to the file as of this draft.

---

## Edit 0 — a comparability trap to avoid

An external review stated that the new control (test PPL 42.18) "lines up with the old 42.07 val
number, so this is apples-to-apples." **It does not.** The two differ in three ways at once:

| | thesis 42.07 | new control 42.18 |
|---|---|---|
| architecture | mixture-of-experts | dense GPT |
| parameters | 22.4 M | 44.8 M |
| split | original `val.txt`, 28,240 tokens | new `test.txt`, 15,744 tokens |

The numerical agreement is coincidence. Do not write that the new control reproduces the old
result. A same-architecture MoE bridge run on the new splits is queued (§Edit 1, [PENDING]); until
it lands, the honest framing is that Track A was **re-run with a different architecture on a newly
constructed test split**, and the old number is the prior setting, not a comparable control.

---

## Edit 1 — Track A

### 1a. Line 35, `tab:signals-track-qa`

**Current:**
```latex
A & Can a compact model learn the dialogue task from scratch? & The strongest retained from-scratch model reaches validation perplexity 42.07. The result motivates the move to pretrained backbones in the subsequent response experiments. \\
```

**Replacement** [FINAL for the 18.11/42.18 pair; the MoE bridge is [PENDING]]:
```latex
A & Can a compact model learn the dialogue task from scratch? & Only to a limited degree, and the limit is data rather than capacity. On a held-out test split a 44.8M dense GPT reaches perplexity 42.18 from scratch; pretraining the same model on 107M tokens of external dialogue before fine-tuning reduces this to 18.11 ($-57\%$, three seeds). Reducing capacity makes results worse, and regularisation saturates, so the binding constraint is the 545K-token in-domain corpus. \\
```

### 1b. Line 266, Track A paragraph

**Current:** "The strongest retained configuration, a 22.4M-parameter mixture-of-experts model,
reaches validation perplexity 42.07, with the 16.1M dense GPT performing slightly worse."

**Replacement:**
```latex
The sequence begins with Track~A, which tests how well compact language models trained from scratch can model the Oakhaven dialogue corpus. Earlier runs retained a 22.4M-parameter mixture-of-experts model at validation perplexity 42.07. Those runs reported validation perplexity only, and validation also selected the checkpoint; the corpus has since been given a held-out test split, constructed by grouping dialogue blocks on their opening player utterance so that a block and its own growing-prefix variants cannot straddle the split. Re-evaluated in that setting, a 44.8M dense GPT trained from scratch reaches test perplexity 42.18 (three seeds). Pretraining the same architecture on 107M tokens of external dialogue and then fine-tuning on the in-domain corpus reaches 18.11, a 57\% reduction. The from-scratch result is not capacity-limited: a 16M-parameter variant is worse (43.41), and increasing regularisation saturates near 39.4. Every from-scratch configuration peaks between epochs three and six and then overfits, whereas the pretrained model continues improving through epochs ten to twelve. Track~A therefore establishes that compact from-scratch dialogue modelling is bounded by in-domain data volume, which is the concrete form of the motivation for moving to pretrained backbones in the later tracks.
```

### 1c. Line 300, `tab:slm-comparison`

The table is captioned "Reported **validation** result". Either retitle the column to
"Reported result" and mark the split per row, or add rows. Minimal change:

```latex
A & Dense GPT, from scratch (44.8M)          & Test PPL 42.18 \\
A & Dense GPT, external pretrain + fine-tune & Test PPL 18.11 \\
A & Mixture-of-experts language model (22.4M, prior setting) & Val PPL 42.07 \\
```

Caption addition:
```latex
Track~A rows are test perplexity on the held-out split introduced in this revision; the
mixture-of-experts row is the earlier validation-only result and is not directly comparable.
```

### 1d. Track C — the conditioning bug

Track C's placebo table (`tab:placebo`, lines 278--292) reports that real OCEAN/VAD conditioning
(2.90) does not beat shuffled or random controls (2.88--2.91). A defect found in this revision is
directly relevant: in `run_small_lm.py`, `device` was referenced before assignment inside a
`try` block whose `except` swallowed the resulting `UnboundLocalError` and fell back to **zero
conditioning**. Any run configured with an embedding model therefore compared zero conditioning
against zero conditioning.

This does **not** invalidate `tab:placebo`, which uses the TinyLlama `ConditionalDialogueModel`
path rather than `run_small_lm.py`. It does mean any Track-C claim resting on the small-LM
`--embedding-model` A/B must be re-run. Recommended footnote:

```latex
\footnote{A defect in the small-LM conditioning path, identified during this revision, caused runs configured with a sentence-embedding conditioner to fall back silently to zero conditioning. Track-C conclusions in this chapter rest on the TinyLlama conditioning path, which is unaffected; small-LM conditioning comparisons are deferred pending re-runs.}
```

---

## Edit 2 — Track D headline and `response_policy`

### 2.0 Why the validation numbers in these edits differ slightly from the chapter's

The chapter and these edits quote the *same validation split* with small differences. This is not
rounding, and it should be understood before inserting anything, because a reader comparing the two
will notice.

| quantity | chapter (archived June run) | recomputed | cause |
|---|---:|---:|---|
| mean macro-$F_1$ | 0.5425 | 0.5409 | **metric definition** |
| `response_policy` macro-$F_1$ | 0.6210 | 0.6218 | **metric definition** |
| mean accuracy | 0.6875 | 0.6880 | numerical non-determinism |
| mean $\kappa$ | 0.4833 | 0.4843 | numerical non-determinism |

Two independent causes:

1. **Macro-$F_1$ definition.** The archived run used scikit-learn's default, which averages over
   the union of gold and predicted classes. The recomputation averages over classes with gold
   support. For `response_policy` the default counts a class the model predicts but which never
   occurs in gold, scoring it 0 and *depressing* the average — hence the corrected value is slightly
   higher. Across 29 heads the net effect goes the other way.
2. **Numerical non-determinism.** Accuracy and $\kappa$ do not depend on the label set at all, so
   their movement has a different source: the two runs executed on different GPU models, and
   borderline `argmax` ties can flip under different kernels. The shift is ~0.0005 on a mean over
   29 heads × 683 turns, i.e. roughly ten flipped predictions out of ~19,800.

**Recommendation:** quote the recomputed values throughout for internal consistency, and add:

```latex
\footnote{Validation figures were recomputed for this revision under the macro-$F_1$ definition described above; they differ from earlier archived values by at most $0.002$. Accuracy and $\kappa$, which are independent of that definition, differ by $\le 0.001$, consistent with tie-breaking in \texttt{argmax} under different GPU kernels.}
```

Do **not** present the two as though the model changed. It did not.

### 2a. Line 38, `tab:signals-track-qa`

**Current:** "mean accuracy $0.688$, $\kappa=0.483$ ... predicted state supports selective routing
at $F_1=0.686$."

**Replacement** [FINAL for aggregates; response_policy is [SINGLE-SEED]]:
```latex
D & Can the 29-field social state be predicted well enough to support response routing? & Partly, and unevenly. On a held-out test split of 884 turns from 80 unseen episodes, mean accuracy is $0.674$ and mean macro-$F_1$ is $0.534$, close to the validation values ($0.688$, $0.541$), so the aggregate representation generalises. The decision head does not: \texttt{response\_policy} falls from macro-$F_1=0.622$ on validation to $0.427$ on test. Both project acceptance thresholds fail on test (\texttt{response\_policy} $0.427 < 0.75$; stance-delta accuracy $0.571 < 0.70$), while predicted-state routing remains serviceable at $F_1=0.735$. \\
```

### 2b. Lines 333--338, "Which Fields Can Be Recovered?"

Insert after the existing aggregate sentence:

```latex
All figures in this subsection were previously computed on the validation split, which also selected the checkpoint and triggered early stopping, and were therefore optimistically biased by construction. A held-out test split of 884 turns drawn from 80 episodes disjoint from training and validation is used here. The aggregate picture is stable across the two splits: mean accuracy moves from $0.688$ to $0.674$, mean macro-$F_1$ from $0.541$ to $0.534$, and mean balanced accuracy is marginally \emph{higher} on test ($0.547 \rightarrow 0.552$). The 29-field representation as a whole therefore transfers to unseen episodes.

Field-level behaviour does not follow the aggregate. \texttt{response\_policy}, the field with the most operational authority because the router consumes it directly, falls from macro-$F_1=0.622$ on validation to $0.427$ on test, a 31\% relative reduction, with accuracy falling from $0.716$ to $0.623$. Both splits contain nine of its ten classes in the gold labels, so the reduction is not an artefact of class coverage. The weakest test fields are \texttt{tone} ($0.376$), \texttt{familiarity\_delta} ($0.388$), \texttt{risk\_type} ($0.407$), \texttt{dominance\_level} ($0.410$) and \texttt{dominance\_delta} ($0.415$); the strongest are \texttt{valence} ($0.802$), \texttt{duty\_pressure} ($0.777$) and \texttt{face\_pressure} ($0.731$).

\texttt{risk\_type} is diagnostic of the failure mode: it reaches $0.869$ accuracy but only $0.407$ macro-$F_1$ and $0.398$ balanced accuracy, the signature of a head that predicts the majority class and little else. The pattern is consistent---fields with balanced label distributions are recovered, while rare-class and relationship-change fields collapse toward the majority. This occurs despite the training procedure applying three imbalance corrections simultaneously: inverse-frequency class weights, a weighted sampler, and focal loss. The sampler assigns each training record the maximum class weight taken across all 28 single-label fields, so oversampling driven by whichever field happens to be rarest in a record also reshapes the marginal distribution seen by every other head. Whether this contributes to the collapse rather than mitigating it is tested directly in \cref{sec:eval:socialstate:imbalance}.
```

> [SINGLE-SEED] caveat: the test `response_policy` figure is one training run. A 31\% drop quoted
> without an error bar invites a question you cannot currently answer.
>
> **Status:** seeded re-runs are now possible and queued. `llm_finetuning` had no seeding at all —
> the configs carried `seed: 42` but nothing read it, so LoRA init, dropout, sampler draws and
> shuffling were unseeded and repeated runs could not serve as error bars. `train_latent.py` now
> seeds `random`, `torch`, `numpy` and CUDA from `cfg["seed"]`, and the control is queued at seeds
> 43 and 44. When they report, replace the point estimate with mean ± sd over three seeds.
>
> If they have not reported before submission, use the inline hedge:
> ```latex
> macro-$F_1=0.427$ (single training run; seeded replications in progress)
> ```

### 2d. Dangling cross-reference (reviewer item 2)

Edit 2b ends with `\cref{sec:eval:socialstate:imbalance}`, which exists only if Edit 5 lands. If
L1--L4 have not reported before submission, replace that final sentence with a self-contained one
that makes the same point without forward-referencing:

```latex
Whether these corrections contribute to the collapse rather than mitigating it is not established by the present analysis and is left to future work.
```

### 2c. Metric definition footnote

```latex
\footnote{Macro-$F_1$ is averaged over classes with gold support in the split under evaluation. Classes absent from a split cannot be scored, and counting them as zero would penalise the model for the split's composition; a schema-wide variant that does count them is also recorded, for comparability across splits, and is lower (\texttt{response\_policy}: $0.560$ validation, $0.384$ test). This differs from the scikit-learn default, which averages over the union of gold and predicted classes and so additionally counts classes the model predicts but that never occur.}
```

---

## Edit 3 — Reconciling the routing numbers

Three different routing $F_1$ values exist across the thesis and results directory. They are three
different measurements, not a contradiction:

| value | n | split | system |
|---:|---:|---|---|
| 0.686 | 363 unique turns | validation | full model, deduplicated turns (current thesis §routing) |
| 0.6721 | 683 | validation | full model, masking-ablation baseline (currently commented out at line ~349) |
| 0.7345 | 884 | **test** | full model, `routing_mode: predicted` |

Suggested sentence for the routing subsection:

```latex
Routing has been evaluated in three configurations that are easily conflated. Across 363 unique validation turns the predicted-state router reaches $F_1=0.686$; the same router evaluated over all 683 validation turns, the baseline for the field-replacement analysis, reaches $0.672$; and on the 884-turn held-out test split it reaches $0.735$. The differences reflect deduplication and split composition rather than disagreement between runs.
```

**On why routing improves on test while `response_policy` degrades.** This will be asked. The
answer is in the existing masking ablations and does not require new work: routing is a binary
decision produced by an OR over four fields, and `response_policy` is not the dominant term.
Replacing each field with its teacher label changes validation $F_1$ by:

| field replaced with gold | $\Delta F_1$ |
|---|---:|
| `secrecy_pressure` | **+0.1159** |
| `reveal_decision` | +0.0381 |
| `response_policy` | +0.0190 |
| `value_conflict` | +0.0148 |

and masking each field with its majority class changes it by:

| field masked to majority | $\Delta F_1$ |
|---|---:|
| `value_conflict` | **−0.0904** |
| `response_policy` | −0.0785 |
| `reveal_decision` | +0.0027 |
| `secrecy_pressure` | +0.0480 |

A ten-class macro-$F_1$ can therefore degrade substantially while a two-class OR-gated routing
decision holds, because routing needs only the coarse careful/ordinary bit and draws it from four
fields of which `value_conflict` and `secrecy_pressure` matter at least as much.

Note the commented-out paragraph at line ~349 already contains the $0.672 \rightarrow 0.788 \rightarrow 0.893$
figures and states their provenance is unresolved. The provenance is now resolved: $0.672$ is the
683-turn validation baseline in `eval_results/masking_ablations.json`. The paragraph can be restored
with that clarification.

---

## Edit 4 — The head-ablation matrix must not be quoted as it stands

`eval_results/ablation_matrix.md` currently reports:

| experiment | heads | routing $F_1$ |
|---|---:|---:|
| exp_a_routing_only | 4 | 0.698 |
| exp_b_plus_affect | 7 | 0.686 |
| exp_c_plus_relational | 6 | 0.666 |
| exp_d_full_29head | 28 | **0.604** |

Read at face value this says **more fields make routing monotonically worse, and the full 29-field
state is the worst configuration** — an argument against the chapter's central claim, sitting
uncorrected in the results directory.

It should not be quoted in either direction, because those ablations were **trained on their own
evaluation split**: `run_head_ablation.py` defaulted its training file to
`cfg["data"]["test_heads_file"]`, `eval.yaml` has no `train_heads_file` key, and no caller in
`scripts/experiments.sh` or `scripts/slurm_experiments.sh` passed `--train-heads-file`. The default
now raises rather than falling back, and corrected ablations training on `train_heads.jsonl` and
evaluating on the test trace are **[PENDING]**.

If the corrected matrix reproduces the monotone decline, that is a genuine and reportable negative
result about the 29-field design and should be argued honestly rather than omitted. If it does not,
the earlier table was an artefact and must be replaced.

---

## Edit 5 — New subsection [PENDING]

Reserve `\subsection{Do the Imbalance Corrections Help?}` with
`\label{sec:eval:socialstate:imbalance}` (referenced by Edit 2b). Four configurations are running,
each isolating one variable, all selecting on `val/mean_macro_f1` rather than the noisier
`val/response_policy_f1`:

| run | change |
|---|---|
| L1_control | current recipe |
| L2_nosampler | weighted sampler removed |
| L3_meanpool | + mean pooling instead of last-token |
| L4_ctx1024 | + 1024-token context, 8 epochs |

L2 is the direct test of the sampler hypothesis asserted in Edit 2b. If it raises
`response_policy` test macro-$F_1$, the chapter gains a mechanism and a remedy; if it does not, the
honest claim becomes that the limitation is architectural or data-bound, which is a materially
different sentence. **Do not write this subsection until L1--L4 report.**

---

## Summary of what is safe to insert now

| edit | status |
|---|---|
| 0 — avoid the false apples-to-apples claim | act now |
| 1a/1b/1c — Track A test numbers | insert now; MoE bridge row pending |
| 1d — Track C conditioning-bug footnote | insert now |
| 2a/2b/2c — Track D test numbers, `response_policy` | insert now **with single-seed hedge** |
| 3 — routing reconciliation | insert now |
| 4 — ablation matrix withdrawal | insert now (as a caveat); corrected numbers pending |
| 5 — imbalance subsection | pending L1--L4 |
