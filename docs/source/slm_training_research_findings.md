# Evidence-backed priorities for 15–16M dialogue LMs

This note maps primary-source evidence to the current SLM pipeline. Numerical
recommendations below are experiment starting points, not claims that one setting
is universally optimal.

## 1. Fix vocabulary allocation first

`run_small_lm.py` always loads GPT-2's 50,257-token tokenizer. In the 256-wide,
four-layer GPT, `small_lm_architectures.py` ties input and output weights, which is
good practice: weight tying can reduce size and improve perplexity
([Press and Wolf, 2017](https://aclanthology.org/E17-2025/)). Even tied, however,
the token table is `50,257 × 256 = 12,865,792` parameters. From the checked-in
architecture, the full GPT is about 16.09M parameters, so the vocabulary consumes
about **80%** of it. The same architecture is about 7.27M parameters with a 16K
vocabulary and 5.27M with an 8,192 vocabulary; the saved budget can be reassigned
to width/depth while holding total parameters fixed.

Vocabulary size is a real scaling dimension and should be selected jointly with
model and compute budget; the controlled study begins at 33M parameters and finds
that larger models/compute favor larger vocabularies
([Tao et al., 2024](https://arxiv.org/abs/2407.13623)). It does **not** establish
an exact optimum for a 16M English dialogue model. A directly relevant BabyLM
study of few-million-parameter decoder models swept 2,048/4,096/8,192-token
in-domain BPE vocabularies and selected 2,048 on training loss
([Edman et al., 2025](https://aclanthology.org/2025.babylm-main.9.pdf)); that
result is corpus-specific but confirms that GPT-2's vocabulary should not be the
default. Train a byte-level BPE/Unigram tokenizer on training-only dialogue and
ablate **2K, 4K, 8K, and 16K**, retaining compact special tokens for
profile/user/NPC boundaries. Start with **4K–8K**; the sweep must resolve the
sequence-length versus embedding-capacity trade-off. Report:

- embedding parameters and total parameters;
- tokens per character/word and the fraction of sequences truncated at 256;
- validation NLL in **bits per byte** as well as token perplexity.

Per-token perplexities from different tokenizers are not directly comparable
because the prediction units differ. Select on bits/byte plus downstream
generation quality, then compare architectures using the same chosen tokenizer.

## 2. Make the corpus easy enough, not structurally ambiguous

TinyStories demonstrates that below-10M models can learn coherent generation when
the data distribution has deliberately restricted vocabulary and complexity
([Eldan and Li, 2023](https://arxiv.org/abs/2305.07759)). This supports capacity
matching, not deleting dialogue structure. Replace verbose, repeated labels with
reserved tokens such as `<profile>`, `<player>`, `<npc>`, `<eot>`. Keep profile
information only when it is relevant to the target and cap it (for example, 32–48
tokens); additionally train a no-profile ablation. Do not concatenate unrelated
records into one causal stream: pack complete examples with `<eos>` boundaries
and mask padding, while preserving turn boundaries.

Curate by normalized exact and near-duplicate removal, train/validation
decontamination, language/length checks, speaker-alternation validity, repetition,
and response-context relevance. Deduplication has been shown to reduce memorized
emission by roughly 10×, improve efficiency, and correct validation leakage
([Lee et al., 2022](https://arxiv.org/abs/2107.06499)). Keep source and synthetic
provenance on every example and publish before/after counts.

## 3. Budget training in tokens and learn the mixture

Chinchilla found that compute-optimal model size and token count scale together
over models from 70M to 16B parameters
([Hoffmann et al., 2022](https://arxiv.org/abs/2203.15556)). Its approximately
20-token-per-parameter operating point suggests **~320M unique or usefully varied
tokens for 16M parameters** as a planning baseline, but this is an extrapolation
below the paper's tested range and domain-restricted data may saturate earlier.
Replace “3 versus 20 epochs” as the main budget with token/update budgets and
plot held-out loss against consumed tokens. Repeating a small corpus 20 times is
not equivalent to obtaining 20× more information.

There is no evidence-based universal ratio for PersonaChat, CRD3, Empathetic
Dialogues, DailyDialog, and synthetic game dialogue. Mixture proportions materially
affect LM performance; DoReMi learns domain weights with a proxy model and reached
its baseline with 2.6× fewer steps in its large-scale setting
([Xie et al., 2023](https://arxiv.org/abs/2305.10429)). For this project:

1. start with token-balanced source sampling, with a 2× sampling weight on real
   in-domain/game-like turns;
2. train short proxy runs for a small grid of mixture weights and evaluate both
   aggregate and per-source held-out NLL;
3. cap synthetic data initially at 25–35%, increasing it only when game-specific
   metrics improve without degrading natural-dialogue metrics.

An easy-to-hard curriculum is plausible for dialogue: a dialogue-generation study
uses length and pair coherence as difficulty features
([Zhu et al., 2021](https://aclanthology.org/2021.findings-emnlp.111/)).
Use it only as a short warm-up: first 10–15% of updates on short, high-confidence,
single-/two-turn examples, then interleave the full distribution. Compare against
fully shuffled sampling at the same token budget; permanent hard phase boundaries
invite forgetting.

## 4. Use a conventional, testable optimization baseline

The official Pythia suite is a useful same-scale reference: its 14M model uses
six layers, width 128, a 2M-token global batch, and peak LR `1e-3`
([EleutherAI Pythia](https://github.com/EleutherAI/pythia)). Its released training
configuration uses Adam betas `(0.9, 0.95)`, weight decay `0.1`, gradient clipping
`1.0`, zero dropout, 1% warm-up, and a single cosine decay
([official configuration](https://github.com/EleutherAI/pythia/blob/main/models/160M/pythia-160m.yml)).
These are references, not drop-in optima: Pythia's data and token batch are much
larger than this project's.

Replace cosine **warm restarts** with linear warm-up (1–3% of optimizer steps)
followed by one cosine decay to 5–10% of peak LR. Sweep peak LR
`{3e-4, 6e-4, 1e-3}` and effective token batch `{64K, 128K, 256K}` via accumulation.
Keep weight decay 0.1 as a baseline but exclude biases and normalization parameters
from decay. Sweep dropout `{0.0, 0.05, 0.1, 0.15}`; low-data dialogue may benefit
even though data-rich Pythia does not use it. Do not add label smoothing in the
first ablation: it changes the optimized NLL and complicates perplexity comparisons.
Use at least three seeds for winning configurations and early-stop on a smoothed
validation NLL with a patience measured in tokens, not evaluation calls.

Sequence length should follow data rather than convention. Measure complete-example
token-length quantiles under the new tokenizer. Start at 128 if it covers at least
90–95% of curated examples; otherwise retain 256. An informative ablation holds
**tokens per optimizer update and total training tokens fixed** while comparing
128 and 256.

## 5. Filter synthetic dialogue before mixing it

TinyStories' result supports teacher generation with explicit vocabulary/complexity
constraints for very small students
([Eldan and Li, 2023](https://arxiv.org/abs/2305.07759)). Generate compact,
natural surface dialogue while retaining the 29 labels as sidecar metadata rather
than spelling all labels into the SLM text. Vary scenario, speech act, outcome,
persona, affect, dialogue length, and teacher seed/model; avoid reusable template
phrases.

Apply a deterministic gate first (schema validity, allowed labels, speaker order,
length, no prompt artifacts, no secret leakage), then:

- exact normalized hash plus MinHash/LSH near-deduplication across all sources;
- train/validation n-gram decontamination;
- a frozen teacher rubric for context relevance, naturalness, persona/state
  consistency, policy compliance, and response specificity;
- balance across social-state labels and difficulty bins; retain rejection reasons.

Audit a stratified sample by humans. Quality-filter thresholds and synthetic
fraction must be selected on the frozen real validation/test sets, never on a
teacher score alone.

## 6. Treat perplexity as one diagnostic

Reference-overlap metrics correlate weakly with human judgments in open-domain
dialogue, so BLEU should not be a model-selection target
([Liu et al., 2016](https://aclanthology.org/D16-1230/)). Add:

- corpus distinct-1/2 and repeated 3-/4-gram rate for diversity
  ([Li et al., 2016](https://aclanthology.org/N16-1014/));
- a reference-free, multi-aspect relevance/naturalness measure such as USR
  ([Mehri and Eskenazi, 2020](https://aclanthology.org/2020.acl-main.64/));
- MAUVE on sufficiently large, length-matched generated/reference samples to
  detect distributional quality/diversity gaps
  ([Pillutla et al., 2021](https://arxiv.org/abs/2102.01454));
- project-specific policy compliance, secret leakage, contradiction, persona
  consistency, social-state adherence, and counterfactual conditioning sensitivity;
- blinded pairwise human preference on naturalness, coherence, role consistency,
  and game-policy correctness, with confidence intervals.

Use validation NLL for early stopping, but select the final checkpoint/model from a
predeclared multi-metric scorecard. Do not collapse all dialogue qualities into
BLEU or one teacher-judge number.

## Recommended first experiment

Run a factorial screen on the plain GPT before comparing architectures:
`vocab ∈ {2K, 4K, 8K, 16K}`, `seq_len ∈ {128, 256}`, `LR ∈ {3e-4, 6e-4, 1e-3}`,
holding total model parameters, total training tokens, and tokens/update as nearly
constant as possible. Use compact role tokens, deduplicated data, one warm-up +
cosine schedule, and three seeds for the top two settings. Only then rerun GPT,
PrefixGPT, MoE, and Mamba-like under the same tokenizer/data/token budget; otherwise
architecture comparisons are confounded by parameter allocation and data exposure.
