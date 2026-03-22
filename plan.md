Great. Here is a concrete **phase-2 research plan** that builds directly from your first paper and turns it into a stronger, more defensible next study.

Your paper already gives the core framing: explicit social-state architecture improved **Social Presence, Emergence and Play, Out-of-Context, and Perceived Progress** over the small single-agent baseline, while the large-model condition still led on immersion and system-thinking dimensions. You also already identified the right next steps: linking trace features to perceived quality, adding selective routing, validating across settings, and pushing toward sub-billion on-device models. 

What phase 2 should do is **tighten the theory** and **compress the mechanism**.

# 1. Revised phase-2 positioning

I would position the next paper as:

**From multi-agent prompting to compact socially stateful dialogue models for edge-deployable NPCs**

That framing is stronger than “we add more agents” because it says the contribution is not just orchestration. It is a theory of **which latent social states matter**, and how to encode them efficiently in small models.

A clean one-sentence claim would be:

**Believable NPC dialogue can be improved in small on-device models by explicitly modeling socio-emotional state transitions, rather than relying only on monolithic next-token generation.**

That claim fits your current findings very well. Your own paper already argues that architectural transparency and scale are complementary levers, not substitutes.  

# 2. Stronger theoretical framework

Your first abstraction used:

* emotion
* stance
* opinion
* reputation

That is a good start, but for theory these are not fully parallel. They operate at different levels. For phase 2, I would reorganize them into a layered socio-cognitive model.

## Proposed framework: Social State Transition Model

### Layer 1: Appraisal/Affect

This replaces “emotion” as the primary formal object.

Instead of only predicting emotion labels, model:

* perceived valence of the partner’s utterance
* arousal or urgency
* threat / safety
* frustration / relief
* action tendency

This is more defensible because emotion becomes the result of an appraisal process rather than an isolated category. It also gives you a continuous latent state that is easier to compress into a small model.

### Layer 2: Mentalizing / Theory of Mind

This is the biggest theoretical upgrade.

Instead of “opinion” alone, ask:

* what does the NPC think the player believes?
* what does the NPC think the player wants?
* is the player confused, deceptive, probing, bonding, or hostile?

Recent work in 2025 emphasizes that Theory-of-Mind claims for language models are fragile and benchmark-dependent, which actually helps your paper: rather than claiming human-like ToM, you can argue for **explicit, operationalized partner modeling** as a design primitive for interactive dialogue. ([arXiv][1])

### Layer 3: Relational Stance

Keep this. It is one of the most valuable parts of your architecture.

Your current paper already makes stance explicit and traceable, and that is likely one reason the small local model became more stable and interpretable. 

I would retain dimensions like:

* trust
* warmth / affiliation
* respect
* dominance
* familiarity
* dependence

This is probably the most game-useful latent state because it directly shapes line choice and disclosure.

### Layer 4: Social Image / Reputation

Keep reputation, but treat it less like a turn-level “agent” and more like a persistent social field:

* local reputation with this NPC
* public reputation
* faction reputation
* rumor load
* credibility score

That makes it a memory structure rather than a peer of emotion or stance.

### Layer 5: Norm / Role / Value Constraint

This is the most important new addition.

A lot of socially believable dialogue is not just “how I feel” or “what I think of you.” It is:

* what someone like me is supposed to do
* what I am allowed to reveal
* what would violate role, taboo, oath, faction, etiquette, secrecy, or self-image

This layer gives you much more believable conflict and much more interesting failure modes.

## Bottom line

For theory, I would move from:

**four social agents**

to:

**a layered latent social-state model with explicit transitions over affect, mentalizing, stance, reputation, and norm constraints**

That is publishably stronger.

# 3. Revised architecture for phase 2

I would not start phase 2 with four fully separate tiny models. Under sub-billion constraints, that is usually too fragmented.

Recent edge-GenAI surveys consistently emphasize that on-device deployment is constrained by memory, compute, and energy, and that compact/shared architectures usually matter as much as raw parameter count. ([arXiv][2])

## Recommended main architecture

**Shared compact backbone + explicit social-state heads + selective slow-path routing**

Per turn:

1. Input encoder reads:

   * current utterance
   * recent dialogue window
   * compact memory summary
   * world state
   * character profile

2. Social-state predictor outputs:

   * affect vector
   * partner-model / ToM summary
   * stance update
   * disclosure / deception / repair flags
   * norm conflict score

3. Router decides:

   * fast path: generate directly
   * slow path: invoke reflective pass only when conflict/uncertainty is high

4. Response generator conditions on the predicted state and produces the final utterance

5. Memory updater writes back:

   * updated stance
   * belief revision
   * public/private reputation effects
   * unresolved conflict markers

This is the best compromise between:

* interpretability
* latency
* edge feasibility
* trainability

It also aligns directly with your own stated future direction on selective routing. 

# 4. Architecture variants to compare

Your phase-2 paper should compare 3 families.

## A. Shared dense backbone + role heads or adapters

This should be your strongest baseline.

Design:

* one 0.3B–1B backbone
* small LoRA adapters or linear heads for each latent state
* one response head
* optional router

Why:

* easiest to train
* most stable under small budgets
* best control of memory footprint

## B. Sparse MoE with socially typed experts

This is the most directly aligned with your “expert per agent” idea.

Possible experts:

* affect
* ToM
* stance
* repair/de-escalation
* persuasion/deception
* style/persona

But I would be careful with the claim. MoE can reduce active FLOPs, but edge deployment still has memory and routing overhead. MoE surveys and edge deployment work in 2025 repeatedly note that sparse activation does not automatically make deployment trivial on constrained hardware. ([arXiv][3])

So MoE is worth testing, but not assuming as the winner.

## C. Recurrent / hybrid backbone with explicit social readouts

This is the most intellectually interesting option.

Instead of separate “agents,” the recurrence carries implicit interaction state. Then lightweight readout heads decode:

* current stance
* current affect
* other-model summary
* disclosure threshold
* norm pressure

This is elegant for dialogue because the model is inherently stateful. It may also be more edge-friendly than repeatedly invoking a full multi-step transformer pipeline, especially if you keep the explicit state small.

# 5. Data strategy

Your text-game sandbox is exactly the right choice.

Your own paper notes that the current text-only social deduction setting is useful for long-horizon interaction but does not yet strongly force conflicting social information or reputation shocks. That gives you a perfect design target for the next dataset. 

## Build 4 data types

### 1. Teacher-trace data

Generate with a stronger teacher system:

* dialogue turn
* social-state predictions
* belief updates
* conflict markers
* reveal/withhold decisions
* reputation effects

This becomes your main supervision target.

### 2. Counterfactual data

For the same scene, vary one latent variable:

* trust high vs low
* public rumor exists vs absent
* player deceptive vs sincere
* taboo triggered vs not triggered
* NPC threatened vs safe

This is crucial. It lets you test whether the student learned state-sensitive policy rather than just style imitation.

### 3. NPC–NPC self-play

Use this for:

* alliances
* rumor spread
* betrayal
* apology/repair
* competitive negotiation
* reputation cascades

This is probably the cheapest way to generate high-volume socially structured traces.

### 4. Human–NPC evaluation set

Keep this smaller and higher quality.
Use it mainly for:

* held-out evaluation
* human ratings
* behavioral metrics

# 6. What to distill

I would avoid making free-form chain-of-thought the main distillation target.

Your first paper already does something more valuable: it externalizes interpretable intermediate state. That is the right thing to compress. 

## Distill these instead

* affect state
* stance vector
* ToM summary
* norm-conflict score
* repair strategy
* reveal/withhold choice
* final response

So the student objective becomes:

**predict structured latent state + generate the final utterance**

That is much more stable and much easier to evaluate.

If you want a reasoning target, use:

* short decision summaries
* categorical justification labels
* conflict flags

not long unrestricted chain-of-thought.

# 7. Concrete training recipe

## Stage 1: domain-adaptive pretraining or continued pretraining

Train on:

* dialogue-heavy corpora
* roleplay / interactive fiction
* negotiation / argument / persuasion
* social conflict and reconciliation data

Goal:
make the base model comfortable with long conversational dependencies.

## Stage 2: supervised latent-state training

Train the model to predict:

* affect
* ToM
* stance
* norm conflict
* disclosure decision

This is where the main structure is learned.

## Stage 3: response generation conditioned on latent state

The generator sees both the dialogue context and the predicted state.

## Stage 4: joint fine-tuning

Train state prediction and response generation together, with multi-task loss.

Example loss:

* state classification / regression loss
* response NLL
* consistency loss between state and response
* optional ranking loss for good vs bad social outcomes

## Stage 5: routing fine-tuning

Train the router to invoke the slow path only when necessary:

* contradiction
* uncertainty
* norm conflict
* secrecy threshold
* abrupt stance shift

This directly operationalizes the selective-routing idea from your current paper. 

# 8. Experimental matrix

Here is the cleanest ablation plan.

## Study A: Which abstraction matters?

Compare:

* response-only
* response + stance
* response + affect
* response + stance + affect
* response + stance + affect + ToM
* full model with norm/reputation

This answers the theory question.

## Study B: Which architecture is best for edge?

Compare:

* shared dense backbone + heads
* shared backbone + adapters
* sparse MoE
* recurrent/hybrid + heads

Measure:

* quality
* latency
* memory
* energy
* robustness

## Study C: Is explicit state actually causal?

Budget-match the systems:

* same token budget
* same context window
* same parameter budget
* same retrieval access

Then vary only:

* with explicit social-state supervision
* without explicit social-state supervision

This is important because otherwise reviewers can say the gains are just extra scaffolding.

# 9. Evaluation plan

Phase 2 should go beyond subjective questionnaires alone.

## Keep from phase 1

* social presence
* emergence and play
* immersion
* perceived progress
* out-of-context / consistency

Those worked well already. 

## Add behavioral metrics

* trust calibration accuracy
* contradiction recovery rate
* appropriate secrecy maintenance
* reveal timing
* apology / repair success
* stable persona under adversarial probing
* rumor propagation consistency
* deception response appropriateness

## Add mechanistic metrics

* stance drift smoothness
* affect persistence across turns
* ToM prediction accuracy against hidden simulator state
* correlation between trace features and human ratings
* routing precision and recall
* rate of unnecessary slow-path invocation

## Add edge deployment metrics

* median latency
* p95 latency
* peak memory
* energy per turn
* tokens per second

Recent edge-LLM work in 2025 stresses exactly these tradeoffs: feasibility is not only model size, but system-level efficiency under realistic device constraints. ([arXiv][2])

# 10. Suggested hypotheses

These are tighter than broad “multi-agent is better” claims.

## H1

Explicit socio-emotional latent state improves perceived social coherence in small models more than response-only fine-tuning.

## H2

Relational stance and Theory-of-Mind variables contribute more to player-perceived believability than emotion labels alone.

## H3

Selective routing preserves most quality gains while reducing compute cost relative to always-on reflective pipelines.

## H4

Shared-backbone compact models outperform fully separate specialist models at sub-billion scale on the quality-efficiency frontier.

## H5

Structured state distillation transfers social behavior more effectively than distilling free-form reasoning text.

# 11. Suggested paper structure

## Title direction

**Compact Social-State Models for Edge-Deployable Believable NPC Dialogue**

or

**Compressing Social Reasoning for Believable NPCs: Explicit Latent State over Sub-Billion Dialogue Models**

## Sections

1. Introduction
2. Limitations of phase-1 multi-agent prompting
3. Social State Transition Model
4. Compact architecture designs
5. Data generation and distillation
6. Experimental design
7. Results: quality, mechanism, efficiency
8. Discussion: believability vs fluency vs control
9. Limitations and future work

# 12. Most defensible contribution statement

I would make the contribution not:

“we made smaller expert agents”

but:

**we identify a compact, interpretable set of social latent variables that can be distilled into edge-scale dialogue models and produce measurable gains in believable NPC interaction**

That is cleaner, stronger, and more publishable.

# 13. My strongest recommendation

If you only pursue one primary system in phase 2, make it this:

**A 0.5B–1B shared backbone with explicit social-state heads, short decision summaries, and selective slow-path routing.**

That is the safest architecture scientifically and the strongest bridge between your first paper and your new goals.

MoE and recurrent variants should be comparison arms, not the main identity of the study.

# 14. One concise roadmap

For the next 8–12 weeks, I would do this:

Week 1–2: formalize the revised latent-state schema
Week 2–4: generate teacher traces and counterfactual data
Week 4–6: train shared-backbone latent-state model
Week 6–8: add response generation and joint fine-tuning
Week 8–10: implement selective router
Week 10–12: run edge benchmarks and human evaluation

If you want, I can next turn this into a **full research blueprint** with:

* exact latent-state schema
* model diagram
* ablation table
* training objectives
* evaluation sheet
* and a CHI/C&C-style extended abstract draft.

[1]: https://arxiv.org/pdf/2505.00026?utm_source=chatgpt.com "Theory of Mind in Large Language Models"
[2]: https://arxiv.org/html/2502.15816v1?utm_source=chatgpt.com "GenAI at the Edge: Comprehensive Survey ..."
[3]: https://arxiv.org/html/2508.09834v1?utm_source=chatgpt.com "A Survey on Efficient Architectures for Large Language ..."
