Absolutely — this is the part that should be nailed down before architecture, training, or data generation.

Your current paper already has the right instinct: **externalize social state each turn**, instead of leaving it implicit inside one monolithic generator. In the current system, Perception outputs entities/claims, Stance tracks four relational dimensions, Opinion produces a short belief summary with confidence, and Reputation updates longer-horizon social judgments; those intermediate states are then reused by the Response Agent.  

For phase 2, I would formalize this into a **revised latent-state schema** that is more coherent theoretically and easier to supervise.

---

# 1. What the latent state should be

I would define the NPC at turn (t) as carrying a structured latent social state:

[
Z_t = {W, C_t, A_t, M_t, R_t, N_t, D_t}
]

Where:

* (W): stable world and persona state
* (C_t): current conversational interpretation of the turn
* (A_t): affective/appraisal state
* (M_t): model of the other mind
* (R_t): relational stance
* (N_t): normative/role constraints
* (D_t): discourse policy / action decision state

And optionally:

[
G_t = \text{global social field}
]

for faction reputation, public rumor, or shared world-facing reputation.

This is better than “emotion / stance / opinion / reputation” because now each part has a clear job and temporal scale.

---

# 2. Split the state by timescale

This is very important.

Not all latent variables should update at the same speed.

## Slow state

Changes rarely:

* persona
* values
* role
* secret ownership
* faction membership
* long-term relationship priors
* public reputation baseline

## Medium state

Changes over scenes / multi-turn interaction:

* trust
* respect
* familiarity
* suspicion
* player credibility
* unresolved conflict
* social obligation debt

## Fast state

Changes per turn:

* immediate appraisal
* emotional arousal
* inferred user intent
* dialogue act expectation
* reveal / deflect / challenge / soothe decision

If you do not formalize timescale, your model will learn unstable behavior. It will overreact turn-to-turn, and you will not know whether trust or arousal is supposed to move.

---

# 3. The revised latent-state schema

Here is the schema I recommend.

## 3.1 Stable character/world state (W)

This is not predicted every turn; it is mostly conditioned on.

[
W = {p, g, s, f, v, b}
]

Where:

* (p): persona traits and speaking style
* (g): current goals
* (s): secrets and disclosure rules
* (f): faction / social group membership
* (v): value priorities
* (b): biography / backstory anchors

This is your authored state.

It includes the kind of things your paper currently injects via persona, secrets, world context, and memory retrieval. 

### Example

* role: castle guard
* goal: protect vault access
* values: duty > politeness > honesty
* secret: knows the prince moved the chalice
* speaking style: curt, formal

---

## 3.2 Conversational interpretation (C_t)

This replaces a purely surface “perception” pass with a more formal turn representation.

[
C_t = {e_t, q_t, da_t, tone_t, sal_t, risk_t}
]

Where:

* (e_t): entities and references
* (q_t): salient propositions / claim tuples
* (da_t): dialogue act of user turn
* (tone_t): detected tone
* (sal_t): salience ranking
* (risk_t): conversational risk cues

This is the direct successor to your current Perception Agent, which already extracts entities and salient claims. 

### Example

User: “I heard you stole the chalice. Why hide it?”

Then:

* entities: chalice, accusation
* dialogue act: accusation + probe
* tone: confrontational
* salience: theft claim high
* risk: secret exposure risk high

Your current trace example is already close to this. 

---

## 3.3 Appraisal / affect state (A_t)

Do not treat “emotion” as just a label like angry/sad.
Formalize it as a compact appraisal state.

[
A_t = {val_t, ar_t, th_t, ctrl_t, goalimp_t, emo_t}
]

Where:

* (val_t): valence toward current interaction
* (ar_t): arousal / activation
* (th_t): perceived threat
* (ctrl_t): perceived control
* (goalimp_t): goal impediment / support
* (emo_t): optional discrete derived emotion label

This lets you model emotion as the output of appraisal, not a primitive.

### Why this matters

If the user asks a rude question, the NPC may show:

* negative valence
* high threat
* medium arousal
* low goal support

That produces irritation or guardedness.

This is much easier to supervise and much more stable than asking the model to directly infer one fuzzy emotion word.

### Suggested format

Continuous values in small bounded ranges:

* valence: ([-2, 2])
* arousal: ([0, 2])
* threat: ([0, 2])
* control: ([0, 2])

Or discretized bins:

* low / medium / high

For sub-billion models, I would start with **3-bin or 5-bin classification**, not free regression.

---

## 3.4 Mental model of the player (M_t)

This is the biggest conceptual upgrade over “opinion.”

[
M_t = {intent_t, belief_t, know_t, trustw_t, strat_t, emo_other_t}
]

Where:

* (intent_t): what the NPC thinks the player is trying to do
* (belief_t): what the NPC thinks the player currently believes
* (know_t): what the NPC thinks the player knows
* (trustw_t): player trustworthiness / credibility estimate
* (strat_t): inferred interaction strategy
* (emo_other_t): inferred player emotional stance

This is not “human Theory of Mind” in the strong philosophical sense. It is an **operational partner model**.

### Example

After an accusation:

* intent: pressure for confession
* belief: thinks NPC is guilty
* know: knows rumor, not facts
* trustworthiness: medium-low
* strategy: probing / baiting

That is much more precise than “current opinion: suspicious.”

Your current Opinion Agent already produces a short belief summary with confidence. This schema makes that formally decomposable and easier to learn. 

---

## 3.5 Relational stance (R_t)

This is where your current work is already strongest.

Your paper’s four dimensions are:

* affection
* respect
* dominance
* familiarity 

I would preserve them, but formalize them as a state vector:

[
R_t = {aff_t, resp_t, dom_t, fam_t, trust_t, oblig_t}
]

Where:

* (aff_t): warmth / liking
* (resp_t): regard / status recognition
* (dom_t): dominance / submission dynamic
* (fam_t): familiarity / closeness
* (trust_t): confidence in honesty / reliability
* (oblig_t): felt obligation toward the player

I would add **trust** and **obligation**.

Why:

* trust is too central to hide inside “opinion”
* obligation matters a lot in role-based or faction-based games

### Update rule intuition

[
R_t = f_R(R_{t-1}, C_t, A_t, M_t, W)
]

Example:

* accusation from stranger:

  * affection down
  * respect down
  * dominance up
  * familiarity unchanged or slightly up
  * trust down
* sincere apology from ally:

  * affection up
  * trust up
  * dominance down
  * obligation up

### Practical note

For training, I would not start with free text like “slight rise.”
I would encode:

* score bin: very low / low / neutral / high / very high
* delta: -- / - / 0 / + / ++

Your current qualitative labels are good for interpretability, but for a trainable compact model you need more standardized labels. 

---

## 3.6 Norm / role / value constraint state (N_t)

This is the missing piece.

[
N_t = {role_t, taboo_t, duty_t, secrecy_t, face_t, conflict_t}
]

Where:

* (role_t): current role obligations
* (taboo_t): forbidden topics / actions
* (duty_t): priority of institutional or moral duty
* (secrecy_t): pressure against disclosure
* (face_t): need to preserve self-image or status
* (conflict_t): active internal conflict

This formalizes the “value conflict” field in your current system, which is presently free-text and ontology-free. 

### Why this matters

A believable NPC does not only ask:

* “How do I feel about you?”

It also asks:

* “Am I allowed to say this?”
* “Would saying this violate my role?”
* “Will this damage my standing?”
* “Does duty override warmth?”

This layer will strongly improve:

* secret handling
* contradiction resistance
* believable evasion
* moral tension

### Example

Guard confronted about missing artifact:

* duty: high
* secrecy pressure: high
* face concern: high
* internal conflict: medium

Then the model may answer with deflection rather than confession even if affection is moderate.

---

## 3.7 Discourse policy / action decision (D_t)

This is the bridge from latent state to response generation.

[
D_t = {act_t, reveal_t, repair_t, style_t, target_t}
]

Where:

* (act_t): next dialogue policy
* (reveal_t): reveal / partial reveal / withhold / redirect
* (repair_t): whether to de-escalate, challenge, apologize, reframe
* (style_t): tone selection
* (target_t): what conversational objective to optimize now

This state should be predicted before text generation.

### Example action classes

* answer directly
* answer partially
* refuse
* challenge
* deflect
* test the player
* soothe
* threaten
* negotiate
* confess
* ask counter-question

This is the state that your current Response Agent is implicitly using via stance/opinion + secret rules. Formalizing it makes training much cleaner. 

---

## 3.8 Global social field (G_t) (optional but useful)

[
G_t = {rep^{local}_t, rep^{public}_t, rep^{faction}_t, rumor_t}
]

Where:

* (rep^{local}_t): this NPC’s view of player
* (rep^{public}_t): general fame or notoriety
* (rep^{faction}_t): standing with group
* (rumor_t): active circulating claims

This is the cleaned-up version of your Reputation Agent output. In your current design, global reputation and NPC-specific opinion are injected after dialogue ends. 

For phase 2, I would explicitly separate:

* relationship state with this NPC
* world-facing reputation state

That avoids mixing interpersonal trust with public notoriety.

---

# 4. The full update process

Now formalize the dynamics.

At each turn:

[
C_t = f_C(x_t, H_{t-1}, W, G_{t-1})
]

[
A_t = f_A(C_t, R_{t-1}, N_{t-1}, W)
]

[
M_t = f_M(C_t, H_{t-1}, R_{t-1}, W)
]

[
R_t = f_R(R_{t-1}, A_t, M_t, C_t)
]

[
N_t = f_N(N_{t-1}, C_t, W, s)
]

[
D_t = f_D(A_t, M_t, R_t, N_t, G_{t-1}, W)
]

[
y_t = f_Y(x_t, H_{t-1}, W, D_t)
]

And if scene or speaker changes:

[
G_t = f_G(G_{t-1}, H_{1:t}, R_{1:t}, D_{1:t})
]

Where:

* (x_t): user utterance
* (H_{t-1}): prior interaction history
* (y_t): generated NPC response

This is the actual computational story your model should tell.

---

# 5. What should be latent vs explicit

For phase 2, not every variable needs to be visible to the player or logged in natural language.

## Keep explicit and supervised

These should be logged and predicted directly:

* dialogue act
* threat / support appraisal
* player intent class
* player knowledge estimate
* stance dimensions
* trust
* secrecy pressure
* next action policy
* reveal/withhold decision

## Keep latent or compressed

These can stay partly internal:

* continuous hidden social embeddings
* style manifolds
* fine-grained semantic memory interactions

That gives you interpretability without overburdening annotation.

---

# 6. Recommended label inventory

You need a practical schema, not just theory.

Here is the version I would actually use for data generation.

## A. Conversational interpretation labels

* dialogue_act: ask / accuse / threaten / flatter / apologize / negotiate / joke / confess / probe / command
* tone: warm / neutral / confrontational / sarcastic / fearful / evasive
* risk_type: none / secret-risk / face-risk / status-risk / conflict-risk

## B. Affect labels

* valence: negative / neutral / positive
* arousal: low / medium / high
* threat: low / medium / high
* control: low / medium / high

## C. Other-model labels

* player_intent: seek-info / trap / bond / manipulate / test / persuade / intimidate
* player_knowledge: unaware / partial / informed / knows-secret
* player_credibility: low / medium / high

## D. Stance labels

For each of:

* affection
* respect
* dominance
* familiarity
* trust
* obligation

Store:

* current level: VL / L / N / H / VH
* delta: -- / - / 0 / + / ++

## E. Norm labels

* duty_pressure: low / medium / high
* secrecy_pressure: low / medium / high
* face_pressure: low / medium / high
* value_conflict: none / mild / strong

## F. Policy labels

* response_policy: answer / partial / withhold / deflect / challenge / soothe / test / threaten / negotiate
* reveal_decision: none / hint / partial / full
* repair_strategy: none / soften / apologize / clarify / redirect

That is enough to train on and small enough for sub-billion models.

---

# 7. What not to include

A common mistake is to make the schema too philosophical.

I would **not** start with:

* 20 emotion categories
* full moral foundation ontologies
* very abstract “identity states”
* rich free-text chain-of-thought as the supervision target

Those are too expensive, too noisy, and too hard to validate.

Your schema should explain behavior in the game, not imitate every theory of mind paper ever written.

---

# 8. How this dictates model design

Once you adopt this schema, the model design becomes much clearer.

## If you use a shared backbone

You need prediction heads for:

* (C_t)
* (A_t)
* (M_t)
* (R_t)
* (N_t)
* (D_t)

Then generation conditions on these outputs.

## If you use MoE

Experts should align with the schema:

* interpretation expert
* affect expert
* partner-model expert
* stance expert
* norm/policy expert

Not “emotion model, stance model, opinion model, reputation model” in the old sense.

## If you use recurrence / RNN / SSM hybrid

The recurrent hidden state can hold slow and medium dynamics, while supervised readout heads expose:

* stance
* trust
* secrecy pressure
* next policy

So the schema directly dictates which parts need explicit supervision and which parts can live in hidden state.

---

# 9. How this dictates training

Your training targets become:

[
\mathcal{L} =
\lambda_C \mathcal{L}_C +
\lambda_A \mathcal{L}_A +
\lambda_M \mathcal{L}_M +
\lambda_R \mathcal{L}_R +
\lambda_N \mathcal{L}_N +
\lambda_D \mathcal{L}_D +
\lambda_Y \mathcal{L}_Y
]

Where:

* (\mathcal{L}_C): interpretation loss
* (\mathcal{L}_A): affect loss
* (\mathcal{L}_M): partner-model loss
* (\mathcal{L}_R): stance loss
* (\mathcal{L}_N): norm/conflict loss
* (\mathcal{L}_D): dialogue policy loss
* (\mathcal{L}_Y): final response generation loss

This is much better than training directly on response text only.

---

# 10. How this dictates data generation

Now the teacher system should produce per-turn records like:

```json
{
  "input": "I heard you stole the chalice. Why hide it?",
  "C_t": {
    "dialogue_act": "accuse+probe",
    "tone": "confrontational",
    "risk_type": "secret-risk"
  },
  "A_t": {
    "valence": "negative",
    "arousal": "medium",
    "threat": "high",
    "control": "medium"
  },
  "M_t": {
    "player_intent": "trap",
    "player_knowledge": "partial",
    "player_credibility": "medium"
  },
  "R_t": {
    "affection": {"level": "low", "delta": "-"},
    "respect": {"level": "low", "delta": "-"},
    "dominance": {"level": "high", "delta": "+"},
    "familiarity": {"level": "medium", "delta": "0"},
    "trust": {"level": "low", "delta": "-"},
    "obligation": {"level": "low", "delta": "0"}
  },
  "N_t": {
    "duty_pressure": "high",
    "secrecy_pressure": "high",
    "face_pressure": "high",
    "value_conflict": "mild"
  },
  "D_t": {
    "response_policy": "challenge",
    "reveal_decision": "none",
    "repair_strategy": "none"
  },
  "response": "You accuse me without proof. Watch your tone if you want answers."
}
```

That is the type of record you want to generate at scale.

---

# 11. My recommended minimal schema

If you want the smallest viable version, use this:

[
Z_t^{mini} = {C_t, A_t, M_t, R_t, D_t}
]

With:

* (C_t): dialogue act + tone + risk
* (A_t): valence + threat + arousal
* (M_t): player intent + player knowledge + credibility
* (R_t): affection + respect + dominance + familiarity + trust
* (D_t): policy + reveal decision

And keep:

* secrets
* values
* role obligations
* reputation

in conditioned memory until phase 2b.

That is probably the best starting point if you want something trainable soon.

---

# 12. My recommended full schema

If you want the version most worthy of a serious paper, use:

[
Z_t^{full} = {W, C_t, A_t, M_t, R_t, N_t, D_t, G_t}
]

This gives you:

* theoretical coherence
* trainable supervision targets
* clean ablations
* interpretable logs
* direct alignment with architecture

---

# 13. Final recommendation

If I had to choose one exact formalization for your project, I would use:

**Per-turn schema**

* (C_t): interpretation
* (A_t): appraisal/affect
* (M_t): player model
* (R_t): relational stance
* (N_t): norm/secrecy pressure
* (D_t): response policy

**Persistent schema**

* (W): persona/goals/secrets/values
* (G_t): public and faction reputation

That is the cleanest bridge between your current paper and the next phase.

The next best step is to turn this into a **label spec sheet** with exact field names, allowed values, update rules, and annotation prompts for teacher-data generation.
