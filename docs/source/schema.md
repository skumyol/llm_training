Latent State Schema Reference
===============================

Complete specification of the 29 latent state targets.

Component C_t — Contextual Analysis
-------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 15 10 10 65

   * - Field
     - Type
     - N
     - Values
   * - ``dialogue_act``
     - list[str]
     - 10
     - ``ask``, ``accuse``, ``threaten``, ``flatter``, ``apologize``, ``negotiate``, ``joke``, ``confess``, ``probe``, ``command``
   * - ``tone``
     - str
     - 6
     - ``warm``, ``neutral``, ``confrontational``, ``sarcastic``, ``fearful``, ``evasive``
   * - ``risk_type``
     - str
     - 5
     - ``none``, ``secret-risk``, ``face-risk``, ``status-risk``, ``conflict-risk``

Component A_t — Affective Appraisal
--------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 15 10 10 65

   * - Field
     - Type
     - N
     - Values
   * - ``valence``
     - str
     - 3
     - ``negative``, ``neutral``, ``positive``
   * - ``arousal``
     - str
     - 3
     - ``low``, ``medium``, ``high``
   * - ``threat``
     - str
     - 3
     - ``low``, ``medium``, ``high``
   * - ``control``
     - str
     - 3
     - ``low``, ``medium``, ``high``

Component M_t — Player Mental Model
-------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 10 10 62

   * - Field
     - Type
     - N
     - Values
   * - ``player_intent``
     - str
     - 9
     - ``seek-info``, ``trap``, ``bond``, ``manipulate``, ``test``, ``persuade``, ``intimidate``, ``probe``, ``negotiate``
   * - ``player_knowledge``
     - str
     - 4
     - ``unaware``, ``partial``, ``informed``, ``knows-secret``
   * - ``player_credibility``
     - str
     - 3
     - ``low``, ``medium``, ``high``

Component R_t — Relational Stance
-----------------------------------

12 fields: 6 dimensions × 2 attributes (level, delta).

.. list-table::
   :header-rows: 1
   :widths: 18 20 20 42

   * - Dimension
     - Level Values
     - Delta Values
     - Semantics
   * - ``affection``
     - VL, L, N, H, VH
     - --, -, 0, +, ++
     - How much the NPC likes the player
   * - ``respect``
     - VL, L, N, H, VH
     - --, -, 0, +, ++
     - How much the NPC respects the player
   * - ``dominance``
     - VL, L, N, H, VH
     - --, -, 0, +, ++
     - Power balance in the interaction
   * - ``familiarity``
     - VL, L, N, H, VH
     - --, -, 0, +, ++
     - How well the NPC knows the player
   * - ``trust``
     - VL, L, N, H, VH
     - --, -, 0, +, ++
     - How much the NPC trusts the player
   * - ``obligation``
     - VL, L, N, H, VH
     - --, -, 0, +, ++
     - Perceived duty toward the player

- **Level** describes the current absolute value
- **Delta** describes the change from the previous turn

Component N_t — Norm/Value Constraints
-----------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 10 10 62

   * - Field
     - Type
     - N
     - Values
   * - ``duty_pressure``
     - str
     - 3
     - ``low``, ``medium``, ``high``
   * - ``secrecy_pressure``
     - str
     - 3
     - ``low``, ``medium``, ``high``
   * - ``face_pressure``
     - str
     - 3
     - ``low``, ``medium``, ``high``
   * - ``value_conflict``
     - str
     - 3
     - ``none``, ``mild``, ``strong``

Component D_t — Response Policy
----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 10 10 62

   * - Field
     - Type
     - N
     - Values
   * - ``response_policy``
     - str
     - 10
     - ``answer``, ``partial``, ``withhold``, ``deflect``, ``challenge``, ``soothe``, ``test``, ``threaten``, ``negotiate``, ``clarify``
   * - ``reveal_decision``
     - str
     - 4
     - ``none``, ``hint``, ``partial``, ``full``
   * - ``repair_strategy``
     - str
     - 5
     - ``none``, ``soften``, ``apologize``, ``clarify``, ``redirect``

Label Encoding
--------------

.. code-block:: python

   LABEL_MAPS = {
       "dialogue_act":       ["ask","accuse","threaten","flatter","apologize",
                              "negotiate","joke","confess","probe","command"],
       "tone":               ["warm","neutral","confrontational","sarcastic","fearful","evasive"],
       "risk_type":          ["none","secret-risk","face-risk","status-risk","conflict-risk"],
       "valence":            ["negative","neutral","positive"],
       "arousal":            ["low","medium","high"],
       "threat":             ["low","medium","high"],
       "control":            ["low","medium","high"],
       "player_intent":      ["seek-info","trap","bond","manipulate","test",
                              "persuade","intimidate","probe","negotiate"],
       "player_knowledge":   ["unaware","partial","informed","knows-secret"],
       "player_credibility": ["low","medium","high"],
       "duty_pressure":      ["low","medium","high"],
       "secrecy_pressure":   ["low","medium","high"],
       "face_pressure":      ["low","medium","high"],
       "value_conflict":     ["none","mild","strong"],
       "response_policy":    ["answer","partial","withhold","deflect","challenge",
                              "soothe","test","threaten","negotiate","clarify"],
       "reveal_decision":    ["none","hint","partial","full"],
       "repair_strategy":    ["none","soften","apologize","clarify","redirect"],
   }

   # Stance dims (6 × 2 fields)
   LEVEL_LABELS = ["VL","L","N","H","VH"]
   DELTA_LABELS = ["--","-","0","+","++"]
   STANCE_DIMS = ["affection","respect","dominance","familiarity","trust","obligation"]
