# Bibliography and Citation Audit for the NPC Social State Paper

## Executive summary

The bibliography is **not yet submission-ready**. The paper’s current source and reference set contain a mix of strong, correctly grounded citations and several **material bibliographic errors** that would likely be noticed by an EMNLP reviewer. The biggest problems are not small formatting issues; they are cases where the BibTeX entry appears to point to the **wrong paper**, to a **nonexistent venue-version**, or to a **preprint when a final accepted version exists**. The highest-priority fixes are `walker2006individual`, `feng2023dialogue`, `jacqmin2024dialoguestate`, `gosmar2022pippa`, `li2023chatharuhi`, `shao2023character`, `gu2023mamba`, `ploug2025open`, `rimer2025talking`, and `wang2024rolellm`. In particular, the Walker, Feng, Jacqmin, and PIPPA entries are the most concerning because they look like **metadata drift or hallucinated/merged references**, not simple typos. citeturn22search4turn6search4turn8search17turn9search1turn26search6turn14search3turn26search1turn29search0turn29search10turn17search4

The in-text citation usage is **mostly coherent at the level of research direction**, but several claims are **overstated or misattributed**. The clearest examples are: the introduction’s use of **Park et al.** and **Voyager** to support “socially inconsistent outputs” (those works do not directly establish that claim), the use of **Walker** to support “early NPC dialogue systems relied on finite-state machines and hand-authored dialogue trees” (the cited paper is about sentence planning, not NPC dialogue trees), the use of **Gunkel** to support “social simulation” in dialogue systems, and the use of **Mamba** to support a claim that SSMs underperform transformers on structured long-range input, when the cited Mamba paper is explicitly arguing for strong sequence-model performance after its architectural improvements. The conclusion table also compresses prior work too aggressively in a few places, especially for `liu2019multi` and `li2021past`. fileciteturn0file1 fileciteturn0file2 citeturn16search5turn28view0turn22search4turn27view1turn13search0turn14search0turn19search4

A further issue is that the provided `.tex` and the latest PDF appear to be **out of sync**. The PDF contains prose mentions of work such as **Spencer-Oatey (2004)** and **Song et al. (2025)** in the social-state discussion, but the provided `main(2).tex` bibliography/citation inventory does not expose corresponding citation keys in the same way. That kind of source–PDF mismatch is exactly the sort of thing that causes broken references or reviewer confusion at submission time. fileciteturn0file1 fileciteturn0file2 fileciteturn0file3

## What I checked

I audited the cited references against primary or quasi-primary sources, prioritizing **ACL Anthology**, **OpenReview**, **arXiv**, **ACM Digital Library**, **JAIR**, **Cambridge University Press**, **Annual Reviews**, and official institutional publication pages where direct publisher access was limited. For recent 2024–2026 papers, I also checked whether the current BibTeX should cite a **final accepted venue** rather than an earlier arXiv version. Where I could not verify a reference cleanly from an authoritative source, I marked it **unverifiable** rather than guessing. citeturn1search3turn12search5turn14search2turn14search3turn19search5turn25view0turn28view0

In the tables below, the **“Verified source”** column uses clickable citations to the official or best available authoritative page rather than pasting raw URLs inline. That keeps the report readable while still giving you the source URL the citation resolves to.

## Major problems and priority fixes

The most important bibliographic fixes are straightforward but should be done **before submission**. `walker2006individual` should be replaced by the actual **2007 JAIR** article metadata if you intend to cite that sentence-planning paper at all; otherwise, remove it from the NPC dialogue-tree claim because the claim is currently misattributed. `feng2023dialogue` almost certainly refers to the wrong work; the author list matches **Feng et al. 2022, “Dynamic Schema Graph Fusion Network for Multi-Domain Dialogue State Tracking”**, not a 2023 Findings paper with the stated title. `jacqmin2024dialoguestate` similarly appears to be a drifted version of the real **2022 SIGDIAL** survey, not a 2024 arXiv survey by the same authors. `gosmar2022pippa` does not line up with the actual public PIPPA paper, which is **Gosling, Dale, and Zheng 2023**. `li2023chatharuhi` is currently cited as an ACL-style conference paper, but the authoritative record I could verify is **CoRR/arXiv 2308.09597**, with the title ending in **“Large Language Model”** (singular), not “Models.” `shao2023character` should use the final **EMNLP 2023** version, not the arXiv-only entry. `gu2023mamba` should not be listed as ICLR 2024 in its current form; the paper is visible both as an ICLR submission and as a later **COLM 2024** accepted paper, so cite one consistent final form. `ploug2025open` and `rimer2025talking` both need their **full four-author lists** and proper names from the official CoG publication record. `wang2024rolellm` should be normalized to the official ACL Anthology author metadata, which begins with **Noah Wang** and **Z.y. Peng**, not the current mixed form in the `.bib`. citeturn22search4turn6search4turn8search17turn9search1turn26search6turn14search3turn26search1turn29search0turn29search10turn17search4

The most important usage fixes are also clear. The sentence that says current approaches fine-tune LLMs “without explicit social-state representations, yielding fluent but socially inconsistent outputs” overreads **Park et al.** and **Voyager**; those works are good exemplars of agentic/LLM systems without your proposed bottleneck, but they are not direct evidence for “social inconsistency” as stated. The Walker citation should be removed from the hand-authored dialogue-tree sentence unless you replace it with a genuinely relevant source. The “social simulation” example anchored by Gunkel should be replaced by a more appropriate language-agent or social-simulation citation. The Mamba citation should not be used to support a general anti-SSM conclusion. In the conclusion table, the `liu2019multi` and `li2021past` rows should be phrased more carefully so you are not suggesting those papers are direct precursors of your exact multi-head social-state formulation when they are not. fileciteturn0file1 fileciteturn0file2 citeturn16search5turn28view0turn22search4turn27view1turn17search4turn13search0turn14search0turn19search4

## Citation-by-citation audit table

### High-priority corrections

| Citation key | Verified citation | Verified source | Entry type correct | Metadata issues | Usage locations | Usage assessment | Suggested fix |
|---|---|---|---|---|---|---|---|
| `walker2006individual` | Walker, Stent, Mairesse, Prasad. 2007. *Individual and Domain Adaptation in Sentence Planning for Dialogue*. JAIR 30. DOI: 10.1613/jair.2329. | JAIR citeturn22search4 | **N** | Current `.bib` has wrong year, wrong coauthors, and likely inherited authors from earlier sentence-planning work. | Related Work → NPC Dialogue | **Unrelated / Misattributed** | Replace metadata **and** remove from the “early NPC dialogue trees” claim unless you substitute a genuinely relevant source. |
| `feng2023dialogue` | Feng et al. 2022. *Dynamic Schema Graph Fusion Network for Multi-Domain Dialogue State Tracking*. ACL 2022. DOI: 10.18653/v1/2022.acl-long.10. | ACL Anthology citeturn6search4 | **N** | Current entry appears not to match a real 2023 Findings paper; author list matches the 2022 ACL paper instead. | Related Work → Structured Social State and Dialogue Tracking | **Misattributed** | Replace with the actual intended DST paper. If this is the work you meant, cite the 2022 ACL paper. |
| `jacqmin2024dialoguestate` | Jacqmin, Rojas-Barahona, Favre. 2022. *“Do you follow me?”: A Survey of Recent Approaches in Dialogue State Tracking*. SIGDIAL 2022; arXiv:2207.14627. | ACL Anthology / arXiv citeturn8search17turn8search6 | **N** | Current entry points to a 2024 arXiv survey title I could not verify; the authoritative survey by these authors is 2022. | Related Work → Structured Social State and Dialogue Tracking | **Misattributed** | Replace with the 2022 SIGDIAL survey. |
| `gosmar2022pippa` | Gosling, Dale, Zheng. 2023. *PIPPA: A Partially Synthetic Conversational Dataset*. arXiv:2308.05884. | arXiv citeturn9search1 | **N** | Current entry has wrong authors, wrong year, and a mismatched title. | Appendix/Data → External Corpora | **Misattributed** | Replace with the actual PIPPA paper, or else cite the dataset card if that is what you used operationally. |
| `li2023chatharuhi` | Li et al. 2023. *ChatHaruhi: Reviving Anime Character in Reality via Large Language Model*. CoRR abs/2308.09597. | DBLP / arXiv citeturn26search6turn13search2 | **N** | Current entry claims ACL-style inproceedings metadata that I could not verify; official trace is arXiv/CoRR. Title wording also differs. | Related Work → NPC Dialogue and Role-Playing Agents | **Supported if corrected; currently misattributed** | Cite as arXiv/CoRR unless you have a confirmed archival venue. |
| `shao2023character` | Shao et al. 2023. *Character-LLM: A Trainable Agent for Role-Playing*. EMNLP 2023. | ACL Anthology citeturn14search3 | **N** | Current entry remains arXiv-style even though an EMNLP 2023 final version exists. | Related Work → NPC Dialogue and Role-Playing Agents | **Supported** | Replace with the EMNLP 2023 final citation. |
| `gu2023mamba` | Gu and Dao. 2024. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. COLM 2024; earlier arXiv:2312.00752. | OpenReview / COLM / arXiv citeturn26search1turn26search5turn26search9 | **N** | Current entry gives ICLR 2024-style metadata; the accepted venue visible from official sources is COLM 2024. | Results → From-Scratch SLM Baseline | **Overstated / Misattributed** | Update venue and soften the claim; Mamba is not a good citation for “SSMs underperform transformers” in the way the sentence currently states. |
| `gunkel2011computational` | **Unverifiable as cited**. No clean authoritative match found for the exact title/venue combination in the current entry. | Search results inconclusive citeturn10search0turn11search0 | **?** | Exact title/venue pairing could not be confirmed from a strong source. | Related Work → Structured Social State and Dialogue Tracking | **Unrelated / Unverifiable** | Remove or replace with a better-established citation for social simulation in language agents, e.g., Generative Agents or SOTOPIA depending the exact claim. |
| `ploug2025open` | Ploug, Rimer, Skov Petersen, Scirea. 2025. *Open-Ended NPC Dialogue Favors Casual Players: A Pilot Comparison of Three LLM-Driven Dialogue Systems*. 2025 IEEE Conference on Games. DOI: 10.1109/CoG64752.2025.11114150. | SDU official publication page citeturn29search0 | **Y** | Current `.bib` has incomplete/wrong author metadata and simplified names. | Related Work → NPC Dialogue and Role-Playing Agents | **Supported** | Replace author list and normalize venue/booktitle. |
| `rimer2025talking` | Rimer, Ploug, Petersen, Scirea. 2025. *Talking to NPCs: Three LLM-Driven Approaches to Dynamic RPG Dialogue*. 2025 IEEE Conference on Games. DOI: 10.1109/CoG64752.2025.11114413. | SDU official publication page / conference record citeturn29search10turn29search5 | **Y** | Current `.bib` omits Marco Scirea and uses simplified first names. | Related Work → NPC Dialogue and Role-Playing Agents | **Supported** | Replace author list and normalize venue/booktitle. |
| `wang2024rolellm` | Wang, Peng, Que, Liu, Zhou, Wu, et al. 2024. *RoleLLM: Benchmarking, Eliciting, and Enhancing Role-Playing Abilities of Large Language Models*. Findings of ACL 2024. | ACL Anthology citeturn17search4 | **Y** | Current author metadata does not match the official Anthology form; first author should follow the published record. | Related Work → NPC Dialogue and Role-Playing Agents | **Supported** | Replace author field with official ACL Anthology metadata. |
| `wang2023voyager` | Wang et al. 2024. *Voyager: An Open-Ended Embodied Agent with Large Language Models*. Accepted by TMLR; earlier arXiv:2305.16291. | OpenReview / arXiv citeturn28view0turn26search7 | **Y** | Current entry cites the preprint; final accepted TMLR version now exists. | Intro; Related Work; Conclusion table | **Overstated in Intro; supported elsewhere** | Update to TMLR if you want the strongest archival citation; soften the Intro claim. |
| `park2023generative` | Park et al. 2023. *Generative Agents: Interactive Simulacra of Human Behavior*. UIST 2023. DOI: 10.1145/3586183.3606763. | ACM DL / arXiv citeturn16search2turn16search5 | **Y** | Current entry is broadly correct; adding DOI would improve it. | Intro; Related Work; Conclusion table | **Overstated in Intro; supported elsewhere** | Keep for related work; narrow the Intro claim to “do not use an explicit turn-level social-state bottleneck.” |
| `liu2019multi` | Liu et al. 2019. *Multi-Task Deep Neural Networks for Natural Language Understanding*. ACL 2019. | ACL Anthology citeturn14search0 | **Y** | Metadata looks fine. | Conclusion table | **Overstated** | Do not present it as a direct precedent for social-state multi-head trajectory modeling; frame it as generic multi-task classification infrastructure. |
| `li2021past` | Li et al. 2021. *Past, Present, and Future: Conversational Emotion Recognition through Structural Modeling of Psychological Knowledge*. Findings of EMNLP 2021. | ACL Anthology citeturn13search0 | **Y** | Metadata looks fine. | Conclusion table | **Overstated** | The paper is multi-turn CER, not “single-turn attribute classification”; revise the conclusion-table wording. |
| `zhou2025social` | Zhou et al. 2025. *Social World Models*. OpenReview / LAW 2025; arXiv:2509.00559. | OpenReview / arXiv citeturn25view0turn19search3 | **Y** | Current preprint citation is acceptable, but an OpenReview venue-version now exists. | Related Work → Structured Social State and Dialogue Tracking | **Supported** | If you want final-venue precision, cite the OpenReview/LAW 2025 version and expand the venue name explicitly. |

### Remaining verified references

| Citation key | Verified citation | Verified source | Entry type correct | Metadata issues | Usage locations | Usage assessment | Suggested fix |
|---|---|---|---|---|---|---|---|
| `assran2023self` | Assran et al. 2023. *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*. CVPR 2023; arXiv:2301.08243. | CVPR / arXiv citeturn1search4turn1search0 | Y | None material. | Appendix → JEPA details | **Supported** | Keep; add DOI/page info if desired. |
| `bordes2017learning` | Bordes, Boureau, Weston. 2017. *Learning End-to-End Goal-Oriented Dialog*. ICLR 2017. | OpenReview citeturn1search1 | Y | None material. | Related Work → NPC Dialogue | **Supported** | Keep. |
| `brown1987politeness` | Brown and Levinson. 1987. *Politeness: Some Universals in Language Usage*. Cambridge UP. | Cambridge UP citeturn0search2 | Y | None material. | Related Work → Social-state theory | **Supported** | Keep. |
| `buakhaw2025deflanderization` | Buakhaw et al. 2025. *Deflanderization for Game Dialogue: Balancing Character Authenticity with Task Execution in LLM-based NPCs*. arXiv:2510.13586. | arXiv citeturn1search2 | Y | Preprint/challenge paper, not archival. | Related Work → NPC Dialogue | **Supported** | Keep, but signal clearly that it is a recent preprint/challenge system. |
| `buechel2017emobank` | Buechel and Hahn. 2017. *EmoBank: Studying the Impact of Annotation Perspective and Representation Format on Dimensional Emotion Analysis*. EACL 2017, Vol. 2. | ACL Anthology citeturn1search3 | Y | Minor venue-detail omission only. | Experimental setup; Appendix data | **Supported** | Keep. |
| `dathathri2020plug` | Dathathri et al. 2020. *Plug and Play Language Models: A Simple Approach to Controlled Text Generation*. ICLR 2020; arXiv:1912.02164. | arXiv / OpenReview citeturn3search0turn3search4 | Y | None material. | Methods → Conditioning; Conclusion table | **Supported** | Keep. |
| `dettmers2023qlora` | Dettmers et al. 2023. *QLoRA: Efficient Finetuning of Quantized LLMs*. NeurIPS 2023. | OpenReview / arXiv citeturn2search7turn2search1 | Y | None material. | Methods → Conditioning | **Supported** | Keep; optionally upgrade with final NeurIPS metadata if you have the proceedings BibTeX. |
| `digman1990personality` | Digman. 1990. *Personality Structure: Emergence of the Five-Factor Model*. *Annual Review of Psychology*. DOI: 10.1146/annurev.ps.41.020190.002221. | Annual Reviews citeturn3search5 | Y | None material. | Related Work; Conclusion table | **Supported** | Keep. |
| `hu2021lora` | Hu et al. 2022. *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022. | OpenReview citeturn27view0 | Y | None material. | Methods → Conditioning | **Supported** | Keep. |
| `keskar2019ctrl` | Keskar et al. 2019. *CTRL: A Conditional Transformer Language Model for Controllable Generation*. arXiv:1909.05858. | arXiv citeturn12search0 | Y | None material. | Methods → Conditioning; Conclusion table | **Supported** | Keep. |
| `lecun2022path` | LeCun. 2022. *A Path Towards Autonomous Machine Intelligence*. OpenReview. | OpenReview citeturn12search1 | Y | Could also be typed as `@misc`; current use is acceptable. | Appendix → JEPA details | **Supported** | Keep. |
| `lester2021power` | Lester, Al-Rfou, Constant. 2021. *The Power of Scale for Parameter-Efficient Prompt Tuning*. EMNLP 2021. DOI: 10.18653/v1/2021.emnlp-main.243. | ACL Anthology citeturn12search5 | Y | None material. | Methods → Conditioning | **Supported** | Keep. |
| `li2016persona` | Li et al. 2016. *A Persona-Based Neural Conversation Model*. ACL 2016. DOI: 10.18653/v1/P16-1094. | ACL Anthology citeturn12search6 | Y | None material. | Related Work → Social-state signals | **Supported** | Keep. |
| `li2021prefix` | Li and Liang. 2021. *Prefix-Tuning: Optimizing Continuous Prompts for Generation*. ACL-IJCNLP 2021. | ACL Anthology citeturn13search5 | Y | None material. | Methods → Conditioning; Conclusion table | **Supported** | Keep. |
| `mairesse2007using` | Mairesse et al. 2007. *Using Linguistic Cues for the Automatic Recognition of Personality in Conversation and Text*. JAIR 30. DOI: 10.1613/jair.2349. | JAIR citeturn15search0 | Y | None material. | Results → Conditioning encoders | **Supported** | Keep. |
| `mrksic2017neural` | Mrkšić et al. 2017. *Neural Belief Tracker: Data-Driven Dialogue State Tracking*. ACL 2017. DOI: 10.18653/v1/P17-1163. | ACL Anthology citeturn15search1 | Y | None material. | Related Work → DST analogy | **Supported** | Keep. |
| `oatley1987towards` | Oatley and Johnson-Laird. 1987. *Towards a Cognitive Theory of Emotions*. *Cognition and Emotion* 1(1). DOI: 10.1080/02699938708408362. | Taylor & Francis citeturn16search0 | Y | None material. | Related Work; Conclusion table | **Supported** | Keep. |
| `raheja2019dialogue` | Raheja and Tetreault. 2019. *Dialogue Act Classification with Context-Aware Self-Attention*. NAACL 2019. | ACL Anthology citeturn19search4 | Y | None material. | Conclusion table | **Supported** | Keep. |
| `rashkin2019empathetic` | Rashkin et al. 2019. *Towards Empathetic Open-domain Conversation Models: A New Benchmark and Dataset*. ACL 2019. DOI: 10.18653/v1/P19-1534. | ACL Anthology citeturn23search1 | Y | None material. | Experimental setup; Appendix data | **Supported** | Keep. |
| `russell1980circumplex` | Russell. 1980. *A Circumplex Model of Affect*. *Journal of Personality and Social Psychology* 39(6). DOI: 10.1037/h0077714. | APA metadata via author-uploaded record citeturn17search0 | Y | None material. | Related Work; Conclusion table | **Supported** | Keep. |
| `sage2025steering` | Zhang and Jaitly. 2025. *SAGE: Steering Dialog Generation with Future-Aware State-Action Augmentation*. Proceedings of the 4th Workshop on Perspectivist Approaches to NLP. DOI: 10.18653/v1/2025.nlperspectives-1.11. | ACL Anthology citeturn19search5 | Y | None material. | Related Work → Explicit social representations | **Supported** | Keep. |
| `urbanek2019learning` | Urbanek et al. 2019. *Learning to Speak and Act in a Fantasy Text Adventure Game*. EMNLP-IJCNLP 2019. | ACL Anthology citeturn18search0 | Y | None material. | Related Work; Conclusion table; Appendix data | **Supported** | Keep. |
| `zhang2018personalizing` | Zhang et al. 2018. *Personalizing Dialogue Agents: I have a dog, do you have pets too?* ACL 2018. | ACL Anthology citeturn23search0 | Y | None material. | Related Work; Appendix data | **Supported** | Keep. |
| `zhou2018emotional` | Zhou et al. 2018. *Emotional Chatting Machine: Emotional Conversation Generation with Internal and External Memory*. AAAI 2018. DOI: 10.1609/aaai.v32i1.11325. | DBLP / AAAI-linked metadata citeturn18search7turn18search10 | Y | None material. | Related Work → emotion-conditioned generation | **Supported** | Keep. |
| `zhou2024sotopia` | Zhou et al. 2024. *SOTOPIA: Interactive Evaluation for Social Intelligence in Language Agents*. ICLR 2024. | OpenReview / arXiv citeturn18search11turn18search2 | Y | None material. | Related Work → explicit social representations | **Supported** | Keep. |

## Corrected BibTeX entries for the most problematic references

These are compact corrected entries based on the authoritative records above.

```bibtex
@article{walker2007individual,
  author    = {Marilyn A. Walker and Amanda Stent and Fran{\c{c}}ois Mairesse and Rashmi Prasad},
  title     = {Individual and Domain Adaptation in Sentence Planning for Dialogue},
  journal   = {Journal of Artificial Intelligence Research},
  volume    = {30},
  pages     = {413--456},
  year      = {2007},
  doi       = {10.1613/jair.2329},
  url       = {https://jair.org/index.php/jair/article/view/10519}
}
```

```bibtex
@inproceedings{feng-etal-2022-dynamic,
  author    = {Yue Feng and Aldo Lipani and Fanghua Ye and Qiang Zhang and Emine Yilmaz},
  title     = {Dynamic Schema Graph Fusion Network for Multi-Domain Dialogue State Tracking},
  booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  year      = {2022},
  pages     = {115--126},
  doi       = {10.18653/v1/2022.acl-long.10},
  url       = {https://aclanthology.org/2022.acl-long.10/}
}
```

```bibtex
@inproceedings{jacqmin-etal-2022-follow,
  author    = {L{\'e}o Jacqmin and Lina M. Rojas-Barahona and Benoit Favre},
  title     = {{``Do you follow me?''}: A Survey of Recent Approaches in Dialogue State Tracking},
  booktitle = {Proceedings of the 23rd Annual Meeting of the Special Interest Group on Discourse and Dialogue},
  year      = {2022},
  pages     = {336--350},
  doi       = {10.18653/v1/2022.sigdial-1.33},
  url       = {https://aclanthology.org/2022.sigdial-1.33/}
}
```

```bibtex
@article{gosling2023pippa,
  author  = {Tear Gosling and Alpin Dale and Yinhe Zheng},
  title   = {PIPPA: A Partially Synthetic Conversational Dataset},
  journal = {arXiv preprint arXiv:2308.05884},
  year    = {2023},
  url     = {https://arxiv.org/abs/2308.05884}
}
```

```bibtex
@article{li2023chatharuhi,
  author  = {Cheng Li and Ziang Leng and Chenxi Yan and Junyi Shen and Hao Wang and Weishi Mi and Yaying Fei and Xiaoyang Feng and Song Yan and Haosheng Wang and Linkang Zhan and Yaokai Jia and Pingyu Wu and Haozhen Sun},
  title   = {{ChatHaruhi}: Reviving Anime Character in Reality via Large Language Model},
  journal = {CoRR},
  volume  = {abs/2308.09597},
  year    = {2023},
  url     = {https://arxiv.org/abs/2308.09597}
}
```

```bibtex
@inproceedings{shao-etal-2023-character,
  author    = {Yunfan Shao and Linyang Li and Junqi Dai and Xipeng Qiu},
  title     = {Character-{LLM}: A Trainable Agent for Role-Playing},
  booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
  year      = {2023},
  pages     = {13153--13187},
  url       = {https://aclanthology.org/2023.emnlp-main.814/}
}
```

```bibtex
@inproceedings{gu2024mamba,
  author    = {Albert Gu and Tri Dao},
  title     = {Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  booktitle = {First Conference on Language Modeling},
  year      = {2024},
  url       = {https://openreview.net/forum?id=tEYskw1VY2}
}
```

```bibtex
@article{wang2024voyager,
  author  = {Guanzhi Wang and Yuqi Xie and Yunfan Jiang and Ajay Mandlekar and Chaowei Xiao and Yuke Zhu and Linxi Fan and Anima Anandkumar},
  title   = {Voyager: An Open-Ended Embodied Agent with Large Language Models},
  journal = {Transactions on Machine Learning Research},
  year    = {2024},
  url     = {https://openreview.net/forum?id=ehfRiF0R3a}
}
```

```bibtex
@inproceedings{ploug2025open,
  author    = {Rasmus Ploug and Emil Rimer and Anthon Kristian Skov Petersen and Marco Scirea},
  title     = {Open-Ended NPC Dialogue Favors Casual Players: A Pilot Comparison of Three LLM-Driven Dialogue Systems},
  booktitle = {2025 IEEE Conference on Games (CoG)},
  year      = {2025},
  publisher = {IEEE},
  doi       = {10.1109/CoG64752.2025.11114150}
}
```

```bibtex
@inproceedings{rimer2025talking,
  author    = {Emil Rimer and Rasmus Ploug and Anthon Kristian Skov Petersen and Marco Scirea},
  title     = {Talking to NPCs: Three LLM-Driven Approaches to Dynamic RPG Dialogue},
  booktitle = {2025 IEEE Conference on Games (CoG)},
  year      = {2025},
  publisher = {IEEE},
  doi       = {10.1109/CoG64752.2025.11114413}
}
```

## Recommended text-level revisions

The introduction should not say that systems like **Generative Agents** and **Voyager** “yield socially inconsistent outputs” unless you cite work that directly evaluates social inconsistency. A safer version would be: *“Recent LLM-based agents often rely on free-text memories, prompts, or environment state rather than an explicit turn-level social-state bottleneck.”* That formulation is accurately supported by the cited papers. citeturn16search5turn28view0

The sentence about “early NPC dialogue systems” should either lose the Walker citation or be rewritten. Right now it conflates **spoken-dialogue sentence planning** with **game NPC dialogue trees**. If you do not have a clean game-specific source, it is better to write this as a general background statement without over-claiming a citation than to keep a visibly wrong one. citeturn22search4

The “social simulation” part of the related-work paragraph should not rely on `gunkel2011computational`. If you mean **LLM-based multi-agent or social simulation**, cite **Generative Agents**, **SOTOPIA**, or **Social World Models**. If you mean a much older pre-LLM computational-social-science lineage, use a reference that actually belongs to that literature. As written, the citation path is too weak for EMNLP review. citeturn16search5turn18search11turn25view0

The Mamba sentence in the SLM baseline discussion should be revised so it does not imply that the Mamba paper itself establishes that SSMs are broadly worse than transformers on the kind of reasoning you test. A safer wording would be: *“Our SSM baseline converged quickly but plateaued higher; this may reflect the difficulty of modeling the relevant structured dependencies with our specific architecture and training setup.”* That way you report your empirical result without over-borrowing from a citation that argues something stronger and more nuanced. citeturn27view1

The conclusion-table row beginning with `liu2019multi, li2021past, raheja2019dialogue` should be narrowed. `raheja2019dialogue` is a good dialogue-act classification citation, but `liu2019multi` is broad multitask NLU and `li2021past` is multi-turn conversational emotion recognition. Those papers can support “attribute classification groundwork,” but not the exact compressed label you currently give them. citeturn14search0turn13search0turn19search4

## Open questions and limitations

One reference remains genuinely uncertain: `gunkel2011computational`. I did not find a high-confidence authoritative match for the exact title-plus-venue combination in the current `.bib`, so I have treated it as **unverifiable** rather than inventing a correction. That is the one entry I would either delete or replace only after you check your own notes/source manager. citeturn10search0turn11search0

The latest PDF also appears to mention additional works that are not cleanly synchronized with the provided `.tex` citation keys, especially around **Spencer-Oatey** and **Song et al. 2025**. Before submission, make sure the exact PDF you intend to submit was compiled from the exact `.tex` and `.bib` you are editing now. Otherwise you risk fixing the wrong bibliography file. fileciteturn0file1 fileciteturn0file2 fileciteturn0file3

Overall recommendation: **major revision of the bibliography, minor-to-moderate revision of citation phrasing**. The paper’s research story is still coherent, but the reference layer needs one careful cleanup pass so that the terminology, related-work positioning, and supporting evidence look as disciplined as the experiments already do.