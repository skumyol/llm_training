# Results Reassessment for Publication — FINAL (May 21)

**Status:** All previously identified issues are now resolved. The `camera_ready_summary.md` and `comprehensive_results.json` are authoritative.

## Resolution Log

| Previous Issue | Resolution | Evidence |
|---|---|---|
| Routing "unresolved" (F1=1.0 gold only) | ✅ **Resolved.** Routing F1=0.669 with predicted Z_t. | `routing_eval_metrics.json`, `routing_eval_report.md` |
| Leakage "unresolved" (artifacts disagree) | ✅ **Resolved.** Gated leakage=0, ungated=0.076. No disagreement. | `response_eval_metrics.json` |
| Latent κ "estimated, not true" | ✅ **Resolved.** True Cohen's κ=0.484 from confusion matrices. | `latent_eval_report.json`, `comprehensive_results.json` |
| Response eval "missing metrics" | ✅ **Resolved.** BLEU, distinct-n, repetition, length ratio, CIs all computed. | `response_eval_report.md` |
| Multi-seed "required" | ✅ **Resolved.** 3 seeds (42/43/44) for all Track A SLMs. | `comprehensive_results.json` |
| Qwen backbone "1.7B debug" | ✅ **Resolved.** All production runs use Qwen3-4B. | `model_registry.yaml` |
| Joint model "not fully evaluated" | ✅ **Resolved.** Job 1145599 completed May 21 00:31. | `comprehensive_results.json` |

## Final Publishable Claims

- **Architecture benchmark:** Param-matched GPT (22.3M) achieves val_ppl=41.86, outperforming MoE (43.38, 3 seeds) by 3.5% at equal scale. MoE's earlier reported advantage was against a smaller 16M GPT baseline, not a fair comparison.
- **Conditioning:** 12.3% PPL reduction from OCEAN+VAD soft-prefix; placebo shows gain is prefix-capacity, not semantic content. Gemma-4 social-state null: 13.48 vs 13.61 (no gain from gold Z_t XML prefix).
- **Latent state:** Qwen3-4B achieves true κ=0.484 across 28 heads. Affect κ=0.645, Decision κ=0.508. Response_policy F1=0.622. Secret leakage (predicted) = 0.
- **Response generation:** ROUGE-L=0.145, distinct-2=0.638, length ratio=0.91. Zero gated leakage, zero contradictions, zero prompt artifacts.
- **Routing:** F1=0.669 with predicted Z_t (the real generalization test). 54.8% slow-path rate.

## No Remaining Checked-In Artifact Gaps

All eval artifacts are present and self-consistent:
- `comprehensive_results.json` — canonical (symlinked from `_2026_05_20.json`)
- `latent_eval_report.json` / `.md` — per-head true κ
- `response_eval_report.json` / `.md` — all metrics
- `routing_eval_report.json` / `.md` — predicted-state routing
- `paper_tables.md` — preserved for historical comparison
- `camera_ready_summary.md` — final authoritative document
- `model_registry.yaml` — 14 models with checkpoints
