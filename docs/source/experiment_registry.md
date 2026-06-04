# Experiment Registry

This page is the canonical map from scripts to experiments, results, and paper status. It is intentionally conservative: existing scripts remain valid, but new work should start from `scripts/experiments.sh` unless a SLURM wrapper is required.

## Canonical Entry Points

| Scope | Canonical command | Existing wrapper kept for compatibility | Purpose |
|---|---|---|---|
| Local LLM training | `bash scripts/experiments.sh train-llm {latent|response|joint}` | `llm_finetuning/run_train.py` | Train Qwen/Gemma-style latent, response, or joint models |
| Local SLM training | `bash scripts/experiments.sh train-slm {personality|affect|small_lm|dialogue|latent}` | `slm_training/src/train/*` | Train small encoders and from-scratch dialogue/latent models |
| Local evaluation | `bash scripts/experiments.sh eval {latent|response|routing|leakage|calibration|adversarial|all}` | `llm_finetuning/run_eval.py` | Produce evaluation JSON/JSONL under `eval_results/` |
| Paper experiments | `bash scripts/experiments.sh paper {name}` | `llm_finetuning/scripts/*.py` | Run focused paper analyses and ablations |
| SLURM training | `bash scripts/experiments.sh slurm train ...` | `scripts/slurm_train.sh` | Submit LLM/SLM training jobs |
| SLURM GPU experiments | `bash scripts/experiments.sh slurm gpu ...` | `scripts/slurm_experiments.sh` | Submit GPU evaluation, ablation, calibration, validator jobs |
| SLURM CPU analyses | `bash scripts/experiments.sh slurm cpu ...` | `scripts/slurm_cpu.sh` | Submit CPU-only analysis and aggregation jobs |
| SLURM suite | `bash scripts/experiments.sh slurm suite [all|gpu|cpu|ablation]` | `scripts/submit_all_experiments.sh` | Submit the current batch suite |

## Paper Experiment Set

| ID | Command | Output | Paper use | Current status |
|---|---|---|---|---|
| A0 | `paper head_utility` | `eval_results/head_utility/head_utility_report.md`, `head_utility.json` | Which heads matter for routing/compression | Diagnostic, not a headline claim |
| A1 | `paper ablation_a --train` or without `--train` for eval | `eval_results/ablation/exp_a_routing_only/ablation_metrics.json` | Routing-only decision-state baseline | Core compression ablation |
| A2 | `paper ablation_b --train` | `eval_results/ablation/exp_b_plus_affect/ablation_metrics.json` | Incremental value of affect heads | Core compression ablation |
| A3 | `paper ablation_c --train` | `eval_results/ablation/exp_c_plus_relational/ablation_metrics.json` | Incremental value of relational heads | Core compression ablation |
| A4 | `paper ablation_full` | `eval_results/ablation/exp_d_full_29head/ablation_metrics.json` | Full-head comparison point | Core compression ablation |
| A5 | `paper aggregate_ablation` | `eval_results/ablation_matrix.md`, optional TeX | Paper table for A1-A4 | Include only with matching checkpoint provenance |
| B1 | `paper calibrate` | `calibrators/temperature/calibration_summary.json` | Confidence calibration and selective routing prerequisite | Supporting evidence |
| B2 | `paper threshold_sweep` | `eval_results/threshold_sweep.json`, `threshold_sweep.md` | F1/leakage/slow-path tradeoff figure | Supporting figure if predicted `Z_t` exists |
| C1 | `eval response` | `eval_results/response_eval_metrics.json`, `sample_generations.json` | Response quality and leakage diagnostics | Use with leakage caveat from results report |
| C2 | `paper leakage_classifier` | `leakage_classifier/final/` | Learned leakage detector for validator/eval | Tooling artifact, not a paper result alone |
| C3 | `eval leakage` | `eval_results/leakage_eval_metrics.json` | Gated/ungated leakage rates | Only report after reconciling sample-level flags |
| C4 | `paper relational_memory` | `eval_results/relational_memory_eval.json` | Multi-turn memory sensitivity | Optional/secondary |
| D1 | `train-llm latent` and `eval latent` | `eval_results/latent_eval_metrics.json`, `predicted_zt.jsonl` | Latent-head prediction metrics | Prefer rigorous true-kappa/F1 outputs when available |
| D2 | `train-slm latent` | model checkpoints and eval CSV/JSON in `eval_results/` | SLM latent comparison | Secondary model comparison |
| D3 | JEPA configs under `llm_finetuning/configs/train_latent_*jepa*.yaml` | JEPA checkpoints and eval CSV/JSON | Auxiliary objective ablation | Report as uncertain/null unless shuffled-future control is clean |

## Result Directories

| Path | Meaning | Paper guidance |
|---|---|---|
| `eval_results/paper_tables.md` | Publication-oriented summary tables | Convenient, but verify against source JSON/CSV before citing |
| `eval_results/ablation_matrix.md` | Aggregated head-ablation table | Main compression table if all component runs are current |
| `eval_results/head_utility/` | Mutual-information and redundancy diagnostics | Use to motivate compression, not as causal proof |
| `eval_results/head_leakage_corr/` | Head/leakage correlation analysis | Exploratory supporting evidence |
| `eval_results/latent_matrix/` | Cross-model latent comparison artifacts | Use for rigorous latent comparison when filenames match selected checkpoints |
| `eval_results/*_eval_report.md` | Human-readable eval summaries | Good for review; JSON/CSV are authoritative |
| `leakage_classifier/` | Trained leakage classifier checkpoints | Tool artifact consumed by leakage eval and validator |
| `calibrators/` | Temperature/isotonic calibration artifacts | Consumed by selective router and calibration eval |

## Path Rule

From the repository root, prefer paths with the package prefix:

```bash
bash scripts/experiments.sh eval all llm_finetuning/configs/eval.yaml
```

From inside `llm_finetuning/`, package-local commands may still use:

```bash
PYTHONPATH=. python run_eval.py --stage all --config configs/eval.yaml
```

## Minimum-Delta Consolidation Policy

- Do not delete old scripts until all running SLURM jobs and docs have migrated.
- Add new paper experiments to `scripts/experiments.sh` first, then optionally expose them in `slurm_experiments.sh` or `slurm_cpu.sh`.
- Treat `docs/source/experiment_results_report.md` as the evidence-quality ledger for what can be claimed in the paper.
- Treat this file as the navigation map for where each experiment/result lives.
