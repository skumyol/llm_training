#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LLM_DIR="${ROOT_DIR}/llm_finetuning"

usage() {
    cat >&2 <<'USAGE'
Usage:
  bash scripts/experiments.sh train-llm {latent|response|joint} [extra args]
  bash scripts/experiments.sh train-slm {personality|affect|small_lm|dialogue|latent} [extra args]
  bash scripts/experiments.sh eval {latent|response|routing|leakage|calibration|adversarial|all} [config]
  bash scripts/experiments.sh paper {head_utility|ablation_a|ablation_b|ablation_c|ablation_full|aggregate_ablation|calibrate|leakage_classifier|threshold_sweep|relational_memory} [extra args]
  bash scripts/experiments.sh slurm {train|gpu|cpu|suite} [extra args]

Canonical local entrypoint for experiment work. Existing SLURM wrappers remain supported:
  scripts/slurm_train.sh        training jobs
  scripts/slurm_experiments.sh  GPU paper/evaluation jobs
  scripts/slurm_cpu.sh          CPU analysis jobs
  scripts/submit_all_experiments.sh suite submission
USAGE
}

need_arg() {
    local value="${1:-}"
    local name="$2"
    if [ -z "${value}" ]; then
        echo "ERROR: missing ${name}" >&2
        usage
        exit 2
    fi
}

run_llm_python() {
    PYTHONPATH="${LLM_DIR}" python "$@"
}

cmd="${1:-}"
shift 2>/dev/null || true

case "${cmd}" in
    train-llm)
        stage="${1:-}"
        shift 2>/dev/null || true
        need_arg "${stage}" "LLM stage"
        run_llm_python "${LLM_DIR}/run_train.py" \
            --stage "${stage}" \
            --config "${LLM_DIR}/configs/train_${stage}.yaml" \
            "$@"
        ;;

    train-slm)
        stage="${1:-}"
        shift 2>/dev/null || true
        need_arg "${stage}" "SLM stage"
        export PYTHONPATH="${ROOT_DIR}/slm_training:${ROOT_DIR}:${LLM_DIR}"
        case "${stage}" in
            personality) python -m src.train.run_personality "$@" ;;
            affect) python -m src.train.run_affect "$@" ;;
            small_lm) python -m src.train.run_small_lm "$@" ;;
            dialogue) python -m src.train.run_dialogue "$@" ;;
            latent) python -m slm_training.src.train.train_latent_slm "$@" ;;
            *) echo "ERROR: unknown SLM stage: ${stage}" >&2; exit 2 ;;
        esac
        ;;

    eval)
        stage="${1:-all}"
        shift 2>/dev/null || true
        config="${1:-${LLM_DIR}/configs/eval.yaml}"
        if [ $# -gt 0 ]; then
            shift
        fi
        run_llm_python "${LLM_DIR}/run_eval.py" --stage "${stage}" --config "${config}" "$@"
        ;;

    paper)
        experiment="${1:-}"
        shift 2>/dev/null || true
        need_arg "${experiment}" "paper experiment"
        case "${experiment}" in
            head_utility)
                run_llm_python "${LLM_DIR}/scripts/analyze_head_utility.py" \
                    --heads-file data/splits/val_heads.jsonl \
                    --output-dir eval_results/head_utility "$@"
                ;;
            ablation_a)
                run_llm_python "${LLM_DIR}/scripts/run_head_ablation.py" \
                    --config "${LLM_DIR}/configs/eval.yaml" \
                    --heads response_policy reveal_decision value_conflict secrecy_pressure \
                    --name exp_a_routing_only "$@"
                ;;
            ablation_b)
                run_llm_python "${LLM_DIR}/scripts/run_head_ablation.py" \
                    --config "${LLM_DIR}/configs/eval.yaml" \
                    --heads response_policy reveal_decision value_conflict secrecy_pressure valence threat control \
                    --name exp_b_plus_affect "$@"
                ;;
            ablation_c)
                run_llm_python "${LLM_DIR}/scripts/run_head_ablation.py" \
                    --config "${LLM_DIR}/configs/eval.yaml" \
                    --heads response_policy reveal_decision value_conflict secrecy_pressure trust_level respect_level \
                    --name exp_c_plus_relational "$@"
                ;;
            ablation_full)
                run_llm_python "${LLM_DIR}/scripts/run_head_ablation.py" \
                    --config "${LLM_DIR}/configs/eval.yaml" \
                    --heads response_policy reveal_decision value_conflict secrecy_pressure valence threat control trust_level respect_level affection_level familiarity_level dominance_level obligation_level trust_delta respect_delta affection_delta familiarity_delta dominance_delta obligation_delta tone dialogue_act risk_type repair_strategy player_intent player_knowledge player_credibility duty_pressure face_pressure \
                    --name exp_d_full_29head "$@"
                ;;
            aggregate_ablation)
                run_llm_python "${LLM_DIR}/scripts/aggregate_ablation_results.py" \
                    --results-dir eval_results/ablation \
                    --output eval_results/ablation_matrix.md "$@"
                ;;
            calibrate)
                run_llm_python "${LLM_DIR}/scripts/calibrate_head.py" \
                    --config "${LLM_DIR}/configs/eval.yaml" \
                    --method temperature \
                    --calib-heads-file data/splits/val_heads.jsonl \
                    --output-dir calibrators/temperature "$@"
                ;;
            leakage_classifier)
                run_llm_python "${LLM_DIR}/scripts/train_leakage_classifier.py" \
                    --heads-file data/splits/train_heads.jsonl \
                    --sft-file data/splits/train_sft.jsonl \
                    --output-dir leakage_classifier "$@"
                ;;
            threshold_sweep)
                run_llm_python "${LLM_DIR}/scripts/sweep_selective_router.py" \
                    --config "${LLM_DIR}/configs/eval.yaml" \
                    --predicted-zt eval_results/predicted_zt.jsonl \
                    --test-trace data/splits/test_trace.jsonl \
                    --leakage-file eval_results/sample_generations.json \
                    --output eval_results/threshold_sweep.json "$@"
                ;;
            relational_memory)
                run_llm_python "${LLM_DIR}/scripts/eval_relational_memory.py" \
                    --config "${LLM_DIR}/configs/eval.yaml" \
                    --checkpoint checkpoints/latent_predictor_best \
                    --test-heads data/splits/test_heads.jsonl \
                    --output eval_results/relational_memory_eval.json "$@"
                ;;
            *) echo "ERROR: unknown paper experiment: ${experiment}" >&2; exit 2 ;;
        esac
        ;;

    slurm)
        wrapper="${1:-}"
        shift 2>/dev/null || true
        need_arg "${wrapper}" "SLURM wrapper"
        case "${wrapper}" in
            train) sbatch "${ROOT_DIR}/scripts/slurm_train.sh" "$@" ;;
            gpu) sbatch "${ROOT_DIR}/scripts/slurm_experiments.sh" "$@" ;;
            cpu) sbatch "${ROOT_DIR}/scripts/slurm_cpu.sh" "$@" ;;
            suite) bash "${ROOT_DIR}/scripts/submit_all_experiments.sh" "$@" ;;
            *) echo "ERROR: unknown SLURM wrapper: ${wrapper}" >&2; exit 2 ;;
        esac
        ;;

    -h|--help|help|"")
        usage
        ;;

    *)
        echo "ERROR: unknown command: ${cmd}" >&2
        usage
        exit 2
        ;;
esac
