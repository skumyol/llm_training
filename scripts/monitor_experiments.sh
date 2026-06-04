#!/usr/bin/env bash
# =============================================================================
# monitor_experiments.sh — Live monitor for submitted experiment jobs
# =============================================================================
# Usage:
#   bash scripts/monitor_experiments.sh              # show all your jobs
#   bash scripts/monitor_experiments.sh --watch      # refresh every 10s
#   bash scripts/monitor_experiments.sh --failed     # show failed jobs + tail of err logs
#   bash scripts/monitor_experiments.sh --tail JOBID # follow stdout of specific job
#   bash scripts/monitor_experiments.sh --gpu        # show GPU utilization on running nodes
#
# Works with jobs submitted via submit_all_experiments.sh or individually.
# =============================================================================
set -euo pipefail

USER="${USER:-$(whoami)}"
LOG_BASE="/scratch/${USER}/logs"
MODE="${1:-status}"

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ── Helper: print section header ──────────────────────────────────────────────
header() {
    echo ""
    echo "══════════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "══════════════════════════════════════════════════════════════════"
}

# ═══════════════════════════════════════════════════════════════════════════════
# Status mode (default)
# ═══════════════════════════════════════════════════════════════════════════════
if [ "${MODE}" = "status" ] || [ "${MODE}" = "" ]; then
    header "JOB QUEUE ($(date +%H:%M:%S))"

    squeue -u "${USER}" -o "%.12i %.9P %.18j %.8u %.2t %.10M %.6D %.10R %C %b" 2>/dev/null || {
        echo "  squeue not available (are you on the cluster login node?)"
        exit 1
    }

    header "JOB COUNTS"
    local running pending completed failed other
    running=$(squeue -u "${USER}" -t RUNNING -h 2>/dev/null | wc -l)
    pending=$(squeue -u "${USER}" -t PENDING -h 2>/dev/null | wc -l)
    completed=$(squeue -u "${USER}" -t COMPLETED -h 2>/dev/null | wc -l)
    failed=$(squeue -u "${USER}" -t FAILED -h 2>/dev/null | wc -l)
    other=$(squeue -u "${USER}" -h 2>/dev/null | wc -l)
    other=$((other - running - pending))

    echo -e "  ${GREEN}RUNNING${NC}:    ${running}"
    echo -e "  ${YELLOW}PENDING${NC}:    ${pending}"
    echo -e "  ${BLUE}OTHER${NC}:      ${other}"
    echo -e "  ${RED}FAILED${NC}:     ${failed}"

    header "RECENT LOGS (last 5 modified)"
    if [ -d "${LOG_BASE}" ]; then
        ls -lt "${LOG_BASE}"/*.out 2>/dev/null | head -n 5 | while read -r line; do
            echo "  ${line}"
        done
    else
        echo "  No logs found at ${LOG_BASE}"
    fi

# ═══════════════════════════════════════════════════════════════════════════════
# Watch mode (auto-refresh)
# ═══════════════════════════════════════════════════════════════════════════════
elif [ "${MODE}" = "--watch" ] || [ "${MODE}" = "-w" ]; then
    INTERVAL="${2:-10}"
    echo "Refreshing every ${INTERVAL}s. Press Ctrl+C to stop."
    while true; do
        clear
        bash "$0" status
        echo ""
        echo "Last update: $(date)  |  Next refresh in ${INTERVAL}s..."
        sleep "${INTERVAL}"
    done

# ═══════════════════════════════════════════════════════════════════════════════
# Failed jobs mode
# ═══════════════════════════════════════════════════════════════════════════════
elif [ "${MODE}" = "--failed" ] || [ "${MODE}" = "-f" ]; then
    header "FAILED JOBS"

    local failed_jobs
    failed_jobs=$(squeue -u "${USER}" -t FAILED -o "%i %j" -h 2>/dev/null)

    if [ -z "${failed_jobs}" ]; then
        echo -e "  ${GREEN}No failed jobs.${NC}"
    else
        echo "${failed_jobs}" | while read -r jid jname; do
            echo -e "  ${RED}Job ${jid} (${jname})${NC}"

            # Try to find the error log
            err_file="${LOG_BASE}/${jname}_${jid}.err"
            if [ ! -f "${err_file}" ]; then
                # Try wildcard match
                err_file=$(ls -t "${LOG_BASE}"/*"${jid}"*.err 2>/dev/null | head -n 1 || true)
            fi

            if [ -n "${err_file}" ] && [ -f "${err_file}" ]; then
                echo "  Error log: ${err_file}"
                echo "  --- last 20 lines ---"
                tail -n 20 "${err_file}" | sed 's/^/    /'
            else
                echo "  No error log found"
            fi
            echo ""
        done
    fi

    # Also show jobs that exited with non-zero recently (within last hour)
    header "RECENT NON-ZERO EXITS (sacct)"
    sacct -u "${USER}" --state=FAILED,TIMEOUT,CANCELLED --format=JobID,JobName,State,ExitCode,End -X -S now-1hour 2>/dev/null | head -n 20 || echo "  sacct not available or no recent failures"

# ═══════════════════════════════════════════════════════════════════════════════
# Tail specific job
# ═══════════════════════════════════════════════════════════════════════════════
elif [ "${MODE}" = "--tail" ] || [ "${MODE}" = "-t" ]; then
    JOBID="${2:-}"
    if [ -z "${JOBID}" ]; then
        echo "Usage: $0 --tail JOBID"
        exit 1
    fi

    out_file=$(ls -t "${LOG_BASE}"/*"${JOBID}"*.out 2>/dev/null | head -n 1 || true)
    if [ -n "${out_file}" ] && [ -f "${out_file}" ]; then
        echo "Following: ${out_file}"
        tail -f "${out_file}"
    else
        echo "No .out log found for job ${JOBID} in ${LOG_BASE}"
        exit 1
    fi

# ═══════════════════════════════════════════════════════════════════════════════
# GPU utilization mode
# ═══════════════════════════════════════════════════════════════════════════════
elif [ "${MODE}" = "--gpu" ] || [ "${MODE}" = "-g" ]; then
    header "GPU UTILIZATION ON RUNNING NODES"

    # Get nodes where our jobs are running
    nodes=$(squeue -u "${USER}" -t RUNNING -o "%N" -h 2>/dev/null | sort -u)

    if [ -z "${nodes}" ]; then
        echo "  No running jobs with allocated GPUs."
        exit 0
    fi

    for node in ${nodes}; do
        echo ""
        echo "  Node: ${node}"
        ssh "${node}" 'nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv' 2>/dev/null | sed 's/^/    /' || echo "    (cannot SSH to node)"
    done

# ═══════════════════════════════════════════════════════════════════════════════
# Help
# ═══════════════════════════════════════════════════════════════════════════════
else
    echo "Usage: $0 [MODE]"
    echo ""
    echo "Modes:"
    echo "  (none)        Show job queue + counts + recent logs"
    echo "  --watch, -w   Auto-refresh every 10s (pass interval as 2nd arg)"
    echo "  --failed, -f  Show failed jobs with error log excerpts"
    echo "  --tail JOBID  Follow stdout of a specific job"
    echo "  --gpu, -g     Show GPU utilization on running nodes"
    echo ""
    echo "Examples:"
    echo "  bash scripts/monitor_experiments.sh"
    echo "  bash scripts/monitor_experiments.sh --watch 5"
    echo "  bash scripts/monitor_experiments.sh --tail 12345"
    exit 1
fi
