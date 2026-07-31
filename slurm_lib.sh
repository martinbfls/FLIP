#!/bin/bash
###
# slurm_lib.sh
#
# Shared bash library for launching FLIP/BRoADflip experiments on a Slurm
# cluster. This is the Slurm counterpart to the job-pool executor found in
# orchestrate_runs.sh (and its bis/ter/trigger variants), which assigns one
# job at a time to each entry of a hardcoded SSH "MACHINES" pool.
#
# On Slurm there is no need to enumerate hosts: a single sbatch allocation
# reserves a set of GPUs/CPUs, and this library packs many *individual*
# experiments into that allocation as concurrent `srun --exclusive` job
# steps (never `sbatch` per experiment — that would create thousands of
# separate Slurm jobs and defeat scheduling efficiency).
#
# Usage (from a script submitted with sbatch):
#   source "$(dirname "${BASH_SOURCE[0]}")/slurm_lib.sh"
#   JOBS=("python run_experiment.py path/to/config|safe_name" ...)
#   run_job_pool_srun JOBS PHASE_NAME
#
# Each entry of the job array is "COMMAND|SAFE_NAME", exactly like in
# orchestrate_runs.sh, so job-building loops can be copy-pasted unchanged.
###

set -u

# ---------------------------------------------------------------------------
# Per-task resource placeholders — ADAPT THESE to the target cluster/queue.
# They configure the resources requested by *each* srun step (i.e. each
# individual experiment), not the overall sbatch allocation (see the
# #SBATCH header of the calling script for that).
# ---------------------------------------------------------------------------
GPUS_PER_TASK="${GPUS_PER_TASK:-1}"          # TODO: GPUs per experiment
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"          # TODO: CPUs per experiment
MEM_PER_TASK="${MEM_PER_TASK:-16G}"          # TODO: RAM per experiment
SLURM_CONSTRAINT_TASK="${SLURM_CONSTRAINT_TASK:-}"  # TODO: e.g. "a100", leave empty if unused

# Number of experiments to run concurrently inside the allocation. This is
# the Slurm equivalent of N_MACHINES in orchestrate_runs.sh. It must not
# exceed the number of GPUs_PER_TASK-sized slots available in the sbatch
# allocation requested by the parent script, or steps will simply queue and
# wait for a free slot (which is safe, just less parallel than intended).
N_PARALLEL_TASKS="${N_PARALLEL_TASKS:-4}"    # TODO: set to (total GPUs allocated / GPUS_PER_TASK)

# Where per-job logs and completion markers are written. Kept separate from
# the legacy $LOG_DIR used by the SSH-based scripts so both infrastructures
# can be operated side by side without clobbering each other's logs.
LOG_DIR="${LOG_DIR:-$PWD/logs_slurm}"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# run_job_pool_srun JOBS_ARRAY_NAME PHASE_NAME
#
# Consumes an array of "COMMAND|SAFE_NAME" strings and runs them as
# concurrent srun job steps, keeping up to N_PARALLEL_TASKS steps in flight
# at all times (packing), then waits for all of them to complete before
# returning. Mirrors the semantics of run_job_pool() in orchestrate_runs.sh.
# ---------------------------------------------------------------------------
run_job_pool_srun() {
    local -n JOBS=$1
    local PHASE=$2

    local TOTAL=${#JOBS[@]}
    local INDEX=0

    echo "[POOL $PHASE] total jobs = $TOTAL, parallel slots = $N_PARALLEL_TASKS"

    local -a extra_srun_args=()
    if [ -n "$SLURM_CONSTRAINT_TASK" ]; then
        extra_srun_args+=(--constraint="$SLURM_CONSTRAINT_TASK")
    fi

    while (( INDEX < TOTAL )); do
        # -------------------------------------------------------------
        # Wait for a free execution slot (bounded parallelism)
        # -------------------------------------------------------------
        while (( $(jobs -rp | wc -l) >= N_PARALLEL_TASKS )); do
            wait -n || true
        done

        local job="${JOBS[$INDEX]}"
        local cmd safe_name
        IFS='|' read -r cmd safe_name <<< "$job"

        local done_file="$LOG_DIR/${safe_name}.done"
        local log_file="$LOG_DIR/${safe_name}.log"
        rm -f "$done_file"

        echo "[POOL $PHASE] ($((INDEX + 1))/$TOTAL) launching: $safe_name"

        # `--exclusive` here means "this step does not share the resources
        # it is granted with other concurrent steps of the same job", which
        # is what allows many srun steps to run in parallel inside a single
        # sbatch allocation (job packing).
        srun --exclusive -N1 -n1 \
             --cpus-per-task="$CPUS_PER_TASK" \
             --gres="gpu:$GPUS_PER_TASK" \
             --mem="$MEM_PER_TASK" \
             ${extra_srun_args[@]+"${extra_srun_args[@]}"} \
             bash -c "$cmd && touch '$done_file'" > "$log_file" 2>&1 &

        INDEX=$((INDEX + 1))
    done

    # Drain remaining running steps
    wait

    echo "[POOL $PHASE] completed"
}
