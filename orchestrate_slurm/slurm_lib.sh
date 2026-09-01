#!/bin/bash
###
# slurm_lib.sh
#
# Shared bash library for launching FLIP/BRoADflip experiments on a Slurm
# cluster. This is the Slurm counterpart to the job-pool executor found in
# orchestrate_runs.sh (and its bis/ter/trigger variants), which assigns one
# job at a time to each entry of a hardcoded SSH "MACHINES" pool.
#
# On Slurm there is no need to enumerate hosts *and* no need to run our own
# pool executor: each experiment is submitted as its own `sbatch` job asking
# for exactly one GPU, and the Slurm scheduler is what keeps up to
# (GPUs on the partition) experiments running concurrently. This replaces
# the previous run_job_pool_srun() approach, where a single sbatch allocation
# held all the GPUs and every experiment was an `srun --exclusive` step inside
# it -- which capped the whole campaign at the allocation's walltime.
#
# Usage (from a plain login-node shell, NOT from inside sbatch):
#   source "$(dirname "${BASH_SOURCE[0]}")/slurm_lib.sh"
#   preflight_slurm || exit 1
#   JOBS=("python run_experiment.py path/to/config|safe_name" ...)
#   submit_job_pool_slurm JOBS PHASE_NAME [DEPENDENCY_SPEC]
#   # -> job IDs land in the global array SUBMITTED_JOB_IDS
#
# Each entry of the job array is "COMMAND|SAFE_NAME", exactly like in
# orchestrate_runs.sh, so job-building loops can be copy-pasted unchanged.
###

set -u

# ---------------------------------------------------------------------------
# Per-job resources. These are what *one experiment* asks for; there is no
# parent allocation any more, so each of these values is what Slurm sees.
# Tuned for the cypress_dgx partition (single node dgx-n01: 8xA100, 128 CPU,
# ~2 TB RAM, MaxTime=1-00:00:00).
# ---------------------------------------------------------------------------
SLURM_PARTITION="${SLURM_PARTITION:-rtx6k}"

# Space-separated list of partitions to spread GPU jobs across (round-robin
# in submit_job_pool_slurm), to maximize the number of GPUs available to a
# campaign at once when the same jobs are eligible on more than one
# partition. Defaults to just SLURM_PARTITION, so single-partition behavior
# is unchanged unless SLURM_PARTITIONS is set explicitly, e.g.:
#   SLURM_PARTITIONS="cypress_dgx other_dgx_partition" ./orchestrate_...sh
SLURM_PARTITIONS="${SLURM_PARTITIONS:-$SLURM_PARTITION}"
read -ra _SLURM_PARTITIONS_ARR <<< "$SLURM_PARTITIONS"

# R&D Line / account required by this cluster. Run `slurmaccounts` to list
# the accounts you may charge jobs to. Nothing can be submitted without it.
SLURM_ACCOUNT="${SLURM_ACCOUNT:-power}"
GPUS_PER_TASK="${GPUS_PER_TASK:-1}"          # GPUs per experiment
# 8 x 16 = 128 = the whole node, leaving no room for the barrier or for
# other users' jobs. 14 keeps 8 concurrent runs achievable (8 x 14 = 112).
CPUS_PER_TASK="${CPUS_PER_TASK:-14}"         # CPUs per experiment
MEM_PER_TASK="${MEM_PER_TASK:-32G}"          # RAM per experiment
TIME_PER_TASK="${TIME_PER_TASK:-1-00:00:00}" # walltime per experiment

# Partition used for the tiny CPU-only barrier job. Defaults to the first
# entry of SLURM_PARTITIONS; override if a general-purpose CPU partition
# exists.
BARRIER_PARTITION="${BARRIER_PARTITION:-${_SLURM_PARTITIONS_ARR[0]}}"

# Set to 0 if this Slurm build does not support --kill-on-invalid-dep.
# With 0, jobs whose dependency can never be satisfied stay PENDING with
# reason DependencyNeverSatisfied instead of being cancelled: still safe,
# but they must be scancel'ed by hand.
KILL_ON_INVALID_DEP="${KILL_ON_INVALID_DEP:-1}"

# ---------------------------------------------------------------------------
# Python environment. sbatch jobs do not inherit an interactive shell state,
# so the conda env is activated explicitly inside every job script, and the
# resulting interpreter is asserted before the experiment runs.
# ---------------------------------------------------------------------------
# Module providing conda ON THE COMPUTE NODE (dgx-n01 exposes a different
# Lmod catalogue than the login node). Leave EMPTY to skip `module load`
# entirely and rely on CONDA_BASE / the env's own interpreter.
CONDA_MODULE="${CONDA_MODULE:-}"
# Absolute path of the conda installation (the output of `conda info --base`).
# Used when no module provides conda. Leave empty to skip conda activation.
CONDA_BASE="${CONDA_BASE:-}"
CONDA_ENV="${CONDA_ENV:-/shared/data1/Projects/DLWP/j1067582/martin/FLIP/envs/flip_x86_64}"

# Working directory the jobs cd into (the caller normally exports this).
BASE_DIR="${BASE_DIR:-$PWD}"

# Where per-job logs, completion markers and submitted job-id lists are
# written. Kept separate from the legacy $LOG_DIR used by the SSH-based
# scripts so both infrastructures can be operated side by side.
LOG_DIR="${LOG_DIR:-$PWD/logs_slurm}"
mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# preflight_slurm
#
# Cheap checks run ONCE before anything is submitted, so that an unusable
# option or a wrong partition name is discovered before 50 GPU jobs are
# already queued. Returns non-zero on a blocking problem.
# ---------------------------------------------------------------------------
preflight_slurm() {
    local rc=0

    if ! command -v sbatch >/dev/null 2>&1; then
        echo "[PREFLIGHT] sbatch not found in PATH" >&2
        return 1
    fi

    if [ -z "$SLURM_ACCOUNT" ]; then
        echo "[PREFLIGHT] SLURM_ACCOUNT is empty; this cluster requires --account" >&2
        echo "[PREFLIGHT] run 'slurmaccounts' and re-run with SLURM_ACCOUNT=<account>" >&2
        return 1
    fi

    local partition
    for partition in "${_SLURM_PARTITIONS_ARR[@]}"; do
        if ! sinfo -h -p "$partition" -o "%P" 2>/dev/null | grep -q .; then
            echo "[PREFLIGHT] partition '$partition' unknown or unreachable" >&2
            rc=1
        fi
    done

    # Trial submission per partition (--test-only validates and submits
    # nothing). Catches a bad account, an unknown QOS, an association limit,
    # or a partition this account can't use, before 450 real submissions hit
    # the same wall. The dependency flag is included so the exact command
    # shape used later is what gets validated -- and so is the resource
    # request (--gres/--cpus-per-task/--mem): a bare `--wrap=true` job with
    # no resources requested is a DIFFERENT allocation shape from the real
    # per-job submissions in submit_job_slurm, and some partitions reject one
    # shape while happily accepting the other (e.g. GPU-only nodes that
    # cannot schedule a 0-GPU job at all) -- probing the real shape avoids a
    # false negative here (or worse, a false pass that only fails once 50
    # real jobs are already queued).
    local probe probe_rc
    for partition in "${_SLURM_PARTITIONS_ARR[@]}"; do
        probe=$(sbatch --kill-on-invalid-dep=yes --test-only \
                       --account="$SLURM_ACCOUNT" \
                       --partition="$partition" \
                       --gres="gpu:$GPUS_PER_TASK" \
                       --cpus-per-task="$CPUS_PER_TASK" \
                       --mem="$MEM_PER_TASK" \
                       --time=00:01:00 \
                       --wrap=true 2>&1)
        probe_rc=$?

        if [ $probe_rc -ne 0 ]; then
            # Any other rejection (bad account, unknown QOS, limits...) would
            # hit all real submissions to this partition. Surface sbatch's
            # own message and stop here.
            echo "[PREFLIGHT] a trial submission to partition '$partition' was rejected by Slurm:" >&2
            echo "$probe" | sed 's/^/[PREFLIGHT]   /' >&2
            rc=1
        fi
    done

    if [ ! -x "$CONDA_ENV/bin/python" ]; then
        echo "[PREFLIGHT] warning: $CONDA_ENV/bin/python not visible from this host" >&2
        echo "[PREFLIGHT] (harmless if /shared is only mounted on dgx-n01)" >&2
    fi

    return $rc
}

# ---------------------------------------------------------------------------
# join_job_ids ID...
#
# Prints the ids joined by ':', the format expected by --dependency.
# ---------------------------------------------------------------------------
join_job_ids() {
    local IFS=':'
    echo "$*"
}

# ---------------------------------------------------------------------------
# submit_job_slurm COMMAND SAFE_NAME [DEPENDENCY_SPEC] [PARTITION]
#
# Submits ONE experiment as an independent Slurm job and prints its JobID on
# stdout (everything else goes to stderr, so the caller can capture the id).
# Returns non-zero if sbatch failed OR if the returned id is not a plain
# integer -- a malformed id must never reach a --dependency string.
#
# DEPENDENCY_SPEC is passed as-is to --dependency (e.g. "afterok:12345").
# PARTITION defaults to $SLURM_PARTITION when omitted, for callers that
# don't care about spreading jobs across SLURM_PARTITIONS themselves.
# ---------------------------------------------------------------------------
submit_job_slurm() {
    local cmd="$1"
    local safe_name="$2"
    local dependency="${3:-}"
    local partition="${4:-$SLURM_PARTITION}"

    local log_file="$LOG_DIR/${safe_name}.log"
    local done_file="$LOG_DIR/${safe_name}.done"
    rm -f "$done_file"

    local -a dep_args=()
    if [ -n "$dependency" ]; then
        dep_args+=(--dependency="$dependency")
        # If the dependency can never be satisfied (an upstream job failed),
        # cancel the job instead of leaving it queued forever. Failure is
        # then explicit, and no experiment ever starts on missing data.
        [ "$KILL_ON_INVALID_DEP" = "1" ] && dep_args+=(--kill-on-invalid-dep=yes)
    fi

    local jobid
    jobid=$(sbatch --parsable \
           --job-name="$safe_name" \
           --account="$SLURM_ACCOUNT" \
           --partition="$partition" \
           --gres="gpu:$GPUS_PER_TASK" \
           --ntasks=1 \
           --cpus-per-task="$CPUS_PER_TASK" \
           --mem="$MEM_PER_TASK" \
           --time="$TIME_PER_TASK" \
           --output="$log_file" \
           --error="$log_file" \
           ${dep_args[@]+"${dep_args[@]}"} <<EOF
#!/bin/bash
set -euo pipefail

cd "$BASE_DIR"

# Do not let the submitting shell's Python state leak into the job.
unset PYTHONPATH PYTHONHOME || true

# Optional: load the module providing conda on the compute node.
if [ -n "$CONDA_MODULE" ]; then
    if ! command -v module >/dev/null 2>&1 && [ -f /etc/profile.d/modules.sh ]; then
        source /etc/profile.d/modules.sh
    fi
    module load $CONDA_MODULE
fi

# Activate the env if a conda installation can be located. Activation matters
# for envs shipping etc/conda/activate.d hooks (MKL/CUDA library paths); when
# no conda is reachable we fall back to the env's interpreter, and the
# assertion below still guarantees it is the right one.
CONDA_SH=""
if [ -n "${CONDA_BASE}" ] && [ -d "${CONDA_BASE}" ]; then
    CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
    CONDA_SH="\$(conda info --base)/etc/profile.d/conda.sh"
fi
if [ -n "\$CONDA_SH" ] && [ -f "\$CONDA_SH" ]; then
    source "\$CONDA_SH"
    conda activate "$CONDA_ENV"
else
    echo "[slurm] no conda.sh reachable; using the env interpreter directly"
fi

# Belt and braces: do not depend on what 'conda activate' did to PATH.
export PATH="$CONDA_ENV/bin:\$PATH"

# Fail loudly here rather than silently running the wrong interpreter.
if [ "\$(command -v python)" != "$CONDA_ENV/bin/python" ]; then
    echo "[slurm] FATAL: python is \$(command -v python), expected $CONDA_ENV/bin/python" >&2
    exit 1
fi

echo "[slurm] job=\$SLURM_JOB_ID node=\$(hostname) arch=\$(uname -m)"
echo "[slurm] gpus=\${CUDA_VISIBLE_DEVICES:-none} cpus=\${SLURM_CPUS_PER_TASK:-?} python=\$(command -v python)"

$cmd

touch "$done_file"
EOF
    ) || return 1

    # --parsable prints "<jobid>" (or "<jobid>;<cluster>" on a federation).
    jobid="${jobid%%;*}"
    case "$jobid" in
        ''|*[!0-9]*) echo "[SUBMIT] unexpected sbatch output for $safe_name: '$jobid'" >&2; return 1 ;;
    esac

    echo "$jobid"
}

# ---------------------------------------------------------------------------
# submit_barrier_slurm NAME DEPENDENCY_SPEC
#
# Submits a tiny CPU-only job that does nothing but wait for DEPENDENCY_SPEC.
# Used as a single synchronisation point so that the 400 TRAIN jobs each carry
# one dependency instead of a list of 50 GEN job ids.
# ---------------------------------------------------------------------------
submit_barrier_slurm() {
    local name="$1"
    local dependency="$2"

    local -a dep_args=(--dependency="$dependency")
    [ "$KILL_ON_INVALID_DEP" = "1" ] && dep_args+=(--kill-on-invalid-dep=yes)

    local jobid
    jobid=$(sbatch --parsable \
           --job-name="$name" \
           --account="$SLURM_ACCOUNT" \
           --partition="$BARRIER_PARTITION" \
           --ntasks=1 \
           --cpus-per-task=1 \
           --mem=1G \
           --time=00:05:00 \
           "${dep_args[@]}" \
           --output="$LOG_DIR/${name}.log" \
           --wrap="echo '[barrier] all upstream jobs completed successfully'") || return 1

    jobid="${jobid%%;*}"
    case "$jobid" in
        ''|*[!0-9]*) echo "[SUBMIT] unexpected sbatch output for barrier: '$jobid'" >&2; return 1 ;;
    esac

    echo "$jobid"
}

# ---------------------------------------------------------------------------
# submit_job_pool_slurm JOBS_ARRAY_NAME PHASE_NAME [DEPENDENCY_SPEC]
#
# Consumes an array of "COMMAND|SAFE_NAME" strings and submits each entry as
# an independent Slurm job. Same call shape as the old run_job_pool_srun(),
# but it returns as soon as everything is submitted: it does not wait for the
# jobs to run. Submitted ids are left in the global array SUBMITTED_JOB_IDS
# and also written to $LOG_DIR/jobids_PHASE.txt.
#
# Jobs are spread round-robin across SLURM_PARTITIONS (a single partition,
# i.e. the old behavior, when it wasn't set), so a campaign can use more
# than one partition's GPUs concurrently instead of queueing on just one.
#
# Returns non-zero if any submission failed, so the caller can abort before
# building a dependency on an incomplete phase.
# ---------------------------------------------------------------------------
submit_job_pool_slurm() {
    local -n JOBS=$1
    local PHASE=$2
    local DEPENDENCY="${3:-}"

    local -a PARTS
    read -ra PARTS <<< "$SLURM_PARTITIONS"
    local NPARTS=${#PARTS[@]}

    local TOTAL=${#JOBS[@]}
    local INDEX=0
    local FAILED=0

    SUBMITTED_JOB_IDS=()

    local id_file="$LOG_DIR/jobids_${PHASE}.txt"
    : > "$id_file"

    echo "[SUBMIT $PHASE] total jobs = $TOTAL, dependency = ${DEPENDENCY:-none}, partitions = ${PARTS[*]}" >&2

    while (( INDEX < TOTAL )); do
        local job="${JOBS[$INDEX]}"
        local cmd safe_name jobid
        IFS='|' read -r cmd safe_name <<< "$job"
        local partition="${PARTS[$((INDEX % NPARTS))]}"

        if jobid=$(submit_job_slurm "$cmd" "$safe_name" "$DEPENDENCY" "$partition"); then
            SUBMITTED_JOB_IDS+=("$jobid")
            echo "$jobid $safe_name" >> "$id_file"
            echo "[SUBMIT $PHASE] ($((INDEX + 1))/$TOTAL) $safe_name -> job $jobid (partition=$partition)" >&2
        else
            FAILED=$((FAILED + 1))
            echo "[SUBMIT $PHASE] ($((INDEX + 1))/$TOTAL) FAILED to submit: $safe_name" >&2
        fi

        INDEX=$((INDEX + 1))
    done

    echo "[SUBMIT $PHASE] submitted ${#SUBMITTED_JOB_IDS[@]}/$TOTAL jobs (ids in $id_file)" >&2

    if (( FAILED > 0 )); then
        echo "[SUBMIT $PHASE] $FAILED submission(s) failed" >&2
        return 1
    fi
    return 0
}