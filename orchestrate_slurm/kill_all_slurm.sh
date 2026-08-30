#!/bin/bash
###
# kill_all_slurm.sh
#
# Slurm counterpart to kill_all.sh. Instead of SSH-ing into each machine to
# pkill stray "python run_experiment.py" processes, it cancels the current
# user's queued/running Slurm jobs whose job name matches a filter (default
# "flip"), via scancel. Scoped to $USER and a name filter rather than a
# blanket cancellation, to avoid touching unrelated jobs on a shared
# cluster.
#
# Usage:
#   ./kill_all_slurm.sh              # cancels jobs with "flip" in the name
#   ./kill_all_slurm.sh my_job_name  # cancels jobs matching a custom filter
###

set -u

FILTER="${1:-flip}"

echo "Looking up Slurm jobs for user '$USER' matching name filter '$FILTER'..."

JOB_IDS=$(squeue -u "$USER" -h -o "%i %j" | awk -v f="$FILTER" '$2 ~ f {print $1}')

if [ -z "$JOB_IDS" ]; then
    echo "No matching jobs found."
    exit 0
fi

for jobid in $JOB_IDS; do
    echo "Cancelling job $jobid"
    scancel "$jobid"
done

echo "Kill done."
