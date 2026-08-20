#!/bin/bash
###
# orchestrate_runs_slurm.sh
#
# Slurm counterpart to orchestrate_runs.sh: runs the exact same gen_labels
# -> train_user campaign, but instead of round-robining jobs over a
# hardcoded pool of SSH machines, it submits every experiment as its own
# independent Slurm job (see slurm_lib.sh).
#
# The old orchestrate_runs.sh (and its bis/ter/trigger variants) are left
# untouched and remain fully usable on the legacy SSH-pool infrastructure.
#
# ---------------------------------------------------------------------
# SUBMISSION
# ---------------------------------------------------------------------
# This script is NOT submitted with sbatch any more: it is a *submitter*,
# run directly from a login-node shell. It returns as soon as all jobs are
# queued.
#
#     bash orchestrate_runs_slurm.sh
#
# Each experiment becomes one sbatch job asking for 1 GPU / 16 CPU / 32G /
# 24h on cypress_dgx, so Slurm itself schedules up to 8 concurrent runs on
# the 8 A100s of dgx-n01, and each run gets its own 24h walltime regardless
# of how long the whole campaign takes.
#
# Resources and the conda env are configured at the top of slurm_lib.sh and
# can be overridden from the environment, e.g.:
#     CPUS_PER_TASK=12 bash orchestrate_runs_slurm.sh
# ---------------------------------------------------------------------

# NOTE: no `set -e` on purpose (same reasoning as orchestrate_runs_bis.sh):
# one failed submission must not tear down the whole campaign.

# Default to the directory this script lives in, so the repo can sit anywhere
# (it is under /shared, not $HOME) and the script works from any cwd.
BASE_DIR="${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$BASE_DIR" || exit 1
export BASE_DIR

mkdir -p "$BASE_DIR/logs_slurm"

source "$BASE_DIR/slurm_lib.sh"

# Fail before anything is queued if the environment cannot support the plan.
preflight_slurm || { echo "[ABORT] preflight failed, nothing submitted"; exit 1; }

# ==========================================================
# EXPERIMENT GRID (identical to orchestrate_runs.sh)
# ==========================================================

DATASET="cifar"
ATTACK="backdoor"

# Space-separated overrides, e.g.:
#   AGGREGATORS=multikrum POISONERS=optimized bash orchestrate_runs_slurm.sh
read -ra AGGREGATORS <<< "${AGGREGATORS:-mean median krum trmean multikrum}"
read -ra POISONERS   <<< "${POISONERS:-1xs}"

BUDGETS=(150 300 500 1000 1500 2000 2500 5000)
N_CYCLES="${N_CYCLES:-10}"

NUM_CLEAN=7
NUM_POISONED=3
MODEL_FLAG="r32p"

# ==========================================================
# BUILD GEN JOBS (GLOBAL)
# ==========================================================

GEN_JOBS=()

for poisoner in "${POISONERS[@]}"; do
for aggregator in "${AGGREGATORS[@]}"; do
for ((run_id=1; run_id<=N_CYCLES; run_id++)); do

config="federated_experiments/${MODEL_FLAG}/${NUM_POISONED}vs${NUM_CLEAN}/${DATASET}/${ATTACK}/${aggregator}/${poisoner}/gen_labels/${run_id}"

safe="gen_${MODEL_FLAG}_${NUM_POISONED}vs${NUM_CLEAN}_${DATASET}_${ATTACK}_${aggregator}_${poisoner}_${run_id}"

cmd="python run_experiment.py $config"

GEN_JOBS+=("$cmd|$safe")

done
done
done

echo "=============================="
echo "GEN_LABELS SLURM SUBMISSION"
echo "=============================="

EXPECTED_GEN=${#GEN_JOBS[@]}

submit_job_pool_slurm GEN_JOBS GEN
GEN_IDS=("${SUBMITTED_JOB_IDS[@]}")

# The barrier must cover EVERY GEN job. A partially submitted phase would
# produce a barrier on a subset, and TRAIN jobs would start on incomplete
# labels -- so this is a hard abort, not a warning.
if [ ${#GEN_IDS[@]} -ne "$EXPECTED_GEN" ]; then
    echo "[ABORT] only ${#GEN_IDS[@]}/$EXPECTED_GEN GEN jobs were submitted."
    echo "[ABORT] no barrier, no TRAIN. Cancel the partial phase with:"
    echo "        scancel \$(awk '{print \$1}' logs_slurm/jobids_GEN.txt)"
    exit 1
fi

# ==========================================================
# GEN -> TRAIN BARRIER
# ==========================================================
# One tiny CPU job depending on afterok of every GEN job. TRAIN jobs then
# depend on this single barrier instead of carrying a 50-id dependency list.
# If any GEN job fails, the barrier's dependency can never be satisfied:
# the barrier is cancelled, and the TRAIN jobs depending on it are cancelled
# too (--kill-on-invalid-dep). Nothing trains on missing labels.

BARRIER_ID=$(submit_barrier_slurm "flip_barrier_gen" "afterok:$(join_job_ids "${GEN_IDS[@]}")")

if [ -z "$BARRIER_ID" ]; then
    echo "[ERROR] barrier job submission failed, aborting before TRAIN"
    exit 1
fi

echo "[BARRIER] job $BARRIER_ID waits for ${#GEN_IDS[@]} GEN jobs"

# ==========================================================
# BUILD TRAIN JOBS (GLOBAL)
# ==========================================================

TRAIN_JOBS=()

for poisoner in "${POISONERS[@]}"; do
for aggregator in "${AGGREGATORS[@]}"; do
for ((run_id=1; run_id<=N_CYCLES; run_id++)); do
for budget in "${BUDGETS[@]}"; do

config="federated_experiments/${MODEL_FLAG}/${NUM_POISONED}vs${NUM_CLEAN}/${DATASET}/${ATTACK}/${aggregator}/${poisoner}/train_user_${budget}/${run_id}"

safe="train_${MODEL_FLAG}_${NUM_POISONED}vs${NUM_CLEAN}_${DATASET}_${ATTACK}_${aggregator}_${poisoner}_${budget}_${run_id}"

cmd="python run_experiment.py $config"

TRAIN_JOBS+=("$cmd|$safe")

done
done
done
done

echo "=============================="
echo "TRAIN_USER SLURM SUBMISSION"
echo "=============================="

EXPECTED_TRAIN=${#TRAIN_JOBS[@]}

if ! submit_job_pool_slurm TRAIN_JOBS TRAIN "afterok:${BARRIER_ID}"; then
    echo "[WARNING] only ${#SUBMITTED_JOB_IDS[@]}/$EXPECTED_TRAIN TRAIN jobs were submitted."
    echo "[WARNING] GEN and the barrier are queued and safe; resubmit the missing"
    echo "[WARNING] TRAIN jobs against barrier $BARRIER_ID once the cause is fixed."
fi

echo "=============================="
echo "ALL SUBMITTED"
echo "  GEN     : ${#GEN_IDS[@]} jobs   (ids in logs_slurm/jobids_GEN.txt)"
echo "  BARRIER : $BARRIER_ID"
echo "  TRAIN   : ${#SUBMITTED_JOB_IDS[@]} jobs  (ids in logs_slurm/jobids_TRAIN.txt)"
echo ""
echo "  squeue -u \$USER"
echo "  scancel \$(awk '{print \$1}' logs_slurm/jobids_*.txt)   # cancel everything"
echo "=============================="