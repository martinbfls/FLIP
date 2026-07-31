#!/bin/bash
###
# orchestrate_runs_slurm.sh
#
# Slurm counterpart to orchestrate_runs.sh: runs the exact same gen_labels
# -> train_user campaign, but instead of round-robining jobs over a
# hardcoded pool of SSH machines, it packs them as concurrent `srun`
# job steps inside a single Slurm allocation (see slurm_lib.sh).
#
# The old orchestrate_runs.sh (and its bis/ter/trigger variants) are left
# untouched and remain fully usable on the legacy SSH-pool infrastructure.
#
# ---------------------------------------------------------------------
# SUBMISSION
# ---------------------------------------------------------------------
# All #SBATCH directives below are placeholders (commented out) so this
# script does not silently submit with made-up resource values. Either:
#
#   (a) uncomment and edit the #SBATCH lines below, then:
#         sbatch orchestrate_runs_slurm.sh
#
#   (b) or leave them commented and pass resources on the command line:
#         sbatch --job-name=flip_gen_train \
#                --partition=<PARTITION> \
#                --nodes=<N_NODES> \
#                --gres=gpu:<GPUS_PER_NODE> \
#                --cpus-per-task=<CPUS_PER_NODE> \
#                --mem=<MEM_PER_NODE> \
#                --time=<TIME_LIMIT> \
#                orchestrate_runs_slurm.sh
#
# Whichever GPU total you request (nodes * gpus-per-node), set
# N_PARALLEL_TASKS below to (that total / GPUS_PER_TASK) so the pool
# executor uses all of it.
# ---------------------------------------------------------------------

# #SBATCH --job-name=flip_gen_train
# #SBATCH --partition=<PARTITION>
# #SBATCH --nodes=<N_NODES>
# #SBATCH --gres=gpu:<GPUS_PER_NODE>
# #SBATCH --cpus-per-task=<CPUS_PER_NODE>
# #SBATCH --mem=<MEM_PER_NODE>
# #SBATCH --time=<TIME_LIMIT>
# #SBATCH --constraint=<CONSTRAINT>
# #SBATCH --output=logs_slurm/%x-%j.out

set -x
# NOTE: no `set -e` on purpose (same reasoning as orchestrate_runs_bis.sh):
# one failed experiment must not tear down the whole allocation/pool.

BASE_DIR="$HOME/FLIP"
cd "$BASE_DIR" || exit 1

mkdir -p "$BASE_DIR/logs_slurm"

source "$BASE_DIR/slurm_lib.sh"

# ---------------------------------------------------------------------
# Per-task resource placeholders (see slurm_lib.sh for details).
# Fill these in to match what you requested via sbatch/#SBATCH above.
# ---------------------------------------------------------------------
GPUS_PER_TASK="${GPUS_PER_TASK:-1}"       # TODO
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"       # TODO
MEM_PER_TASK="${MEM_PER_TASK:-16G}"       # TODO
N_PARALLEL_TASKS="${N_PARALLEL_TASKS:-4}" # TODO: total GPUs allocated / GPUS_PER_TASK

# ==========================================================
# EXPERIMENT GRID (identical to orchestrate_runs.sh)
# ==========================================================

DATASET="cifar"
ATTACK="backdoor"

AGGREGATORS=("mean" "median" "krum" "trmean" "multikrum")
POISONERS=("1xs")

BUDGETS=(150 300 500 1000 1500 2000 2500 5000)
N_CYCLES=10

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
echo "GEN_LABELS SLURM POOL"
echo "=============================="

run_job_pool_srun GEN_JOBS GEN

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
echo "TRAIN_USER SLURM POOL"
echo "=============================="

run_job_pool_srun TRAIN_JOBS TRAIN

echo "=============================="
echo "ALL DONE"
echo "=============================="
