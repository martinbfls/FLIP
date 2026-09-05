#!/bin/bash
###
# run_trigger_joint_old_objective_port_slurm.sh
#
# Minimal test: port of the old federated_optimizing_trigger objective (trigger_penalty,
# cos(delta, mu_target)+1, reintroduced in modules/federated_optimizing_trigger/utils.py) into
# federated_generate_labels_trigger_joint, under the exact config that broke every robust
# aggregator in the old module (num_honests=0, num_poisoned=1, agg_method="mean"). See
# experiments/federated_experiments/threat_model_direct_trigger_joint_old_objective_port/ for
# the configs this chains.
#
# Single sequential job (`srun` per step, same convention as run_experiment_slurm.sh) -- this
# is a one-off test, not a campaign, so no job-pool/barrier machinery.
#
# Usage:
#   sbatch --partition=<PARTITION> --gres=gpu:1 --cpus-per-task=<CPUS_PER_TASK> \
#          --mem=<MEM_PER_TASK> --time=<TIME_LIMIT> \
#          run_trigger_joint_old_objective_port_slurm.sh
###

# #SBATCH --job-name=trigger_joint_old_objective_port
# #SBATCH --partition=<PARTITION>
# #SBATCH --gres=gpu:1
# #SBATCH --cpus-per-task=<CPUS_PER_TASK>
# #SBATCH --mem=<MEM_PER_TASK>
# #SBATCH --time=<TIME_LIMIT>
# #SBATCH --output=logs_slurm/%x-%j.out

set -e

BASE_DIR="$HOME/FLIP"
cd "$BASE_DIR" || exit 1

EXP_BASE="experiments/federated_experiments/threat_model_direct_trigger_joint_old_objective_port/r32p/cifar"

srun python run_experiment.py "$EXP_BASE/gen_labels_trigger_joint"
srun python run_experiment.py "$EXP_BASE/select_flips"
srun python run_experiment.py "$EXP_BASE/mean/train_user_1500"
srun python run_experiment.py "$EXP_BASE/trmean/train_user_1500"
