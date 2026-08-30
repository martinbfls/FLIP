#!/bin/bash
###
# run_experiment_slurm.sh
#
# Minimal Slurm entry point for a single FLIP/BRoADflip experiment, using
# `srun` for the actual task execution (never `sbatch` per experiment).
# Useful either standalone for one-off/debug runs, or as the basic
# building block that orchestrate_runs_slurm.sh packs many copies of.
#
# ---------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------
# Single run:
#   sbatch --partition=<PARTITION> --gres=gpu:<GPUS_PER_TASK> \
#          --cpus-per-task=<CPUS_PER_TASK> --mem=<MEM_PER_TASK> \
#          --time=<TIME_LIMIT> \
#          run_experiment_slurm.sh <experiment_config_path>
#
# Repeated runs (e.g. one per seed), reusing the SLURM_ARRAY_TASK_ID
# support already built into run_experiment.py / slurmify_path(): any
# "{}" in the config's path-valued fields is substituted with the array
# index, so the same config can fan out to N independent result dirs.
#   sbatch --array=1-<N_REPEATS> --partition=<PARTITION> \
#          --gres=gpu:<GPUS_PER_TASK> --cpus-per-task=<CPUS_PER_TASK> \
#          --mem=<MEM_PER_TASK> --time=<TIME_LIMIT> \
#          run_experiment_slurm.sh <experiment_config_path>
# ---------------------------------------------------------------------

# #SBATCH --job-name=flip_experiment
# #SBATCH --partition=<PARTITION>
# #SBATCH --gres=gpu:<GPUS_PER_TASK>
# #SBATCH --cpus-per-task=<CPUS_PER_TASK>
# #SBATCH --mem=<MEM_PER_TASK>
# #SBATCH --time=<TIME_LIMIT>
# #SBATCH --constraint=<CONSTRAINT>
# #SBATCH --output=logs_slurm/%x-%A_%a.out

set -e

BASE_DIR="$HOME/FLIP"
cd "$BASE_DIR" || exit 1

CONFIG="${1:?Usage: sbatch run_experiment_slurm.sh <experiment_config_path>}"

srun python run_experiment.py "$CONFIG"
