#!/bin/bash
###
# orchestrate_runs_trigger_joint_hardening_all_slurm.sh
#
# Thin wrapper: sets STEPS="1 2 3 4" and execs the shared
# orchestrate_runs_trigger_joint_hardening_slurm.sh -- submits EVERY step of the P0-P7
# hardening protocol in one go ("lancer toutes les experiences d'un coup pour la nuit"). See
# that script's own docstring for the full phase pipeline: train_expert is submitted ONCE
# (shared by all 4 steps, not once per step -- see its "Multi-step submission" section for why
# that matters), and each step's own GEN -> FLIPS -> USER pipeline then runs independently and
# CONCURRENTLY once the shared expert checkpoint is ready, bounded only by how many GPUs the
# Slurm partition can give this account at once.
#
# Generate every step's configs first (each step's own defaults -- P0+P1, P2+P6, P5+match,
# P4+P3 -- see gen_configs_hardening_steps.py's STEP_OVERRIDES):
#   for s in 1 2 3 4; do
#       python -m modules.federated_generate_labels_trigger_joint.gen_configs_hardening_steps --step $s
#   done
#
# Usage:
#   SLURM_ACCOUNT=<account> ./orchestrate_runs_trigger_joint_hardening_all_slurm.sh
#   SLURM_ACCOUNT=<account> DRY_RUN=1 ./orchestrate_runs_trigger_joint_hardening_all_slurm.sh
###
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEPS="1 2 3 4" exec "$SCRIPT_DIR/orchestrate_runs_trigger_joint_hardening_slurm.sh" "$@"
