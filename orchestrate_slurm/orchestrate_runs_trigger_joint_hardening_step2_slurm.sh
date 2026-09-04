#!/bin/bash
###
# orchestrate_runs_trigger_joint_hardening_step2_slurm.sh
#
# Thin wrapper: sets STEP=2 and execs the shared
# orchestrate_runs_trigger_joint_hardening_slurm.sh (see that script's own docstring for the
# full phase pipeline and usage). Generate this step's configs first:
#   python -m modules.federated_generate_labels_trigger_joint.gen_configs_hardening_steps --step 2
###
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP=2 exec "$SCRIPT_DIR/orchestrate_runs_trigger_joint_hardening_slurm.sh" "$@"
