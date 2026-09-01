#!/bin/bash
###
# orchestrate_runs_trigger_joint_axis_sweep_slurm.sh
#
# Single entry point for the three federated_generate_labels_trigger_joint proof-of-concept axis
# comparisons requested together (1-poisoned/0-honest, "mean" aggregation, loosely-constrained
# trigger -- delta_min_frac=0.0, epsilon=1.0, lambda_align=0.0, all gen_configs.py's own current
# defaults):
#
#   - init:                    "stripe" (baseline) vs "random"
#                               (gen_configs_init_compare.py / orchestrate_runs_trigger_joint_
#                                init_compare_slurm.sh)
#   - expert_retrain_interval: 0 [disabled] vs 1 [retrain every outer iteration]
#                               (gen_configs_expert_retrain_compare.py / orchestrate_runs_
#                                trigger_joint_expert_retrain_compare_slurm.sh)
#   - detach_param_dist:       False [param_dist stays fully differentiable, the module's
#                               original behavior] vs True [param_dist.detach() in mtt_term_k's
#                               denominator]
#                               (gen_configs_detach_param_dist_compare.py / orchestrate_runs_
#                                trigger_joint_detach_param_dist_compare_slurm.sh)
#
# These are three INDEPENDENT one-axis-at-a-time comparisons (each holds the other two axes
# fixed at its own campaign's gen_configs.py-inherited default), not a cross product -- run
# analyze_axis_compare.py after all three complete to read CTA/ASR off each axis independently.
#
# This script only (1) regenerates all three campaigns' configs and (2) calls all three
# campaigns' own Slurm launchers in sequence -- it does not duplicate their job-building logic.
# Every env var a sibling launcher accepts (DRY_RUN, SLURM_ACCOUNT, BUDGETS, SEEDS, ...) is
# forwarded to all three (only launcher-specific axis env vars -- INITS/INTERVALS/TAGS -- are
# NOT shared, since each campaign has its own).
#
# Usage:
#   SLURM_ACCOUNT=<account> ./orchestrate_runs_trigger_joint_axis_sweep_slurm.sh
#   SLURM_ACCOUNT=<account> DRY_RUN=1 ./orchestrate_runs_trigger_joint_axis_sweep_slurm.sh
#   SKIP_GEN=1 ./orchestrate_runs_trigger_joint_axis_sweep_slurm.sh   # configs already generated
###

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export BASE_DIR
cd "$BASE_DIR"

SKIP_GEN="${SKIP_GEN:-0}"

if [ "$SKIP_GEN" != "1" ]; then
    echo "[GEN] python -m modules.federated_generate_labels_trigger_joint.gen_configs_init_compare"
    python -m modules.federated_generate_labels_trigger_joint.gen_configs_init_compare
    echo "[GEN] python -m modules.federated_generate_labels_trigger_joint.gen_configs_expert_retrain_compare"
    python -m modules.federated_generate_labels_trigger_joint.gen_configs_expert_retrain_compare
    echo "[GEN] python -m modules.federated_generate_labels_trigger_joint.gen_configs_detach_param_dist_compare"
    python -m modules.federated_generate_labels_trigger_joint.gen_configs_detach_param_dist_compare
else
    echo "[GEN] SKIP_GEN=1 -- assuming all three campaigns' configs already exist on disk."
fi

echo
echo "[SUBMIT] init_compare campaign (stripe vs random)"
"$SCRIPT_DIR/orchestrate_runs_trigger_joint_init_compare_slurm.sh"

echo
echo "[SUBMIT] expert_retrain_compare campaign (interval=0 vs interval=1)"
"$SCRIPT_DIR/orchestrate_runs_trigger_joint_expert_retrain_compare_slurm.sh"

echo
echo "[SUBMIT] detach_param_dist_compare campaign (False vs True)"
"$SCRIPT_DIR/orchestrate_runs_trigger_joint_detach_param_dist_compare_slurm.sh"

echo
echo "[DONE] all three axis-comparison campaigns submitted (or dry-run printed above)."
echo "[NOTE] once all finish: python scripts/show_results/show_results.py, then"
echo "[NOTE] python scripts/show_results/analyze_axis_compare.py for the head-to-head summary."
