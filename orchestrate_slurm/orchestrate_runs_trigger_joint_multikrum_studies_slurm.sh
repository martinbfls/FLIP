#!/bin/bash
###
# orchestrate_runs_trigger_joint_multikrum_studies_slurm.sh
#
# Single entry point for the four federated_generate_labels_trigger_joint campaigns built
# around the base config (1-poisoned/0-honest/mean attack generation, epsilon=1.0,
# delta_min_frac=0, lambda_align=0, expert_retrain_interval=1, detach_param_dist=False,
# budgets=[500, 2000]) and the question "does this attack survive Multi-Krum in a federated
# (3-poisoned/7-honest) deployment":
#
#   - federated_multikrum_compare : the base attack itself, single-user (1v0/mean) vs
#                                    federated (3v7/multikrum)
#                                    (gen_configs_federated_multikrum_compare.py /
#                                     orchestrate_runs_trigger_joint_federated_multikrum_slurm.sh)
#   - gradmatch_ablation           : grad_match relerr / cosine / off, EACH deployed both
#                                    single-user AND federated
#                                    (gen_configs_gradmatch_ablation.py /
#                                     orchestrate_runs_trigger_joint_gradmatch_ablation_slurm.sh)
#   - epsilon_sweep                : epsilon 1.0 down to 16/255, EACH deployed both single-user
#                                    AND federated
#                                    (gen_configs_epsilon_sweep.py /
#                                     orchestrate_runs_trigger_joint_epsilon_sweep_slurm.sh)
#   - lpips_compare                : base vs +LPIPS, EACH deployed both single-user AND
#                                    federated
#                                    (gen_configs_lpips_compare.py /
#                                     orchestrate_runs_trigger_joint_lpips_compare_slurm.sh)
#
# These are four INDEPENDENT campaigns (each isolates its own axis against the base config;
# the isolated-factor studies additionally cross their axis with single-user vs federated),
# not a cross product of all four together -- run scripts/show_results/show_results.py after
# all four complete for the CTA/ASR tables/plots and the Multi-Krum poison-selection plots
# (each study gets its own dedicated figures, per _federated_branch.py / show_results.py's own
# "never mix incomparable axes" convention).
#
# This script only (1) regenerates all four campaigns' configs and (2) calls all four
# campaigns' own Slurm launchers in sequence -- it does not duplicate their job-building logic.
# Every env var a sibling launcher accepts (DRY_RUN, SLURM_ACCOUNT, BUDGETS, SEEDS,
# EXPERT_TIME/GEN_TIME/FLIPS_TIME/USER_TIME, ...) is forwarded to all four (only
# launcher-specific axis env vars -- TAGS/EPS_TAGS -- are NOT shared, since each campaign has
# its own).
#
# Requires the `lpips` package installed wherever the lpips_compare campaign's GEN jobs
# actually run (see orchestrate_runs_trigger_joint_lpips_compare_slurm.sh's own note).
#
# Usage:
#   SLURM_ACCOUNT=<account> ./orchestrate_runs_trigger_joint_multikrum_studies_slurm.sh
#   SLURM_ACCOUNT=<account> DRY_RUN=1 ./orchestrate_runs_trigger_joint_multikrum_studies_slurm.sh
#   SKIP_GEN=1 ./orchestrate_runs_trigger_joint_multikrum_studies_slurm.sh   # configs already generated
###

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export BASE_DIR
cd "$BASE_DIR"

SKIP_GEN="${SKIP_GEN:-0}"

if [ "$SKIP_GEN" != "1" ]; then
    echo "[GEN] python -m modules.federated_generate_labels_trigger_joint.gen_configs_federated_multikrum_compare"
    python -m modules.federated_generate_labels_trigger_joint.gen_configs_federated_multikrum_compare
    echo "[GEN] python -m modules.federated_generate_labels_trigger_joint.gen_configs_gradmatch_ablation"
    python -m modules.federated_generate_labels_trigger_joint.gen_configs_gradmatch_ablation
    echo "[GEN] python -m modules.federated_generate_labels_trigger_joint.gen_configs_epsilon_sweep"
    python -m modules.federated_generate_labels_trigger_joint.gen_configs_epsilon_sweep
    echo "[GEN] python -m modules.federated_generate_labels_trigger_joint.gen_configs_lpips_compare"
    python -m modules.federated_generate_labels_trigger_joint.gen_configs_lpips_compare
else
    echo "[GEN] SKIP_GEN=1 -- assuming all four campaigns' configs already exist on disk."
fi

echo
echo "[SUBMIT] federated_multikrum_compare campaign (base attack: single-user vs federated)"
"$SCRIPT_DIR/orchestrate_runs_trigger_joint_federated_multikrum_slurm.sh"

echo
echo "[SUBMIT] gradmatch_ablation campaign (relerr / cosine / off, single-user + federated)"
"$SCRIPT_DIR/orchestrate_runs_trigger_joint_gradmatch_ablation_slurm.sh"

echo
echo "[SUBMIT] epsilon_sweep campaign (1.0 -> 16/255, single-user + federated)"
"$SCRIPT_DIR/orchestrate_runs_trigger_joint_epsilon_sweep_slurm.sh"

echo
echo "[SUBMIT] lpips_compare campaign (base vs +LPIPS, single-user + federated)"
"$SCRIPT_DIR/orchestrate_runs_trigger_joint_lpips_compare_slurm.sh"

echo
echo "[DONE] all four campaigns submitted (or dry-run printed above)."
echo "[NOTE] once all finish: python scripts/show_results/show_results.py for every CTA/ASR"
echo "[NOTE] table/plot AND the Multi-Krum poison-selection plots (global + per-worker for the"
echo "[NOTE] base attack, global-by-variant for each isolated-factor study)."
