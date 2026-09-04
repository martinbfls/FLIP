#!/bin/bash
###
# orchestrate_runs_trigger_joint_hardening_slurm.sh
#
# Shared Slurm launcher for the P0-P7 robustness-hardening protocol's Etapes 1-4 (see the
# accompanying diagnostic writeup's "Protocole experimental"). NOT called directly -- each step
# has its own thin wrapper (orchestrate_runs_trigger_joint_hardening_step{1,2,3,4}_slurm.sh)
# that just sets STEP and execs this script, so "un script de lancement par etape" is satisfied
# without duplicating this file four times.
#
# Configs for STEP must already exist -- generate them first:
#   python -m modules.federated_generate_labels_trigger_joint.gen_configs_hardening_steps \
#       --step $STEP [--defense-agg <agg>]
#
# Each step is ONE cell, TWO branches (single-user 1v0/mean undefended, and a defended
# federated_3vs7_<agg> branch -- trmean for step 1 per the diagnostic's own instruction,
# multikrum for steps 2-4, overridable via DEFENSE_AGG at generation time):
#
#   [EXPERT]  train_expert                            (shared across all 4 steps' seed0)
#        |
#   [GEN]     federated_generate_labels_trigger_joint  (ONE cell for this step,
#        |                                              metrics_log_path set --
#        |                                              this is what "regarder
#        |                                              cos_delta_to_init, mag_active_rate,
#        |                                              delta_sign_flip_rate" reads)
#   [FLIPS]   federated_select_flips x2:
#               - single-user (1v0)
#               - defended    (3v7, federated_3vs7_<agg>/)
#        |
#   [USER]    federated_train_user x2 per budget:
#               - single-user (1v0, mean)
#               - defended    (3v7, agg=<agg>)
#
# Usage (from BASE_DIR):
#   STEP=1 SLURM_ACCOUNT=<account> ./orchestrate_runs_trigger_joint_hardening_slurm.sh
#   STEP=1 SLURM_ACCOUNT=<account> DRY_RUN=1 ./orchestrate_runs_trigger_joint_hardening_slurm.sh
###

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export BASE_DIR
cd "$BASE_DIR"

# shellcheck source=slurm_lib.sh
source "$SCRIPT_DIR/slurm_lib.sh"

STEP="${STEP:?Set STEP=1|2|3|4 (or use the per-step wrapper script).}"
DRY_RUN="${DRY_RUN:-0}"

EXPERT_TIME="${EXPERT_TIME:-0-00:10:00}"
GEN_TIME="${GEN_TIME:-1-00:00:00}"
FLIPS_TIME="${FLIPS_TIME:-00:10:00}"
USER_TIME="${USER_TIME:-0-01:00:00}"

MODEL_FLAG="${MODEL_FLAG:-r32p}"
DATASET="${DATASET:-cifar}"
SEED="${SEED:-0}"
read -ra BUDGETS <<< "${BUDGETS:-500 2000}"

# Defense aggregator for this step's federated_3vs7 branch -- must match whatever
# gen_configs_hardening_steps.py was actually invoked with for this STEP (its own default per
# step: trmean for step 1, multikrum for steps 2-4 -- see STEP_OVERRIDES there).
case "$STEP" in
  1) DEFAULT_DEFENSE_AGG="trmean" ;;
  *) DEFAULT_DEFENSE_AGG="multikrum" ;;
esac
DEFENSE_AGG="${DEFENSE_AGG:-$DEFAULT_DEFENSE_AGG}"

case "$STEP" in
  1) TAG="step1_p0_p1" ;;
  2) TAG="step2_p2_p6" ;;
  3) TAG="step3_p5_match" ;;
  4) TAG="step4_p4_p3" ;;
  *) echo "[ABORT] STEP must be 1, 2, 3 or 4 (got $STEP)" >&2; exit 1 ;;
esac

EXP_BASE_REL="${EXP_BASE_REL:-federated_experiments/threat_model_direct_trigger_joint_hardening_steps}"
FED_TAG="federated_3vs7_${DEFENSE_AGG}"

MISSING=0
require_config() {
    local path="$1"
    if [ ! -f "experiments/$path/config.toml" ]; then
        echo "[CONFIG] missing: experiments/$path/config.toml" >&2
        MISSING=$((MISSING + 1))
    fi
}

cell="$EXP_BASE_REL/$TAG/$MODEL_FLAG/$DATASET/seed${SEED}"

expert_cfg="$EXP_BASE_REL/train_expert/${MODEL_FLAG}_1xs/seed${SEED}"
gen_cfg="$cell/gen_labels_trigger_joint"
flips_su_cfg="$cell/select_flips"
flips_fed_cfg="$cell/$FED_TAG/select_flips"

require_config "$expert_cfg"
require_config "$gen_cfg"
require_config "$flips_su_cfg"
require_config "$flips_fed_cfg"

EXPERT_JOBS=("python run_experiment.py $expert_cfg|hardening_expert_${TAG}")
GEN_JOBS=("python run_experiment.py $gen_cfg|hardening_gen_${TAG}")
FLIPS_JOBS=(
    "python run_experiment.py $flips_su_cfg|hardening_flips_single_${TAG}"
    "python run_experiment.py $flips_fed_cfg|hardening_flips_fed_${TAG}"
)
USER_JOBS=()
for budget in "${BUDGETS[@]}"; do
    user_su_cfg="$cell/train_user_${budget}"
    user_fed_cfg="$cell/$FED_TAG/train_user_${budget}"
    require_config "$user_su_cfg"
    require_config "$user_fed_cfg"
    USER_JOBS+=("python run_experiment.py $user_su_cfg|hardening_user_single_${TAG}_${budget}")
    USER_JOBS+=("python run_experiment.py $user_fed_cfg|hardening_user_fed_${TAG}_${budget}")
done

if [ "$MISSING" -gt 0 ]; then
    echo "[ABORT] $MISSING config(s) missing -- generate them first:" >&2
    echo "        python -m modules.federated_generate_labels_trigger_joint.gen_configs_hardening_steps --step $STEP --defense-agg $DEFENSE_AGG" >&2
    exit 1
fi

echo "[PLAN] hardening protocol STEP=$STEP tag=$TAG exp_base=$EXP_BASE_REL"
echo "[PLAN] model=$MODEL_FLAG dataset=$DATASET seed=$SEED budgets=${BUDGETS[*]}"
echo "[PLAN] defended branch: $FED_TAG (3vs7, agg=$DEFENSE_AGG)"
echo "[PLAN] phases: EXPERT -> GEN -> FLIPS(single+defended) -> USER(single+defended)"
if [ "$STEP" = "1" ]; then
    echo "[PLAN] Etape 1 (P0+P1): after GEN completes, read metrics_log_path (see"
    echo "       gen_labels_trigger_joint/logs/metrics.json under $cell) for cos_delta_to_init,"
    echo "       mag_active_rate, delta_sign_flip_rate BEFORE trusting the USER-phase results."
fi

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY-RUN] nothing submitted."
    printf '[DRY-RUN] %s\n' "${EXPERT_JOBS[@]}" "${GEN_JOBS[@]}" "${FLIPS_JOBS[@]}" "${USER_JOBS[@]}"
    exit 0
fi

preflight_slurm || exit 1

TIME_PER_TASK="$EXPERT_TIME"
submit_job_pool_slurm EXPERT_JOBS "hardening_expert" "" || exit 1
expert_barrier=$(submit_barrier_slurm "hardening_barrier_expert_${TAG}" "afterok:$(join_job_ids "${SUBMITTED_JOB_IDS[@]}")") || exit 1
echo "[PHASE] EXPERT submitted -> barrier $expert_barrier"

TIME_PER_TASK="$GEN_TIME"
submit_job_pool_slurm GEN_JOBS "hardening_gen" "afterok:$expert_barrier" || exit 1
gen_barrier=$(submit_barrier_slurm "hardening_barrier_gen_${TAG}" "afterok:$(join_job_ids "${SUBMITTED_JOB_IDS[@]}")") || exit 1
echo "[PHASE] GEN submitted -> barrier $gen_barrier"

TIME_PER_TASK="$FLIPS_TIME"
submit_job_pool_slurm FLIPS_JOBS "hardening_flips" "afterok:$gen_barrier" || exit 1
flips_barrier=$(submit_barrier_slurm "hardening_barrier_flips_${TAG}" "afterok:$(join_job_ids "${SUBMITTED_JOB_IDS[@]}")") || exit 1
echo "[PHASE] FLIPS submitted -> barrier $flips_barrier"

TIME_PER_TASK="$USER_TIME"
submit_job_pool_slurm USER_JOBS "hardening_user" "afterok:$flips_barrier" || exit 1
echo "[PHASE] USER submitted (${#USER_JOBS[@]} jobs)."

echo "[DONE] STEP=$STEP campaign submitted; job ids in $LOG_DIR/jobids_*.txt"
