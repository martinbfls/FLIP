#!/bin/bash
###
# orchestrate_trigger_joint_slurm.sh
#
# Full-chain Slurm launcher for the `federated_generate_labels_trigger_joint`
# threat model (real coupling: the expert step is differentiable, so delta
# receives gradient through the federated aggregate AND through L_bd).
#
# Chain, one Slurm job per cell, with afterok barriers between phases:
#
#   [EXPERT]  train_expert                           (optional, shared)
#        |
#   [GEN]     federated_generate_labels_trigger_joint
#        |
#   [FLIPS]   federated_select_flips
#        |
#   [USER]    federated_train_user
#
# Structurally identical to orchestrate_trigger_slurm.sh: the two differ ONLY
# by the GEN module and by the extra guardrails below. Keeping them parallel
# is deliberate -- diffing the two files should show the formulation change
# and nothing else.
#
# TWO GUARDRAILS SPECIFIC TO THIS MODULE
#
#  1. checkpoint_sampling defaults to "biased" here and "uniform" in the
#     sibling script. Leaving them different makes an indirect-vs-joint
#     comparison cross that factor. The script refuses to run unless the
#     value is set explicitly, so the choice is always conscious.
#
#  2. delta_min is computed as a fraction of ||delta_init||_2 BEFORE the
#     epsilon clamp, and can therefore be structurally unreachable: once
#     clamped, ||delta||_2 <= epsilon*sqrt(numel). A campaign launched on
#     such a cell measures nothing (the magnitude floor stays active for
#     every batch and dominates the loss). The check below aborts instead.
#
# Usage:
#   SLURM_ACCOUNT=<account> CHECKPOINT_SAMPLING=uniform ./orchestrate_trigger_joint_slurm.sh
#   SLURM_ACCOUNT=<account> DRY_RUN=1 CHECKPOINT_SAMPLING=biased ./orchestrate_trigger_joint_slurm.sh
###

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export BASE_DIR
cd "$BASE_DIR"

# shellcheck source=slurm_lib.sh
source "$SCRIPT_DIR/slurm_lib.sh"

# ---------------------------------------------------------------------------
# Campaign definition -- edit here.
# ---------------------------------------------------------------------------
CAMPAIGN="${CAMPAIGN:-joint_v1}"
CONFIG_ROOT="${CONFIG_ROOT:-$BASE_DIR/experiments/$CAMPAIGN/joint}"

SEEDS=(${SEEDS:-0 1 2})
BUDGETS=(${BUDGETS:-500 1500 5000})
AGGS=(${AGGS:-mean trmean})

NUM_POISONED="${NUM_POISONED:-3}"
NUM_HONESTS="${NUM_HONESTS:-7}"

# No default: forcing an explicit value is guardrail 1 (see header).
CHECKPOINT_SAMPLING="${CHECKPOINT_SAMPLING:-}"

# Trigger constraint / anti-collapse parameters, for the feasibility check below.
EPSILON="${EPSILON:-0.031}"                 # 8/255 -- NOT 1.0, which saturates
DELTA_MIN_FRAC="${DELTA_MIN_FRAC:-0.5}"
DELTA_INIT_NORM="${DELTA_INIT_NORM:-115.7}" # ||delta_init||_2 for strength=6.0, 3x32x32
IMAGE_NUMEL="${IMAGE_NUMEL:-3072}"          # 3*32*32

RUN_EXPERT="${RUN_EXPERT:-0}"
EXPERT_CONFIG="${EXPERT_CONFIG:-$CONFIG_ROOT/train_expert.toml}"

DRY_RUN="${DRY_RUN:-0}"

EXPERT_TIME="${EXPERT_TIME:-1-00:00:00}"
GEN_TIME="${GEN_TIME:-1-00:00:00}"
FLIPS_TIME="${FLIPS_TIME:-02:00:00}"
USER_TIME="${USER_TIME:-1-00:00:00}"

# ---------------------------------------------------------------------------
# Guardrail 1 -- checkpoint sampling must be an explicit, conscious choice.
# ---------------------------------------------------------------------------
if [ -z "$CHECKPOINT_SAMPLING" ]; then
    cat >&2 <<'MSG'
[ABORT] CHECKPOINT_SAMPLING is unset.

  This module defaults to "biased" and the indirect module to "uniform".
  Running both at their defaults makes an indirect-vs-joint comparison cross
  the checkpoint-sampling factor on top of the coupling factor being studied.

  Set it explicitly, and set the SAME value in orchestrate_trigger_slurm.sh:
      CHECKPOINT_SAMPLING=uniform ./orchestrate_trigger_joint_slurm.sh
MSG
    exit 1
fi

# ---------------------------------------------------------------------------
# Guardrail 2 -- is the magnitude floor reachable at all?
#   delta_min             = DELTA_MIN_FRAC * ||delta_init||_2
#   max reachable ||d||_2 = EPSILON * sqrt(numel)
# ---------------------------------------------------------------------------
DELTA_MIN=$(python - <<PY
print(${DELTA_MIN_FRAC} * ${DELTA_INIT_NORM})
PY
)
MAX_L2=$(python - <<PY
import math
print(${EPSILON} * math.sqrt(${IMAGE_NUMEL}))
PY
)
UNREACHABLE=$(python - <<PY
print(1 if ${DELTA_MIN} > ${MAX_L2} else 0)
PY
)

if [ "$UNREACHABLE" = "1" ]; then
    cat >&2 <<MSG
[ABORT] the magnitude floor is structurally unreachable for this cell.

  delta_min             = ${DELTA_MIN_FRAC} * ${DELTA_INIT_NORM} = ${DELTA_MIN}
  max reachable ||d||_2 = ${EPSILON} * sqrt(${IMAGE_NUMEL})      = ${MAX_L2}

  Once delta is clamped to ||delta||_inf <= ${EPSILON}, its L2 norm can never
  exceed ${MAX_L2}, so relu(delta_min - ||delta||_2) stays active on every
  batch and dominates the loss whatever lambda_align does. Under
  trigger_constraint="projection" the alternating projection additionally
  returns a delta that VIOLATES epsilon (known bug, not yet fixed).

  Lower DELTA_MIN_FRAC, or rebase delta_min on epsilon*sqrt(numel) in the
  module, before spending GPU hours here.
MSG
    exit 1
fi

echo "[CHECK] magnitude floor reachable: delta_min=${DELTA_MIN} <= max ||d||_2=${MAX_L2}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
MISSING=0

require_config() {
    local path="$1"
    if [ ! -f "$path" ]; then
        echo "[CONFIG] missing: $path" >&2
        MISSING=$((MISSING + 1))
    fi
}

cell_tag() {
    printf 's%s_b%s_%s_%svs%s' "$1" "$2" "$3" "$NUM_POISONED" "$NUM_HONESTS"
}

# ---------------------------------------------------------------------------
# Build the job lists.
# ---------------------------------------------------------------------------
GEN_JOBS=()
FLIPS_JOBS=()
USER_JOBS=()

for seed in "${SEEDS[@]}"; do
  for budget in "${BUDGETS[@]}"; do
    for agg in "${AGGS[@]}"; do
      tag="$(cell_tag "$seed" "$budget" "$agg")"

      gen_cfg="$CONFIG_ROOT/gen/${tag}.toml"
      flips_cfg="$CONFIG_ROOT/flips/${tag}.toml"
      user_cfg="$CONFIG_ROOT/user/${tag}.toml"

      require_config "$gen_cfg"
      require_config "$flips_cfg"
      require_config "$user_cfg"

      GEN_JOBS+=("python run_experiment.py $gen_cfg|joint_gen_${tag}")
      FLIPS_JOBS+=("python run_experiment.py $flips_cfg|joint_flips_${tag}")
      USER_JOBS+=("python run_experiment.py $user_cfg|joint_user_${tag}")
    done
  done
done

if [ "$RUN_EXPERT" = "1" ]; then
    require_config "$EXPERT_CONFIG"
fi

if [ "$MISSING" -gt 0 ]; then
    echo "[ABORT] $MISSING config(s) missing -- generate them first:" >&2
    echo "        python -m modules.federated_generate_labels_trigger_joint.gen_configs" >&2
    exit 1
fi

echo "[PLAN] campaign=$CAMPAIGN module=federated_generate_labels_trigger_joint"
echo "[PLAN] cells=${#GEN_JOBS[@]} (seeds=${#SEEDS[@]} x budgets=${#BUDGETS[@]} x aggs=${#AGGS[@]})"
echo "[PLAN] num_poisoned=$NUM_POISONED num_honests=$NUM_HONESTS checkpoint_sampling=$CHECKPOINT_SAMPLING"
echo "[PLAN] epsilon=$EPSILON delta_min_frac=$DELTA_MIN_FRAC"
echo "[PLAN] phases: $([ "$RUN_EXPERT" = 1 ] && echo 'EXPERT -> ')GEN -> FLIPS -> USER"

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY-RUN] nothing submitted."
    printf '[DRY-RUN] %s\n' "${GEN_JOBS[@]}"
    exit 0
fi

preflight_slurm || exit 1

# ---------------------------------------------------------------------------
# Submit.
# ---------------------------------------------------------------------------
DEP=""

if [ "$RUN_EXPERT" = "1" ]; then
    TIME_PER_TASK="$EXPERT_TIME"
    EXPERT_JOBS=("python run_experiment.py $EXPERT_CONFIG|joint_expert")
    submit_job_pool_slurm EXPERT_JOBS "expert" "" || exit 1
    barrier=$(submit_barrier_slurm "joint_barrier_expert" "afterok:$(join_job_ids "${SUBMITTED_JOB_IDS[@]}")") || exit 1
    DEP="afterok:$barrier"
    echo "[PHASE] EXPERT done -> barrier $barrier"
else
    echo "[PHASE] EXPERT skipped (RUN_EXPERT=0) -- expert checkpoints assumed present."
    echo "[PHASE] they MUST be the same ones used by the two sibling campaigns."
fi

TIME_PER_TASK="$GEN_TIME"
submit_job_pool_slurm GEN_JOBS "gen" "$DEP" || exit 1
gen_barrier=$(submit_barrier_slurm "joint_barrier_gen" "afterok:$(join_job_ids "${SUBMITTED_JOB_IDS[@]}")") || exit 1
echo "[PHASE] GEN submitted -> barrier $gen_barrier"

TIME_PER_TASK="$FLIPS_TIME"
submit_job_pool_slurm FLIPS_JOBS "flips" "afterok:$gen_barrier" || exit 1
flips_barrier=$(submit_barrier_slurm "joint_barrier_flips" "afterok:$(join_job_ids "${SUBMITTED_JOB_IDS[@]}")") || exit 1
echo "[PHASE] FLIPS submitted -> barrier $flips_barrier"

TIME_PER_TASK="$USER_TIME"
submit_job_pool_slurm USER_JOBS "user" "afterok:$flips_barrier" || exit 1
echo "[PHASE] USER submitted."

echo "[DONE] campaign $CAMPAIGN submitted; job ids in $LOG_DIR/jobids_*.txt"
echo "[NOTE] this module builds a second-order graph (create_graph=True) on the"
echo "[NOTE] expert step: memory is ~2.6x the non-differentiable variant on a CPU"
echo "[NOTE] proxy measurement. Watch the first GEN job for OOM before the rest start."