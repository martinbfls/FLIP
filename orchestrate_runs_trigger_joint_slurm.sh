#!/bin/bash
###
# orchestrate_runs_trigger_joint_slurm.sh
#
# Full-chain Slurm launcher for the `federated_generate_labels_trigger_joint`
# threat model (real coupling: the expert step is differentiable, so delta
# receives gradient through the federated aggregate AND through L_bd).
#
# Chain, one Slurm job per cell, with afterok barriers between phases --
# same job-pool/barrier mechanics as orchestrate_runs_slurm.sh (the Slurm
# launcher for federated_generate_labels), extended to 4 phases because this
# module's gen_configs.py writes train_expert / gen_trigger_joint /
# select_flips / train_user as FOUR SEPARATE config files (the
# federated_generate_labels module bundles train_expert + gen_labels +
# select_flips into a single "gen_labels" config, so it only needs a
# GEN -> TRAIN barrier):
#
#   [EXPERT]  train_expert                           (optional, shared)
#        |
#   [GEN]     federated_generate_labels_trigger_joint
#        |
#   [FLIPS]   federated_select_flips
#        |
#   [USER]    federated_train_user
#
# Sweep axes and config paths below MUST match
# modules/federated_generate_labels_trigger_joint/gen_configs.py (the module
# that actually writes these configs) -- gen_configs.py has no env-var/CLI
# overrides of its own, so this grid is filled in BY HAND below and has to be
# kept in sync manually every time the grid in gen_configs.py changes.
# (Deliberately not read out of gen_configs.py via `python -c ...`: that
# import pulls in torch/toml, which are only available once the conda env is
# activated, and this script is meant to be runnable from a bare login shell
# before that happens.)
#
# Usage:
#   1. Generate the configs (from BASE_DIR):
#        python -m modules.federated_generate_labels_trigger_joint.gen_configs
#   2. Submit the campaign:
#        SLURM_ACCOUNT=<account> ./orchestrate_runs_trigger_joint_slurm.sh
#        SLURM_ACCOUNT=<account> DRY_RUN=1 ./orchestrate_runs_trigger_joint_slurm.sh
###

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$SCRIPT_DIR}"
export BASE_DIR
cd "$BASE_DIR"

# shellcheck source=slurm_lib.sh
source "$SCRIPT_DIR/slurm_lib.sh"

RUN_EXPERT="${RUN_EXPERT:-0}"
DRY_RUN="${DRY_RUN:-0}"

EXPERT_TIME="${EXPERT_TIME:-0-00:10:00}"
GEN_TIME="${GEN_TIME:-1-00:00:00}"
FLIPS_TIME="${FLIPS_TIME:-00:10:00}"
USER_TIME="${USER_TIME:-0-01:00:00}"

# ---------------------------------------------------------------------------
# CAMPAIGN GRID -- fill in by hand, kept in sync with
# modules/federated_generate_labels_trigger_joint/gen_configs.py. Values
# below are that file's CURRENT defaults; re-copy them here whenever the
# grid in gen_configs.py changes. All overridable from the environment.
# ---------------------------------------------------------------------------
read -ra MODEL_FLAGS <<< "${MODEL_FLAGS:-r32p}"
read -ra DATASETS    <<< "${DATASETS:-cifar}"
read -ra AGG_METHODS <<< "${AGG_METHODS:-mean multikrum}"
read -ra SEEDS       <<< "${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
read -ra BUDGETS     <<< "${BUDGETS:-150 300 500 1000 2000 2500 5000}"

NUM_POISONED="${NUM_POISONED:-3}"
NUM_HONESTS="${NUM_HONESTS:-7}"

# gen_configs.py's EXP_BASE, relative to experiments/ (what run_experiment.py
# expects: it prepends "experiments/" and appends "/config.toml" itself).
EXP_BASE_REL="${EXP_BASE_REL:-federated_experiments/threat_model_direct_trigger_joint}"

# ---------------------------------------------------------------------------
# Helpers -- mirror gen_configs.py's cell_name()/directory layout exactly.
# ---------------------------------------------------------------------------
MISSING=0

require_config() {
    local path="$1"
    if [ ! -f "experiments/$path/config.toml" ]; then
        echo "[CONFIG] missing: experiments/$path/config.toml" >&2
        MISSING=$((MISSING + 1))
    fi
}

cell_dir() {
    # model_flag dataset agg_method seed
    printf '%s/%s/%s/%svs%s/%s/seed%s' "$EXP_BASE_REL" "$1" "$2" "$NUM_POISONED" "$NUM_HONESTS" "$3" "$4"
}

# ---------------------------------------------------------------------------
# Build the job lists, following gen_configs.py's own nesting
# (model_flag -> dataset -> agg_method -> seed -> budget) and directory
# layout (cell_name / gen_labels_trigger_joint, select_flips, train_user_*).
# train_expert is per (model_flag, seed) -- a different expert per seed, per
# gen_configs.py -- shared only across dataset/agg_method for a fixed
# (model_flag, seed), so it is deduplicated on that pair instead of
# resubmitted per cell.
# ---------------------------------------------------------------------------
EXPERT_JOBS=()
GEN_JOBS=()
FLIPS_JOBS=()
USER_JOBS=()

SEEN_EXPERT=""

for model_flag in "${MODEL_FLAGS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for agg in "${AGG_METHODS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        expert_key="${model_flag}_seed${seed}"
        if [[ " $SEEN_EXPERT " != *" $expert_key "* ]]; then
          SEEN_EXPERT="$SEEN_EXPERT $expert_key"
          expert_cfg="$EXP_BASE_REL/train_expert/${model_flag}_1xs/seed${seed}"
          require_config "$expert_cfg"
          EXPERT_JOBS+=("python run_experiment.py $expert_cfg|joint_expert_${expert_key}")
        fi

        cell="$(cell_dir "$model_flag" "$dataset" "$agg" "$seed")"
        tag="${model_flag}_${dataset}_${agg}_seed${seed}"

        gen_cfg="$cell/gen_labels_trigger_joint"
        flips_cfg="$cell/select_flips"

        require_config "$gen_cfg"
        require_config "$flips_cfg"

        GEN_JOBS+=("python run_experiment.py $gen_cfg|joint_gen_${tag}")
        FLIPS_JOBS+=("python run_experiment.py $flips_cfg|joint_flips_${tag}")

        for budget in "${BUDGETS[@]}"; do
          user_cfg="$cell/train_user_${budget}"
          require_config "$user_cfg"
          USER_JOBS+=("python run_experiment.py $user_cfg|joint_user_${tag}_${budget}")
        done
      done
    done
  done
done

if [ "$MISSING" -gt 0 ]; then
    echo "[ABORT] $MISSING config(s) missing -- generate them first:" >&2
    echo "        python -m modules.federated_generate_labels_trigger_joint.gen_configs" >&2
    echo "[ABORT] (a missing cell is also what a delta_min-infeasible or otherwise" >&2
    echo "[ABORT] refused cell looks like -- gen_configs.py prints REFUSED for those" >&2
    echo "[ABORT] instead of writing a config; re-run it above to see which.)" >&2
    exit 1
fi

echo "[PLAN] module=federated_generate_labels_trigger_joint exp_base=$EXP_BASE_REL"
echo "[PLAN] model_flags=${MODEL_FLAGS[*]} datasets=${DATASETS[*]} agg_methods=${AGG_METHODS[*]}"
echo "[PLAN] seeds=${SEEDS[*]} budgets=${BUDGETS[*]} num_poisoned=$NUM_POISONED num_honests=$NUM_HONESTS"
echo "[PLAN] cells=${#GEN_JOBS[@]} (=model_flags x datasets x agg_methods x seeds), train_user jobs=${#USER_JOBS[@]}"
echo "[PLAN] phases: $([ "$RUN_EXPERT" = 1 ] && echo 'EXPERT -> ')GEN -> FLIPS -> USER"

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY-RUN] nothing submitted."
    [ "$RUN_EXPERT" = "1" ] && printf '[DRY-RUN] %s\n' "${EXPERT_JOBS[@]}"
    printf '[DRY-RUN] %s\n' "${GEN_JOBS[@]}"
    printf '[DRY-RUN] %s\n' "${FLIPS_JOBS[@]}"
    printf '[DRY-RUN] %s\n' "${USER_JOBS[@]}"
    exit 0
fi

preflight_slurm || exit 1

# ---------------------------------------------------------------------------
# Submit -- same job-pool/barrier pattern as orchestrate_runs_slurm.sh: each
# phase is submitted as a pool, then a single tiny barrier job (afterok on
# every job of that phase) is what the next phase depends on, so a 400-job
# phase carries one dependency id downstream instead of 400.
# ---------------------------------------------------------------------------
DEP=""

if [ "$RUN_EXPERT" = "1" ]; then
    TIME_PER_TASK="$EXPERT_TIME"
    submit_job_pool_slurm EXPERT_JOBS "joint_expert" "" || exit 1
    expert_barrier=$(submit_barrier_slurm "joint_barrier_expert" "afterok:$(join_job_ids "${SUBMITTED_JOB_IDS[@]}")") || exit 1
    DEP="afterok:$expert_barrier"
    echo "[PHASE] EXPERT submitted (${#EXPERT_JOBS[@]} jobs) -> barrier $expert_barrier"
else
    echo "[PHASE] EXPERT skipped (RUN_EXPERT=0) -- expert checkpoints assumed present."
    echo "[PHASE] they MUST be the same ones used by the sibling indirect campaign."
fi

TIME_PER_TASK="$GEN_TIME"
submit_job_pool_slurm GEN_JOBS "joint_gen" "$DEP" || exit 1
gen_barrier=$(submit_barrier_slurm "joint_barrier_gen" "afterok:$(join_job_ids "${SUBMITTED_JOB_IDS[@]}")") || exit 1
echo "[PHASE] GEN submitted (${#GEN_JOBS[@]} jobs) -> barrier $gen_barrier"

TIME_PER_TASK="$FLIPS_TIME"
submit_job_pool_slurm FLIPS_JOBS "joint_flips" "afterok:$gen_barrier" || exit 1
flips_barrier=$(submit_barrier_slurm "joint_barrier_flips" "afterok:$(join_job_ids "${SUBMITTED_JOB_IDS[@]}")") || exit 1
echo "[PHASE] FLIPS submitted (${#FLIPS_JOBS[@]} jobs) -> barrier $flips_barrier"

TIME_PER_TASK="$USER_TIME"
submit_job_pool_slurm USER_JOBS "joint_user" "afterok:$flips_barrier" || exit 1
echo "[PHASE] USER submitted (${#USER_JOBS[@]} jobs)."

echo "[DONE] campaign submitted; job ids in $LOG_DIR/jobids_*.txt"
echo "[NOTE] this module builds a second-order graph (create_graph=True) on the"
echo "[NOTE] expert step: memory is ~2.6x the non-differentiable variant on a CPU"
echo "[NOTE] proxy measurement. Watch the first GEN job for OOM before the rest start."
