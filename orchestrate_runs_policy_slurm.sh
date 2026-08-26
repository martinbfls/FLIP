#!/bin/bash
###
# orchestrate_runs_policy_slurm.sh
#
# Full-chain Slurm launcher for the `federated_optimizing_trigger_policy`
# threat model -- problem (P^mean): trigger delta and class-pair policy u
# optimised jointly against the mean-reachable deviation set.
#
# Chain, one Slurm job per cell, with afterok barriers between phases:
#
#   [EXPERT]  train_expert                        (optional, shared)
#        |
#   [POLICY]  federated_optimizing_trigger_policy  (GPU; also retrains its own
#        |                                          model at every outer step)
#   [FLIPS]   federated_policy_to_flips            (CPU-bound, minutes)
#        |
#   [USER]    federated_train_user
#
# Differences from the two `direct` orchestrators, all structural:
#   - FLIPS is federated_policy_to_flips, not federated_select_flips, and it
#     consumes a .npz policy artifact rather than labels.npy;
#   - num_honests / num_poisoned MUST match between the POLICY and FLIPS
#     configs -- the downstream module raises a ValueError on divergence, so
#     a mismatch fails at run time rather than silently producing wrong flip
#     counts. The check below catches it before submission instead;
#   - beta is the LOCAL corruption rate (fraction of ONE corrupted worker's
#     own shard), not the global budget. beta_global = gamma * beta.
#
# Sweep axes and config paths below MUST match
# modules/federated_optimizing_trigger_policy/gen_configs.py (the module that
# actually writes these configs) -- gen_configs.py has no env-var/CLI
# overrides of its own, so this grid is filled in BY HAND below and has to be
# kept in sync manually every time the grid in gen_configs.py changes.
# (Deliberately not read out of gen_configs.py via `python -c ...`: that
# import needs the conda env active, and this script is meant to be runnable
# from a bare login shell before that happens.)
#
# NOTE: gen_configs.py sweeps a TARGET GLOBAL flip budget (BUDGETS_TARGET,
# converted internally to this module's own LOCAL beta -- see its
# resolve_beta_gamma docstring), not beta directly, and does NOT use
# agg_method as a path/config axis at all ("(P^mean) has no agg_method of its
# own -- mean aggregation IS the formulation", see AGG_METHODS' comment
# there). This orchestrator mirrors that: no AGGS axis here either -- sweep
# BUDGETS_TARGET, not a per-aggregator grid.
#
# Usage:
#   1. Generate the configs (from BASE_DIR):
#        python -m modules.federated_optimizing_trigger_policy.gen_configs
#   2. Submit the campaign:
#        SLURM_ACCOUNT=<account> ./orchestrate_runs_policy_slurm.sh
#        SLURM_ACCOUNT=<account> DRY_RUN=1 ./orchestrate_runs_policy_slurm.sh
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

EXPERT_TIME="${EXPERT_TIME:-1-00:00:00}"
POLICY_TIME="${POLICY_TIME:-1-00:00:00}"
FLIPS_TIME="${FLIPS_TIME:-01:00:00}"     # CPU-bound, minutes
FLIPS_MEM="${FLIPS_MEM:-16G}"
USER_TIME="${USER_TIME:-1-00:00:00}"

# ---------------------------------------------------------------------------
# CAMPAIGN GRID -- fill in by hand, kept in sync with
# modules/federated_optimizing_trigger_policy/gen_configs.py. Values below
# are that file's CURRENT defaults; re-copy them here whenever the grid in
# gen_configs.py changes. All overridable from the environment.
# ---------------------------------------------------------------------------
read -ra MODEL_FLAGS    <<< "${MODEL_FLAGS:-r32p}"
read -ra DATASETS       <<< "${DATASETS:-cifar}"
read -ra SEEDS          <<< "${SEEDS:-0}"
# TARGET GLOBAL flip budgets (gen_configs.py's BUDGETS_TARGET) -- NOT beta_local. beta_local is
# derived internally from these exactly as gen_configs.py's resolve_beta_gamma does.
read -ra BUDGETS_TARGET <<< "${BUDGETS_TARGET:-1500}"

NUM_POISONED="${NUM_POISONED:-3}"
NUM_HONESTS="${NUM_HONESTS:-7}"

# gen_configs.py's EXP_BASE, relative to experiments/ (what run_experiment.py expects: it
# prepends "experiments/" and appends "/config.toml" itself).
EXP_BASE_REL="${EXP_BASE_REL:-federated_experiments/threat_model_expert_policy}"

# ---------------------------------------------------------------------------
# Helpers -- mirror gen_configs.py's cell_name()/directory layout exactly:
#   cell_dir       = EXP_BASE/<model>/<dataset>/<p>vs<h>/budget<budget>/seed<seed>
#   train_expert   = EXP_BASE/train_expert/<model>_1xs                    (per model, shared)
#   policy_opt     = cell_dir/policy_opt
#   policy_to_flips= cell_dir/policy_to_flips
#   train_user_*   = cell_dir/train_user_<budget>   (predicted_budget == budget_target exactly,
#                    see gen_configs.py's resolve_beta_gamma/predicted_budget derivation)
# ---------------------------------------------------------------------------
MISSING=0
MISMATCH=0

require_config() {
    local path="$1"
    if [ ! -f "experiments/$path/config.toml" ]; then
        echo "[CONFIG] missing: experiments/$path/config.toml" >&2
        MISSING=$((MISSING + 1))
    fi
}

# federated_policy_to_flips raises if its num_honests/num_poisoned differ from what the policy
# artifact recorded. Catch it here rather than at run time.
check_worker_counts() {
    local policy_cfg="$1" flips_cfg="$2" tag="$3"
    local p_np p_nh f_np f_nh
    p_np=$(grep -E '^\s*num_poisoned\s*=' "experiments/$policy_cfg/config.toml" | head -1 | tr -dc '0-9')
    p_nh=$(grep -E '^\s*num_honests\s*='  "experiments/$policy_cfg/config.toml" | head -1 | tr -dc '0-9')
    f_np=$(grep -E '^\s*num_poisoned\s*=' "experiments/$flips_cfg/config.toml"  | head -1 | tr -dc '0-9')
    f_nh=$(grep -E '^\s*num_honests\s*='  "experiments/$flips_cfg/config.toml"  | head -1 | tr -dc '0-9')
    if [ "$p_np" != "$f_np" ] || [ "$p_nh" != "$f_nh" ]; then
        echo "[CONFIG] worker-count mismatch in $tag: policy=($p_np,$p_nh) flips=($f_np,$f_nh)" >&2
        MISMATCH=$((MISMATCH + 1))
    fi
}

cell_dir() {
    # model_flag dataset budget_target seed
    printf '%s/%s/%s/%svs%s/budget%s/seed%s' "$EXP_BASE_REL" "$1" "$2" "$NUM_POISONED" "$NUM_HONESTS" "$3" "$4"
}

# ---------------------------------------------------------------------------
# Build the job lists, following gen_configs.py's own nesting
# (model_flag -> dataset -> budget_target -> seed) and directory layout.
# train_expert is per model_flag only and shared across cells with that
# model, so it is deduplicated instead of resubmitted per cell.
# ---------------------------------------------------------------------------
EXPERT_JOBS=()
POLICY_JOBS=()
FLIPS_JOBS=()
USER_JOBS=()

SEEN_EXPERT=""

for model_flag in "${MODEL_FLAGS[@]}"; do
  if [[ " $SEEN_EXPERT " != *" $model_flag "* ]]; then
    SEEN_EXPERT="$SEEN_EXPERT $model_flag"
    expert_cfg="$EXP_BASE_REL/train_expert/${model_flag}_1xs"
    require_config "$expert_cfg"
    EXPERT_JOBS+=("python run_experiment.py $expert_cfg|pol_expert_${model_flag}")
  fi

  for dataset in "${DATASETS[@]}"; do
    for budget_target in "${BUDGETS_TARGET[@]}"; do
      for seed in "${SEEDS[@]}"; do
        cell="$(cell_dir "$model_flag" "$dataset" "$budget_target" "$seed")"
        tag="${model_flag}_${dataset}_budget${budget_target}_seed${seed}"

        policy_cfg="$cell/policy_opt"
        flips_cfg="$cell/policy_to_flips"
        user_cfg="$cell/train_user_${budget_target}"

        require_config "$policy_cfg"
        require_config "$flips_cfg"
        require_config "$user_cfg"

        if [ -f "experiments/$policy_cfg/config.toml" ] && [ -f "experiments/$flips_cfg/config.toml" ]; then
            check_worker_counts "$policy_cfg" "$flips_cfg" "$tag"
        fi

        POLICY_JOBS+=("python run_experiment.py $policy_cfg|pol_policy_${tag}")
        FLIPS_JOBS+=("python run_experiment.py $flips_cfg|pol_flips_${tag}")
        USER_JOBS+=("python run_experiment.py $user_cfg|pol_user_${tag}")
      done
    done
  done
done

if [ "$MISSING" -gt 0 ]; then
    echo "[ABORT] $MISSING config(s) missing -- generate them first:" >&2
    echo "        python -m modules.federated_optimizing_trigger_policy.gen_configs" >&2
    echo "[ABORT] (a missing cell is also what a beta_local>1-infeasible cell looks like --" >&2
    echo "[ABORT] gen_configs.py prints REFUSED for those instead of writing a config; re-run" >&2
    echo "[ABORT] it above to see which.)" >&2
    exit 1
fi

if [ "$MISMATCH" -gt 0 ]; then
    echo "[ABORT] $MISMATCH cell(s) with policy/flips worker-count mismatch." >&2
    echo "        federated_policy_to_flips would raise a ValueError at run time." >&2
    exit 1
fi

echo "[PLAN] module=federated_optimizing_trigger_policy exp_base=$EXP_BASE_REL"
echo "[PLAN] model_flags=${MODEL_FLAGS[*]} datasets=${DATASETS[*]} budgets_target=${BUDGETS_TARGET[*]}"
echo "[PLAN] seeds=${SEEDS[*]} num_poisoned=$NUM_POISONED num_honests=$NUM_HONESTS"
echo "[PLAN] cells=${#POLICY_JOBS[@]} (=model_flags x datasets x budgets_target x seeds)"
echo "[PLAN] phases: $([ "$RUN_EXPERT" = 1 ] && echo 'EXPERT -> ')POLICY -> FLIPS -> USER"

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY-RUN] nothing submitted."
    [ "$RUN_EXPERT" = "1" ] && printf '[DRY-RUN] %s\n' "${EXPERT_JOBS[@]}"
    printf '[DRY-RUN] %s\n' "${POLICY_JOBS[@]}"
    printf '[DRY-RUN] %s\n' "${FLIPS_JOBS[@]}"
    printf '[DRY-RUN] %s\n' "${USER_JOBS[@]}"
    exit 0
fi

preflight_slurm || exit 1

# ---------------------------------------------------------------------------
# Submit -- job-pool/barrier pattern: each phase is submitted as a pool, then
# a single tiny barrier job (afterok on every job of that phase) is what the
# next phase depends on, so a large phase carries one dependency id
# downstream instead of one per job.
# ---------------------------------------------------------------------------
DEP=""

if [ "$RUN_EXPERT" = "1" ]; then
    TIME_PER_TASK="$EXPERT_TIME"
    submit_job_pool_slurm EXPERT_JOBS "expert" "" || exit 1
    expert_barrier=$(submit_barrier_slurm "pol_barrier_expert" "afterok:$(join_job_ids "${SUBMITTED_JOB_IDS[@]}")") || exit 1
    DEP="afterok:$expert_barrier"
    echo "[PHASE] EXPERT submitted (${#EXPERT_JOBS[@]} jobs) -> barrier $expert_barrier"
else
    echo "[PHASE] EXPERT skipped (RUN_EXPERT=0) -- expert checkpoints assumed present."
    echo "[PHASE] they MUST be the same ones used by the two sibling campaigns."
fi

TIME_PER_TASK="$POLICY_TIME"
submit_job_pool_slurm POLICY_JOBS "policy" "$DEP" || exit 1
policy_barrier=$(submit_barrier_slurm "pol_barrier_policy" "afterok:$(join_job_ids "${SUBMITTED_JOB_IDS[@]}")") || exit 1
echo "[PHASE] POLICY submitted (${#POLICY_JOBS[@]} jobs) -> barrier $policy_barrier"

# FLIPS is CPU-bound (drawing indices, writing .npy shards); it still requests
# one GPU because --gres=gpu:0 is rejected on some Slurm builds. Lower the
# walltime and memory instead, so these short jobs do not hold a full slot.
TIME_PER_TASK="$FLIPS_TIME"
MEM_PER_TASK="$FLIPS_MEM"
submit_job_pool_slurm FLIPS_JOBS "flips" "afterok:$policy_barrier" || exit 1
flips_barrier=$(submit_barrier_slurm "pol_barrier_flips" "afterok:$(join_job_ids "${SUBMITTED_JOB_IDS[@]}")") || exit 1
echo "[PHASE] FLIPS submitted (${#FLIPS_JOBS[@]} jobs) -> barrier $flips_barrier"

TIME_PER_TASK="$USER_TIME"
MEM_PER_TASK="${MEM_PER_TASK_USER:-32G}"
submit_job_pool_slurm USER_JOBS "user" "afterok:$flips_barrier" || exit 1
echo "[PHASE] USER submitted (${#USER_JOBS[@]} jobs)."

echo "[DONE] campaign submitted; job ids in $LOG_DIR/jobids_*.txt"
echo "[NOTE] the POLICY phase retrains its own model at every outer step, so its"
echo "[NOTE] walltime scales with n_steps x epochs -- check the first job's rate"
echo "[NOTE] before the whole pool starts competing for the node."
