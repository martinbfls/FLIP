#!/bin/bash
###
# orchestrate_policy_slurm.sh
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
# Usage:
#   SLURM_ACCOUNT=<account> ./orchestrate_policy_slurm.sh
#   SLURM_ACCOUNT=<account> DRY_RUN=1 ./orchestrate_policy_slurm.sh
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
CAMPAIGN="${CAMPAIGN:-policy_v1}"
CONFIG_ROOT="${CONFIG_ROOT:-$BASE_DIR/experiments/$CAMPAIGN/policy}"

SEEDS=(${SEEDS:-0 1 2})
# LOCAL corruption rates. beta_global = gamma * beta_local; with gamma = 0.3,
# beta_local in {0.033, 0.10, 0.33} gives beta_global in {0.01, 0.03, 0.10}.
BETAS=(${BETAS:-0.033 0.10 0.33})
AGGS=(${AGGS:-mean trmean})

NUM_POISONED="${NUM_POISONED:-3}"
NUM_HONESTS="${NUM_HONESTS:-7}"

# Class prior of the least-represented class, for the saturation index.
# CIFAR-10 balanced -> 0.1.
MIN_PI="${MIN_PI:-0.1}"

RUN_EXPERT="${RUN_EXPERT:-0}"
EXPERT_CONFIG="${EXPERT_CONFIG:-$CONFIG_ROOT/train_expert.toml}"

DRY_RUN="${DRY_RUN:-0}"

EXPERT_TIME="${EXPERT_TIME:-1-00:00:00}"
POLICY_TIME="${POLICY_TIME:-1-00:00:00}"
FLIPS_TIME="${FLIPS_TIME:-01:00:00}"     # CPU-bound, minutes
FLIPS_MEM="${FLIPS_MEM:-16G}"
USER_TIME="${USER_TIME:-1-00:00:00}"

GAMMA=$(python - <<PY
print(${NUM_POISONED} / (${NUM_POISONED} + ${NUM_HONESTS}))
PY
)

echo "[CHECK] gamma = num_poisoned/(num_poisoned+num_honests) = $GAMMA"

# ---------------------------------------------------------------------------
# Guardrails on the budget, evaluated per cell BEFORE any submission.
#
#   beta_local <= 1        (lem:beta-bar; beta_local > 1 is infeasible)
#   s_beta = beta_global / (gamma * min_y pi_y)
#            > 1 means the per-class capacities bind: the lambda = beta
#            constraint loses its theoretical justification there
#            (prop:budget-match assumes beta <= gamma * min_y pi_y), so
#            lambda should be swept rather than locked.
# ---------------------------------------------------------------------------
INFEASIBLE=0
SATURATED=0

for beta in "${BETAS[@]}"; do
    read -r beta_global s_beta feasible <<< "$(python - <<PY
g = ${GAMMA}
b_local = ${beta}
b_global = g * b_local
s = b_global / (g * ${MIN_PI})
print(f"{b_global:.5f} {s:.3f} {int(b_local <= 1.0)}")
PY
)"
    if [ "$feasible" = "0" ]; then
        echo "[ABORT] beta_local=$beta > 1 : a worker cannot flip more than its own shard." >&2
        INFEASIBLE=$((INFEASIBLE + 1))
        continue
    fi
    printf '[CHECK] beta_local=%-6s beta_global=%-8s s_beta=%-6s' "$beta" "$beta_global" "$s_beta"
    if python -c "import sys; sys.exit(0 if ${s_beta} > 1.0 else 1)"; then
        echo "  SATURATED -- per-class caps bind, lambda=beta not justified here"
        SATURATED=$((SATURATED + 1))
    else
        echo "  ok"
    fi
done

if [ "$INFEASIBLE" -gt 0 ]; then
    echo "[ABORT] $INFEASIBLE infeasible budget(s); fix BETAS and re-run." >&2
    exit 1
fi

if [ "$SATURATED" -gt 0 ]; then
    cat >&2 <<MSG

[WARN] $SATURATED of ${#BETAS[@]} budgets are in the saturated regime (s_beta > 1).
       There, prop:budget-match does not hold and lambda = beta has no
       theoretical backing: the extra budget beyond gamma*pi_source cannot be
       spent on the source class at all. Sweep lambda_poison explicitly in the
       generated configs rather than leaving it at "beta", or restrict BETAS to
       the unsaturated cells, before drawing conclusions from these runs.

MSG
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
MISSING=0
MISMATCH=0

require_config() {
    local path="$1"
    if [ ! -f "$path" ]; then
        echo "[CONFIG] missing: $path" >&2
        MISSING=$((MISSING + 1))
    fi
}

# federated_policy_to_flips raises if its num_honests/num_poisoned differ from
# what the policy artifact recorded. Catch it here rather than at run time.
check_worker_counts() {
    local policy_cfg="$1" flips_cfg="$2" tag="$3"
    local p_np p_nh f_np f_nh
    p_np=$(grep -E '^\s*num_poisoned\s*=' "$policy_cfg" | head -1 | tr -dc '0-9')
    p_nh=$(grep -E '^\s*num_honests\s*='  "$policy_cfg" | head -1 | tr -dc '0-9')
    f_np=$(grep -E '^\s*num_poisoned\s*=' "$flips_cfg"  | head -1 | tr -dc '0-9')
    f_nh=$(grep -E '^\s*num_honests\s*='  "$flips_cfg"  | head -1 | tr -dc '0-9')
    if [ "$p_np" != "$f_np" ] || [ "$p_nh" != "$f_nh" ]; then
        echo "[CONFIG] worker-count mismatch in $tag: policy=($p_np,$p_nh) flips=($f_np,$f_nh)" >&2
        MISMATCH=$((MISMATCH + 1))
    fi
}

cell_tag() {   # seed beta agg
    printf 's%s_b%s_%s_%svs%s' "$1" "${2//./p}" "$3" "$NUM_POISONED" "$NUM_HONESTS"
}

# ---------------------------------------------------------------------------
# Build the job lists.
# ---------------------------------------------------------------------------
POLICY_JOBS=()
FLIPS_JOBS=()
USER_JOBS=()

for seed in "${SEEDS[@]}"; do
  for beta in "${BETAS[@]}"; do
    for agg in "${AGGS[@]}"; do
      tag="$(cell_tag "$seed" "$beta" "$agg")"

      policy_cfg="$CONFIG_ROOT/policy/${tag}.toml"
      flips_cfg="$CONFIG_ROOT/flips/${tag}.toml"
      user_cfg="$CONFIG_ROOT/user/${tag}.toml"

      require_config "$policy_cfg"
      require_config "$flips_cfg"
      require_config "$user_cfg"

      if [ -f "$policy_cfg" ] && [ -f "$flips_cfg" ]; then
          check_worker_counts "$policy_cfg" "$flips_cfg" "$tag"
      fi

      POLICY_JOBS+=("python run_experiment.py $policy_cfg|pol_policy_${tag}")
      FLIPS_JOBS+=("python run_experiment.py $flips_cfg|pol_flips_${tag}")
      USER_JOBS+=("python run_experiment.py $user_cfg|pol_user_${tag}")
    done
  done
done

if [ "$RUN_EXPERT" = "1" ]; then
    require_config "$EXPERT_CONFIG"
fi

if [ "$MISSING" -gt 0 ]; then
    echo "[ABORT] $MISSING config(s) missing -- generate them first:" >&2
    echo "        python -m modules.federated_optimizing_trigger_policy.gen_configs" >&2
    exit 1
fi

if [ "$MISMATCH" -gt 0 ]; then
    echo "[ABORT] $MISMATCH cell(s) with policy/flips worker-count mismatch." >&2
    echo "        federated_policy_to_flips would raise a ValueError at run time." >&2
    exit 1
fi

echo "[PLAN] campaign=$CAMPAIGN module=federated_optimizing_trigger_policy"
echo "[PLAN] cells=${#POLICY_JOBS[@]} (seeds=${#SEEDS[@]} x betas=${#BETAS[@]} x aggs=${#AGGS[@]})"
echo "[PLAN] num_poisoned=$NUM_POISONED num_honests=$NUM_HONESTS gamma=$GAMMA"
echo "[PLAN] phases: $([ "$RUN_EXPERT" = 1 ] && echo 'EXPERT -> ')POLICY -> FLIPS -> USER"

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY-RUN] nothing submitted."
    printf '[DRY-RUN] %s\n' "${POLICY_JOBS[@]}"
    exit 0
fi

preflight_slurm || exit 1

# ---------------------------------------------------------------------------
# Submit.
# ---------------------------------------------------------------------------
DEP=""

if [ "$RUN_EXPERT" = "1" ]; then
    TIME_PER_TASK="$EXPERT_TIME"
    EXPERT_JOBS=("python run_experiment.py $EXPERT_CONFIG|pol_expert")
    submit_job_pool_slurm EXPERT_JOBS "expert" "" || exit 1
    barrier=$(submit_barrier_slurm "pol_barrier_expert" "afterok:$(join_job_ids "${SUBMITTED_JOB_IDS[@]}")") || exit 1
    DEP="afterok:$barrier"
    echo "[PHASE] EXPERT done -> barrier $barrier"
else
    echo "[PHASE] EXPERT skipped (RUN_EXPERT=0) -- expert checkpoints assumed present."
    echo "[PHASE] they MUST be the same ones used by the two sibling campaigns."
fi

TIME_PER_TASK="$POLICY_TIME"
submit_job_pool_slurm POLICY_JOBS "policy" "$DEP" || exit 1
policy_barrier=$(submit_barrier_slurm "pol_barrier_policy" "afterok:$(join_job_ids "${SUBMITTED_JOB_IDS[@]}")") || exit 1
echo "[PHASE] POLICY submitted -> barrier $policy_barrier"

# FLIPS is CPU-bound (drawing indices, writing .npy shards); it still requests
# one GPU because --gres=gpu:0 is rejected on some Slurm builds. Lower the
# walltime and memory instead, so these short jobs do not hold a full slot.
TIME_PER_TASK="$FLIPS_TIME"
MEM_PER_TASK="$FLIPS_MEM"
submit_job_pool_slurm FLIPS_JOBS "flips" "afterok:$policy_barrier" || exit 1
flips_barrier=$(submit_barrier_slurm "pol_barrier_flips" "afterok:$(join_job_ids "${SUBMITTED_JOB_IDS[@]}")") || exit 1
echo "[PHASE] FLIPS submitted -> barrier $flips_barrier"

TIME_PER_TASK="$USER_TIME"
MEM_PER_TASK="${MEM_PER_TASK_USER:-32G}"
submit_job_pool_slurm USER_JOBS "user" "afterok:$flips_barrier" || exit 1
echo "[PHASE] USER submitted."

echo "[DONE] campaign $CAMPAIGN submitted; job ids in $LOG_DIR/jobids_*.txt"
echo "[NOTE] the POLICY phase retrains its own model at every outer step, so its"
echo "[NOTE] walltime scales with n_steps x epochs -- check the first job's rate"
echo "[NOTE] before the whole pool starts competing for the node."