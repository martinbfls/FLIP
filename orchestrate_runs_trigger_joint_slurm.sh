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
# Sweep axes and config paths below are read DIRECTLY from
# modules/federated_generate_labels_trigger_joint/gen_configs.py (the module
# that actually writes these configs) instead of being duplicated here --
# gen_configs.py has no env-var/CLI overrides of its own, so any hardcoded
# copy of its constants silently drifts out of sync the first time someone
# edits the grid there. Edit the grid in gen_configs.py; this script follows.
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

EXPERT_TIME="${EXPERT_TIME:-1-00:00:00}"
GEN_TIME="${GEN_TIME:-1-00:00:00}"
FLIPS_TIME="${FLIPS_TIME:-02:00:00}"
USER_TIME="${USER_TIME:-1-00:00:00}"

# ---------------------------------------------------------------------------
# Pull the sweep grid + directory layout straight out of gen_configs.py, as
# one JSON blob, so this script can never disagree with the module that
# actually wrote the configs on disk.
# ---------------------------------------------------------------------------
PLAN_JSON=$(python - <<'PY'
import json
from pathlib import Path

from modules.federated_generate_labels_trigger_joint import gen_configs as g

exp_root = Path("experiments").resolve()
exp_base_rel = str(g.EXP_BASE.relative_to(exp_root))

print(json.dumps({
    "model_flags": g.MODEL_FLAGS,
    "datasets": g.DATASETS,
    "agg_methods": g.AGG_METHODS,
    "seeds": g.SEEDS,
    "budgets": g.BUDGETS,
    "num_poisoned": g.NUM_POISONED,
    "num_honests": g.NUM_HONESTS,
    "checkpoint_sampling": g.CHECKPOINT_SAMPLING,
    "indirect_checkpoint_sampling": g.INDIRECT_MODULE_CHECKPOINT_SAMPLING,
    "exp_base_rel": exp_base_rel,
}))
PY
)

read -ra MODEL_FLAGS <<< "$(python -c "import json,sys; print(' '.join(json.loads(sys.argv[1])['model_flags']))" "$PLAN_JSON")"
read -ra DATASETS    <<< "$(python -c "import json,sys; print(' '.join(json.loads(sys.argv[1])['datasets']))" "$PLAN_JSON")"
read -ra AGG_METHODS <<< "$(python -c "import json,sys; print(' '.join(json.loads(sys.argv[1])['agg_methods']))" "$PLAN_JSON")"
read -ra SEEDS       <<< "$(python -c "import json,sys; print(' '.join(str(x) for x in json.loads(sys.argv[1])['seeds']))" "$PLAN_JSON")"
read -ra BUDGETS     <<< "$(python -c "import json,sys; print(' '.join(str(x) for x in json.loads(sys.argv[1])['budgets']))" "$PLAN_JSON")"

NUM_POISONED=$(python -c "import json,sys; print(json.loads(sys.argv[1])['num_poisoned'])" "$PLAN_JSON")
NUM_HONESTS=$(python -c "import json,sys; print(json.loads(sys.argv[1])['num_honests'])" "$PLAN_JSON")
CHECKPOINT_SAMPLING=$(python -c "import json,sys; print(json.loads(sys.argv[1])['checkpoint_sampling'])" "$PLAN_JSON")
INDIRECT_CHECKPOINT_SAMPLING=$(python -c "import json,sys; print(json.loads(sys.argv[1])['indirect_checkpoint_sampling'])" "$PLAN_JSON")
EXP_BASE_REL=$(python -c "import json,sys; print(json.loads(sys.argv[1])['exp_base_rel'])" "$PLAN_JSON")

if [ "$CHECKPOINT_SAMPLING" != "$INDIRECT_CHECKPOINT_SAMPLING" ]; then
    echo "[NOTE] gen_configs.py's CHECKPOINT_SAMPLING=$CHECKPOINT_SAMPLING differs from the" >&2
    echo "[NOTE] sibling indirect module's default ($INDIRECT_CHECKPOINT_SAMPLING) -- an" >&2
    echo "[NOTE] indirect-vs-joint comparison crosses this factor too. This is only a" >&2
    echo "[NOTE] property of the already-generated configs; nothing to set here." >&2
fi

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
# train_expert is per model_flag only and shared across cells with that
# model, so it is deduplicated instead of resubmitted per cell.
# ---------------------------------------------------------------------------
EXPERT_JOBS=()
GEN_JOBS=()
FLIPS_JOBS=()
USER_JOBS=()

SEEN_EXPERT=""

for model_flag in "${MODEL_FLAGS[@]}"; do
  if [[ " $SEEN_EXPERT " != *" $model_flag "* ]]; then
    SEEN_EXPERT="$SEEN_EXPERT $model_flag"
    expert_cfg="$EXP_BASE_REL/train_expert/${model_flag}_1xs"
    require_config "$expert_cfg"
    EXPERT_JOBS+=("python run_experiment.py $expert_cfg|joint_expert_${model_flag}")
  fi

  for dataset in "${DATASETS[@]}"; do
    for agg in "${AGG_METHODS[@]}"; do
      for seed in "${SEEDS[@]}"; do
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
    submit_job_pool_slurm EXPERT_JOBS "expert" "" || exit 1
    expert_barrier=$(submit_barrier_slurm "joint_barrier_expert" "afterok:$(join_job_ids "${SUBMITTED_JOB_IDS[@]}")") || exit 1
    DEP="afterok:$expert_barrier"
    echo "[PHASE] EXPERT submitted (${#EXPERT_JOBS[@]} jobs) -> barrier $expert_barrier"
else
    echo "[PHASE] EXPERT skipped (RUN_EXPERT=0) -- expert checkpoints assumed present."
    echo "[PHASE] they MUST be the same ones used by the sibling indirect campaign."
fi

TIME_PER_TASK="$GEN_TIME"
submit_job_pool_slurm GEN_JOBS "gen" "$DEP" || exit 1
gen_barrier=$(submit_barrier_slurm "joint_barrier_gen" "afterok:$(join_job_ids "${SUBMITTED_JOB_IDS[@]}")") || exit 1
echo "[PHASE] GEN submitted (${#GEN_JOBS[@]} jobs) -> barrier $gen_barrier"

TIME_PER_TASK="$FLIPS_TIME"
submit_job_pool_slurm FLIPS_JOBS "flips" "afterok:$gen_barrier" || exit 1
flips_barrier=$(submit_barrier_slurm "joint_barrier_flips" "afterok:$(join_job_ids "${SUBMITTED_JOB_IDS[@]}")") || exit 1
echo "[PHASE] FLIPS submitted (${#FLIPS_JOBS[@]} jobs) -> barrier $flips_barrier"

TIME_PER_TASK="$USER_TIME"
submit_job_pool_slurm USER_JOBS "user" "afterok:$flips_barrier" || exit 1
echo "[PHASE] USER submitted (${#USER_JOBS[@]} jobs)."

echo "[DONE] campaign submitted; job ids in $LOG_DIR/jobids_*.txt"
echo "[NOTE] this module builds a second-order graph (create_graph=True) on the"
echo "[NOTE] expert step: memory is ~2.6x the non-differentiable variant on a CPU"
echo "[NOTE] proxy measurement. Watch the first GEN job for OOM before the rest start."
