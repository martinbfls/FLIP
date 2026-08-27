#!/bin/bash
###
# scripts/run_policy_campaign.sh
#
# Etape 6.2 of the "solveur QP couple, etapes 1 a 5" follow-up task: ONE command that generates
# the policy_solver comparison campaign's configs, validates them, and submits the full chain
# (train_expert [optional, SHARED across every cell] -> policy -> policy_to_flips -> train_user)
# with afterok barriers between phases -- mirrors orchestrate_runs_policy_slurm.sh's structure,
# but drives modules/federated_optimizing_trigger_policy/gen_configs.py's
# generate_policy_solver_campaign (config GENERATION lives there, in Python, reusing its own
# schema-backed validate_config -- this script does not duplicate that logic in bash).
#
# Grid (the SOLVER axis is the factor of interest; overridable from the environment):
#   SOLVERS = "descent qp"                # rem:solver's solver (b) vs (a)
#   SEEDS   = "0 1 2"
#   BETAS   = "0.033 0.10 0.33"           # LOCAL rate, swept directly (NOT a target global
#                                           budget -- see generate_cell's beta_local override)
#   AGGS    = "mean trmean"               # LOGGING-ONLY, see generate_policy_solver_campaign's
#                                           docstring: no real trimmed-mean aggregation exists in
#                                           this module -- cells differing only in agg are
#                                           functionally IDENTICAL runs under a different name.
# 2 x 3 x 3 x 2 = 36 cells, 3 jobs each (policy/flips/user) = 108 jobs, plus 1 shared expert job
# if RUN_EXPERT=1.
#
# Usage:
#   SLURM_ACCOUNT=<account> ./scripts/run_policy_campaign.sh
#   SLURM_ACCOUNT=<account> DRY_RUN=1 ./scripts/run_policy_campaign.sh
###

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$(dirname "$SCRIPT_DIR")}"
export BASE_DIR
cd "$BASE_DIR"

# shellcheck source=../slurm_lib.sh
source "$BASE_DIR/slurm_lib.sh"

RUN_EXPERT="${RUN_EXPERT:-0}"
DRY_RUN="${DRY_RUN:-0}"

EXPERT_TIME="${EXPERT_TIME:-1-00:00:00}"
POLICY_TIME="${POLICY_TIME:-1-00:00:00}"
FLIPS_TIME="${FLIPS_TIME:-01:00:00}"
FLIPS_MEM="${FLIPS_MEM:-16G}"
USER_TIME="${USER_TIME:-1-00:00:00}"

read -ra SOLVERS <<< "${SOLVERS:-descent qp}"
read -ra SEEDS   <<< "${SEEDS:-0 1 2}"
read -ra BETAS   <<< "${BETAS:-0.033 0.10 0.33}"
read -ra AGGS    <<< "${AGGS:-mean trmean}"

MODEL_FLAG="${MODEL_FLAG:-r32p}"
DATASET="${DATASET:-cifar}"

# PYTHON_BIN resolution: gen_configs.py needs `toml` (schema validation) -- a bare `python3` on
# the login node's default PATH is frequently a system interpreter without it, even though the
# project's own conda env (slurm_lib.sh's CONDA_ENV, used inside every submitted job) has it.
# Prefer an explicit PYTHON_BIN; otherwise fall back to that conda env's interpreter (checked
# for `toml` before use) rather than failing on whatever `python3` happens to resolve to here.
if [ -n "${PYTHON_BIN:-}" ]; then
    : # explicit override wins, no detection needed
elif command -v python3 >/dev/null 2>&1 && python3 -c "import toml" >/dev/null 2>&1; then
    PYTHON_BIN="python3"
elif [ -x "${CONDA_ENV:-/shared/data1/Projects/DLWP/j1067582/martin/FLIP/envs/flip_x86_64}/bin/python" ]; then
    PYTHON_BIN="${CONDA_ENV:-/shared/data1/Projects/DLWP/j1067582/martin/FLIP/envs/flip_x86_64}/bin/python"
    echo "[NOTE] python3 lacks 'toml' -- using $PYTHON_BIN instead (set PYTHON_BIN to override)."
else
    PYTHON_BIN="python3"
fi
if ! "$PYTHON_BIN" -c "import toml" >/dev/null 2>&1; then
    echo "[ABORT] $PYTHON_BIN has no 'toml' module -- activate the project's conda env (e.g." >&2
    echo "        'conda activate flip_x86_64') or set PYTHON_BIN to an interpreter that has it." >&2
    exit 1
fi

echo "[NOTE] AGGS (${AGGS[*]}) is a LOGGING-ONLY axis (same convention as"
echo "[NOTE] gen_configs.py's own AGG_METHODS): federated_optimizing_trigger_policy implements"
echo "[NOTE] (P^mean), mean aggregation only -- no trimmed-mean variant exists in this module."
echo "[NOTE] Cells differing only in agg are functionally IDENTICAL runs under a different name."

# ---------------------------------------------------------------------------
# Step 1: generate + validate every cell's configs via gen_configs.py's own API (Etape 6.1) --
# this is also where infeasible cells (beta_local > 1) are caught; a worker-count mismatch
# between policy/flips configs cannot occur here (both templates share the SAME NUM_POISONED/
# NUM_HONESTS constants for every cell), but is still checked below defensively.
# ---------------------------------------------------------------------------
GEN_OUTPUT="$("$PYTHON_BIN" - "$MODEL_FLAG" "$DATASET" "${SOLVERS[*]}" "${SEEDS[*]}" "${BETAS[*]}" "${AGGS[*]}" <<'PYEOF'
import sys
sys.path.insert(0, ".")
import modules.federated_optimizing_trigger_policy.gen_configs as gc

model_flag, dataset, solvers_s, seeds_s, betas_s, aggs_s = sys.argv[1:7]
solvers = solvers_s.split()
seeds = [int(x) for x in seeds_s.split()]
betas = [float(x) for x in betas_s.split()]
aggs = aggs_s.split()

# Config generation always happens for real (cheap, idempotent local file writes) --
# DRY_RUN governs only whether this script goes on to SUBMIT anything to Slurm, below.
cells, refused = gc.generate_policy_solver_campaign(
    solvers=solvers, seeds=seeds, betas=betas, aggs=aggs,
    model_flag=model_flag, dataset=dataset, dry_run=False,
)

print(f"CELLS_TOTAL={len(cells)}")
print(f"REFUSED_TOTAL={len(refused)}")
for solver, seed, beta_local, agg, reason in refused:
    print(f"REFUSED\t{solver}\t{seed}\t{beta_local}\t{agg}\t{reason}")
for c in cells:
    policy_cfg = flips_cfg = user_cfg = ""
    for p in c["paths"]:
        s = str(p)
        if "policy_opt" in s:
            policy_cfg = s
        elif "policy_to_flips" in s:
            flips_cfg = s
        elif "train_user" in s:
            user_cfg = s
    print(
        f"CELL\t{c['solver']}\t{c['seed']}\t{c['beta_local']}\t{c['agg']}\t"
        f"{c['beta_global']:.6f}\t{c['s_beta']:.4f}\t{policy_cfg}\t{flips_cfg}\t{user_cfg}"
    )
PYEOF
)"

CELLS_TOTAL=$(echo "$GEN_OUTPUT" | grep '^CELLS_TOTAL=' | cut -d= -f2)
REFUSED_TOTAL=$(echo "$GEN_OUTPUT" | grep '^REFUSED_TOTAL=' | cut -d= -f2)

echo "[PLAN] generated $CELLS_TOTAL cell(s), $REFUSED_TOTAL refused (infeasible: beta_local > 1)."
if [ "$REFUSED_TOTAL" -gt 0 ]; then
    echo "$GEN_OUTPUT" | grep '^REFUSED\b' | while IFS=$'\t' read -r _ solver seed beta agg reason; do
        echo "[ABORT] REFUSED solver=$solver seed=$seed beta=$beta agg=$agg: $reason" >&2
    done
    echo "[ABORT] $REFUSED_TOTAL infeasible cell(s) -- adjust BETAS or NUM_POISONED/NUM_HONESTS." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 2: per-cell s_beta display + config-existence + worker-count-mismatch checks (defensive
# -- config generation above already used a schema-validated, single code path for both
# templates, so a mismatch here would indicate a gen_configs.py bug, not a user config error).
# ---------------------------------------------------------------------------
MISSING=0
MISMATCH=0

EXPERT_JOBS=()
POLICY_JOBS=()
FLIPS_JOBS=()
USER_JOBS=()
SEEN_EXPERT=""

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

echo "$GEN_OUTPUT" | grep '^CELL\b' | while IFS=$'\t' read -r _ solver seed beta agg beta_global s_beta policy_cfg flips_cfg user_cfg; do
    warn=""
    awk -v s="$s_beta" 'BEGIN{exit !(s > 1)}' && warn=" [WARNING: s_beta > 1 -- lambda=beta not theoretically justified, rem:saturated]"
    echo "[CELL] solver=$solver seed=$seed beta_local=$beta agg=$agg beta_global=$beta_global s_beta=$s_beta$warn"
done

while IFS=$'\t' read -r _ solver seed beta agg beta_global s_beta policy_cfg flips_cfg user_cfg; do
    tag="${solver}_${agg}_beta${beta}_seed${seed}"

    for cfg in "$policy_cfg" "$flips_cfg" "$user_cfg"; do
        if [ ! -f "$cfg" ]; then
            echo "[CONFIG] missing: $cfg" >&2
            MISSING=$((MISSING + 1))
        fi
    done
    if [ -f "$policy_cfg" ] && [ -f "$flips_cfg" ]; then
        check_worker_counts "$policy_cfg" "$flips_cfg" "$tag"
    fi

    if [[ " $SEEN_EXPERT " != *" ${MODEL_FLAG} "* ]]; then
        SEEN_EXPERT="$SEEN_EXPERT $MODEL_FLAG"
        expert_cfg="$(dirname "$(dirname "$policy_cfg")" | sed "s#$BASE_DIR/experiments/##")"
        # Shared expert config path: EXP_BASE/train_expert/<model>_1xs -- identical for every
        # cell (Etape 6.2's "experts shared, generated once" requirement), so this only fires
        # for the FIRST cell processed.
        expert_rel="federated_experiments/threat_model_expert_policy/train_expert/${MODEL_FLAG}_1xs"
        EXPERT_JOBS+=("python run_experiment.py $expert_rel|pol_expert_${MODEL_FLAG}")
    fi

    rel_policy="${policy_cfg#"$BASE_DIR/experiments/"}"
    rel_flips="${flips_cfg#"$BASE_DIR/experiments/"}"
    rel_user="${user_cfg#"$BASE_DIR/experiments/"}"
    rel_policy="${rel_policy%/config.toml}"
    rel_flips="${rel_flips%/config.toml}"
    rel_user="${rel_user%/config.toml}"

    POLICY_JOBS+=("python run_experiment.py $rel_policy|camp_policy_${tag}")
    FLIPS_JOBS+=("python run_experiment.py $rel_flips|camp_flips_${tag}")
    USER_JOBS+=("python run_experiment.py $rel_user|camp_user_${tag}")
done <<< "$(echo "$GEN_OUTPUT" | grep '^CELL\b')"

if [ "$MISSING" -gt 0 ]; then
    echo "[ABORT] $MISSING config(s) missing after generation -- this should not happen; check" >&2
    echo "        gen_configs.py's generate_policy_solver_campaign for a write failure." >&2
    exit 1
fi
if [ "$MISMATCH" -gt 0 ]; then
    echo "[ABORT] $MISMATCH cell(s) with policy/flips worker-count mismatch." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Step 3: resume support -- a cell whose FINAL artifact (train_user's output) already exists is
# dropped from all three job lists (policy/flips/user), so a re-run of this script only submits
# what is actually missing.
#
# NOTE: output_dir for the train_user config is THE SAME DIRECTORY config.toml itself was just
# written into (see gen_configs.py's TRAIN_USER_TEMPLATE: output_dir = "{train_user_dir}",
# and the config is written to train_user_dir/config.toml) -- so "output_dir is non-empty" is
# true the instant config generation runs, on every cell, on every invocation, whether or not
# the job ever executed. The actual completion signal federated_train_user writes is
# paccs.npy/caccs.npy (np.save'd only at the very end of its run -- also what
# collect_policy_campaign.py itself reads) -- check for those specifically, not "any file".
# ---------------------------------------------------------------------------
filter_done() {
    local -n jobs_ref=$1
    local kept=()
    local job cfg_rel out_dir
    for job in "${jobs_ref[@]}"; do
        cfg_rel="${job%%|*}"
        cfg_rel="${cfg_rel#python run_experiment.py }"
        out_dir=$(grep -E '^\s*output_dir\s*=' "experiments/${cfg_rel}/config.toml" 2>/dev/null \
            | head -1 | sed -E 's/^[^"]*"([^"]*)".*/\1/')
        if [ -n "$out_dir" ] && [ -f "$out_dir/paccs.npy" ] && [ -f "$out_dir/caccs.npy" ]; then
            continue
        fi
        kept+=("$job")
    done
    jobs_ref=("${kept[@]}")
}

filter_done USER_JOBS
# Only resubmit flips/policy for cells whose USER job still needs to run -- a simple
# tag-suffix intersection (USER_JOBS' safe names end the same way FLIPS_JOBS'/POLICY_JOBS' do).
if [ "${#USER_JOBS[@]}" -lt "$CELLS_TOTAL" ]; then
    declare -A remaining_tags
    for job in "${USER_JOBS[@]}"; do
        tag="${job##*camp_user_}"
        remaining_tags["$tag"]=1
    done
    filtered_policy=()
    for job in "${POLICY_JOBS[@]}"; do
        tag="${job##*camp_policy_}"
        [ -n "${remaining_tags[$tag]:-}" ] && filtered_policy+=("$job")
    done
    POLICY_JOBS=("${filtered_policy[@]}")
    filtered_flips=()
    for job in "${FLIPS_JOBS[@]}"; do
        tag="${job##*camp_flips_}"
        [ -n "${remaining_tags[$tag]:-}" ] && filtered_flips+=("$job")
    done
    FLIPS_JOBS=("${filtered_flips[@]}")
fi

echo "[PLAN] module=federated_optimizing_trigger_policy solver-comparison campaign"
echo "[PLAN] solvers=${SOLVERS[*]} seeds=${SEEDS[*]} betas=${BETAS[*]} aggs=${AGGS[*]}"
echo "[PLAN] cells=$CELLS_TOTAL (=solvers x seeds x betas x aggs); after resume-filtering:"
echo "[PLAN]   policy=${#POLICY_JOBS[@]} flips=${#FLIPS_JOBS[@]} user=${#USER_JOBS[@]} jobs"
echo "[PLAN] phases: $([ "$RUN_EXPERT" = 1 ] && echo 'EXPERT [SHARED] -> ')POLICY -> FLIPS -> USER"

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY-RUN] nothing submitted."
    [ "$RUN_EXPERT" = "1" ] && printf '[DRY-RUN] %s\n' "${EXPERT_JOBS[@]}"
    printf '[DRY-RUN] %s\n' "${POLICY_JOBS[@]}"
    printf '[DRY-RUN] %s\n' "${FLIPS_JOBS[@]}"
    printf '[DRY-RUN] %s\n' "${USER_JOBS[@]}"
    exit 0
fi

if [ "${#USER_JOBS[@]}" -eq 0 ]; then
    echo "[DONE] every cell's final artifacts already exist -- nothing to submit."
    exit 0
fi

preflight_slurm || exit 1

DEP=""

if [ "$RUN_EXPERT" = "1" ]; then
    TIME_PER_TASK="$EXPERT_TIME"
    submit_job_pool_slurm EXPERT_JOBS "camp_expert" "" || exit 1
    expert_barrier=$(submit_barrier_slurm "camp_barrier_expert" "afterok:$(join_job_ids "${SUBMITTED_JOB_IDS[@]}")") || exit 1
    DEP="afterok:$expert_barrier"
    echo "[PHASE] EXPERT submitted (${#EXPERT_JOBS[@]} job, SHARED across every cell) -> barrier $expert_barrier"
else
    echo "[PHASE] EXPERT skipped (RUN_EXPERT=0) -- shared expert checkpoints assumed present."
fi

TIME_PER_TASK="$POLICY_TIME"
if [ "${#POLICY_JOBS[@]}" -gt 0 ]; then
    submit_job_pool_slurm POLICY_JOBS "camp_policy" "$DEP" || exit 1
    policy_barrier=$(submit_barrier_slurm "camp_barrier_policy" "afterok:$(join_job_ids "${SUBMITTED_JOB_IDS[@]}")") || exit 1
    echo "[PHASE] POLICY submitted (${#POLICY_JOBS[@]} jobs) -> barrier $policy_barrier"
    DEP="afterok:$policy_barrier"
fi

TIME_PER_TASK="$FLIPS_TIME"
MEM_PER_TASK="$FLIPS_MEM"
if [ "${#FLIPS_JOBS[@]}" -gt 0 ]; then
    submit_job_pool_slurm FLIPS_JOBS "camp_flips" "$DEP" || exit 1
    flips_barrier=$(submit_barrier_slurm "camp_barrier_flips" "afterok:$(join_job_ids "${SUBMITTED_JOB_IDS[@]}")") || exit 1
    echo "[PHASE] FLIPS submitted (${#FLIPS_JOBS[@]} jobs) -> barrier $flips_barrier"
    DEP="afterok:$flips_barrier"
fi

TIME_PER_TASK="$USER_TIME"
MEM_PER_TASK="${MEM_PER_TASK_USER:-32G}"
submit_job_pool_slurm USER_JOBS "camp_user" "$DEP" || exit 1
echo "[PHASE] USER submitted (${#USER_JOBS[@]} jobs)."
user_barrier=$(submit_barrier_slurm "camp_barrier_user" "afterok:$(join_job_ids "${SUBMITTED_JOB_IDS[@]}")") || exit 1

# ---------------------------------------------------------------------------
# Step 4 (Etape 6.3, automatic): once every USER job has completed, collect_policy_campaign.py
# runs by itself -- this is what makes the whole campaign (submit -> wait -> collect) a single
# command instead of a submit-then-remember-to-collect-later manual step.
# ---------------------------------------------------------------------------
CAMPAIGN_EXP_BASE="$BASE_DIR/experiments/federated_experiments/threat_model_expert_policy"
TIME_PER_TASK="00:15:00"
MEM_PER_TASK="8G"
GPUS_PER_TASK="0"
CPUS_PER_TASK="1"
collect_jobid=$(submit_job_slurm \
    "python scripts/collect_policy_campaign.py $CAMPAIGN_EXP_BASE" \
    "camp_collect" "afterok:$user_barrier") || exit 1
echo "[PHASE] COLLECT submitted (job $collect_jobid, after all USER jobs) -> $CAMPAIGN_EXP_BASE/{campaign_results.csv,report.md}"

echo "[DONE] campaign submitted end-to-end; job ids in $LOG_DIR/jobids_*.txt"
echo "[NEXT] nothing further needed -- report.md lands in $CAMPAIGN_EXP_BASE once camp_collect finishes."
