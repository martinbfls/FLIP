#!/bin/bash
###
# orchestrate_runs_trigger_joint_hardening_slurm.sh
#
# Shared Slurm launcher for the P0-P7 robustness-hardening protocol's Etapes 1-4 (see the
# accompanying diagnostic writeup's "Protocole experimental"). NOT called directly for a
# single step -- orchestrate_runs_trigger_joint_hardening_step{1,2,3,4}_slurm.sh set STEPS to
# one step and exec this script; orchestrate_runs_trigger_joint_hardening_all_slurm.sh sets
# STEPS="1 2 3 4" to submit every step in one go (see "Multi-step submission" below).
#
# Bash 3.2 compatible on purpose (no `declare -A`, no `mapfile`/`readarray`) -- macOS's system
# bash is 3.2, and that's what this script's own DRY_RUN was actually verified against before
# handing this over; a bash-4-only script could not be checked without cluster access.
#
# Configs for every requested step must already exist -- generate them first (once per step):
#   python -m modules.federated_generate_labels_trigger_joint.gen_configs_hardening_steps \
#       --step $STEP [--defense-aggs trmean multikrum]
#
# Each step is ONE cell, 1 + len(DEFENSE_AGGS) branches (single-user 1v0/mean undefended, and
# one defended federated_3vs7_<agg> branch PER entry of DEFENSE_AGGS -- every step tests BOTH
# trmean and multikrum by default, 2026-09-05 fix: comparing two steps' own single-defense
# numbers used to confound "did this step's fix help" with "is this just a harder defense than
# the other step was tested against" -- see gen_configs_hardening_steps.py's DEFENSE_AGGS):
#
#   [EXPERT]  train_expert                            (SHARED across every requested step --
#        |                                              submitted ONCE, not once per step; see
#        |                                              "Multi-step submission" below)
#   [GEN]     federated_generate_labels_trigger_joint  (ONE cell per step, metrics_log_path
#        |                                              set -- this is what "regarder
#        |                                              cos_delta_to_init, mag_active_rate,
#        |                                              delta_sign_flip_rate" reads)
#   [FLIPS]   federated_select_flips, single-user + one per DEFENSE_AGGS entry
#        |
#   [USER]    federated_train_user x budgets, single-user + one per DEFENSE_AGGS entry
#
# Multi-step submission: train_expert writes to a SEED-KEYED path shared by every step (see
# gen_configs_hardening_steps.py -- train_expert_dir does not depend on the step's own tag), so
# submitting it once PER requested step would have several Slurm jobs racing to write the SAME
# checkpoint files concurrently -- silently corrupting them or wasting redundant GPU-hours, not
# just "inefficient". This script always submits EXACTLY ONE train_expert job (regardless of
# how many steps are requested) and makes every step's own GEN phase depend on that single
# barrier -- the steps' GEN/FLIPS/USER phases then proceed independently and CONCURRENTLY
# (bounded only by the Slurm partition's available GPUs), which is what makes
# orchestrate_runs_trigger_joint_hardening_all_slurm.sh a genuine "submit everything, let it
# run overnight" launcher rather than 4 sequential campaigns.
#
# Usage (from BASE_DIR):
#   STEP=1 SLURM_ACCOUNT=<account> ./orchestrate_runs_trigger_joint_hardening_slurm.sh
#   STEPS="1 2 3 4" SLURM_ACCOUNT=<account> ./orchestrate_runs_trigger_joint_hardening_slurm.sh
#   STEP=1 SLURM_ACCOUNT=<account> DRY_RUN=1 ./orchestrate_runs_trigger_joint_hardening_slurm.sh
###

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="${BASE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
export BASE_DIR
cd "$BASE_DIR"

# shellcheck source=slurm_lib.sh
source "$SCRIPT_DIR/slurm_lib.sh"

STEPS="${STEPS:-${STEP:-}}"
if [ -z "$STEPS" ]; then
    echo "[ABORT] Set STEPS=\"1 2 3 4\" (or STEP=<n> for a single step, or use one of the" >&2
    echo "        per-step / hardening_all wrapper scripts)." >&2
    exit 1
fi
read -ra STEP_LIST <<< "$STEPS"

DRY_RUN="${DRY_RUN:-0}"

EXPERT_TIME="${EXPERT_TIME:-0-00:10:00}"
GEN_TIME="${GEN_TIME:-1-00:00:00}"
FLIPS_TIME="${FLIPS_TIME:-00:10:00}"
USER_TIME="${USER_TIME:-0-01:00:00}"

MODEL_FLAG="${MODEL_FLAG:-r32p}"
DATASET="${DATASET:-cifar}"
SEED="${SEED:-0}"
read -ra BUDGETS <<< "${BUDGETS:-500 2000}"

# Defended aggregators tested for EVERY step -- must match whatever
# gen_configs_hardening_steps.py was actually invoked with (its own default: DEFENSE_AGGS =
# ["trmean", "multikrum"], see that module).
read -ra DEFENSE_AGGS <<< "${DEFENSE_AGGS:-trmean multikrum}"

EXP_BASE_REL="${EXP_BASE_REL:-federated_experiments/threat_model_direct_trigger_joint_hardening_steps}"

step_tag() {
    case "$1" in
        1) echo "step1_p0_p1" ;;
        2) echo "step2_p2_p6" ;;
        3) echo "step3_p5_match" ;;
        4) echo "step4_p4_p3" ;;
        *) echo "[ABORT] STEP must be 1, 2, 3 or 4 (got $1)" >&2; exit 1 ;;
    esac
}

MISSING=0
require_config() {
    local path="$1"
    if [ ! -f "experiments/$path/config.toml" ]; then
        echo "[CONFIG] missing: experiments/$path/config.toml" >&2
        MISSING=$((MISSING + 1))
    fi
}

expert_cfg="$EXP_BASE_REL/train_expert/${MODEL_FLAG}_1xs/seed${SEED}"

# ---------------------------------------------------------------------------
# Pass 1: validate every config exists (EXPERT + every requested step's every branch) BEFORE
# submitting anything -- a single missing file aborts the whole batch rather than leaving a
# half-submitted DAG. Plain existence checks only, no arrays built here (kept bash-3.2-simple:
# see the file's own header for why).
# ---------------------------------------------------------------------------
require_config "$expert_cfg"
for step in "${STEP_LIST[@]}"; do
    tag="$(step_tag "$step")"
    cell="$EXP_BASE_REL/$tag/$MODEL_FLAG/$DATASET/seed${SEED}"
    require_config "$cell/gen_labels_trigger_joint"
    require_config "$cell/select_flips"
    for budget in "${BUDGETS[@]}"; do
        require_config "$cell/train_user_${budget}"
    done
    for agg in "${DEFENSE_AGGS[@]}"; do
        fed_tag="federated_3vs7_${agg}"
        require_config "$cell/$fed_tag/select_flips"
        for budget in "${BUDGETS[@]}"; do
            require_config "$cell/$fed_tag/train_user_${budget}"
        done
    done
done

if [ "$MISSING" -gt 0 ]; then
    echo "[ABORT] $MISSING config(s) missing -- generate them first, per step:" >&2
    echo "        python -m modules.federated_generate_labels_trigger_joint.gen_configs_hardening_steps --step <n> --defense-aggs ${DEFENSE_AGGS[*]}" >&2
    exit 1
fi

echo "[PLAN] hardening protocol STEPS=\"${STEP_LIST[*]}\" exp_base=$EXP_BASE_REL"
echo "[PLAN] model=$MODEL_FLAG dataset=$DATASET seed=$SEED budgets=${BUDGETS[*]}"
DEFENSE_AGGS_CSV="$(IFS=,; echo "${DEFENSE_AGGS[*]}")"
echo "[PLAN] defended branches (every step): federated_3vs7_{$DEFENSE_AGGS_CSV}"
echo "[PLAN] EXPERT submitted ONCE, shared by all requested steps; each step's own"
echo "       GEN -> FLIPS -> USER pipeline then runs independently/concurrently."
if [[ " ${STEP_LIST[*]} " == *" 1 "* ]]; then
    echo "[PLAN] Etape 1 (P0+P1): after its GEN completes, read metrics_log_path (see"
    echo "       gen_labels_trigger_joint/logs/metrics.json under that cell) for"
    echo "       cos_delta_to_init, mag_active_rate, delta_sign_flip_rate BEFORE trusting"
    echo "       the USER-phase results."
fi

# ---------------------------------------------------------------------------
# Pass 2: build + (print or submit) each phase's job list. `EXPERT_JOBS` etc. are plain
# indexed arrays, rebuilt fresh per step inside the loop (bash 3.2 has no associative arrays
# to cache them across two passes, and rebuilding a handful of strings is cheap).
# ---------------------------------------------------------------------------
EXPERT_JOBS=("python run_experiment.py $expert_cfg|hardening_expert")

if [ "$DRY_RUN" = "1" ]; then
    printf '[DRY-RUN] %s\n' "${EXPERT_JOBS[@]}"
fi

if [ "$DRY_RUN" != "1" ]; then
    preflight_slurm || exit 1

    TIME_PER_TASK="$EXPERT_TIME"
    submit_job_pool_slurm EXPERT_JOBS "hardening_expert" "" || exit 1
    expert_barrier=$(submit_barrier_slurm "hardening_barrier_expert" "afterok:$(join_job_ids "${SUBMITTED_JOB_IDS[@]}")") || exit 1
    echo "[PHASE] EXPERT submitted -> barrier $expert_barrier (shared by ${#STEP_LIST[@]} step(s))"
fi

for step in "${STEP_LIST[@]}"; do
    tag="$(step_tag "$step")"
    cell="$EXP_BASE_REL/$tag/$MODEL_FLAG/$DATASET/seed${SEED}"

    GEN_JOBS=("python run_experiment.py $cell/gen_labels_trigger_joint|hardening_gen_${tag}")

    FLIPS_JOBS=("python run_experiment.py $cell/select_flips|hardening_flips_single_${tag}")
    for agg in "${DEFENSE_AGGS[@]}"; do
        FLIPS_JOBS+=("python run_experiment.py $cell/federated_3vs7_${agg}/select_flips|hardening_flips_${agg}_${tag}")
    done

    USER_JOBS=()
    for budget in "${BUDGETS[@]}"; do
        USER_JOBS+=("python run_experiment.py $cell/train_user_${budget}|hardening_user_single_${tag}_${budget}")
    done
    for agg in "${DEFENSE_AGGS[@]}"; do
        for budget in "${BUDGETS[@]}"; do
            USER_JOBS+=("python run_experiment.py $cell/federated_3vs7_${agg}/train_user_${budget}|hardening_user_${agg}_${tag}_${budget}")
        done
    done

    if [ "$DRY_RUN" = "1" ]; then
        printf '[DRY-RUN] %s\n' "${GEN_JOBS[@]}" "${FLIPS_JOBS[@]}" "${USER_JOBS[@]}"
        continue
    fi

    TIME_PER_TASK="$GEN_TIME"
    submit_job_pool_slurm GEN_JOBS "hardening_gen_${tag}" "afterok:$expert_barrier" || exit 1
    gen_barrier=$(submit_barrier_slurm "hardening_barrier_gen_${tag}" "afterok:$(join_job_ids "${SUBMITTED_JOB_IDS[@]}")") || exit 1
    echo "[PHASE step $step] GEN submitted -> barrier $gen_barrier"

    TIME_PER_TASK="$FLIPS_TIME"
    submit_job_pool_slurm FLIPS_JOBS "hardening_flips_${tag}" "afterok:$gen_barrier" || exit 1
    flips_barrier=$(submit_barrier_slurm "hardening_barrier_flips_${tag}" "afterok:$(join_job_ids "${SUBMITTED_JOB_IDS[@]}")") || exit 1
    echo "[PHASE step $step] FLIPS submitted -> barrier $flips_barrier"

    TIME_PER_TASK="$USER_TIME"
    submit_job_pool_slurm USER_JOBS "hardening_user_${tag}" "afterok:$flips_barrier" || exit 1
    echo "[PHASE step $step] USER submitted (${#USER_JOBS[@]} jobs)."
done

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY-RUN] nothing submitted."
    exit 0
fi

echo "[DONE] STEPS=\"${STEP_LIST[*]}\" campaign submitted; job ids in $LOG_DIR/jobids_*.txt"
