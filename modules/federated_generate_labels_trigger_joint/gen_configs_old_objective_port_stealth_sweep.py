"""
Stealth-regularization sweep on top of the old-objective-port winning config (see
gen_configs_old_objective_port.py's own docstring for the base chain and the
num_honests=0/num_poisoned=1/agg_method="mean" generation-time config that broke every robust
aggregator). This generator does NOT touch that campaign's own files -- it is a separate,
sibling campaign that starts from the SAME generation-time hyperparameters (including
lambda_joint=1.0, iterations=25 -- the actual on-disk winning config, which drifted from
gen_configs_old_objective_port.py's own GEN_ITERATIONS=5/no-lambda_joint template; see
STEALTH_BASE_OVERRIDES below) and varies exactly the knobs that affect the trigger's visual/
perceptual footprint:

  - epsilon           (L_infinity budget on delta -- smaller = less visible)
  - lambda_tv         (total-variation penalty -- smoother, less patchy delta)
  - lambda_lpips      (perceptual (LPIPS) similarity penalty -- requires the `lpips` package,
                        see run_module.py's lazy import; 0.0 cells never need it)
  - lambda_penalty    (trigger_penalty(delta, mu) weight -- NOT a visual-stealth term, see
                        run_module.py's own note that this is the OLD unbounded anti-alignment
                        penalty, distinct from trigger_penalty_hinge's stealth ceiling; it IS
                        part of what makes this config break robust aggregators, so it's swept
                        here as a robustness-vs-conspicuousness tradeoff axis, not a pure
                        stealth knob)

One-at-a-time (OFAT) around the baseline, same convention as gen_configs_epsilon_sweep.py /
gen_configs_lpips_compare.py / gen_configs_gradmatch_lambda_sweep.py -- NOT a combinatorial
grid (this module's GEN step retrains an expert from scratch every outer iteration; a full
cross-product would be far too expensive). Edit STEALTH_GRID below to add/remove cells or turn
this into a combined-candidate confirmation run once individual axes have been ranked (same
"combined candidate" workflow gen_configs.py's own REGULARIZATION_GRID docstring describes).

Chain per tag (mirrors gen_configs_old_objective_port.py's `generate()` exactly, just nested
one level deeper under EXP_BASE/MODEL_FLAG/DATASET/<tag>/):
    [BOOTSTRAP]  train_expert -- REUSED, not regenerated: this campaign assumes
                 gen_configs_old_objective_port's own bootstrap (poisoner="1xs") has already
                 been generated and trained once (`python -m ...gen_configs_old_objective_port`
                 then the BOOTSTRAP phase of its orchestrator) -- delta/regularization changes
                 don't affect the clean bootstrap expert, so there is nothing tag-specific to
                 retrain here.
    [GEN]        federated_generate_labels_trigger_joint (one per tag)
    [FLIPS]      federated_select_flips (one per tag, shared across every deployment agg_method)
    [USER]       federated_train_user (agg_method x budget = 3 x 3 = 9 jobs per tag)

Run `python -m modules.federated_generate_labels_trigger_joint.gen_configs_old_objective_port_stealth_sweep --print-grid`
for the authoritative (tag, agg_method, budget) cell list.
"""

import argparse
from pathlib import Path

from modules.base_utils.config_validation import (
    write_config,
    validate_config_file as validate_config,
)
from modules.federated_generate_labels_trigger_joint.gen_configs import (
    check_delta_min_feasible,
)
from modules.federated_generate_labels_trigger_joint.gen_configs_old_objective_port import (
    BOOTSTRAP_OUTPUT_DIR,
    CLUSTER_ROOT,
    DATASET,
    DEPLOY_AGG_METHODS,
    DEPLOY_BUDGETS,
    DEPLOY_NUM_HONESTS,
    DEPLOY_NUM_POISONED,
    GEN_AGG_METHOD,
    GEN_EXPERT_RETRAIN_CHECKPOINT_ITERS,
    GEN_EXPERT_RETRAIN_EPOCHS,
    GEN_EXPERT_RETRAIN_INTERVAL,
    GEN_INIT,
    GEN_LR_DELTA,
    GEN_N_CHECKPOINTS_PER_STEP,
    GEN_NUM_HONESTS,
    GEN_NUM_POISONED,
    GEN_TRIGGER_CONSTRAINT,
    JOINT_TRIGGER_TEMPLATE,
    LR,
    MILESTONES,
    MODEL_FLAG,
    SELECT_FLIPS_TEMPLATE,
    SOURCE_LABEL,
    TARGET_LABEL,
    TRAIN_USER_TEMPLATE,
    WD,
)

# --------------------------------------------------------------------------- #
# Locally-extended copy of gen_configs_old_objective_port's JOINT_TRIGGER_TEMPLATE: splices in
# lambda_joint (not exposed by that template at all -- the on-disk winning config that this
# whole campaign ports has it hand-set to 1.0, see run_module.py's lambda_joint docstring) and
# turns its hardcoded `lambda_tv = 0.0` / `lambda_lpips = 0.0` lines into format slots, WITHOUT
# touching gen_configs_old_objective_port.py itself (every other consumer of its
# JOINT_TRIGGER_TEMPLATE keeps working unchanged).
# --------------------------------------------------------------------------- #
_JOINT_TRIGGER_TEMPLATE_STEALTH = JOINT_TRIGGER_TEMPLATE.replace(
    "lambda_delta = {lambda_delta}\n",
    "lambda_delta = {lambda_delta}\nlambda_joint = {lambda_joint}\n",
).replace(
    "lambda_tv = 0.0\nlambda_lpips = 0.0\n",
    "lambda_tv = {lambda_tv}\nlambda_lpips = {lambda_lpips}\n",
)
assert "{lambda_joint}" in _JOINT_TRIGGER_TEMPLATE_STEALTH, (
    "JOINT_TRIGGER_TEMPLATE's lambda_delta line changed shape -- update the splice."
)
assert "{lambda_tv}" in _JOINT_TRIGGER_TEMPLATE_STEALTH and (
    "{lambda_lpips}" in _JOINT_TRIGGER_TEMPLATE_STEALTH
), "JOINT_TRIGGER_TEMPLATE's lambda_tv/lambda_lpips lines changed shape -- update the splice."

EXP_BASE = Path(
    "experiments/federated_experiments/threat_model_direct_trigger_joint_old_objective_port_stealth_sweep"
).resolve()

MODULE_NAME = "federated_generate_labels_trigger_joint"

# --------------------------------------------------------------------------- #
# Baseline generation-time config: gen_configs_old_objective_port's own GEN_* constants, PLUS
# the two knobs its template doesn't expose (lambda_joint) or has stale defaults for
# (iterations) -- see module docstring. This IS the actual on-disk winning reference config the
# user validated, not gen_configs_old_objective_port.py's own (drifted) template defaults.
# --------------------------------------------------------------------------- #
STEALTH_BASE = {
    "epsilon": 1.0,
    "lr_delta": GEN_LR_DELTA,
    "lambda_bd": 1.0,
    "lambda_penalty": 1.0,
    "lambda_delta": 0.0,
    "lambda_joint": 1.0,
    "lambda_tv": 0.0,
    "lambda_lpips": 0.0,
    "iterations": 25,
}

# --------------------------------------------------------------------------- #
# STEALTH_GRID -- edit this for a real campaign. One-at-a-time around STEALTH_BASE (see module
# docstring for why not a combinatorial grid). "baseline" reproduces the reference config
# exactly, as the comparison point every other row is measured against.
# --------------------------------------------------------------------------- #
STEALTH_GRID = [
    ("baseline", {}),
    ("eps_0p50", {"epsilon": 0.5}),
    ("eps_0p25", {"epsilon": 0.25}),
    ("eps_0p125", {"epsilon": 0.125}),
    ("tv_0p01", {"lambda_tv": 0.01}),
    ("tv_0p05", {"lambda_tv": 0.05}),
    ("tv_0p10", {"lambda_tv": 0.10}),
    ("lpips_0p1", {"lambda_lpips": 0.1}),
    ("lpips_0p5", {"lambda_lpips": 0.5}),
    ("lpips_1p0", {"lambda_lpips": 1.0}),
    ("penalty_0p5", {"lambda_penalty": 0.5}),
    ("penalty_2p0", {"lambda_penalty": 2.0}),
    # Hybrid candidate: slightly reduced epsilon combined with a mild LPIPS penalty, to check
    # whether the two axes compound (mixed reduction in visible footprint) or whether reducing
    # epsilon alone dilutes the robust-aggregator-breaking effect that lpips_0p1 (epsilon=1.0)
    # showed -- eps_0p50/eps_0p25 alone collapsed multikrum ASR (worst-case pta down to
    # 0.013-0.04) despite lpips_0p1 alone being the most robust cell in the grid (worst-case
    # pta=0.279), so this cell is NOT assumed to inherit lpips_0p1's robustness.
    ("eps_0p50_lpips_0p1", {"epsilon": 0.5, "lambda_lpips": 0.1}),
]

DELTA_MIN_FRAC = 0.0  # lambda_mag=0.0 (hardcoded in JOINT_TRIGGER_TEMPLATE) makes this inert
# either way (see run_module.py's trigger_constraint=="penalty" branch) -- kept at 0.0 to match
# the baseline reference config's own (implicit, default-delta_min_frac=0.8-but-never-applied)
# behavior. check_delta_min_feasible is still called below as a cheap guard in case a future
# STEALTH_GRID row starts touching lambda_mag/trigger_constraint.


def trigger_output_path(module_dir):
    run_tag = f"{GEN_NUM_POISONED}vs{GEN_NUM_HONESTS}"
    return (
        module_dir
        / "trigger"
        / f"opt_trig_direct_joint_{GEN_INIT}_{MODEL_FLAG}_{DATASET}_{run_tag}.pt"
    )


def generate_cell(tag, overrides, dry_run=False):
    cfg = {**STEALTH_BASE, **overrides}

    feasible, delta_min, max_reachable = check_delta_min_feasible(
        DATASET, cfg["epsilon"], DELTA_MIN_FRAC,
    )
    if not feasible:
        reason = (
            f"delta_min_frac={DELTA_MIN_FRAC} -> delta_min={delta_min:.4f} > "
            f"epsilon*sqrt(numel)={max_reachable:.4f} at epsilon={cfg['epsilon']} -- "
            "structurally unreachable post-clamp; refusing to generate this cell."
        )
        print(f"REFUSED [{tag}]: {reason}")
        return [], reason

    bootstrap_dir = f"{CLUSTER_ROOT}/out/checkpoints/old_objective_port_r32p_1xs_bootstrap"
    cell_dir = EXP_BASE / MODEL_FLAG / DATASET / tag
    module_dir = cell_dir / "gen_labels_trigger_joint"
    flips_dir = cell_dir / "select_flips"

    configs = {
        module_dir / "config.toml": _JOINT_TRIGGER_TEMPLATE_STEALTH.format(
            bootstrap_dir=bootstrap_dir,
            cell_dir=module_dir,
            model_flag=MODEL_FLAG,
            dataset=DATASET,
            source_label=SOURCE_LABEL,
            target_label=TARGET_LABEL,
            init=GEN_INIT,
            epsilon=cfg["epsilon"],
            lr_delta=cfg["lr_delta"],
            lambda_bd=cfg["lambda_bd"],
            lambda_penalty=cfg["lambda_penalty"],
            lambda_delta=cfg["lambda_delta"],
            lambda_joint=cfg["lambda_joint"],
            lambda_tv=cfg["lambda_tv"],
            lambda_lpips=cfg["lambda_lpips"],
            gen_num_honests=GEN_NUM_HONESTS,
            gen_num_poisoned=GEN_NUM_POISONED,
            gen_agg_method=GEN_AGG_METHOD,
            trigger_constraint=GEN_TRIGGER_CONSTRAINT,
            n_checkpoints_per_step=GEN_N_CHECKPOINTS_PER_STEP,
            expert_retrain_interval=GEN_EXPERT_RETRAIN_INTERVAL,
            expert_retrain_epochs=GEN_EXPERT_RETRAIN_EPOCHS,
            expert_retrain_checkpoint_iters=GEN_EXPERT_RETRAIN_CHECKPOINT_ITERS,
            lr=LR,
            wd=WD,
            milestones=MILESTONES,
            iterations=cfg["iterations"],
        ),
        flips_dir / "config.toml": SELECT_FLIPS_TEMPLATE.format(
            budgets=DEPLOY_BUDGETS,
            module_dir=module_dir,
            flips_dir=flips_dir,
            deploy_num_honests=DEPLOY_NUM_HONESTS,
            deploy_num_poisoned=DEPLOY_NUM_POISONED,
        ),
    }

    trigger_path = trigger_output_path(module_dir)
    for agg_method in DEPLOY_AGG_METHODS:
        for budget in DEPLOY_BUDGETS:
            train_user_dir = cell_dir / agg_method / f"train_user_{budget}"
            configs[train_user_dir / "config.toml"] = TRAIN_USER_TEMPLATE.format(
                flips_dir=flips_dir,
                train_user_dir=train_user_dir,
                model_flag=MODEL_FLAG,
                dataset=DATASET,
                source_label=SOURCE_LABEL,
                target_label=TARGET_LABEL,
                trigger_path=trigger_path,
                budget=budget,
                deploy_num_honests=DEPLOY_NUM_HONESTS,
                deploy_num_poisoned=DEPLOY_NUM_POISONED,
                agg_method=agg_method,
                lr=LR,
                wd=WD,
                milestones=MILESTONES,
            )

    paths = []
    for path, content in configs.items():
        assert "out/checkpoints" not in str(path), f"Refusing to write under out/checkpoints/: {path}"
        if dry_run:
            paths.append(path)
            continue
        write_config(path, content)
        validate_config(path)
        paths.append(path)

    return paths, None


def generate_all(dry_run=False):
    all_paths, refused = [], []
    for tag, overrides in STEALTH_GRID:
        paths, reason = generate_cell(tag, overrides, dry_run=dry_run)
        all_paths += paths
        if reason:
            refused.append((tag, reason))
    return all_paths, refused


def list_grid():
    """Enumerates this campaign's (tag, agg_method, budget) cells as plain dicts, without
    writing any config -- the single source of truth for the grid's shape, so
    orchestrate_slurm/orchestrate_runs_trigger_joint_old_objective_port_stealth_sweep_slurm.sh
    doesn't need to duplicate STEALTH_GRID/DEPLOY_AGG_METHODS/DEPLOY_BUDGETS by hand."""
    return [
        {"tag": tag, "agg_method": agg_method, "budget": budget}
        for tag, _ in STEALTH_GRID
        for agg_method in DEPLOY_AGG_METHODS
        for budget in DEPLOY_BUDGETS
    ]


def list_tags():
    return [tag for tag, _ in STEALTH_GRID]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--print-grid",
        action="store_true",
        help="print the (tag, agg_method, budget) sweep cells as JSON and exit, without writing configs",
    )
    parser.add_argument(
        "--print-tags",
        action="store_true",
        help="print just the STEALTH_GRID tags (one per line) and exit, without writing configs",
    )
    args = parser.parse_args()

    if args.print_grid:
        import json

        print(json.dumps(list_grid(), indent=2))
        raise SystemExit(0)

    if args.print_tags:
        for tag in list_tags():
            print(tag)
        raise SystemExit(0)

    paths, refused = generate_all(dry_run=args.dry_run)

    if args.dry_run:
        print(f"\n[DRY RUN] {MODULE_NAME} stealth_sweep: {len(paths)} config files would be written.")
        for p in paths:
            print(f"  {p}")
    else:
        print(f"\n{MODULE_NAME} stealth_sweep: {len(paths)} config files written and schema-validated.")

    if refused:
        print(f"\n{len(refused)} cell(s) REFUSED (delta_min infeasible):")
        for tag, reason in refused:
            print(f"  [{tag}] {reason}")
