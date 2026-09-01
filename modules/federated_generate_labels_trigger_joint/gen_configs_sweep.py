"""
Hyperparameter sweep generator for federated_generate_labels_trigger_joint.

Purpose: find good optimization hyperparameters for the label/trigger generation step
(lr_delta, lambda_bd, n_iterations) and explore the trigger's regularization/constraint knobs
(epsilon, gamma_stealth, lambda_delta, trigger_constraint, align_kappa, lambda_align,
lambda_mag, delta_min_frac), in the 1-poisoned/0-honest proof-of-concept regime. A SEPARATE
file from gen_configs.py on purpose, so the real campaign's sweep axes/behavior there are left
untouched -- this script only READS gen_configs.py (templates, write_config, validate_config,
check_delta_min_feasible) and defines its own axes/cell layout on top.

Design -- one-at-a-time (OFAT), not a full cross-product: a full grid over ~11 axes would be
thousands of cells. BASELINE below fixes every axis at gen_configs.py's current (relaxed)
defaults; each sweep cell overrides exactly ONE axis away from BASELINE (see build_cells()).
This is an exploratory search meant to find a good operating point and to see, per axis,
whether the campaign's current default is actually a good choice -- not a statistically-powered
multi-seed comparison, hence SEEDS defaults to a single seed; re-run the winning cell(s) across
more seeds afterwards if a final number is needed.

Only ONE train_user budget is generated per cell (BUDGETS below) -- "un unique train_user pour
un budget de 1500" is enough to read a cell's CTA/ASR off without paying for the main
campaign's full budget sweep.
"""
import argparse
from pathlib import Path

from modules.federated_generate_labels_trigger_joint.gen_configs import (
    ALPHA_CKPT,
    CHECKPOINT_ITERS,
    CHECKPOINT_SAMPLING,
    CLUSTER_ROOT,
    EPOCHS_EXPERT,
    JOINT_TRIGGER_TEMPLATE,
    LEARNING_RATE,
    MILESTONE,
    SELECT_FLIPS_TEMPLATE,
    SOURCE_LABEL,
    TARGET_LABEL,
    TRAIN_EXPERT_TEMPLATE,
    TRAIN_PCT,
    TRAIN_USER_TEMPLATE,
    WANDB_ENABLED,
    WANDB_ENTITY,
    WANDB_MODE,
    WANDB_PROJECT,
    WEIGHT_DECAY,
    check_delta_min_feasible,
    validate_config,
    wandb_block,
    write_config,
)

# --------------------------------------------------------------------------- #
# Fixed axes for this sweep -- the proof-of-concept regime requested: 1 poisoned worker,
# 0 honest workers, one dataset/model, a single train_user budget.
# --------------------------------------------------------------------------- #
NUM_POISONED = 1
NUM_HONESTS = 0
AGG_METHOD = "mean"  # only one worker total (1 poisoned + 0 honest) -- aggregator choice is
# moot here; fixed to keep the sweep's cell count down.
DATASET = "cifar"
MODEL_FLAG = "r32p"
SEEDS = [0]  # exploratory HP search, not a seeded-variance comparison -- see module docstring.
BUDGETS = [1500]

EXP_BASE = Path(
    "experiments/federated_experiments/threat_model_direct_trigger_joint_sweep"
).resolve()

MODULE_NAME = "federated_generate_labels_trigger_joint_sweep"

# --------------------------------------------------------------------------- #
# Baseline -- gen_configs.py's current (relaxed) defaults. Every sweep cell starts here and
# overrides exactly one key; see SWEEP_AXES below. Kept as literal values (not imported) so
# this sweep's baseline is pinned and doesn't silently drift if gen_configs.py's own defaults
# are tuned again later -- update deliberately if you want the sweep centered on a new point.
# --------------------------------------------------------------------------- #
BASELINE = dict(
    lr_delta=1e-2,
    lambda_bd=2.0,
    n_iterations=15,
    epsilon=1.0,
    gamma_stealth=0.3,
    lambda_delta=0.0,
    trigger_constraint="penalty",
    align_kappa=0.3,
    lambda_align=0.3,
    lambda_mag=0.3,
    delta_min_frac=0.01,
)

# --------------------------------------------------------------------------- #
# Sweep axes, OFAT around BASELINE. OPTIMIZATION_AXES tune how labels/trigger are optimized;
# REGULARIZATION_AXES are the trigger's constraint/regularization knobs.
# --------------------------------------------------------------------------- #
OPTIMIZATION_AXES = {
    "lr_delta": [1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
    "lambda_bd": [0.5, 1.0, 2.0, 4.0, 8.0],
    "n_iterations": [10, 15, 25, 40],
}

REGULARIZATION_AXES = {
    "epsilon": [0.05, 0.1, 0.3, 1.0],
    "gamma_stealth": [0.0, 0.1, 0.3, 1.0],
    "lambda_delta": [0.0, 0.01, 0.1, 1.0],
    "trigger_constraint": ["penalty", "projection"],
    "align_kappa": [0.0, 0.3, 0.6],
    "lambda_align": [0.0, 0.3, 1.0],
    "lambda_mag": [0.0, 0.3, 1.0],
    "delta_min_frac": [0.0, 0.01, 0.05],
}

SWEEP_AXES = {**OPTIMIZATION_AXES, **REGULARIZATION_AXES}


def _slug(value):
    return str(value).replace(".", "p").replace("-", "m")


def build_cells():
    """[("baseline", {}), ("lr_delta_0p001", {"lr_delta": 0.001}), ...] -- one baseline cell
    plus one cell per (axis, value) that differs from BASELINE[axis] (float-safe compare), so
    BASELINE's own value in an axis's list doesn't produce a duplicate of the baseline cell."""
    cells = [("baseline", {})]
    for axis, values in SWEEP_AXES.items():
        base_value = BASELINE[axis]
        for value in values:
            if isinstance(value, float) and isinstance(base_value, float):
                if abs(value - base_value) < 1e-12:
                    continue
            elif value == base_value:
                continue
            cells.append((f"{axis}_{_slug(value)}", {axis: value}))
    return cells


def generate_cell(tag, overrides, seed, dry_run=False):
    params = {**BASELINE, **overrides}

    feasible, delta_min, max_reachable = check_delta_min_feasible(
        DATASET, params["epsilon"], params["delta_min_frac"],
    )
    if not feasible:
        reason = (
            f"delta_min_frac={params['delta_min_frac']} -> delta_min={delta_min:.4f} > "
            f"epsilon*sqrt(numel)={max_reachable:.4f} at epsilon={params['epsilon']} -- "
            "structurally unreachable post-clamp; refusing to generate this cell."
        )
        print(f"REFUSED [{tag} seed{seed}]: {reason}")
        return [], reason

    lr = LEARNING_RATE.get(MODEL_FLAG, 0.1)
    wd = WEIGHT_DECAY.get(MODEL_FLAG, 2e-4)
    milestones = MILESTONE.get(MODEL_FLAG, [75, 125])

    cell_dir = (
        EXP_BASE
        / f"{MODEL_FLAG}/{DATASET}/{NUM_POISONED}vs{NUM_HONESTS}/seed{seed}/{tag}"
    )
    # Shared across every cell of this seed (lr_delta/lambda_bd/etc. don't affect the expert
    # training step at all) -- one train_expert per seed for the whole sweep, not per cell.
    train_expert_dir = EXP_BASE / f"train_expert/{MODEL_FLAG}_1xs/seed{seed}"
    module_dir = cell_dir / "gen_labels_trigger_joint"
    flips_dir = cell_dir / "select_flips"

    configs = {
        train_expert_dir / "config.toml": TRAIN_EXPERT_TEMPLATE.format(
            cluster_root=CLUSTER_ROOT,
            model_flag=MODEL_FLAG,
            dataset=DATASET,
            seed=seed,
            source_label=SOURCE_LABEL,
            target_label=TARGET_LABEL,
            checkpoint_iters=CHECKPOINT_ITERS,
            epochs=EPOCHS_EXPERT,
            lr=lr,
            wd=wd,
            milestones=milestones,
            wandb_block_train_expert=wandb_block(
                "train_expert", f"train_expert/{MODEL_FLAG}/{DATASET}/seed{seed}",
                enabled=WANDB_ENABLED, project=WANDB_PROJECT,
                mode=WANDB_MODE, entity=WANDB_ENTITY, group=MODULE_NAME,
            ),
        ),
        module_dir / "config.toml": JOINT_TRIGGER_TEMPLATE.format(
            cluster_root=CLUSTER_ROOT,
            model_flag=MODEL_FLAG,
            dataset=DATASET,
            seed=seed,
            cell_dir=module_dir,
            source_label=SOURCE_LABEL,
            target_label=TARGET_LABEL,
            epsilon=params["epsilon"],
            lr_delta=params["lr_delta"],
            lambda_bd=params["lambda_bd"],
            lambda_delta=params["lambda_delta"],
            # Gradient-mismatch penalty (added to gen_configs.py/run_module.py after this sweep
            # script was written, schema-optional, 0.0 = disabled): pinned off here so this
            # OFAT hyperparameter search's own axes stay the only thing varying between cells.
            lambda_gradmatch=0.0,
            gradmatch_eps=1e-8,
            gradmatch_metric="relerr",
            train_pct=TRAIN_PCT,
            num_honests=NUM_HONESTS,
            num_poisoned=NUM_POISONED,
            agg_method=AGG_METHOD,
            gamma_stealth=params["gamma_stealth"],
            checkpoint_sampling=CHECKPOINT_SAMPLING,
            alpha_ckpt=ALPHA_CKPT,
            n_iterations=params["n_iterations"],
            trigger_constraint=params["trigger_constraint"],
            align_kappa=params["align_kappa"],
            lambda_align=params["lambda_align"],
            lambda_mag=params["lambda_mag"],
            delta_min_frac=params["delta_min_frac"],
            # expert_retrain_* is a gen_configs.py feature added after this sweep script was
            # written (schema-optional, 0 = disabled); pinned to disabled here so this
            # exploratory HP sweep's frozen-trajectory behavior is unchanged by its addition.
            expert_retrain_interval=0,
            expert_retrain_epochs=EPOCHS_EXPERT,
            expert_retrain_checkpoint_iters=CHECKPOINT_ITERS,
            lr=lr,
            wd=wd,
            milestones=milestones,
            wandb_block_module=wandb_block(
                "federated_generate_labels_trigger_joint",
                f"{MODULE_NAME}/{tag}/seed{seed}",
                enabled=WANDB_ENABLED, project=WANDB_PROJECT,
                mode=WANDB_MODE, entity=WANDB_ENTITY, group=MODULE_NAME,
            ),
        ),
        flips_dir / "config.toml": SELECT_FLIPS_TEMPLATE.format(
            budgets=BUDGETS,
            module_dir=module_dir,
            flips_dir=flips_dir,
            num_honests=NUM_HONESTS,
            num_poisoned=NUM_POISONED,
        ),
    }
    for budget in BUDGETS:
        train_user_dir = cell_dir / f"train_user_{budget}"
        configs[train_user_dir / "config.toml"] = TRAIN_USER_TEMPLATE.format(
            flips_dir=flips_dir,
            train_user_dir=train_user_dir,
            model_flag=MODEL_FLAG,
            dataset=DATASET,
            source_label=SOURCE_LABEL,
            target_label=TARGET_LABEL,
            budget=budget,
            num_honests=NUM_HONESTS,
            num_poisoned=NUM_POISONED,
            agg_method=AGG_METHOD,
            lr=lr,
            wd=wd,
            milestones=milestones,
            wandb_block_train_user=wandb_block(
                "federated_train_user",
                f"train_user/{MODEL_FLAG}/{DATASET}/{tag}/{budget}/seed{seed}",
                enabled=WANDB_ENABLED, project=WANDB_PROJECT,
                mode=WANDB_MODE, entity=WANDB_ENTITY, group=MODULE_NAME,
            ),
        )

    paths = []
    for path, content in configs.items():
        assert "out/checkpoints" not in str(path), (
            f"Refusing to write under out/checkpoints/: {path}"
        )
        if dry_run:
            paths.append(path)
            continue
        write_config(path, content)
        validate_config(path)
        paths.append(path)

    return paths, None


def generate_all_configs(dry_run=False):
    all_paths, refused = [], []
    cells = build_cells()
    for seed in SEEDS:
        for tag, overrides in cells:
            paths, reason = generate_cell(tag, overrides, seed, dry_run=dry_run)
            all_paths += paths
            if reason:
                refused.append((tag, seed, reason))
    return all_paths, refused, cells


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths, refused, cells = generate_all_configs(dry_run=args.dry_run)

    if args.dry_run:
        print(
            f"\n[DRY RUN] {MODULE_NAME}: {len(cells)} sweep cells x {len(SEEDS)} seed(s), "
            f"{len(paths)} config files would be written."
        )
        for p in paths:
            print(f"  {p}")
    else:
        print(
            f"\n{MODULE_NAME}: {len(cells)} sweep cells x {len(SEEDS)} seed(s), "
            f"{len(paths)} config files written and schema-validated."
        )

    if refused:
        print(f"\n{len(refused)} cell(s) REFUSED (delta_min infeasible):")
        for tag, seed, reason in refused:
            print(f"  [{tag} seed{seed}] {reason}")
