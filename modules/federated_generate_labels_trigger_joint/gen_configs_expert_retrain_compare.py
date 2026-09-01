"""
Live expert-retraining comparison generator for federated_generate_labels_trigger_joint.

Purpose: isolate the effect of expert_retrain_interval (see run_module.py's run() docstring,
"Live expert retraining") -- 0 (disabled: a single frozen expert trajectory throughout, the
module's original behavior) vs 1 (retrain a fresh expert against the CURRENT trigger every
single outer iteration, the most aggressive live-retraining setting, closing the H1 gap
documented in run_module.py) -- in the 1-poisoned/0-honest proof-of-concept regime, at the SAME
full budget sweep as the main campaign (gen_configs.py's BUDGETS), against "mean" aggregation
only (the only aggregator that means anything with a single worker total). A SEPARATE file from
gen_configs.py on purpose, so the real (3-poisoned/7-honest) campaign's axes/behavior there are
left untouched -- this script only READS gen_configs.py (templates, write_config,
validate_config, check_delta_min_feasible, trigger_output_path, and every regularization/
optimization constant EXCEPT the one this comparison explicitly varies). Structurally identical
to gen_configs_init_compare.py/gen_configs_gradmatch_metric.py (its own siblings), kept
deliberately so, to be easy to diff/compare.

Two cells (one per expert_retrain_interval value), sharing every other hyperparameter --
including delta_min_frac=0.0, epsilon=1.0, lambda_align=0.0 (gen_configs.py's own current
defaults, imported rather than overridden: this comparison is meant to run at a loosely
constrained trigger, per the request that motivated it) -- so any CTA/ASR difference between the
two cells is attributable to expert_retrain_interval alone, not a confounded hyperparameter
change. expert_retrain_epochs/expert_retrain_checkpoint_iters are left at gen_configs.py's own
defaults in both cells (irrelevant when interval=0 disables retraining outright).
"""
import argparse
from pathlib import Path

from modules.federated_generate_labels_trigger_joint.gen_configs import (
    ALIGN_KAPPA,
    ALPHA_CKPT,
    CHECKPOINT_ITERS,
    CHECKPOINT_SAMPLING,
    CLUSTER_ROOT,
    DELTA_MIN_FRAC,
    EPOCHS_EXPERT,
    EPSILON,
    EXPERT_RETRAIN_CHECKPOINT_ITERS,
    EXPERT_RETRAIN_EPOCHS,
    GAMMA_STEALTH,
    GRADMATCH_EPS,
    GRADMATCH_METRIC,
    INIT,
    JOINT_TRIGGER_TEMPLATE,
    LAMBDA_ALIGN,
    LAMBDA_BD,
    LAMBDA_DELTA,
    LAMBDA_GRADMATCH,
    LAMBDA_MAG,
    LEARNING_RATE,
    LR_DELTA,
    MILESTONE,
    N_CHECKPOINTS_PER_STEP,
    N_ITERATIONS,
    SELECT_FLIPS_TEMPLATE,
    SOURCE_LABEL,
    TARGET_LABEL,
    TRAIN_EXPERT_TEMPLATE,
    TRAIN_PCT,
    TRAIN_USER_TEMPLATE,
    TRIGGER_CONSTRAINT,
    WANDB_ENABLED,
    WANDB_ENTITY,
    WANDB_MODE,
    WANDB_PROJECT,
    WEIGHT_DECAY,
    check_delta_min_feasible,
    draw_rng_seed,
    trigger_output_path,
    validate_config,
    wandb_block,
    write_config,
)

# --------------------------------------------------------------------------- #
# Fixed axes for this comparison: 1 poisoned worker, 0 honest workers ("mean" vs "multikrum"
# is a no-op distinction with a single worker total -- fixed to "mean"), one dataset/model,
# one seed, the SAME full budget sweep as the main campaign.
# --------------------------------------------------------------------------- #
NUM_POISONED = 1
NUM_HONESTS = 0
AGG_METHOD = "mean"
DATASET = "cifar"
MODEL_FLAG = "r32p"
SEEDS = [0]
BUDGETS = [150, 300, 500, 1000, 2000, 2500, 5000]

# The axis this comparison varies -- see module docstring. 0 = disabled (frozen expert
# trajectory throughout), 1 = retrain the expert against the current trigger every single outer
# iteration (the most aggressive live-retraining setting).
EXPERT_RETRAIN_INTERVALS = [0, 1]

EXP_BASE = Path(
    "experiments/federated_experiments/threat_model_direct_trigger_joint_expert_retrain_compare"
).resolve()

MODULE_NAME = "federated_generate_labels_trigger_joint_expert_retrain_compare"


def cell_name(expert_retrain_interval, seed):
    return (
        f"{MODEL_FLAG}/{DATASET}/{NUM_POISONED}vs{NUM_HONESTS}/{AGG_METHOD}/"
        f"retrain_interval_{expert_retrain_interval}/seed{seed}"
    )


def generate_cell(expert_retrain_interval, seed, budgets, dry_run=False):
    feasible, delta_min, max_reachable = check_delta_min_feasible(
        DATASET, EPSILON, DELTA_MIN_FRAC, init=INIT,
    )
    if not feasible:
        reason = (
            f"delta_min_frac={DELTA_MIN_FRAC} -> delta_min={delta_min:.4f} > "
            f"epsilon*sqrt(numel)={max_reachable:.4f} at epsilon={EPSILON} -- "
            "structurally unreachable post-clamp; refusing to generate this cell."
        )
        print(f"REFUSED [retrain_interval={expert_retrain_interval} seed{seed}]: {reason}")
        return [], reason

    lr = LEARNING_RATE.get(MODEL_FLAG, 0.1)
    wd = WEIGHT_DECAY.get(MODEL_FLAG, 2e-4)
    milestones = MILESTONE.get(MODEL_FLAG, [75, 125])
    # Real RNG seed (see gen_configs.py's draw_rng_seed docstring) -- cached per
    # (MODEL_FLAG, seed) so both interval cells sharing this same train_expert cell (see
    # train_expert_dir below) get the SAME actual value.
    rng_seed = draw_rng_seed(MODEL_FLAG, seed)

    cell_dir = EXP_BASE / cell_name(expert_retrain_interval, seed)
    # Shared across both interval cells of this seed (expert_retrain_interval doesn't affect
    # the INITIAL train_expert step at all, only what gen_labels_trigger_joint does with it
    # afterwards) -- one train_expert per seed for the whole comparison, not per cell.
    train_expert_dir = EXP_BASE / f"train_expert/{MODEL_FLAG}_1xs/seed{seed}"
    module_dir = cell_dir / "gen_labels_trigger_joint"
    flips_dir = cell_dir / "select_flips"

    configs = {
        train_expert_dir / "config.toml": TRAIN_EXPERT_TEMPLATE.format(
            cluster_root=CLUSTER_ROOT,
            model_flag=MODEL_FLAG,
            dataset=DATASET,
            seed=seed,
            rng_seed=rng_seed,
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
            rng_seed=rng_seed,
            cell_dir=module_dir,
            source_label=SOURCE_LABEL,
            target_label=TARGET_LABEL,
            epsilon=EPSILON,
            lr_delta=LR_DELTA,
            lambda_bd=LAMBDA_BD,
            lambda_delta=LAMBDA_DELTA,
            lambda_gradmatch=LAMBDA_GRADMATCH,
            gradmatch_eps=GRADMATCH_EPS,
            gradmatch_metric=GRADMATCH_METRIC,
            # detach_param_dist (added to gen_configs.py/run_module.py after this comparison
            # script was written, schema-optional, false = the module's original,
            # fully-differentiable-denominator behavior): pinned off here so
            # expert_retrain_interval stays the only axis varying between this script's two
            # cells -- the detach comparison lives in gen_configs_detach_param_dist_compare.py
            # instead.
            detach_param_dist="false",
            train_pct=TRAIN_PCT,
            num_honests=NUM_HONESTS,
            num_poisoned=NUM_POISONED,
            agg_method=AGG_METHOD,
            gamma_stealth=GAMMA_STEALTH,
            checkpoint_sampling=CHECKPOINT_SAMPLING,
            alpha_ckpt=ALPHA_CKPT,
            n_iterations=N_ITERATIONS,
            trigger_constraint=TRIGGER_CONSTRAINT,
            align_kappa=ALIGN_KAPPA,
            lambda_align=LAMBDA_ALIGN,
            lambda_mag=LAMBDA_MAG,
            delta_min_frac=DELTA_MIN_FRAC,
            init=INIT,
            # The axis this comparison varies -- see module docstring.
            expert_retrain_interval=expert_retrain_interval,
            expert_retrain_epochs=EXPERT_RETRAIN_EPOCHS,
            expert_retrain_checkpoint_iters=EXPERT_RETRAIN_CHECKPOINT_ITERS,
            n_checkpoints_per_step=N_CHECKPOINTS_PER_STEP,
            lr=lr,
            wd=wd,
            milestones=milestones,
            wandb_block_module=wandb_block(
                "federated_generate_labels_trigger_joint",
                f"{MODULE_NAME}/retrain_interval_{expert_retrain_interval}/seed{seed}",
                enabled=WANDB_ENABLED, project=WANDB_PROJECT,
                mode=WANDB_MODE, entity=WANDB_ENTITY, group=MODULE_NAME,
            ),
        ),
        flips_dir / "config.toml": SELECT_FLIPS_TEMPLATE.format(
            budgets=budgets,
            module_dir=module_dir,
            flips_dir=flips_dir,
            num_honests=NUM_HONESTS,
            num_poisoned=NUM_POISONED,
        ),
    }
    trigger_path = trigger_output_path(module_dir, MODEL_FLAG, DATASET, NUM_POISONED, NUM_HONESTS, INIT)
    for budget in budgets:
        train_user_dir = cell_dir / f"train_user_{budget}"
        configs[train_user_dir / "config.toml"] = TRAIN_USER_TEMPLATE.format(
            flips_dir=flips_dir,
            train_user_dir=train_user_dir,
            model_flag=MODEL_FLAG,
            dataset=DATASET,
            source_label=SOURCE_LABEL,
            target_label=TARGET_LABEL,
            trigger_path=trigger_path,
            budget=budget,
            num_honests=NUM_HONESTS,
            num_poisoned=NUM_POISONED,
            agg_method=AGG_METHOD,
            lr=lr,
            wd=wd,
            milestones=milestones,
            wandb_block_train_user=wandb_block(
                "federated_train_user",
                f"train_user/{MODEL_FLAG}/{DATASET}/retrain_interval_{expert_retrain_interval}/{budget}/seed{seed}",
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
    for seed in SEEDS:
        for expert_retrain_interval in EXPERT_RETRAIN_INTERVALS:
            paths, reason = generate_cell(expert_retrain_interval, seed, BUDGETS, dry_run=dry_run)
            all_paths += paths
            if reason:
                refused.append((expert_retrain_interval, seed, reason))
    return all_paths, refused


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths, refused = generate_all_configs(dry_run=args.dry_run)

    if args.dry_run:
        print(
            f"\n[DRY RUN] {MODULE_NAME}: {len(EXPERT_RETRAIN_INTERVALS)} interval(s) x "
            f"{len(SEEDS)} seed(s), {len(paths)} config files would be written."
        )
        for p in paths:
            print(f"  {p}")
    else:
        print(
            f"\n{MODULE_NAME}: {len(EXPERT_RETRAIN_INTERVALS)} interval(s) x {len(SEEDS)} "
            f"seed(s), {len(paths)} config files written and schema-validated."
        )

    if refused:
        print(f"\n{len(refused)} cell(s) REFUSED (delta_min infeasible):")
        for expert_retrain_interval, seed, reason in refused:
            print(f"  [retrain_interval={expert_retrain_interval} seed{seed}] {reason}")
