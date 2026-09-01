"""
Trigger-init comparison generator for federated_generate_labels_trigger_joint.

Purpose: isolate the effect of the trigger's initialization (see init_delta,
modules/federated_optimizing_trigger/utils.py) -- "stripe" (the module's original default: a
deterministic sinusoidal pattern) vs "random" (strength * torch.rand(mu_shape), drawn from
torch's GLOBAL RNG -- NOT reproducible across runs/processes unless torch itself is seeded
beforehand) -- in the 1-poisoned/0-honest proof-of-concept regime, at the SAME full budget
sweep as the main campaign (gen_configs.py's BUDGETS), against "mean" aggregation only (the
only aggregator that means anything with a single worker total). A SEPARATE file from
gen_configs.py on purpose, so the real (3-poisoned/7-honest) campaign's axes/behavior there are
left untouched -- this script only READS gen_configs.py (templates, write_config,
validate_config, check_delta_min_feasible, and every regularization/optimization constant
EXCEPT the one this comparison explicitly varies). Structurally identical to
gen_configs_gradmatch_metric.py (its own sibling comparison, over gradmatch_metric instead of
init) -- kept deliberately so, to be easy to diff/compare.

Two cells (one per init value), sharing every other hyperparameter, so any CTA/ASR difference
between the two cells is attributable to init alone, not a confounded hyperparameter change.
gradmatch_metric is left at gen_configs.py's own default ("relerr") in both cells -- this
comparison is orthogonal to that one.

NOTE (feasibility check, see gen_configs.py's check_delta_min_feasible docstring): for
init="random", delta_min = delta_min_frac * ||delta_init||_2 is computed against a torch-seeded
(seed=0, locally forked) draw for a STABLE printed pre-check number -- the actual training
run's own delta_init draw is unseeded (global RNG) and will differ slightly in exact value,
though a uniform random vector's norm concentrates tightly around its expectation for this
trigger shape (numel~3000), so the feasibility verdict itself is a stable proxy in practice.
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
    EXPERT_RETRAIN_INTERVAL,
    GAMMA_STEALTH,
    GRADMATCH_EPS,
    GRADMATCH_METRIC,
    JOINT_TRIGGER_TEMPLATE,
    LAMBDA_ALIGN,
    LAMBDA_BD,
    LAMBDA_DELTA,
    LAMBDA_GRADMATCH,
    LAMBDA_MAG,
    LEARNING_RATE,
    LR_DELTA,
    MILESTONE,
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

# The axis this comparison varies -- see module docstring.
INITS = ["stripe", "random"]

EXP_BASE = Path(
    "experiments/federated_experiments/threat_model_direct_trigger_joint_init_compare"
).resolve()

MODULE_NAME = "federated_generate_labels_trigger_joint_init_compare"


def cell_name(init, seed):
    return f"{MODEL_FLAG}/{DATASET}/{NUM_POISONED}vs{NUM_HONESTS}/{AGG_METHOD}/{init}/seed{seed}"


def generate_cell(init, seed, budgets, dry_run=False):
    feasible, delta_min, max_reachable = check_delta_min_feasible(
        DATASET, EPSILON, DELTA_MIN_FRAC, init=init,
    )
    if not feasible:
        reason = (
            f"delta_min_frac={DELTA_MIN_FRAC} -> delta_min={delta_min:.4f} > "
            f"epsilon*sqrt(numel)={max_reachable:.4f} at epsilon={EPSILON}, init={init!r} -- "
            "structurally unreachable post-clamp; refusing to generate this cell."
        )
        print(f"REFUSED [{init} seed{seed}]: {reason}")
        return [], reason

    lr = LEARNING_RATE.get(MODEL_FLAG, 0.1)
    wd = WEIGHT_DECAY.get(MODEL_FLAG, 2e-4)
    milestones = MILESTONE.get(MODEL_FLAG, [75, 125])
    # Real RNG seed (see gen_configs.py's draw_rng_seed docstring) -- cached per
    # (MODEL_FLAG, seed) so both init cells sharing this same train_expert cell (see
    # train_expert_dir below) get the SAME actual value.
    rng_seed = draw_rng_seed(MODEL_FLAG, seed)

    cell_dir = EXP_BASE / cell_name(init, seed)
    # Shared across both init cells of this seed (init doesn't affect expert training at all --
    # only the LABEL-generation step's own trigger optimization) -- one train_expert per seed
    # for the whole comparison, not per cell.
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
            init=init,
            expert_retrain_interval=EXPERT_RETRAIN_INTERVAL,
            expert_retrain_epochs=EXPERT_RETRAIN_EPOCHS,
            expert_retrain_checkpoint_iters=EXPERT_RETRAIN_CHECKPOINT_ITERS,
            lr=lr,
            wd=wd,
            milestones=milestones,
            wandb_block_module=wandb_block(
                "federated_generate_labels_trigger_joint",
                f"{MODULE_NAME}/{init}/seed{seed}",
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
    for budget in budgets:
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
                f"train_user/{MODEL_FLAG}/{DATASET}/{init}/{budget}/seed{seed}",
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
        for init in INITS:
            paths, reason = generate_cell(init, seed, BUDGETS, dry_run=dry_run)
            all_paths += paths
            if reason:
                refused.append((init, seed, reason))
    return all_paths, refused


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths, refused = generate_all_configs(dry_run=args.dry_run)

    if args.dry_run:
        print(
            f"\n[DRY RUN] {MODULE_NAME}: {len(INITS)} init(s) x {len(SEEDS)} "
            f"seed(s), {len(paths)} config files would be written."
        )
        for p in paths:
            print(f"  {p}")
    else:
        print(
            f"\n{MODULE_NAME}: {len(INITS)} init(s) x {len(SEEDS)} seed(s), "
            f"{len(paths)} config files written and schema-validated."
        )

    if refused:
        print(f"\n{len(refused)} cell(s) REFUSED (delta_min infeasible):")
        for init, seed, reason in refused:
            print(f"  [{init} seed{seed}] {reason}")
