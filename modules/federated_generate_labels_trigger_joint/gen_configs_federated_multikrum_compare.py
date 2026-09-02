"""
Single-user vs federated Multi-Krum comparison generator for
federated_generate_labels_trigger_joint.

Purpose: the attack (labels_syn + trigger delta) is optimized ONCE, at gen_configs.py's own
1-poisoned/0-honest/mean "base config" -- NOT regenerated in a federated setting (per request:
"Je ne veux pas generer les labels en federe mais uniquement tester l'attaque qui a ete
optimisee avec la configuration de base"). From that single gen_labels_trigger_joint output,
TWO downstream branches are generated per (seed, budget), both reading the SAME labels.npy /
true.npy / trigger .pt:

  1. "single-user" branch (existing structure, cell_dir/select_flips + cell_dir/train_user_*):
     1 poisoned worker, 0 honest workers, mean aggregation -- unchanged from gen_configs.py's
     own convention, so it stays directly comparable to every other campaign in this family.
  2. "federated" branch (new, nested under cell_dir/federated_3vs7_multikrum/): the SAME
     flipped/clean indices repartitioned across 3 poisoned + 7 honest workers
     (federated_select_flips.utils.partition_across_workers, called with different
     num_honests/num_poisoned than the attack-generation step -- select_flips only needs
     labels.npy/true.npy, it never depends on how many workers GENERATED the attack), then
     trained with agg_method="multikrum" and track_poison_selection=true (see
     modules/base_utils/util.py's mini_train_multi and modules/federated_train_user/
     run_module.py) so each train_user_{budget} run also produces
     multikrum_poison_stats.json, tracing how often a poisoned worker's flipped-label
     gradient actually gets selected by Multi-Krum.

The two branches nested under one shared cell_dir keep the "which experiments are directly
comparable" relationship explicit in the directory structure itself, per request ("conserver
une organisation claire permettant d'identifier facilement les experiences single-user et
federees").

Only READS gen_configs.py (templates, write_config, validate_config, check_delta_min_feasible,
every regularization/optimization constant) -- does not modify it or any other existing
gen_configs_*.py.
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
    trigger_output_path,
    validate_config,
    wandb_block,
    write_config,
)

# Locally-extended copy of TRAIN_USER_TEMPLATE -- adds `track_poison_selection = {track_poison_selection}`
# without touching the shared constant in gen_configs.py (every OTHER generator's
# TRAIN_USER_TEMPLATE.format() call never passes this field).
_TRAIN_USER_TEMPLATE_TRACKED = TRAIN_USER_TEMPLATE.replace(
    'agg_method = "{agg_method}"\n',
    'agg_method = "{agg_method}"\n'
    "track_poison_selection = {track_poison_selection}\n",
)
assert _TRAIN_USER_TEMPLATE_TRACKED != TRAIN_USER_TEMPLATE, (
    "TRAIN_USER_TEMPLATE's agg_method line changed shape -- update the splice."
)

# --------------------------------------------------------------------------- #
# Attack-generation axes (unchanged from gen_configs.py's own 1v0/mean base config): the
# label/trigger optimization itself is generated ONCE per seed, never regenerated federated.
# --------------------------------------------------------------------------- #
NUM_POISONED = 1
NUM_HONESTS = 0
AGG_METHOD = "mean"
DATASET = "cifar"
MODEL_FLAG = "r32p"
SEEDS = [0]
BUDGETS = [500, 2000]

# Federated deployment axes (only affect select_flips/train_user, NOT attack generation).
FED_NUM_POISONED = 3
FED_NUM_HONESTS = 7
FED_AGG_METHOD = "multikrum"
FED_TAG = f"federated_{FED_NUM_POISONED}vs{FED_NUM_HONESTS}_{FED_AGG_METHOD}"

EXP_BASE = Path(
    "experiments/federated_experiments/threat_model_direct_trigger_joint_federated_multikrum"
).resolve()

MODULE_NAME = "federated_generate_labels_trigger_joint_federated_multikrum"


def cell_name(seed):
    return f"{MODEL_FLAG}/{DATASET}/{NUM_POISONED}vs{NUM_HONESTS}/{AGG_METHOD}/seed{seed}"


def _select_flips_config(flips_dir, module_dir, budgets, num_honests, num_poisoned):
    return SELECT_FLIPS_TEMPLATE.format(
        budgets=budgets,
        module_dir=module_dir,
        flips_dir=flips_dir,
        num_honests=num_honests,
        num_poisoned=num_poisoned,
    )


def _train_user_config(
    flips_dir, train_user_dir, budget, num_honests, num_poisoned, agg_method,
    trigger_path, lr, wd, milestones, wandb_run_name, track_poison_selection,
):
    return _TRAIN_USER_TEMPLATE_TRACKED.format(
        flips_dir=flips_dir,
        train_user_dir=train_user_dir,
        model_flag=MODEL_FLAG,
        dataset=DATASET,
        source_label=SOURCE_LABEL,
        target_label=TARGET_LABEL,
        trigger_path=trigger_path,
        budget=budget,
        num_honests=num_honests,
        num_poisoned=num_poisoned,
        agg_method=agg_method,
        track_poison_selection="true" if track_poison_selection else "false",
        lr=lr,
        wd=wd,
        milestones=milestones,
        wandb_block_train_user=wandb_block(
            "federated_train_user", wandb_run_name,
            enabled=WANDB_ENABLED, project=WANDB_PROJECT,
            mode=WANDB_MODE, entity=WANDB_ENTITY, group=MODULE_NAME,
        ),
    )


def generate_cell(seed, budgets, dry_run=False):
    feasible, delta_min, max_reachable = check_delta_min_feasible(
        DATASET, EPSILON, DELTA_MIN_FRAC,
    )
    if not feasible:
        reason = (
            f"delta_min_frac={DELTA_MIN_FRAC} -> delta_min={delta_min:.4f} > "
            f"epsilon*sqrt(numel)={max_reachable:.4f} at epsilon={EPSILON} -- structurally "
            "unreachable post-clamp; refusing to generate this cell."
        )
        print(f"REFUSED [seed{seed}]: {reason}")
        return [], reason

    lr = LEARNING_RATE.get(MODEL_FLAG, 0.1)
    wd = WEIGHT_DECAY.get(MODEL_FLAG, 2e-4)
    milestones = MILESTONE.get(MODEL_FLAG, [75, 125])
    rng_seed = draw_rng_seed(MODEL_FLAG, seed)

    cell_dir = EXP_BASE / cell_name(seed)
    train_expert_dir = EXP_BASE / f"train_expert/{MODEL_FLAG}_1xs/seed{seed}"
    module_dir = cell_dir / "gen_labels_trigger_joint"
    flips_dir = cell_dir / "select_flips"
    fed_dir = cell_dir / FED_TAG
    fed_flips_dir = fed_dir / "select_flips"

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
        # Attack generation: ONE cell, 1v0/mean -- never regenerated for the federated branch.
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
            init="stripe",
            expert_retrain_interval=EXPERT_RETRAIN_INTERVAL,
            expert_retrain_epochs=EXPERT_RETRAIN_EPOCHS,
            expert_retrain_checkpoint_iters=EXPERT_RETRAIN_CHECKPOINT_ITERS,
            n_checkpoints_per_step=1,
            lr=lr,
            wd=wd,
            milestones=milestones,
            wandb_block_module=wandb_block(
                "federated_generate_labels_trigger_joint",
                f"{MODULE_NAME}/seed{seed}",
                enabled=WANDB_ENABLED, project=WANDB_PROJECT,
                mode=WANDB_MODE, entity=WANDB_ENTITY, group=MODULE_NAME,
            ),
        ),
        # Single-user branch: same convention as every other campaign in this family.
        flips_dir / "config.toml": _select_flips_config(
            flips_dir, module_dir, budgets, NUM_HONESTS, NUM_POISONED,
        ),
        # Federated branch: repartitions the SAME labels.npy/true.npy across 3v7 workers.
        fed_flips_dir / "config.toml": _select_flips_config(
            fed_flips_dir, module_dir, budgets, FED_NUM_HONESTS, FED_NUM_POISONED,
        ),
    }

    trigger_path = trigger_output_path(module_dir, MODEL_FLAG, DATASET, NUM_POISONED, NUM_HONESTS, "stripe")

    for budget in budgets:
        train_user_dir = cell_dir / f"train_user_{budget}"
        configs[train_user_dir / "config.toml"] = _train_user_config(
            flips_dir=flips_dir,
            train_user_dir=train_user_dir,
            budget=budget,
            num_honests=NUM_HONESTS,
            num_poisoned=NUM_POISONED,
            agg_method=AGG_METHOD,
            trigger_path=trigger_path,
            lr=lr, wd=wd, milestones=milestones,
            wandb_run_name=f"train_user/{MODEL_FLAG}/{DATASET}/single_user/{budget}/seed{seed}",
            track_poison_selection=False,
        )

        fed_train_user_dir = fed_dir / f"train_user_{budget}"
        configs[fed_train_user_dir / "config.toml"] = _train_user_config(
            flips_dir=fed_flips_dir,
            train_user_dir=fed_train_user_dir,
            budget=budget,
            num_honests=FED_NUM_HONESTS,
            num_poisoned=FED_NUM_POISONED,
            agg_method=FED_AGG_METHOD,
            trigger_path=trigger_path,
            lr=lr, wd=wd, milestones=milestones,
            wandb_run_name=f"train_user/{MODEL_FLAG}/{DATASET}/{FED_TAG}/{budget}/seed{seed}",
            track_poison_selection=True,
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
        paths, reason = generate_cell(seed, BUDGETS, dry_run=dry_run)
        all_paths += paths
        if reason:
            refused.append((seed, reason))
    return all_paths, refused


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths, refused = generate_all_configs(dry_run=args.dry_run)

    if args.dry_run:
        print(
            f"\n[DRY RUN] {MODULE_NAME}: {len(SEEDS)} seed(s) x {len(BUDGETS)} budget(s) x "
            f"2 branches (single-user, {FED_TAG}), {len(paths)} config files would be written."
        )
        for p in paths:
            print(f"  {p}")
    else:
        print(
            f"\n{MODULE_NAME}: {len(SEEDS)} seed(s) x {len(BUDGETS)} budget(s) x 2 branches "
            f"(single-user, {FED_TAG}), {len(paths)} config files written and schema-validated."
        )

    if refused:
        print(f"\n{len(refused)} cell(s) REFUSED (delta_min infeasible):")
        for seed, reason in refused:
            print(f"  [seed{seed}] {reason}")
