"""
LPIPS perceptual-loss comparison generator for federated_generate_labels_trigger_joint.

Purpose: isolate the effect of adding a perceptual (LPIPS) penalty on the trigger delta (see
utils.lpips_penalty / run_module.py's lambda_lpips) -- two cells, sharing every other
hyperparameter with gen_configs.py's own current defaults (the "base config"):

  1. "base"  -- lambda_lpips = 0.0 (the module's behavior before this term existed)
  2. "lpips" -- lambda_lpips = LAMBDA_LPIPS (first, unsweeped guess -- re-tune once this
                comparison's L_lpips_mean / expert_asr / CTA-ASR results are in)

Only READS gen_configs.py (templates, write_config, validate_config, check_delta_min_feasible,
every regularization/optimization constant). JOINT_TRIGGER_TEMPLATE does not itself expose a
lambda_lpips slot (adding one there would break every OTHER generator's .format() call, which
never passes it) -- this file builds its own locally-extended copy of the template instead,
by splicing one extra `lambda_lpips = {lambda_lpips}` line in after `n_checkpoints_per_step`,
without modifying gen_configs.py.
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
from modules.federated_generate_labels_trigger_joint._federated_branch import (
    FED_TAG,
    federated_branch_configs,
)

# Locally-extended copy of JOINT_TRIGGER_TEMPLATE (see module docstring) -- adds one line,
# `lambda_lpips = {lambda_lpips}`, right after n_checkpoints_per_step, without touching the
# shared constant in gen_configs.py.
_JOINT_TRIGGER_TEMPLATE_LPIPS = JOINT_TRIGGER_TEMPLATE.replace(
    "n_checkpoints_per_step = {n_checkpoints_per_step}\n",
    "n_checkpoints_per_step = {n_checkpoints_per_step}\n"
    "lambda_lpips = {lambda_lpips}\n",
)
assert _JOINT_TRIGGER_TEMPLATE_LPIPS != JOINT_TRIGGER_TEMPLATE, (
    "JOINT_TRIGGER_TEMPLATE's n_checkpoints_per_step line changed shape -- update the splice."
)

# --------------------------------------------------------------------------- #
# Fixed axes: 1 poisoned worker, 0 honest workers, "mean" aggregation, one dataset/model,
# one seed, the reduced budget set requested for this study family.
# --------------------------------------------------------------------------- #
NUM_POISONED = 1
NUM_HONESTS = 0
AGG_METHOD = "mean"
DATASET = "cifar"
MODEL_FLAG = "r32p"
SEEDS = [0]
BUDGETS = [500, 2000]

# The axis this comparison varies. LAMBDA_LPIPS is a first, unsweeped guess (same order of
# magnitude as LAMBDA_ALIGN/LAMBDA_MAG) -- re-tune once this comparison's results are in.
LAMBDA_LPIPS = 1.0
LPIPS_VARIANTS = [("base", 0.0), ("lpips", LAMBDA_LPIPS)]

EXP_BASE = Path(
    "experiments/federated_experiments/threat_model_direct_trigger_joint_lpips_compare"
).resolve()

MODULE_NAME = "federated_generate_labels_trigger_joint_lpips_compare"


def cell_name(tag, seed):
    return f"{MODEL_FLAG}/{DATASET}/{NUM_POISONED}vs{NUM_HONESTS}/{AGG_METHOD}/{tag}/seed{seed}"


def generate_cell(tag, lambda_lpips, seed, budgets, dry_run=False):
    feasible, delta_min, max_reachable = check_delta_min_feasible(
        DATASET, EPSILON, DELTA_MIN_FRAC,
    )
    if not feasible:
        reason = (
            f"delta_min_frac={DELTA_MIN_FRAC} -> delta_min={delta_min:.4f} > "
            f"epsilon*sqrt(numel)={max_reachable:.4f} at epsilon={EPSILON} -- structurally "
            "unreachable post-clamp; refusing to generate this cell."
        )
        print(f"REFUSED [{tag} seed{seed}]: {reason}")
        return [], reason

    lr = LEARNING_RATE.get(MODEL_FLAG, 0.1)
    wd = WEIGHT_DECAY.get(MODEL_FLAG, 2e-4)
    milestones = MILESTONE.get(MODEL_FLAG, [75, 125])
    rng_seed = draw_rng_seed(MODEL_FLAG, seed)

    cell_dir = EXP_BASE / cell_name(tag, seed)
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
        module_dir / "config.toml": _JOINT_TRIGGER_TEMPLATE_LPIPS.format(
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
            lambda_lpips=lambda_lpips,
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
            budgets=budgets,
            module_dir=module_dir,
            flips_dir=flips_dir,
            num_honests=NUM_HONESTS,
            num_poisoned=NUM_POISONED,
        ),
    }
    trigger_path = trigger_output_path(module_dir, MODEL_FLAG, DATASET, NUM_POISONED, NUM_HONESTS, "stripe")
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
                f"train_user/{MODEL_FLAG}/{DATASET}/{tag}/{budget}/seed{seed}",
                enabled=WANDB_ENABLED, project=WANDB_PROJECT,
                mode=WANDB_MODE, entity=WANDB_ENTITY, group=MODULE_NAME,
            ),
        )

    # Federated Multi-Krum branch (3-poisoned/7-honest, track_poison_selection=true), nested
    # under cell_dir/FED_TAG -- reads the SAME labels.npy/true.npy/trigger this cell's
    # single-user branch above already reads. See _federated_branch.py's own docstring.
    configs.update(federated_branch_configs(
        module_dir=module_dir, cell_dir=cell_dir, budgets=budgets,
        model_flag=MODEL_FLAG, dataset=DATASET, source_label=SOURCE_LABEL,
        target_label=TARGET_LABEL, trigger_path=trigger_path, lr=lr, wd=wd,
        milestones=milestones, module_name=MODULE_NAME,
        wandb_run_name_prefix=f"train_user/{MODEL_FLAG}/{DATASET}/{tag}/seed{seed}",
        wandb_enabled=WANDB_ENABLED, wandb_project=WANDB_PROJECT,
        wandb_mode=WANDB_MODE, wandb_entity=WANDB_ENTITY,
    ))

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
        for tag, lambda_lpips in LPIPS_VARIANTS:
            paths, reason = generate_cell(tag, lambda_lpips, seed, BUDGETS, dry_run=dry_run)
            all_paths += paths
            if reason:
                refused.append((tag, seed, reason))
    return all_paths, refused


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths, refused = generate_all_configs(dry_run=args.dry_run)

    if args.dry_run:
        print(
            f"\n[DRY RUN] {MODULE_NAME}: {len(LPIPS_VARIANTS)} variant(s) x {len(SEEDS)} "
            f"seed(s), {len(paths)} config files would be written."
        )
        for p in paths:
            print(f"  {p}")
    else:
        print(
            f"\n{MODULE_NAME}: {len(LPIPS_VARIANTS)} variant(s) x {len(SEEDS)} seed(s), "
            f"{len(paths)} config files written and schema-validated."
        )

    if refused:
        print(f"\n{len(refused)} cell(s) REFUSED (delta_min infeasible):")
        for tag, seed, reason in refused:
            print(f"  [{tag} seed{seed}] {reason}")
