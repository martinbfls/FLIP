"""
Config generator for the federated_generate_labels_trigger campaign chain:
train_expert -> federated_generate_labels_trigger -> federated_select_flips ->
federated_train_user.

Conventions follow modules/base_utils/gen_configs.py: string .format() templates, one
config.toml per pipeline stage under experiments/federated_experiments/..., deterministic
directory names derived from the sweep axes, a write_config(path, content) helper.

Every generated config is validated against its schema (schemas/<module>.toml) the same way
run_experiment.py does -- key presence only, run_experiment.py does not check types -- so a
config that "looks right" but doesn't actually match the schema (extra/missing keys) is caught
at generation time, not on the cluster. This is the exact failure mode that made the
anti-collapse regularizer unusable for a whole session (schema/code drift caught only when a
real run failed) -- see docs/threat_models_audit.md.
"""
import argparse
import os
from pathlib import Path

import toml

from modules.base_utils.gen_configs import wandb_block

# Weights & Biases mirroring (see modules/base_utils/experiment_tracker.py). Off by
# default -- flip WANDB_ENABLED to True to have every config in this campaign carry a
# [<module>.wandb] table (requires `wandb login` or WANDB_API_KEY in the environment that
# runs them). Local plots/metrics under experiments/.../{plots,logs}/ are unaffected
# either way.
WANDB_ENABLED = False
WANDB_PROJECT = "flip"
WANDB_ENTITY = None
WANDB_MODE = "online"

# --------------------------------------------------------------------------- #
# Sweep axes -- edit these for a real campaign. Defaults match the grid agreed
# for this threat-model family (docs/policy_module_audit_report.md, Bloc B).
# --------------------------------------------------------------------------- #
NUM_POISONED = 3
NUM_HONESTS = 7
SEEDS = [0]
BUDGETS = [1500]
AGG_METHODS = ["mean"]
DATASETS = ["cifar"]
MODEL_FLAGS = ["r32p"]
SOURCE_LABEL = 9
TARGET_LABEL = 4

CHECKPOINT_SAMPLING = "uniform"  # this module's own prior default -- see the joint generator's
                                  # cross-module warning
ALPHA_CKPT = 0.01
TRAIN_PCT = 1.0
EPOCHS_EXPERT = 20
CHECKPOINT_ITERS = 50
N_ITERATIONS = 15

# --------------------------------------------------------------------------- #
# REGULARIZATION_GRID -- trigger regularization knobs, kept separate from the sweep axes
# above so a real campaign's regularization sweep is easy to find and edit in one place.
# `single_cell` mode (see generate_single_cell / --single-cell) fixes ALL of these to the
# defaults below and sweeps only the main axes (SEEDS/BUDGETS/AGG_METHODS).
# --------------------------------------------------------------------------- #
# Relaxed 2026-08-28, kept aligned with the sibling federated_generate_labels_trigger_joint
# generator (same reasoning: EPSILON=0.1/GAMMA_STEALTH=1.0 were tight enough to degrade
# CTA/ASR well below what this attack can actually do -- this is a proof-of-concept campaign,
# not a stealth-optimized one). Keep these two generators' values equal so the
# indirect-vs-joint comparison isn't confounded by this axis.
EPSILON = 0.3         # L_infinity bound on the trigger delta -- larger allows a stronger/more
                      # visible perturbation; too small can make the backdoor unreachable.
                      # (was 0.1)
LR_DELTA = 1e-2       # Adam learning rate for the trigger optimization.
LAMBDA_BD = 2.0       # weight of the backdoor-efficacy loss (kappa in the P^mean/P^direct
                      # formulas) -- higher pushes harder for backdoor success at the cost of
                      # the MTT alignment term. Raised (was 1.0) to lean further into backdoor
                      # efficacy now that GAMMA_STEALTH is relaxed.
GAMMA_STEALTH = 0.3   # scalar stealth/backdoor loss weight multiplying grand_loss (UNRELATED
                      # to federated_optimizing_trigger_policy's gamma = num_poisoned/
                      # (num_poisoned+num_honests) -- disjoint concept, see gen_configs.py's
                      # docstring / schema note). Lowered (was 1.0): stealth is secondary for
                      # this proof-of-concept.
LAMBDA_DELTA = 0.0    # L2 norm penalty on delta ("lambda_trigger_l2") -- 0.0 (schema default)
                      # leaves the trigger magnitude unregularized beyond the epsilon clamp.

LEARNING_RATE = {"r32p": 0.1, "r18": 0.1, "vgg": 0.01}
WEIGHT_DECAY = {"r32p": 2e-4, "r18": 2e-4, "vgg": 2e-4}
MILESTONE = {"r32p": [75, 125], "r18": [75, 125], "vgg": [125]}

# Absolute cluster root for real artifacts (out/checkpoints, experiment outputs) -- edit for
# your own cluster mount, or set FLIP_CLUSTER_ROOT. Kept as ONE constant (unlike
# modules/base_utils/gen_configs.py, which repeats the literal path per template) so a
# campaign can be retargeted in one place.
CLUSTER_ROOT = "/shared/data1/Projects/DLWP/j1067582/martin/FLIP"

EXP_BASE = Path("experiments/federated_experiments/threat_model_direct_trigger").resolve()

MODULE_NAME = "federated_generate_labels_trigger"


def write_config(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"[OK] Config written to {path}")


def validate_config(path: Path):
    """
    Same check run_experiment.py performs before running a module: every schema key (unless
    listed under [OPTIONAL]) must be present in the config, and every config key must exist in
    the schema. Raises AssertionError on any mismatch -- run at GENERATION time, not only
    discovered when the config is actually executed.
    """
    exp_toml = toml.load(path)
    for module_name, module_config in exp_toml.items():
        schema_path = Path("schemas") / f"{module_name}.toml"
        assert schema_path.exists(), f"Malformed module! Schema {schema_path} does not exist."
        schema = toml.load(schema_path)
        optionals = list(schema.get("OPTIONAL", {}).keys())

        schema_keys = set(schema[module_name].keys())
        config_keys = set(module_config.keys())

        missing = [k for k in schema_keys - config_keys if k not in optionals]
        extra = [k for k in config_keys - schema_keys if k not in optionals]
        assert not missing, (
            f"Malformed config ({path}, [{module_name}]): missing required keys {missing}"
        )
        assert not extra, (
            f"Malformed config ({path}, [{module_name}]): unknown keys {extra} "
            "(not in schema, not optional)"
        )


TRAIN_EXPERT_TEMPLATE = """[train_expert]
output_dir = "{cluster_root}/out/checkpoints/{model_flag}_1xs/seed{seed}/0/"
model = "{model_flag}"
dataset = "{dataset}"
trainer = "sgd"
source_label = {source_label}
target_label = {target_label}
poisoner = "1xs"
checkpoint_iters = {checkpoint_iters}
epochs = {epochs}
optim_kwargs = {{lr = {lr}, momentum = 0.9, nesterov = true, weight_decay = {wd}}}
scheduler_kwargs = {{milestones = {milestones}, gamma = 0.1}}
{wandb_block_train_expert}"""

DIRECT_TRIGGER_TEMPLATE = """[federated_generate_labels_trigger]
input_pths = "{cluster_root}/out/checkpoints/{model_flag}_1xs/seed{seed}/{{}}/model_{{}}_{{}}.pth"
opt_pths = "{cluster_root}/out/checkpoints/{model_flag}_1xs/seed{seed}/{{}}/model_{{}}_{{}}_opt.pth"
output_dir = "{cell_dir}/labels/"
output_dir_trigger = "{cell_dir}/trigger"
expert_model = "{model_flag}"
dataset = "{dataset}"
source_label = {source_label}
target_label = {target_label}

epsilon = {epsilon}
lr_delta = {lr_delta}
lambda_bd = {lambda_bd}
lambda_delta = {lambda_delta}

train_pct = {train_pct}
num_honests = {num_honests}
num_poisoned = {num_poisoned}
agg_method = "{agg_method}"
attack = "backdoor"
gamma_stealth = {gamma_stealth}
checkpoint_sampling = "{checkpoint_sampling}"
alpha_ckpt = {alpha_ckpt}
{wandb_block_module}
[federated_generate_labels_trigger.expert_config]
experts = 1
min = 0
max = 20
trajectories = [50, 100, 150, 200]

[federated_generate_labels_trigger.attack_config]
iterations = {n_iterations}
one_hot_temp = 5
"""

SELECT_FLIPS_TEMPLATE = """[federated_select_flips]
budgets = {budgets}
input_label_glob = "{module_dir}/labels/labels.npy"
true_labels = "{module_dir}/labels/true.npy"
output_dir = "{flips_dir}"
num_honests = {num_honests}
num_poisoned = {num_poisoned}
"""

TRAIN_USER_TEMPLATE = """[federated_train_user]
input_labels = "{flips_dir}/"
output_dir = "{train_user_dir}"
user_model = "{model_flag}"
trainer = "sgd"
dataset = "{dataset}"
source_label = {source_label}
target_label = {target_label}
poisoner = "1xs"
budget = {budget}
num_honests = {num_honests}
num_poisoned = {num_poisoned}
agg_method = "{agg_method}"
optim_kwargs = {{lr = {lr}, momentum = 0.9, nesterov = true, weight_decay = {wd}}}
schedule_kwargs = {{milestones = {milestones}, gamma = 0.1}}
{wandb_block_train_user}"""


def cell_name(model_flag, dataset, agg_method, seed):
    return f"{model_flag}/{dataset}/{NUM_POISONED}vs{NUM_HONESTS}/{agg_method}/seed{seed}"


def generate_cell(model_flag, dataset, agg_method, seed, budgets, dry_run=False):
    """One full chain (train_expert -> module -> select_flips -> train_user, ONE train_user
    config per budget) for one sweep cell. Returns the list of config paths (written, or that
    WOULD be written under dry_run)."""
    lr = LEARNING_RATE.get(model_flag, 0.1)
    wd = WEIGHT_DECAY.get(model_flag, 2e-4)
    milestones = MILESTONE.get(model_flag, [75, 125])

    cell_dir = EXP_BASE / cell_name(model_flag, dataset, agg_method, seed)
    # One train_expert per seed (not shared/deduped across seeds) -- kept aligned with the
    # sibling federated_generate_labels_trigger_joint generator.
    train_expert_dir = EXP_BASE / f"train_expert/{model_flag}_1xs/seed{seed}"
    module_dir = cell_dir / "gen_labels_trigger"
    flips_dir = cell_dir / "select_flips"

    configs = {
        train_expert_dir / "config.toml": TRAIN_EXPERT_TEMPLATE.format(
            cluster_root=CLUSTER_ROOT, model_flag=model_flag, dataset=dataset, seed=seed,
            source_label=SOURCE_LABEL, target_label=TARGET_LABEL,
            checkpoint_iters=CHECKPOINT_ITERS, epochs=EPOCHS_EXPERT, lr=lr, wd=wd,
            milestones=milestones,
            wandb_block_train_expert=wandb_block(
                "train_expert", f"train_expert/{model_flag}/{dataset}/seed{seed}",
                enabled=WANDB_ENABLED, project=WANDB_PROJECT,
                mode=WANDB_MODE, entity=WANDB_ENTITY, group=model_flag,
            ),
        ),
        module_dir / "config.toml": DIRECT_TRIGGER_TEMPLATE.format(
            cluster_root=CLUSTER_ROOT, model_flag=model_flag, dataset=dataset, seed=seed,
            cell_dir=module_dir, source_label=SOURCE_LABEL, target_label=TARGET_LABEL,
            epsilon=EPSILON, lr_delta=LR_DELTA, lambda_bd=LAMBDA_BD, lambda_delta=LAMBDA_DELTA,
            train_pct=TRAIN_PCT,
            num_honests=NUM_HONESTS, num_poisoned=NUM_POISONED, agg_method=agg_method,
            gamma_stealth=GAMMA_STEALTH, checkpoint_sampling=CHECKPOINT_SAMPLING,
            alpha_ckpt=ALPHA_CKPT, n_iterations=N_ITERATIONS,
            wandb_block_module=wandb_block(
                MODULE_NAME, f"{MODULE_NAME}/{model_flag}/{dataset}/{agg_method}/seed{seed}",
                enabled=WANDB_ENABLED, project=WANDB_PROJECT,
                mode=WANDB_MODE, entity=WANDB_ENTITY, group=model_flag,
            ),
        ),
        flips_dir / "config.toml": SELECT_FLIPS_TEMPLATE.format(
            budgets=budgets, module_dir=module_dir, flips_dir=flips_dir,
            num_honests=NUM_HONESTS, num_poisoned=NUM_POISONED,
        ),
    }
    for budget in budgets:
        train_user_dir = cell_dir / f"train_user_{budget}"
        configs[train_user_dir / "config.toml"] = TRAIN_USER_TEMPLATE.format(
            flips_dir=flips_dir, train_user_dir=train_user_dir, model_flag=model_flag,
            dataset=dataset, source_label=SOURCE_LABEL, target_label=TARGET_LABEL,
            budget=budget, num_honests=NUM_HONESTS, num_poisoned=NUM_POISONED,
            agg_method=agg_method, lr=lr, wd=wd, milestones=milestones,
            wandb_block_train_user=wandb_block(
                "federated_train_user",
                f"train_user/{model_flag}/{dataset}/{agg_method}/{budget}/seed{seed}",
                enabled=WANDB_ENABLED, project=WANDB_PROJECT,
                mode=WANDB_MODE, entity=WANDB_ENTITY, group=model_flag,
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

    return paths


def generate_single_cell(dry_run=True):
    """Exactly one cell (first model/dataset/agg_method/seed, first budget), regularization
    fixed at REGULARIZATION_GRID's defaults -- the minimal preliminary campaign used to sanity
    check the chain before spending a real sweep's compute."""
    return generate_cell(
        MODEL_FLAGS[0], DATASETS[0], AGG_METHODS[0], SEEDS[0], BUDGETS[:1], dry_run=dry_run,
    )


def generate_minimal_campaign(dry_run=True):
    """B3: the minimal exploitable campaign -- 3 seeds x 3 budgets x 2 aggregators, one
    model/dataset. Shown before generating the full grid."""
    all_paths = []
    for model_flag in MODEL_FLAGS[:1]:
        for dataset in DATASETS[:1]:
            for agg_method in AGG_METHODS[:2]:
                for seed in SEEDS[:3]:
                    all_paths += generate_cell(
                        model_flag, dataset, agg_method, seed, BUDGETS[:3], dry_run=dry_run,
                    )
    return all_paths


def generate_all_configs(dry_run=False):
    all_paths = []
    for model_flag in MODEL_FLAGS:
        for dataset in DATASETS:
            for agg_method in AGG_METHODS:
                for seed in SEEDS:
                    all_paths += generate_cell(
                        model_flag, dataset, agg_method, seed, BUDGETS, dry_run=dry_run,
                    )
    return all_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--minimal", action="store_true", help="B3 minimal campaign only")
    parser.add_argument(
        "--single-cell", action="store_true",
        help="exactly one cell, REGULARIZATION_GRID fixed at defaults -- the minimal "
             "preliminary campaign",
    )
    args = parser.parse_args()
    assert not (args.minimal and args.single_cell), "pass at most one of --minimal/--single-cell"

    if args.single_cell:
        gen_fn = generate_single_cell
    elif args.minimal:
        gen_fn = generate_minimal_campaign
    else:
        gen_fn = generate_all_configs
    paths = gen_fn(dry_run=args.dry_run)

    if args.single_cell:
        n_budgets, n_seeds, n_aggs, n_cells = 1, 1, 1, 1
    else:
        n_budgets = len(BUDGETS[:3]) if args.minimal else len(BUDGETS)
        n_seeds = len(SEEDS[:3]) if args.minimal else len(SEEDS)
        n_aggs = len(AGG_METHODS[:2]) if args.minimal else len(AGG_METHODS)
        n_cells = n_seeds * n_aggs * (1 if args.minimal else len(MODEL_FLAGS) * len(DATASETS))
    n_configs_per_cell = 3 + n_budgets  # train_expert (shared) + module + select_flips + N train_user

    if args.dry_run:
        print(f"\n[DRY RUN] {MODULE_NAME}: {n_cells} sweep cells, {len(paths)} config files "
              f"would be written (~{n_configs_per_cell} per cell: train_expert (shared across "
              f"seeds/aggs), module, select_flips, {n_budgets} train_user).")
        for p in paths:
            print(f"  {p}")
    else:
        print(f"\n{MODULE_NAME}: {len(paths)} config files written and schema-validated.")
