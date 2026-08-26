"""
Config generator for the federated_generate_labels_trigger_joint campaign chain:
train_expert -> federated_generate_labels_trigger_joint -> federated_select_flips ->
federated_train_user.

Conventions follow modules/base_utils/gen_configs.py (see also
federated_generate_labels_trigger/gen_configs.py, the sibling generator for the INDIRECT
module -- kept deliberately structurally identical so the two are easy to diff/compare).

Two guardrails specific to this module (docs/policy_module_audit_report.md, Bloc B):

  1. checkpoint_sampling divergence warning: if this generator's CHECKPOINT_SAMPLING differs
     from federated_generate_labels_trigger's own default ("uniform"), an indirect-vs-joint
     comparison crosses that factor too, confounding the direct-vs-joint comparison this
     threat-model family is meant to support.
  2. delta_min_frac feasibility refusal: delta_min = delta_min_frac * ||delta_init||_2 is
     computed BEFORE delta is ever clamped to [-epsilon, epsilon] (a known, NOT-corrected bug
     in the module itself -- see docs/threat_models_audit.md) -- so it can demand a magnitude
     the trigger can never reach post-clamp (max reachable ||delta||_2 = epsilon*sqrt(numel)).
     A campaign generated against an infeasible delta_min_frac would measure nothing (expert_asr
     collapses for reasons unrelated to whatever is being swept) -- refused outright, not just
     warned about.
"""
import argparse
import math
import os
from pathlib import Path

import toml

from modules.federated_optimizing_trigger.utils import init_delta

# --------------------------------------------------------------------------- #
# Sweep axes -- edit these for a real campaign. Defaults match the grid agreed
# for this threat-model family (docs/policy_module_audit_report.md, Bloc B) and are kept
# ALIGNED with federated_generate_labels_trigger/gen_configs.py's own defaults wherever the
# axis is shared (num_poisoned/num_honests/budgets/agg_method/epsilon/expert provenance) --
# see the checkpoint_sampling note below for the one axis that is NOT aligned by default.
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

# This module's own prior default is "biased" -- federated_generate_labels_trigger's is
# "uniform". Left at "biased" here (each module's own historical default, per the schema) --
# a cross-module comparison MUST set both to the SAME value; see the warning this generator
# prints when they differ.
CHECKPOINT_SAMPLING = "biased"
INDIRECT_MODULE_CHECKPOINT_SAMPLING = "uniform"  # mirrors the sibling generator's default

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
EPSILON = 0.1        # L_infinity bound on the trigger delta -- larger allows a stronger/more
                      # visible perturbation; too small can make the backdoor unreachable.
LR_DELTA = 1e-2       # Adam learning rate for the trigger optimization.
LAMBDA_BD = 1.0       # weight of the backdoor-efficacy loss (kappa in the P^mean/P^direct
                      # formulas) -- higher pushes harder for backdoor success at the cost of
                      # the matching term.
GAMMA_STEALTH = 1.0   # scalar stealth/backdoor loss weight multiplying grand_loss (UNRELATED
                      # to federated_optimizing_trigger_policy's gamma -- disjoint concept).
# lambda_trigger_l2 (schema's lambda_delta): the L2-norm penalty on delta. Kept at 0.0 in THIS
# module specifically -- the descent toward a null trigger (delta -> 0) is exactly the
# collapse mode this module's anti-collapse machinery (trigger_constraint/align_kappa/
# lambda_align/lambda_mag/delta_min_frac, below) was built to detect and prevent. An L2
# penalty on delta would passively encourage that same collapse, working against the floor
# terms below -- do not raise this without also reconsidering delta_min_frac.
LAMBDA_DELTA = 0.0

TRIGGER_CONSTRAINT = "penalty"
ALIGN_KAPPA = 0.6
LAMBDA_ALIGN = 1.0
LAMBDA_MAG = 1.0
# NOTE: the schema's own default is 0.5, but delta_min_frac=0.5 is INFEASIBLE at EPSILON=0.1
# (see the guard below: 0.5*||delta_init||_2 ~= 115.7 >> 0.1*sqrt(3*32*32) ~= 5.5) -- this is
# exactly the bug the guard exists to catch. Set here to the largest value that stays feasible
# at EPSILON=0.1 (with a small safety margin), so the generator is usable out of the box;
# raise EPSILON instead of DELTA_MIN_FRAC if a stronger magnitude floor is genuinely wanted.
DELTA_MIN_FRAC = 0.02

LEARNING_RATE = {"r32p": 0.1, "r18": 0.1, "vgg": 0.01}
WEIGHT_DECAY = {"r32p": 2e-4, "r18": 2e-4, "vgg": 2e-4}
MILESTONE = {"r32p": [75, 125], "r18": [75, 125], "vgg": [125]}

CLUSTER_ROOT = "/shared/data1/Projects/DLWP/j1067582/martin/FLIP"

EXP_BASE = Path("experiments/federated_experiments/threat_model_direct_trigger_joint").resolve()

MODULE_NAME = "federated_generate_labels_trigger_joint"

# Trigger image shape used for the delta_min feasibility check -- only the small (32x32)
# datasets are supported here (needs_big_ims models/datasets would need a different shape;
# not in the default sweep, see MODEL_FLAGS/DATASETS above).
_TRIGGER_SHAPE = {"cifar": (3, 32, 32), "cifar_100": (3, 32, 32), "svhn": (3, 32, 32)}
_STRENGTH, _FREQ = 6.0, 16  # must match init_delta's call in run_module.py exactly


def write_config(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"[OK] Config written to {path}")


def validate_config(path: Path):
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


def check_delta_min_feasible(dataset, epsilon, delta_min_frac):
    """
    docs/threat_models_audit.md's delta_min-unreachable bug, checked BEFORE generating a
    config: delta_min = delta_min_frac * ||delta_init||_2 (computed pre-clamp, strength=6.0,
    exactly matching run_module.py's own init_delta call) must not exceed
    epsilon*sqrt(numel), the max ||delta||_2 reachable once delta IS clamped to
    [-epsilon,epsilon] post-init. Returns (feasible: bool, delta_min: float, max_reachable:
    float) -- never raises itself, callers decide whether to refuse.
    """
    if dataset not in _TRIGGER_SHAPE:
        raise ValueError(
            f"check_delta_min_feasible: no known trigger shape for dataset={dataset!r} -- "
            f"add it to _TRIGGER_SHAPE (only {list(_TRIGGER_SHAPE)} are supported)."
        )
    shape = _TRIGGER_SHAPE[dataset]
    delta_init = init_delta(
        shape, horizontal=True, strength=_STRENGTH, freq=_FREQ, device="cpu", init="stripe",
    )
    delta_min = delta_min_frac * delta_init.detach().norm().item()
    numel = shape[0] * shape[1] * shape[2]
    max_reachable = epsilon * math.sqrt(numel)
    return delta_min <= max_reachable, delta_min, max_reachable


TRAIN_EXPERT_TEMPLATE = """[train_expert]
output_dir = "{cluster_root}/out/checkpoints/{model_flag}_1xs/0/"
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
"""

JOINT_TRIGGER_TEMPLATE = """[federated_generate_labels_trigger_joint]
input_pths = "{cluster_root}/out/checkpoints/{model_flag}_1xs/{{}}/model_{{}}_{{}}.pth"
opt_pths = "{cluster_root}/out/checkpoints/{model_flag}_1xs/{{}}/model_{{}}_{{}}_opt.pth"
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

trigger_constraint = "{trigger_constraint}"
align_kappa = {align_kappa}
lambda_align = {lambda_align}
lambda_mag = {lambda_mag}
delta_min_frac = {delta_min_frac}

[federated_generate_labels_trigger_joint.expert_config]
experts = 1
min = 0
max = 20
trajectories = [50, 100, 150, 200]

[federated_generate_labels_trigger_joint.attack_config]
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
"""


def cell_name(model_flag, dataset, agg_method, seed):
    return f"{model_flag}/{dataset}/{NUM_POISONED}vs{NUM_HONESTS}/{agg_method}/seed{seed}"


def generate_cell(model_flag, dataset, agg_method, seed, budgets, dry_run=False,
                   delta_min_frac=None):
    delta_min_frac = DELTA_MIN_FRAC if delta_min_frac is None else delta_min_frac

    if CHECKPOINT_SAMPLING != INDIRECT_MODULE_CHECKPOINT_SAMPLING:
        print(
            f"WARNING: checkpoint_sampling={CHECKPOINT_SAMPLING!r} differs from the sibling "
            f"federated_generate_labels_trigger generator's {INDIRECT_MODULE_CHECKPOINT_SAMPLING!r} "
            "-- an indirect-vs-joint comparison crosses this factor too. Set both generators' "
            "CHECKPOINT_SAMPLING to the same value to remove it as a confound."
        )

    feasible, delta_min, max_reachable = check_delta_min_feasible(
        dataset, EPSILON, delta_min_frac,
    )
    if not feasible:
        reason = (
            f"delta_min_frac={delta_min_frac} -> delta_min={delta_min:.4f} > "
            f"epsilon*sqrt(numel)={max_reachable:.4f} at epsilon={EPSILON} -- structurally "
            "unreachable post-clamp (docs/threat_models_audit.md); refusing to generate this "
            "cell, it would measure nothing."
        )
        print(f"REFUSED [{model_flag}/{dataset}/{agg_method}/seed{seed}]: {reason}")
        return [], reason

    lr = LEARNING_RATE.get(model_flag, 0.1)
    wd = WEIGHT_DECAY.get(model_flag, 2e-4)
    milestones = MILESTONE.get(model_flag, [75, 125])

    cell_dir = EXP_BASE / cell_name(model_flag, dataset, agg_method, seed)
    train_expert_dir = EXP_BASE / f"train_expert/{model_flag}_1xs"
    module_dir = cell_dir / "gen_labels_trigger_joint"
    flips_dir = cell_dir / "select_flips"

    configs = {
        train_expert_dir / "config.toml": TRAIN_EXPERT_TEMPLATE.format(
            cluster_root=CLUSTER_ROOT, model_flag=model_flag, dataset=dataset,
            source_label=SOURCE_LABEL, target_label=TARGET_LABEL,
            checkpoint_iters=CHECKPOINT_ITERS, epochs=EPOCHS_EXPERT, lr=lr, wd=wd,
            milestones=milestones,
        ),
        module_dir / "config.toml": JOINT_TRIGGER_TEMPLATE.format(
            cluster_root=CLUSTER_ROOT, model_flag=model_flag, dataset=dataset,
            cell_dir=module_dir, source_label=SOURCE_LABEL, target_label=TARGET_LABEL,
            epsilon=EPSILON, lr_delta=LR_DELTA, lambda_bd=LAMBDA_BD, lambda_delta=LAMBDA_DELTA,
            train_pct=TRAIN_PCT,
            num_honests=NUM_HONESTS, num_poisoned=NUM_POISONED, agg_method=agg_method,
            gamma_stealth=GAMMA_STEALTH, checkpoint_sampling=CHECKPOINT_SAMPLING,
            alpha_ckpt=ALPHA_CKPT, n_iterations=N_ITERATIONS,
            trigger_constraint=TRIGGER_CONSTRAINT, align_kappa=ALIGN_KAPPA,
            lambda_align=LAMBDA_ALIGN, lambda_mag=LAMBDA_MAG, delta_min_frac=delta_min_frac,
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


def generate_single_cell(dry_run=True):
    """Exactly one cell (first model/dataset/agg_method/seed, first budget), regularization
    fixed at REGULARIZATION_GRID's defaults -- the minimal preliminary campaign used to sanity
    check the chain before spending a real sweep's compute."""
    paths, reason = generate_cell(
        MODEL_FLAGS[0], DATASETS[0], AGG_METHODS[0], SEEDS[0], BUDGETS[:1], dry_run=dry_run,
    )
    refused = [(MODEL_FLAGS[0], DATASETS[0], AGG_METHODS[0], SEEDS[0], reason)] if reason else []
    return paths, refused


def generate_minimal_campaign(dry_run=True):
    all_paths, refused = [], []
    for model_flag in MODEL_FLAGS[:1]:
        for dataset in DATASETS[:1]:
            for agg_method in AGG_METHODS[:2]:
                for seed in SEEDS[:3]:
                    paths, reason = generate_cell(
                        model_flag, dataset, agg_method, seed, BUDGETS[:3], dry_run=dry_run,
                    )
                    all_paths += paths
                    if reason:
                        refused.append((model_flag, dataset, agg_method, seed, reason))
    return all_paths, refused


def generate_all_configs(dry_run=False):
    all_paths, refused = [], []
    for model_flag in MODEL_FLAGS:
        for dataset in DATASETS:
            for agg_method in AGG_METHODS:
                for seed in SEEDS:
                    paths, reason = generate_cell(
                        model_flag, dataset, agg_method, seed, BUDGETS, dry_run=dry_run,
                    )
                    all_paths += paths
                    if reason:
                        refused.append((model_flag, dataset, agg_method, seed, reason))
    return all_paths, refused


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
    paths, refused = gen_fn(dry_run=args.dry_run)

    if args.dry_run:
        print(f"\n[DRY RUN] {MODULE_NAME}: {len(paths)} config files would be written.")
        for p in paths:
            print(f"  {p}")
    else:
        print(f"\n{MODULE_NAME}: {len(paths)} config files written and schema-validated.")

    if refused:
        print(f"\n{len(refused)} cell(s) REFUSED (delta_min infeasible):")
        for model_flag, dataset, agg_method, seed, reason in refused:
            print(f"  [{model_flag}/{dataset}/{agg_method}/seed{seed}] {reason}")
