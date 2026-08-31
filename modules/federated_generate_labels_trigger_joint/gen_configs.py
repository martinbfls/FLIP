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
from modules.base_utils.gen_configs import wandb_block

# Weights & Biases mirroring (see modules/base_utils/experiment_tracker.py). Off by
# default -- flip WANDB_ENABLED to True to have every config in this campaign carry a
# [<module>.wandb] table (requires `wandb login` or WANDB_API_KEY in the environment that
# runs them). Local plots/metrics under experiments/.../{plots,logs}/ are unaffected
# either way. gen_configs_sweep.py (which reuses these templates) inherits this toggle.
WANDB_ENABLED = False
WANDB_PROJECT = "flip"
WANDB_ENTITY = None
WANDB_MODE = "online"

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
BUDGETS = [150, 300, 500, 1000, 2000, 2500, 5000]
# Validation run (2026-08-30) of the sweep's "combined candidate" (see
# analyze_trigger_joint_sweep.py's output and the REGULARIZATION_GRID note below): single
# seed, full budget sweep, "mean" only -- with NUM_HONESTS=0 there is only one worker total,
# so "mean" vs "multikrum" aggregation is a no-op distinction here (see
# gen_configs_sweep.py's own AGG_METHOD comment); dropped to avoid doubling this validation
# campaign's cell count for a factor that can't actually vary at NUM_HONESTS=0.
AGG_METHODS = ["mean", "multikrum"]
DATASETS = ["cifar"]
MODEL_FLAGS = ["r32p"]
SOURCE_LABEL = 9
TARGET_LABEL = 4

# This module's own prior default is "biased" -- federated_generate_labels_trigger's is
# "uniform". Left at "biased" here (each module's own historical default, per the schema) --
# a cross-module comparison MUST set both to the SAME value; see the warning this generator
# prints when they differ.
CHECKPOINT_SAMPLING = "biased"
INDIRECT_MODULE_CHECKPOINT_SAMPLING = (
    "uniform"  # mirrors the sibling generator's default
)

ALPHA_CKPT = 0.01
TRAIN_PCT = 1.0
EPOCHS_EXPERT = 20
CHECKPOINT_ITERS = 50
N_ITERATIONS = 25  # Validation run (2026-08-30): sweep's best value for this axis alone
# (ASR 1.0000 vs baseline's 0.9860 at n_iterations=15, see analyze_trigger_joint_sweep.py) --
# part of the "combined candidate" below. (was 15)

# Live expert retraining (2026-08-31): every EXPERT_RETRAIN_INTERVAL outer iterations (each
# one full epoch over mtt_dataset, per N_ITERATIONS above), retrain a fresh expert against the
# CURRENT trigger for EXPERT_RETRAIN_EPOCHS epochs -- closes the H1 gap documented in
# run_module.py's run() docstring ("a design difference from the policy module's
# live-retraining architecture"). 0 disables entirely (unchanged behavior: a frozen expert
# trajectory throughout, as before this feature). EXPERT_RETRAIN_EPOCHS/_CHECKPOINT_ITERS
# default to the SAME values as the initial train_expert step above (EPOCHS_EXPERT/
# CHECKPOINT_ITERS) so the retrained trajectory is comparable in depth/granularity to the one
# it replaces -- override independently below if a shorter/longer retrain is wanted.
EXPERT_RETRAIN_INTERVAL = 5
EXPERT_RETRAIN_EPOCHS = EPOCHS_EXPERT
EXPERT_RETRAIN_CHECKPOINT_ITERS = CHECKPOINT_ITERS

# --------------------------------------------------------------------------- #
# REGULARIZATION_GRID -- trigger regularization knobs, kept separate from the sweep axes
# above so a real campaign's regularization sweep is easy to find and edit in one place.
# `single_cell` mode (see generate_single_cell / --single-cell) fixes ALL of these to the
# defaults below and sweeps only the main axes (SEEDS/BUDGETS/AGG_METHODS).
# --------------------------------------------------------------------------- #
# Relaxed 2026-08-28 (see git history for that pass's reasoning), then VALIDATED 2026-08-30
# against the one-at-a-time hyperparameter sweep in gen_configs_sweep.py (30 cells, 1
# poisoned/0 honest, single budget=1500 -- see analyze_trigger_joint_sweep.py's ranking).
# EPSILON/GAMMA_STEALTH/LAMBDA_DELTA below are the sweep's "combined candidate" (its own
# best-per-axis value substituted in for each axis that beat the 2026-08-28 baseline) --
# NOTE this combination was never itself a sweep cell (one-at-a-time sweeps don't see axis
# interactions): this run (SEEDS/BUDGETS above, full budget sweep) IS that confirmation run.
EPSILON = (
    0.05  # L_infinity bound on the trigger delta. Sweep's best value for this axis
)
# (ASR 0.9900 vs baseline's 0.9860 at epsilon=1.0) -- counterintuitively SMALLER than the
# 2026-08-28 baseline, not larger; a tighter perturbation bound apparently helped THIS
# checkpoint/init combination rather than hurting it. CAUTION: at DELTA_MIN_FRAC=0.01 below,
# this leaves only a ~1.2x feasibility margin (delta_min~=2.31 vs max_reachable~=2.77) --
# thin enough that a small change to init strength/freq could flip the A3-style feasibility
# guard below to REFUSED; re-check its printed delta_min/max_reachable if this generator's
# init constants ever change. (was 1.0)
LR_DELTA = (
    1e-2  # Adam learning rate for the trigger optimization. Sweep confirmed baseline
)
# already best on this axis -- unchanged.
LAMBDA_BD = 2.0  # weight of the backdoor-efficacy loss (kappa in the P^mean/P^direct
# formulas) -- higher pushes harder for backdoor success at the cost of
# the matching term. Sweep confirmed baseline already best on this axis -- unchanged.
GAMMA_STEALTH = (
    1.0  # scalar stealth/backdoor loss weight multiplying grand_loss (UNRELATED
)
# to federated_optimizing_trigger_policy's gamma -- disjoint concept). Sweep's best value for
# this axis (ASR 0.9940 vs baseline's 0.9860 at gamma_stealth=0.3) -- back up at the ORIGINAL
# (pre-2026-08-28) value: at NUM_HONESTS=0/NUM_POISONED=1 this proof-of-concept apparently
# doesn't need stealth traded off against L_bd the way the relaxation pass assumed. (was 0.3)
# lambda_trigger_l2 (schema's lambda_delta): the L2-norm penalty on delta. Sweep's best value
# for this axis (ASR 0.9870 vs baseline's 0.9860 at lambda_delta=0.0) -- CAUTION: this is a
# small margin over baseline, and raising lambda_delta off 0.0 is exactly what the ORIGINAL
# comment here warned against ("do not raise this without also reconsidering delta_min_frac"
# -- an L2 penalty passively encourages the delta->0 collapse this module's anti-collapse
# floor terms below exist to prevent). Kept as the sweep's own suggestion for this
# confirmation run, but if CTA/ASR degrade at the larger budgets, LAMBDA_DELTA is the first
# axis to revert to 0.0.
LAMBDA_DELTA = 1.0

# TRIGGER_CONSTRAINT/ALIGN_KAPPA/LAMBDA_ALIGN/LAMBDA_MAG/DELTA_MIN_FRAC: sweep confirmed the
# 2026-08-28 baseline already best on every one of these axes -- left unchanged for this
# confirmation run.
TRIGGER_CONSTRAINT = "penalty"
ALIGN_KAPPA = (
    0.3  # directional floor on cos(delta, mu_target) -- lowered (was 0.6): easier
)
# to satisfy, so L_align stays inactive more of the time instead of competing with L_bd.
LAMBDA_ALIGN = (
    0.3  # weight of the (now easier-to-satisfy) directional floor. Lowered (was
)
# 1.0) so it still guards against collapse without dominating the loss.
LAMBDA_MAG = 0.3  # weight of the magnitude floor. Lowered (was 1.0), same reasoning.
# NOTE (updated 2026-08-30 for EPSILON=0.05 above, was EPSILON=0.3's ~14x/EPSILON=0.1's ~2x):
# delta_min = 0.01*||delta_init||_2 ~= 2.31 vs max_reachable = 0.05*sqrt(3*32*32) ~= 2.77 --
# only a ~1.2x feasibility margin now, thin enough to be worth re-reading the feasibility
# guard's printed values below rather than assuming this comment stays accurate if EPSILON,
# DELTA_MIN_FRAC, or the init constants (_STRENGTH/_FREQ) change again.
DELTA_MIN_FRAC = 0.01

LEARNING_RATE = {"r32p": 0.1, "r18": 0.1, "vgg": 0.01}
WEIGHT_DECAY = {"r32p": 2e-4, "r18": 2e-4, "vgg": 2e-4}
MILESTONE = {"r32p": [75, 125], "r18": [75, 125], "vgg": [125]}

CLUSTER_ROOT = "/shared/data1/Projects/DLWP/j1067582/martin/FLIP"

EXP_BASE = Path(
    "experiments/federated_experiments/threat_model_direct_trigger_joint"
).resolve()

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
        assert schema_path.exists(), (
            f"Malformed module! Schema {schema_path} does not exist."
        )
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
        shape,
        horizontal=True,
        strength=_STRENGTH,
        freq=_FREQ,
        device="cpu",
        init="stripe",
    )
    delta_min = delta_min_frac * delta_init.detach().norm().item()
    numel = shape[0] * shape[1] * shape[2]
    max_reachable = epsilon * math.sqrt(numel)
    return delta_min <= max_reachable, delta_min, max_reachable


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

JOINT_TRIGGER_TEMPLATE = """[federated_generate_labels_trigger_joint]
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

trigger_constraint = "{trigger_constraint}"
align_kappa = {align_kappa}
lambda_align = {lambda_align}
lambda_mag = {lambda_mag}
delta_min_frac = {delta_min_frac}

expert_retrain_interval = {expert_retrain_interval}
expert_retrain_epochs = {expert_retrain_epochs}
expert_retrain_checkpoint_iters = {expert_retrain_checkpoint_iters}
expert_retrain_optim_kwargs = {{lr = {lr}, momentum = 0.9, nesterov = true, weight_decay = {wd}}}
expert_retrain_scheduler_kwargs = {{milestones = {milestones}, gamma = 0.1}}
{wandb_block_module}
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
{wandb_block_train_user}"""


def cell_name(model_flag, dataset, agg_method, seed):
    return (
        f"{model_flag}/{dataset}/{NUM_POISONED}vs{NUM_HONESTS}/{agg_method}/seed{seed}"
    )


def generate_cell(
    model_flag, dataset, agg_method, seed, budgets, dry_run=False, delta_min_frac=None
):
    delta_min_frac = DELTA_MIN_FRAC if delta_min_frac is None else delta_min_frac

    if CHECKPOINT_SAMPLING != INDIRECT_MODULE_CHECKPOINT_SAMPLING:
        print(
            f"WARNING: checkpoint_sampling={CHECKPOINT_SAMPLING!r} differs from the sibling "
            f"federated_generate_labels_trigger generator's {INDIRECT_MODULE_CHECKPOINT_SAMPLING!r} "
            "-- an indirect-vs-joint comparison crosses this factor too. Set both generators' "
            "CHECKPOINT_SAMPLING to the same value to remove it as a confound."
        )

    feasible, delta_min, max_reachable = check_delta_min_feasible(
        dataset,
        EPSILON,
        delta_min_frac,
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
    # One train_expert per seed (not shared/deduped across seeds): a real proof-of-concept
    # sweep needs the label-generation step to see a genuinely different expert per seed, not
    # the same checkpoint replayed under every seed.
    train_expert_dir = EXP_BASE / f"train_expert/{model_flag}_1xs/seed{seed}"
    module_dir = cell_dir / "gen_labels_trigger_joint"
    flips_dir = cell_dir / "select_flips"

    configs = {
        train_expert_dir / "config.toml": TRAIN_EXPERT_TEMPLATE.format(
            cluster_root=CLUSTER_ROOT,
            model_flag=model_flag,
            dataset=dataset,
            seed=seed,
            source_label=SOURCE_LABEL,
            target_label=TARGET_LABEL,
            checkpoint_iters=CHECKPOINT_ITERS,
            epochs=EPOCHS_EXPERT,
            lr=lr,
            wd=wd,
            milestones=milestones,
            wandb_block_train_expert=wandb_block(
                "train_expert", f"train_expert/{model_flag}/{dataset}/seed{seed}",
                enabled=WANDB_ENABLED, project=WANDB_PROJECT,
                mode=WANDB_MODE, entity=WANDB_ENTITY, group=model_flag,
            ),
        ),
        module_dir / "config.toml": JOINT_TRIGGER_TEMPLATE.format(
            cluster_root=CLUSTER_ROOT,
            model_flag=model_flag,
            dataset=dataset,
            seed=seed,
            cell_dir=module_dir,
            source_label=SOURCE_LABEL,
            target_label=TARGET_LABEL,
            epsilon=EPSILON,
            lr_delta=LR_DELTA,
            lambda_bd=LAMBDA_BD,
            lambda_delta=LAMBDA_DELTA,
            train_pct=TRAIN_PCT,
            num_honests=NUM_HONESTS,
            num_poisoned=NUM_POISONED,
            agg_method=agg_method,
            gamma_stealth=GAMMA_STEALTH,
            checkpoint_sampling=CHECKPOINT_SAMPLING,
            alpha_ckpt=ALPHA_CKPT,
            n_iterations=N_ITERATIONS,
            trigger_constraint=TRIGGER_CONSTRAINT,
            align_kappa=ALIGN_KAPPA,
            lambda_align=LAMBDA_ALIGN,
            lambda_mag=LAMBDA_MAG,
            delta_min_frac=delta_min_frac,
            expert_retrain_interval=EXPERT_RETRAIN_INTERVAL,
            expert_retrain_epochs=EXPERT_RETRAIN_EPOCHS,
            expert_retrain_checkpoint_iters=EXPERT_RETRAIN_CHECKPOINT_ITERS,
            lr=lr,
            wd=wd,
            milestones=milestones,
            wandb_block_module=wandb_block(
                MODULE_NAME,
                f"{MODULE_NAME}/{model_flag}/{dataset}/{agg_method}/seed{seed}",
                enabled=WANDB_ENABLED, project=WANDB_PROJECT,
                mode=WANDB_MODE, entity=WANDB_ENTITY, group=model_flag,
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
            model_flag=model_flag,
            dataset=dataset,
            source_label=SOURCE_LABEL,
            target_label=TARGET_LABEL,
            budget=budget,
            num_honests=NUM_HONESTS,
            num_poisoned=NUM_POISONED,
            agg_method=agg_method,
            lr=lr,
            wd=wd,
            milestones=milestones,
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

    return paths, None


def generate_single_cell(dry_run=True):
    """Exactly one cell (first model/dataset/agg_method/seed, first budget), regularization
    fixed at REGULARIZATION_GRID's defaults -- the minimal preliminary campaign used to sanity
    check the chain before spending a real sweep's compute."""
    paths, reason = generate_cell(
        MODEL_FLAGS[0],
        DATASETS[0],
        AGG_METHODS[0],
        SEEDS[0],
        BUDGETS[:1],
        dry_run=dry_run,
    )
    refused = (
        [(MODEL_FLAGS[0], DATASETS[0], AGG_METHODS[0], SEEDS[0], reason)]
        if reason
        else []
    )
    return paths, refused


def generate_minimal_campaign(dry_run=True):
    all_paths, refused = [], []
    for model_flag in MODEL_FLAGS[:1]:
        for dataset in DATASETS[:1]:
            for agg_method in AGG_METHODS[:2]:
                for seed in SEEDS[:3]:
                    paths, reason = generate_cell(
                        model_flag,
                        dataset,
                        agg_method,
                        seed,
                        BUDGETS[:3],
                        dry_run=dry_run,
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
                        model_flag,
                        dataset,
                        agg_method,
                        seed,
                        BUDGETS,
                        dry_run=dry_run,
                    )
                    all_paths += paths
                    if reason:
                        refused.append((model_flag, dataset, agg_method, seed, reason))
    return all_paths, refused


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--minimal", action="store_true", help="B3 minimal campaign only"
    )
    parser.add_argument(
        "--single-cell",
        action="store_true",
        help="exactly one cell, REGULARIZATION_GRID fixed at defaults -- the minimal "
        "preliminary campaign",
    )
    args = parser.parse_args()
    assert not (args.minimal and args.single_cell), (
        "pass at most one of --minimal/--single-cell"
    )

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
        print(
            f"\n{MODULE_NAME}: {len(paths)} config files written and schema-validated."
        )

    if refused:
        print(f"\n{len(refused)} cell(s) REFUSED (delta_min infeasible):")
        for model_flag, dataset, agg_method, seed, reason in refused:
            print(f"  [{model_flag}/{dataset}/{agg_method}/seed{seed}] {reason}")
