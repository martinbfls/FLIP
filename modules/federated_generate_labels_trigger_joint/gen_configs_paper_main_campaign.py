"""
Main-results campaign for the paper: the three finalist stealth configs selected off
gen_configs_old_objective_port_stealth_sweep.py's own grid (see that module's docstring for the
full stealth-regularization sweep this was picked from) -- "baseline", "eps_0p50_lpips_0p1"
(epsilon=0.5, lambda_lpips=0.1 -- the hybrid config confirmed to COMPOUND rather than dilute the
robust-aggregator-breaking effect, see that module's STEALTH_GRID comment) and "eps_16_255"
(epsilon=16/255, no extra regularization -- the plain low-epsilon control) -- now scaled up
across models/datasets/seeds and a full deployment-aggregator grid, for the paper's headline
numbers.

Unlike gen_configs_old_objective_port_stealth_sweep.py (single model=r32p/dataset=cifar, no seed
axis, one shared global bootstrap expert), THIS generator adds:

  - MODEL_FLAGS = [r32p, convnext_micro], DATASETS = [cifar, svhn]
  - SEEDS = 0..9 (10 seeds) -- each with its OWN bootstrap train_expert (poisoner="1xs"),
    keyed by (model_flag, dataset, seed) -- see _TRAIN_EXPERT_TEMPLATE_DATASET below (the
    upstream gen_configs.py TRAIN_EXPERT_TEMPLATE this is spliced from keys checkpoints only by
    (model_flag, seed), which would silently collide cifar/svhn checkpoints under the SAME path
    for a fixed (model_flag, seed) -- not a problem upstream since that generator never sweeps
    more than one dataset at a time, but a real correctness bug here).
  - `seed = {rng_seed}` wired into the joint-trigger config (absent in
    gen_configs_old_objective_port.py's own template, since that campaign never needed
    reproducible expert-retraining rounds across a seed axis -- see schemas/
    federated_generate_labels_trigger_joint.toml's own `seed` doc: reseeds every
    expert_retrain_interval round to replicate the ORIGINAL bootstrap expert's init/data order).

Deployment (per generated trigger): TWO branches, exactly as requested --
  - single_user: num_poisoned=1/num_honests=0/agg_method="mean" (same convention as
    gen_configs_old_objective_port.py's own generation-time split, now ALSO used to deploy)
  - federated ("federated_3vs7"): num_poisoned=3/num_honests=7, swept over
    DEPLOY_AGG_METHODS_FEDERATED = [mean, krum, multikrum, median, trmean]
  both swept over the SAME DEPLOY_BUDGETS = [0, 150, 300, 500, 1000, 2000, 2500, 5000].
  track_poison_selection=true only for {multikrum, krum} (the only two agg_methods
  util.py's mini_train_multi actually records selection stats for, per schemas/
  federated_train_user.toml's own doc) -- false for mean/median/trmean.

Grid size (see --print-grid for the authoritative enumeration): 2 models x 2 datasets x 10 seeds
x 3 tags = 120 generation cells, each deploying single_user (8 budgets) + federated (5 agg x 8
budgets = 40) = 48 train_user cells -> 5760 train_user configs total, plus 120 gen + 240 flips
(single_user + federated, one each per gen cell) + 40 bootstrap (train_expert, shared across the
3 tags for a fixed (model_flag, dataset, seed)) = 6160 config files / eventual Slurm jobs. This
is the full paper campaign, not a preliminary sweep -- expect it to take a while both to generate
and to run.

Run:
  python -m modules.federated_generate_labels_trigger_joint.gen_configs_paper_main_campaign --print-grid
  python -m modules.federated_generate_labels_trigger_joint.gen_configs_paper_main_campaign --dry-run
  python -m modules.federated_generate_labels_trigger_joint.gen_configs_paper_main_campaign
then orchestrate_slurm/orchestrate_runs_trigger_joint_paper_main_campaign_slurm.sh to submit.
"""

import argparse
from pathlib import Path

from modules.base_utils.config_validation import (
    write_config,
    validate_config_file as validate_config,
)
from modules.federated_generate_labels_trigger_joint.gen_configs import (
    CLUSTER_ROOT,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MILESTONE,
    TRAIN_EXPERT_TEMPLATE,
    check_delta_min_feasible,
    draw_rng_seed,
    wandb_block,
    WANDB_ENABLED,
    WANDB_PROJECT,
    WANDB_MODE,
    WANDB_ENTITY,
)
from modules.federated_generate_labels_trigger_joint.gen_configs_old_objective_port import (
    SOURCE_LABEL,
    TARGET_LABEL,
    GEN_INIT,
    GEN_TRIGGER_CONSTRAINT,
    GEN_N_CHECKPOINTS_PER_STEP,
    GEN_EXPERT_RETRAIN_INTERVAL,
    GEN_EXPERT_RETRAIN_EPOCHS,
    GEN_EXPERT_RETRAIN_CHECKPOINT_ITERS,
    SELECT_FLIPS_TEMPLATE,
    TRAIN_USER_TEMPLATE,
)
from modules.federated_generate_labels_trigger_joint.gen_configs_old_objective_port_stealth_sweep import (
    STEALTH_BASE,
    STEALTH_GRID,
    _JOINT_TRIGGER_TEMPLATE_STEALTH,
)

# --------------------------------------------------------------------------- #
# The three finalist tags, pulled straight off STEALTH_GRID by name so their hyperparameters
# can never drift from what was actually validated in the stealth sweep -- edit STEALTH_GRID
# in gen_configs_old_objective_port_stealth_sweep.py, not here, if one of these configs changes.
# --------------------------------------------------------------------------- #
_REQUESTED_TAGS = ["baseline", "eps_0p50_lpips_0p1", "eps_16_255"]
_STEALTH_GRID_BY_TAG = dict(STEALTH_GRID)
for _tag in _REQUESTED_TAGS:
    assert _tag in _STEALTH_GRID_BY_TAG, (
        f"'{_tag}' not found in gen_configs_old_objective_port_stealth_sweep.STEALTH_GRID -- "
        "has it been renamed/removed?"
    )
CONFIG_TAGS = [(tag, _STEALTH_GRID_BY_TAG[tag]) for tag in _REQUESTED_TAGS]

# --------------------------------------------------------------------------- #
# Sweep axes.
# --------------------------------------------------------------------------- #
MODEL_FLAGS = ["r32p", "convnext_micro"]
DATASETS = ["cifar", "svhn"]
SEEDS = list(range(10))

GEN_NUM_POISONED = 1
GEN_NUM_HONESTS = 0
GEN_AGG_METHOD = "mean"

DEPLOY_BUDGETS = [0, 150, 300, 500, 1000, 2000, 2500, 5000]

DEPLOY_SINGLE_USER_NUM_POISONED = 1
DEPLOY_SINGLE_USER_NUM_HONESTS = 0
DEPLOY_SINGLE_USER_AGG_METHOD = "mean"

DEPLOY_FEDERATED_NUM_POISONED = 3
DEPLOY_FEDERATED_NUM_HONESTS = 7
DEPLOY_AGG_METHODS_FEDERATED = ["mean", "krum", "multikrum", "median", "trmean"]
FEDERATED_TAG = "federated_3vs7"
# Only these two agg_methods are actually instrumented by util.py's mini_train_multi (see
# schemas/federated_train_user.toml's own track_poison_selection doc) -- true for both, false
# (the template default) for mean/median/trmean.
_TRACKED_AGG_METHODS = {"multikrum", "krum"}

MODULE_NAME = "federated_generate_labels_trigger_joint"

EXP_BASE = Path(
    "experiments/federated_experiments/threat_model_direct_trigger_joint_paper_main_campaign"
).resolve()

# --------------------------------------------------------------------------- #
# Locally-extended copies: dataset-aware bootstrap checkpoint path (see module docstring), and
# track_poison_selection spliced into TRAIN_USER_TEMPLATE (imported from
# gen_configs_old_objective_port.py, which has neither field).
# --------------------------------------------------------------------------- #
_TRAIN_EXPERT_TEMPLATE_DATASET = TRAIN_EXPERT_TEMPLATE.replace(
    'output_dir = "{cluster_root}/out/checkpoints/{model_flag}_1xs/seed{seed}/0/"\n',
    'output_dir = "{cluster_root}/out/checkpoints/{model_flag}_{dataset}_1xs/seed{seed}/0/"\n',
).replace(
    'poisoner = "1xs"\n',
    'poisoner = "1xs"\n'
    "budget = {budget}\n",
)
assert _TRAIN_EXPERT_TEMPLATE_DATASET != TRAIN_EXPERT_TEMPLATE, (
    "TRAIN_EXPERT_TEMPLATE's output_dir/poisoner lines changed shape -- update the splice."
)

# get_matching_datasets' own no-budget default is len(train_data)//n_classes (modules/base_utils/
# datasets.py) -- exactly right for CIFAR (5000 per class, perfectly balanced) but WRONG for
# SVHN, whose classes are heavily imbalanced (digit '9' -- SOURCE_LABEL here -- has only ~4659
# eligible train examples, far under the 7325 that formula assumes): the un-set default budget
# crashed every SVHN bootstrap cell with "Budget requires 7325 poisoned samples, but only 4659
# eligible samples are available." Pin an explicit, per-dataset-safe budget instead of relying on
# that default -- 5000 for cifar (bit-identical to the previous unset-default behavior), a value
# safely under SVHN's smallest class count for svhn.
BOOTSTRAP_BUDGET = {"cifar": 5000, "svhn": 4500}

_JOINT_TRIGGER_TEMPLATE_MAIN = _JOINT_TRIGGER_TEMPLATE_STEALTH.replace(
    "expert_retrain_scheduler_kwargs = {{milestones = {milestones}, gamma = 0.1}}\n",
    "expert_retrain_scheduler_kwargs = {{milestones = {milestones}, gamma = 0.1}}\n"
    "seed = {rng_seed}\n",
)
assert _JOINT_TRIGGER_TEMPLATE_MAIN != _JOINT_TRIGGER_TEMPLATE_STEALTH, (
    "_JOINT_TRIGGER_TEMPLATE_STEALTH's expert_retrain_scheduler_kwargs line changed shape -- "
    "update the splice."
)

_TRAIN_USER_TEMPLATE_TRACKABLE = TRAIN_USER_TEMPLATE.replace(
    'agg_method = "{agg_method}"\n',
    'agg_method = "{agg_method}"\n'
    "track_poison_selection = {track_poison_selection}\n",
)
assert _TRAIN_USER_TEMPLATE_TRACKABLE != TRAIN_USER_TEMPLATE, (
    "TRAIN_USER_TEMPLATE's agg_method line changed shape -- update the splice."
)

DELTA_MIN_FRAC = 0.0  # inert either way -- lambda_mag=0.0 is hardcoded in
# _JOINT_TRIGGER_TEMPLATE_STEALTH (see gen_configs_old_objective_port_stealth_sweep.py's own
# DELTA_MIN_FRAC comment); kept only so check_delta_min_feasible's guard still runs.


def bootstrap_output_dir(model_flag, dataset, seed):
    return f"{CLUSTER_ROOT}/out/checkpoints/{model_flag}_{dataset}_1xs/seed{seed}"


def cell_name(model_flag, dataset, tag, seed):
    return f"{model_flag}/{dataset}/{tag}/seed{seed}"


def bootstrap_cell_name(model_flag, dataset, seed):
    return f"train_expert/{model_flag}_{dataset}_1xs/seed{seed}"


def _train_user_configs(
    *, base_dir, budgets, num_honests, num_poisoned, agg_methods, flips_dir, model_flag,
    dataset, trigger_path, lr, wd, milestones,
):
    configs = {}
    for agg_method in agg_methods:
        for budget in budgets:
            train_user_dir = base_dir / agg_method / f"train_user_{budget}"
            configs[train_user_dir / "config.toml"] = _TRAIN_USER_TEMPLATE_TRACKABLE.format(
                flips_dir=flips_dir,
                train_user_dir=train_user_dir,
                model_flag=model_flag,
                dataset=dataset,
                source_label=SOURCE_LABEL,
                target_label=TARGET_LABEL,
                trigger_path=trigger_path,
                budget=budget,
                deploy_num_honests=num_honests,
                deploy_num_poisoned=num_poisoned,
                agg_method=agg_method,
                track_poison_selection=(
                    "true" if agg_method in _TRACKED_AGG_METHODS else "false"
                ),
                lr=lr,
                wd=wd,
                milestones=milestones,
            )
    return configs


def generate_bootstrap_cell(model_flag, dataset, seed, dry_run=False):
    lr = LEARNING_RATE.get(model_flag, 0.1)
    wd = WEIGHT_DECAY.get(model_flag, 2e-4)
    milestones = MILESTONE.get(model_flag, [75, 125])
    rng_seed = draw_rng_seed(model_flag, (dataset, seed))

    bootstrap_cfg_dir = EXP_BASE / bootstrap_cell_name(model_flag, dataset, seed)

    content = _TRAIN_EXPERT_TEMPLATE_DATASET.format(
        cluster_root=CLUSTER_ROOT,
        model_flag=model_flag,
        dataset=dataset,
        seed=seed,
        rng_seed=rng_seed,
        source_label=SOURCE_LABEL,
        target_label=TARGET_LABEL,
        budget=BOOTSTRAP_BUDGET[dataset],
        checkpoint_iters=GEN_EXPERT_RETRAIN_CHECKPOINT_ITERS,
        epochs=GEN_EXPERT_RETRAIN_EPOCHS,
        lr=lr,
        wd=wd,
        milestones=milestones,
        wandb_block_train_expert=wandb_block(
            "train_expert", f"train_expert/{model_flag}/{dataset}/seed{seed}",
            enabled=WANDB_ENABLED, project=WANDB_PROJECT,
            mode=WANDB_MODE, entity=WANDB_ENTITY, group=MODULE_NAME,
        ),
    )
    path = bootstrap_cfg_dir / "config.toml"
    assert "out/checkpoints" not in str(path), f"Refusing to write under out/checkpoints/: {path}"
    if not dry_run:
        write_config(path, content)
        validate_config(path)
    return [path]


def generate_cell(model_flag, dataset, tag, overrides, seed, dry_run=False):
    cfg = {**STEALTH_BASE, **overrides}

    feasible, delta_min, max_reachable = check_delta_min_feasible(
        dataset, cfg["epsilon"], DELTA_MIN_FRAC,
    )
    if not feasible:
        reason = (
            f"delta_min_frac={DELTA_MIN_FRAC} -> delta_min={delta_min:.4f} > "
            f"epsilon*sqrt(numel)={max_reachable:.4f} at epsilon={cfg['epsilon']} -- "
            "structurally unreachable post-clamp; refusing to generate this cell."
        )
        print(f"REFUSED [{model_flag}/{dataset}/{tag}/seed{seed}]: {reason}")
        return [], reason

    lr = LEARNING_RATE.get(model_flag, 0.1)
    wd = WEIGHT_DECAY.get(model_flag, 2e-4)
    milestones = MILESTONE.get(model_flag, [75, 125])
    rng_seed = draw_rng_seed(model_flag, (dataset, seed))

    bootstrap_dir = bootstrap_output_dir(model_flag, dataset, seed)
    cell_dir = EXP_BASE / cell_name(model_flag, dataset, tag, seed)
    module_dir = cell_dir / "gen_labels_trigger_joint"

    configs = {
        module_dir / "config.toml": _JOINT_TRIGGER_TEMPLATE_MAIN.format(
            bootstrap_dir=bootstrap_dir,
            cell_dir=module_dir,
            model_flag=model_flag,
            dataset=dataset,
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
            rng_seed=rng_seed,
            lr=lr,
            wd=wd,
            milestones=milestones,
            iterations=cfg["iterations"],
        ),
    }

    trigger_path = (
        module_dir / "trigger"
        / f"opt_trig_direct_joint_{GEN_INIT}_{model_flag}_{dataset}_{GEN_NUM_POISONED}vs{GEN_NUM_HONESTS}.pt"
    )

    # -- single_user branch: 1-poisoned/0-honest/mean, same split as generation time. --
    single_user_flips_dir = cell_dir / "select_flips"
    configs[single_user_flips_dir / "config.toml"] = SELECT_FLIPS_TEMPLATE.format(
        budgets=DEPLOY_BUDGETS,
        module_dir=module_dir,
        flips_dir=single_user_flips_dir,
        deploy_num_honests=DEPLOY_SINGLE_USER_NUM_HONESTS,
        deploy_num_poisoned=DEPLOY_SINGLE_USER_NUM_POISONED,
    )
    configs.update(_train_user_configs(
        base_dir=cell_dir, budgets=DEPLOY_BUDGETS,
        num_honests=DEPLOY_SINGLE_USER_NUM_HONESTS, num_poisoned=DEPLOY_SINGLE_USER_NUM_POISONED,
        agg_methods=[DEPLOY_SINGLE_USER_AGG_METHOD], flips_dir=single_user_flips_dir,
        model_flag=model_flag, dataset=dataset, trigger_path=trigger_path,
        lr=lr, wd=wd, milestones=milestones,
    ))

    # -- federated branch: 3-poisoned/7-honest, swept over DEPLOY_AGG_METHODS_FEDERATED. --
    fed_dir = cell_dir / FEDERATED_TAG
    fed_flips_dir = fed_dir / "select_flips"
    configs[fed_flips_dir / "config.toml"] = SELECT_FLIPS_TEMPLATE.format(
        budgets=DEPLOY_BUDGETS,
        module_dir=module_dir,
        flips_dir=fed_flips_dir,
        deploy_num_honests=DEPLOY_FEDERATED_NUM_HONESTS,
        deploy_num_poisoned=DEPLOY_FEDERATED_NUM_POISONED,
    )
    configs.update(_train_user_configs(
        base_dir=fed_dir, budgets=DEPLOY_BUDGETS,
        num_honests=DEPLOY_FEDERATED_NUM_HONESTS, num_poisoned=DEPLOY_FEDERATED_NUM_POISONED,
        agg_methods=DEPLOY_AGG_METHODS_FEDERATED, flips_dir=fed_flips_dir,
        model_flag=model_flag, dataset=dataset, trigger_path=trigger_path,
        lr=lr, wd=wd, milestones=milestones,
    ))

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
    for model_flag in MODEL_FLAGS:
        for dataset in DATASETS:
            for seed in SEEDS:
                all_paths += generate_bootstrap_cell(model_flag, dataset, seed, dry_run=dry_run)
            for tag, overrides in CONFIG_TAGS:
                for seed in SEEDS:
                    paths, reason = generate_cell(
                        model_flag, dataset, tag, overrides, seed, dry_run=dry_run,
                    )
                    all_paths += paths
                    if reason:
                        refused.append((model_flag, dataset, tag, seed, reason))
    return all_paths, refused


def list_grid():
    """Enumerates this campaign's (model_flag, dataset, tag, seed, branch, agg_method, budget)
    cells as plain dicts, without writing any config -- the single source of truth for the
    grid's shape, so orchestrate_slurm/orchestrate_runs_trigger_joint_paper_main_campaign_slurm.sh
    doesn't need to duplicate these axes by hand."""
    cells = []
    for model_flag in MODEL_FLAGS:
        for dataset in DATASETS:
            for tag, _ in CONFIG_TAGS:
                for seed in SEEDS:
                    for budget in DEPLOY_BUDGETS:
                        cells.append({
                            "model_flag": model_flag, "dataset": dataset, "tag": tag,
                            "seed": seed, "branch": "single_user",
                            "agg_method": DEPLOY_SINGLE_USER_AGG_METHOD, "budget": budget,
                        })
                    for agg_method in DEPLOY_AGG_METHODS_FEDERATED:
                        for budget in DEPLOY_BUDGETS:
                            cells.append({
                                "model_flag": model_flag, "dataset": dataset, "tag": tag,
                                "seed": seed, "branch": FEDERATED_TAG,
                                "agg_method": agg_method, "budget": budget,
                            })
    return cells


def list_tags():
    return [tag for tag, _ in CONFIG_TAGS]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--print-grid", action="store_true",
        help="print the full (model, dataset, tag, seed, branch, agg_method, budget) cell list "
             "as JSON and exit, without writing configs",
    )
    parser.add_argument(
        "--print-tags", action="store_true",
        help="print just the three finalist tags (one per line) and exit, without writing configs",
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
        print(f"\n[DRY RUN] {MODULE_NAME} paper_main_campaign: {len(paths)} config files would be written.")
    else:
        print(f"\n{MODULE_NAME} paper_main_campaign: {len(paths)} config files written and schema-validated.")

    if refused:
        print(f"\n{len(refused)} cell(s) REFUSED (delta_min infeasible):")
        for model_flag, dataset, tag, seed, reason in refused:
            print(f"  [{model_flag}/{dataset}/{tag}/seed{seed}] {reason}")
