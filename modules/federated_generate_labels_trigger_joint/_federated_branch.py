"""
Shared helper for adding a federated Multi-Krum (3-poisoned/7-honest) deployment branch to a
federated_generate_labels_trigger_joint comparison campaign, alongside its existing single-user
(1-poisoned/0-honest/mean) branch.

Used by gen_configs_federated_multikrum_compare.py's sibling isolated-factor studies
(gen_configs_gradmatch_ablation.py, gen_configs_epsilon_sweep.py, gen_configs_lpips_compare.py)
so each can ALSO be checked against Multi-Krum, not just against "mean" -- same rationale as
gen_configs_federated_multikrum_compare.py itself (see its own docstring): the attack (labels/
trigger) is generated ONCE per variant, at that study's own 1-poisoned/0-honest/mean setting,
and the federated branch only repartitions the SAME labels.npy/true.npy across 3v7 workers and
retrains with agg_method="multikrum" -- it never regenerates the attack.

Purely additive -- imported by no pre-existing script (predates this file), so it cannot change
any existing campaign's behavior. FED_TAG matches gen_configs_federated_multikrum_compare.py's
own naming exactly (same FED_NUM_POISONED/FED_NUM_HONESTS/FED_AGG_METHOD), so every campaign in
this family nests its federated branch under the same directory name.
"""
from modules.federated_generate_labels_trigger_joint.gen_configs import (
    SELECT_FLIPS_TEMPLATE,
    TRAIN_USER_TEMPLATE,
    wandb_block,
)

FED_NUM_POISONED = 3
FED_NUM_HONESTS = 7
FED_AGG_METHOD = "multikrum"
FED_TAG = f"federated_{FED_NUM_POISONED}vs{FED_NUM_HONESTS}_{FED_AGG_METHOD}"

# Locally-extended copy of TRAIN_USER_TEMPLATE -- adds `track_poison_selection = {...}` without
# touching the shared constant in gen_configs.py (every OTHER generator's TRAIN_USER_TEMPLATE.
# format() call never passes this field, and gen_configs.py itself never sees this module).
_TRAIN_USER_TEMPLATE_TRACKED = TRAIN_USER_TEMPLATE.replace(
    'agg_method = "{agg_method}"\n',
    'agg_method = "{agg_method}"\n'
    "track_poison_selection = {track_poison_selection}\n",
)
assert _TRAIN_USER_TEMPLATE_TRACKED != TRAIN_USER_TEMPLATE, (
    "TRAIN_USER_TEMPLATE's agg_method line changed shape -- update the splice."
)


def federated_branch_configs(
    *, module_dir, cell_dir, budgets, model_flag, dataset, source_label, target_label,
    trigger_path, lr, wd, milestones, module_name, wandb_run_name_prefix,
    wandb_enabled, wandb_project, wandb_mode, wandb_entity,
):
    """Returns a {path: content} dict for the federated_3vs7_multikrum branch (one select_flips
    + one train_user_{budget} per budget, track_poison_selection=true) nested under
    cell_dir/FED_TAG -- merge this into the caller's own `configs` dict alongside its
    single-user branch (`configs.update(federated_branch_configs(...))`). Reads the SAME
    module_dir (labels.npy/true.npy) and trigger_path the caller's own single-user branch
    already reads -- the attack itself is never regenerated here.
    """
    fed_dir = cell_dir / FED_TAG
    fed_flips_dir = fed_dir / "select_flips"

    configs = {
        fed_flips_dir / "config.toml": SELECT_FLIPS_TEMPLATE.format(
            budgets=budgets,
            module_dir=module_dir,
            flips_dir=fed_flips_dir,
            num_honests=FED_NUM_HONESTS,
            num_poisoned=FED_NUM_POISONED,
        ),
    }

    for budget in budgets:
        train_user_dir = fed_dir / f"train_user_{budget}"
        configs[train_user_dir / "config.toml"] = _TRAIN_USER_TEMPLATE_TRACKED.format(
            flips_dir=fed_flips_dir,
            train_user_dir=train_user_dir,
            model_flag=model_flag,
            dataset=dataset,
            source_label=source_label,
            target_label=target_label,
            trigger_path=trigger_path,
            budget=budget,
            num_honests=FED_NUM_HONESTS,
            num_poisoned=FED_NUM_POISONED,
            agg_method=FED_AGG_METHOD,
            track_poison_selection="true",
            lr=lr,
            wd=wd,
            milestones=milestones,
            wandb_block_train_user=wandb_block(
                "federated_train_user",
                f"{wandb_run_name_prefix}/{FED_TAG}/{budget}",
                enabled=wandb_enabled, project=wandb_project,
                mode=wandb_mode, entity=wandb_entity, group=module_name,
            ),
        )

    return configs
