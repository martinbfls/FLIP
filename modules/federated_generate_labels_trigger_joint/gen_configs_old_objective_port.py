"""
Config generator for the old-objective-port campaign chain:
train_expert (bootstrap) -> federated_generate_labels_trigger_joint -> federated_select_flips ->
federated_train_user, sweeping the DEPLOYMENT (victim) aggregator and budget.

Ports the old federated_optimizing_trigger objective (trigger_penalty, cos(delta,mu_target)+1 --
reintroduced in modules/federated_optimizing_trigger/utils.py after trigger_penalty_hinge
replaced it) into federated_generate_labels_trigger_joint, under the exact generation-time
config that broke every robust aggregator in the old module: num_honests=0, num_poisoned=1,
agg_method="mean". See run_module.py's L_pen assignment for the code-side half of this port, and
this file's own GEN_* constants below for the full rationale (expert bootstrap, retrain timing,
epsilon, etc.) -- carried over unchanged from the hand-written configs this generator replaces
(experiments/federated_experiments/threat_model_direct_trigger_joint_old_objective_port/).

Unlike modules/federated_generate_labels_trigger_joint/gen_configs.py (whose sweep axes are the
GENERATION-time model/dataset/agg_method/seed), the generation-time config here is FIXED (that's
the whole point of the port -- reproduce the one config that worked) and what's actually swept is
the DEPLOYMENT (victim) side: agg_method in {mean, trmean, multikrum} x budget in {500, 1500,
5000}, independent of the 0-honest/1-poisoned split used to generate the trigger (same 3-poisoned
/7-honest split already used elsewhere in the repo to evaluate a generated trigger against
trmean/multikrum, e.g. threat_model_direct_trigger/r32p/cifar/3vs7 and
gen_configs_federated_multikrum_compare.py). select_flips is shared across every agg_method (it
depends only on the worker split and budget list, not on the aggregator), so this campaign is
1 train_expert + 1 gen_labels_trigger_joint + 1 select_flips + 9 train_user configs, not 9
independent generation runs.

NOTE on B1/B2 (ruled out, not a gap): the CURRENT on-disk federated_optimizing_trigger/
run_module.py (B1_k/B2_k, a gradient-shift-vs-label-flip-polytope feasibility term) is a LATER
rewrite ("Finalize federated_optimizing_trigger as a non-federated B1/B2 objective") of the
module -- it is NOT the code that actually produced the winning num_honests=0/num_poisoned=1/
agg="mean" config this campaign ports. That run used an earlier, genuinely FEDERATED version of
the module (build_worker_loaders + federated_aggregate/agg() mixing honest+poisoned client
gradients into agg_mix, cosine_grad_loss(agg_mix, mu_poison) as L_match), whose objective was:
    L_tot = lambda_match*L_match + lambda_adv*L_adv + lambda_penalty*L_pen + lambda_delta*||delta||
with L_pen = trigger_penalty(delta, mu) (ported above) and L_adv == this module's own L_bd. At
num_honests=0/num_poisoned=1, agg_mix reduces to the single poisoned worker's own gradient
(mu_poison), making L_match = 1 - cos(g, g) = 0 structurally -- so GEN_LAMBDA_BD/GEN_LAMBDA_PENALTY/
GEN_LAMBDA_DELTA below already cover the objective's only non-vanishing terms; there is nothing
further to port for THIS specific config. lambda_match would need porting only for a FUTURE cell
with num_honests>0 at generation time (this campaign's generation-time split stays 0/1, so it
doesn't apply here).
"""

import argparse
from pathlib import Path

from modules.base_utils.config_validation import (
    write_config,
    validate_config_file as validate_config,
)

CLUSTER_ROOT = "/shared/data1/Projects/DLWP/j1067582/martin/FLIP"
EXP_BASE = Path(
    "experiments/federated_experiments/threat_model_direct_trigger_joint_old_objective_port"
).resolve()

MODEL_FLAG = "r32p"
DATASET = "cifar"
SOURCE_LABEL = 9
TARGET_LABEL = 4
LR = 0.1
WD = 2e-4
MILESTONES = [75, 125]

# --------------------------------------------------------------------------- #
# Generation-time config: FIXED, not swept -- see module docstring above.
# --------------------------------------------------------------------------- #
GEN_NUM_HONESTS = 0
GEN_NUM_POISONED = 1
GEN_AGG_METHOD = "mean"

GEN_INIT = "stripe"
GEN_EPSILON = 1.0  # the old opt_trigger winning config's own value (was mistakenly ported as
# 0.1 -- a different, unrelated smoke-test config's value -- in this campaign's first pass).
GEN_LR_DELTA = 1e-2
GEN_LAMBDA_BD = 1.0        # = lambda_adv of the old objective (L_bd == L_adv)
GEN_LAMBDA_PENALTY = 1.0   # from the old opt_trigger config that broke every aggregator
GEN_LAMBDA_DELTA = 0.0     # idem (old config's lambda_delta was 0.0)
GEN_TRIGGER_CONSTRAINT = "penalty"
GEN_N_CHECKPOINTS_PER_STEP = 5  # was 1 (old module's own accumulation bug meant only 1 counted
# regardless of num_chckpt); raised to 5 as a real multi-checkpoint test now that the campaign
# has moved past a literal single-checkpoint port. Requires pool_size >= 5 -- run_module.py's
# own default (15) is used, not overridden here.

GEN_EXPERT_RETRAIN_INTERVAL = 1        # every outer iteration, see module docstring's "Missing
# piece" note above for what this replaces (the old module's step-0 mini_train aliasing
# expert_path/output_dir).
GEN_EXPERT_RETRAIN_EPOCHS = 20         # = train_expert's own `epochs` (bootstrap and original)
GEN_EXPERT_RETRAIN_CHECKPOINT_ITERS = 50  # = train_expert's own `checkpoint_iters`

GEN_ITERATIONS = 5  # attack_config.iterations: the number of OUTER iterations of this module's
# own loop, each one full epoch over mtt_dataset (`for batches in zip(*loaders)`) -- i.e. one
# full pass updating labels_syn/delta once per batch, THEN (since expert_retrain_interval=1)
# retraining the expert from scratch against the drifted delta before the next pass. The old
# module's winning config was n_steps=1 -- a single retrain-then-optimize round; iterations=5
# here is a starting point to run that round several times over and see whether
# cos_delta_to_init keeps moving as the expert keeps re-chasing the trigger, not something
# itself derived from the old module.

# Bootstrap expert (../train_expert/r32p_1xs_bootstrap/config.toml in the hand-written version):
# poisoner="1xs" (StripePoisoner(strength=6, freq=16)) is bit-identical to the raw, UNCLAMPED
# init_delta(strength=6.0, freq=16, horizontal=True) stripe both the old module's delta_eval
# (captured before any clamp_) and this module's own `delta` (its own clamp_ only runs AFTER
# optimizer_delta.step(), so the first forward pass also sees the raw stripe) start from -- so
# this bootstrap expert is trained against exactly the trigger both modules actually begin with.
BOOTSTRAP_OUTPUT_DIR = f"{CLUSTER_ROOT}/out/checkpoints/old_objective_port_r32p_1xs_bootstrap/0/"

# --------------------------------------------------------------------------- #
# Deployment (victim) sweep -- what this generator actually varies.
# --------------------------------------------------------------------------- #
DEPLOY_AGG_METHODS = ["mean", "trmean", "multikrum"]
DEPLOY_BUDGETS = [500, 1500, 5000]
DEPLOY_NUM_HONESTS = 7
DEPLOY_NUM_POISONED = 3

MODULE_NAME = "federated_generate_labels_trigger_joint"

TRAIN_EXPERT_BOOTSTRAP_TEMPLATE = """[train_expert]
output_dir = "{output_dir}"
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
input_pths = "{bootstrap_dir}/{{}}/model_{{}}_{{}}.pth"
opt_pths = "{bootstrap_dir}/{{}}/model_{{}}_{{}}_opt.pth"
output_dir = "{cell_dir}/labels/"
output_dir_trigger = "{cell_dir}/trigger"
expert_model = "{model_flag}"
dataset = "{dataset}"
source_label = {source_label}
target_label = {target_label}

init = "{init}"
epsilon = {epsilon}
lr_delta = {lr_delta}
lambda_bd = {lambda_bd}
lambda_penalty = {lambda_penalty}
lambda_delta = {lambda_delta}

num_honests = {gen_num_honests}
num_poisoned = {gen_num_poisoned}
agg_method = "{gen_agg_method}"

lambda_align = 0.0
lambda_mag = 0.0
lambda_tv = 0.0
lambda_lpips = 0.0
lambda_gradmatch = 0.0
lambda_match = 0.0
trigger_constraint = "{trigger_constraint}"
n_checkpoints_per_step = {n_checkpoints_per_step}

expert_retrain_interval = {expert_retrain_interval}
expert_retrain_epochs = {expert_retrain_epochs}
expert_retrain_checkpoint_iters = {expert_retrain_checkpoint_iters}
expert_retrain_optim_kwargs = {{lr = {lr}, momentum = 0.9, nesterov = true, weight_decay = {wd}}}
expert_retrain_scheduler_kwargs = {{milestones = {milestones}, gamma = 0.1}}

checkpoint_sampling = "biased"
alpha_ckpt = 0.01
train_pct = 1.0
gamma_stealth = 1.0
attack = "backdoor"

metrics_log_path = "{cell_dir}/metrics.json"

[federated_generate_labels_trigger_joint.expert_config]
experts = 1
min = 0
max = 20
trajectories = [50, 100, 150, 200]

[federated_generate_labels_trigger_joint.attack_config]
iterations = {iterations}
one_hot_temp = 5
"""

SELECT_FLIPS_TEMPLATE = """[federated_select_flips]
budgets = {budgets}
input_label_glob = "{module_dir}/labels/labels.npy"
true_labels = "{module_dir}/labels/true.npy"
output_dir = "{flips_dir}"
num_honests = {deploy_num_honests}
num_poisoned = {deploy_num_poisoned}
"""

TRAIN_USER_TEMPLATE = """[federated_train_user]
input_labels = "{flips_dir}/"
output_dir = "{train_user_dir}"
user_model = "{model_flag}"
trainer = "sgd"
dataset = "{dataset}"
source_label = {source_label}
target_label = {target_label}
poisoner = "optimized"
delta = "{trigger_path}"
budget = {budget}
num_honests = {deploy_num_honests}
num_poisoned = {deploy_num_poisoned}
agg_method = "{agg_method}"
optim_kwargs = {{lr = {lr}, momentum = 0.9, nesterov = true, weight_decay = {wd}}}
schedule_kwargs = {{milestones = {milestones}, gamma = 0.1}}
"""


def trigger_output_path(module_dir):
    """Path to the .pt trigger gen_labels_trigger_joint saves (run_module.py's run():
    torch.save(delta, output_dir_trigger/f"opt_trig_direct_joint_{init}_{model_flag}_{dataset}_
    {num_poisoned}vs{num_honests}.pt")) -- train_user needs this to poison with the ACTUALLY
    optimized trigger (poisoner="optimized") rather than the fixed "1xs" stripe pattern."""
    run_tag = f"{GEN_NUM_POISONED}vs{GEN_NUM_HONESTS}"
    return (
        module_dir
        / "trigger"
        / f"opt_trig_direct_joint_{GEN_INIT}_{MODEL_FLAG}_{DATASET}_{run_tag}.pt"
    )


def generate(dry_run=False):
    bootstrap_dir = f"{CLUSTER_ROOT}/out/checkpoints/old_objective_port_r32p_1xs_bootstrap"
    bootstrap_cfg_dir = EXP_BASE / "train_expert/r32p_1xs_bootstrap"
    cell_dir = EXP_BASE / f"{MODEL_FLAG}/{DATASET}"
    module_dir = cell_dir / "gen_labels_trigger_joint"
    flips_dir = cell_dir / "select_flips"

    configs = {
        bootstrap_cfg_dir / "config.toml": TRAIN_EXPERT_BOOTSTRAP_TEMPLATE.format(
            output_dir=BOOTSTRAP_OUTPUT_DIR,
            model_flag=MODEL_FLAG,
            dataset=DATASET,
            source_label=SOURCE_LABEL,
            target_label=TARGET_LABEL,
            checkpoint_iters=GEN_EXPERT_RETRAIN_CHECKPOINT_ITERS,
            epochs=GEN_EXPERT_RETRAIN_EPOCHS,
            lr=LR,
            wd=WD,
            milestones=MILESTONES,
        ),
        module_dir / "config.toml": JOINT_TRIGGER_TEMPLATE.format(
            bootstrap_dir=bootstrap_dir,
            cell_dir=module_dir,
            model_flag=MODEL_FLAG,
            dataset=DATASET,
            source_label=SOURCE_LABEL,
            target_label=TARGET_LABEL,
            init=GEN_INIT,
            epsilon=GEN_EPSILON,
            lr_delta=GEN_LR_DELTA,
            lambda_bd=GEN_LAMBDA_BD,
            lambda_penalty=GEN_LAMBDA_PENALTY,
            lambda_delta=GEN_LAMBDA_DELTA,
            gen_num_honests=GEN_NUM_HONESTS,
            gen_num_poisoned=GEN_NUM_POISONED,
            gen_agg_method=GEN_AGG_METHOD,
            trigger_constraint=GEN_TRIGGER_CONSTRAINT,
            n_checkpoints_per_step=GEN_N_CHECKPOINTS_PER_STEP,
            expert_retrain_interval=GEN_EXPERT_RETRAIN_INTERVAL,
            expert_retrain_epochs=GEN_EXPERT_RETRAIN_EPOCHS,
            expert_retrain_checkpoint_iters=GEN_EXPERT_RETRAIN_CHECKPOINT_ITERS,
            lr=LR,
            wd=WD,
            milestones=MILESTONES,
            iterations=GEN_ITERATIONS,
        ),
        flips_dir / "config.toml": SELECT_FLIPS_TEMPLATE.format(
            budgets=DEPLOY_BUDGETS,
            module_dir=module_dir,
            flips_dir=flips_dir,
            deploy_num_honests=DEPLOY_NUM_HONESTS,
            deploy_num_poisoned=DEPLOY_NUM_POISONED,
        ),
    }

    trigger_path = trigger_output_path(module_dir)
    for agg_method in DEPLOY_AGG_METHODS:
        for budget in DEPLOY_BUDGETS:
            train_user_dir = cell_dir / agg_method / f"train_user_{budget}"
            configs[train_user_dir / "config.toml"] = TRAIN_USER_TEMPLATE.format(
                flips_dir=flips_dir,
                train_user_dir=train_user_dir,
                model_flag=MODEL_FLAG,
                dataset=DATASET,
                source_label=SOURCE_LABEL,
                target_label=TARGET_LABEL,
                trigger_path=trigger_path,
                budget=budget,
                deploy_num_honests=DEPLOY_NUM_HONESTS,
                deploy_num_poisoned=DEPLOY_NUM_POISONED,
                agg_method=agg_method,
                lr=LR,
                wd=WD,
                milestones=MILESTONES,
            )

    paths = []
    for path, content in configs.items():
        assert "out/checkpoints" not in str(path), f"Refusing to write under out/checkpoints/: {path}"
        if dry_run:
            paths.append(path)
            continue
        write_config(path, content)
        validate_config(path)
        paths.append(path)

    return paths


def list_grid():
    """Enumerates the deployment sweep cells (agg_method x budget) as plain dicts, without
    writing any config -- the single source of truth for the grid's shape, so
    orchestrate_slurm/run_trigger_joint_old_objective_port_sweep_slurm.sh doesn't need to
    duplicate DEPLOY_AGG_METHODS/DEPLOY_BUDGETS by hand."""
    return [
        {"agg_method": agg_method, "budget": budget}
        for agg_method in DEPLOY_AGG_METHODS
        for budget in DEPLOY_BUDGETS
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--print-grid",
        action="store_true",
        help="print the deployment sweep cells as JSON and exit, without writing configs",
    )
    args = parser.parse_args()

    if args.print_grid:
        import json

        print(json.dumps(list_grid(), indent=2))
        raise SystemExit(0)

    paths = generate(dry_run=args.dry_run)

    if args.dry_run:
        print(f"\n[DRY RUN] {MODULE_NAME} old_objective_port: {len(paths)} config files would be written.")
        for p in paths:
            print(f"  {p}")
    else:
        print(f"\n{MODULE_NAME} old_objective_port: {len(paths)} config files written and schema-validated.")
