"""
Config generator for the federated_optimizing_trigger_policy campaign chain:
train_expert -> federated_optimizing_trigger_policy -> federated_policy_to_flips ->
federated_train_user.

Conventions follow modules/base_utils/gen_configs.py (see also the sibling generators under
federated_generate_labels_trigger/ and federated_generate_labels_trigger_joint/).

This module's "budget" axis is NOT a free choice like the other two modules' `budgets` list:
the realized flip count is DERIVED at run time from beta (this module's own LOCAL rate) via
gamma*n_train, not chosen directly (federated_policy_to_flips has no `budgets` parameter at
all -- see its schema). To keep the "provenance et budget des experts" nuisance factor aligned
across all three module generators (docs/policy_module_audit_report.md, Bloc B3), BETA_LOCAL_
VALUES below is computed FROM the same target global budgets (BUDGETS_TARGET) the other two
generators sweep directly, via beta_local = (budget/n_train) / gamma -- see A1's beta_local/
beta_global distinction.

A1/A2 corrections (docs/policy_module_audit_report.md) are exposed and enforced here exactly
as in the corrected module: beta is the LOCAL rate (commented per-cell with the derived
beta_global/gamma/s_beta/lambda_poison), a cell with beta_local > 1 is refused outright
(lem:beta-bar: infeasible, no allocation of a global budget under gamma exists), and s_beta > 1
is a warning (rem:saturated: lambda=beta is not theoretically justified there, still generated).
"""

import argparse
import os
from pathlib import Path

import toml

# --------------------------------------------------------------------------- #
# Sweep axes -- edit these for a real campaign. num_poisoned/num_honests, agg_method*, dataset,
# model, checkpoint provenance ("1xs") kept ALIGNED with the other two generators.
# --------------------------------------------------------------------------- #
NUM_POISONED = 3
NUM_HONESTS = 7
SEEDS = [1]
# Target GLOBAL flip budgets, aligned with the other two generators' BUDGETS -- converted to
# this module's own LOCAL beta below (beta_local = (budget/n_train) / gamma).
BUDGETS_TARGET = [1500]
AGG_METHODS = ["mean"]  # logged only -- (P^mean) has no agg_method of its own (mean
# aggregation is the (P^mean) formulation itself); kept as an
# axis purely so directory names/campaign size line up 1:1
# with the other two modules for comparison.
DATASETS = ["cifar"]
MODEL_FLAGS = ["r32p"]
SOURCE_LABEL = 9
TARGET_LABEL = 4

N_TRAIN = {"cifar": 50000, "cifar_100": 50000, "svhn": 73257}

LAMBDA_POISON = (
    "beta"  # "beta" (A1-corrected: resolves to beta_global) or a numeric override
)
ALPHA_CKPT = 0.01
NUM_CHCKPT = 4
CHECKPOINT_ITERS = 50
# train_expert's own bootstrap-checkpoint epochs -- a DIFFERENT quantity from POLICY_EPOCHS
# below (the per-outer-step expert-retraining epochs inside federated_optimizing_trigger_
# policy itself). Deliberately a separate constant: reusing one EPOCHS constant for both was
# the exact ambiguity behind the "expert trains 200 epochs instead of 20" investigation --
# see run_module.py's expert_epochs assertion at the top of each outer step.
EPOCHS_EXPERT = 20

# --------------------------------------------------------------------------- #
# REGULARIZATION_GRID -- trigger/policy regularization knobs, kept separate from the sweep
# axes above so a real campaign's regularization sweep is easy to find and edit in one place.
# `single_cell` mode (see generate_single_cell / --single-cell) fixes ALL of these to the
# defaults below and sweeps only the main axes (SEEDS/BUDGETS_TARGET).
# --------------------------------------------------------------------------- #
EPSILON = 0.1  # L_infinity bound on the trigger delta -- larger allows a stronger/more
# visible perturbation; too small can make the backdoor unreachable.
LR_DELTA = 1e-2  # Adam learning rate for the trigger optimization.
LR_POLICY = 1e-2  # Adam learning rate for the policy u optimization.
LAMBDA_BD = 1.0  # weight of the backdoor-efficacy loss (kappa in the P^mean formula) --
# higher pushes harder for backdoor success at the cost of the B2
# alignment term.
LAMBDA_B2 = 1.0  # EXPERIMENTAL (diagnostics task, Experiment D): weight on B2 itself,
# alongside LAMBDA_BD's weight on L_bd -- eq:P has no free coefficient
# here (implicitly 1); raised in EXPERIMENT_VARIANTS below to counteract
# a low grad_delta_ratio (delta's gradient dominated by L_bd, B2's own
# signal to delta comparatively weak). 1.0 = unweighted, the original
# behavior.
LAMBDA_DELTA = 0.0  # L2 norm penalty on delta ("lambda_trigger_l2") -- 0.0 (schema
# default) leaves the trigger magnitude unregularized beyond the
# epsilon clamp.
NORMALIZATION = (
    "rho"  # B2's denominator: "rho" (eq:rho, non-saturating) or "v" (legacy,
)
# saturates once v leaves the reachable set). See schema for the tradeoff.
DIAG_EVERY = 50  # frequency (in batches) of the QP diagnostic (B2_qp) against the
# co-descended policy -- lower gives a tighter Danskin-gap read, at
# the cost of solving the QP more often.
N_STEPS = 1  # number of outer (retrain expert + optimize trigger/policy) steps.
POLICY_EPOCHS = 20  # epochs of expert retraining PER outer step -- distinct from both
# EPOCHS_EXPERT (the bootstrap expert above) and federated_train_user's
# own epochs (unset in TRAIN_USER_TEMPLATE below, so it falls back to
# modules/base_utils/util.py's DEFAULT_SGD_EPOCHS=200).

# --------------------------------------------------------------------------- #
# Diagnostics (modules/federated_optimizing_trigger_policy/diagnostics.py) -- every generated
# cell writes a diagnostics.jsonl under its own policy dir (diag_path below) with the CHEAP
# diagnostics on by default (discretization gap, gradient balance, span projection, direction/
# amplitude decomposition, constraint activity -- no extra model forward/backward passes beyond
# what B2_qp already costs), so a run is diagnosable out of the box per prelim/SPEC.md's
# "observe before correcting" instruction. diag_actual_gradient (Diagnostic D) is the ONE
# expensive diagnostic (materializes flips + real forward/backward passes against
# class_samples_raw) and stays OFF by default -- flip DIAG_ACTUAL_GRADIENT to True (and set
# DIAG_ACTUAL_GRADIENT_EVERY>0 to throttle it) once the cheap diagnostics have narrowed down
# where B2 is failing.
#
# DIAG_QP_CONVERGENCE is ON here (unlike diag_actual_gradient): the first diagnosis round
# (B2=0.0415 vs B2_qp=0.0248, B2_qp vs B2_qp_relaxed barely differing) raised the question of
# whether the diagnostic QP solver itself (diag_qp_iters=50) had actually converged -- this
# re-solves the SAME (Q, c) at 50/200/1000 iterations (all warm-started from the SAME point,
# see project_gradient_descent_local's docstring) and logs qp_<a>_vs_<b>_relative_improvement
# for each consecutive pair, so that question is answered directly from diagnostics.jsonl.
# --------------------------------------------------------------------------- #
DIAG_QP_ITERS = 50
DIAG_QP_CONVERGENCE = True
DIAG_QP_CHECK_ITERS = [50, 200, 1000]
DIAG_POLICY_NNZ_THRESHOLD = 1e-8
DIAG_POLICY_TOPK = 10
DIAG_POLICY_FULL_VECTOR = False
DIAG_DISCRETIZATION = True
DIAG_GRADIENT_BALANCE = True
DIAG_ACTUAL_GRADIENT = False
DIAG_ACTUAL_GRADIENT_EVERY = 0
DIAG_CONSTRAINT_TOL = 1e-8
DIAG_SPAN_PROJECTION = True
DIAG_DIRECTION_SCALING = True
DIAG_ONESHOT_GAP = False  # Etape 0 audit ("switch to the exact QP solver" task): EXPENSIVE
                          # (K+1 extra QP solves per diagnostic batch) -- OFF by default for
                          # normal campaigns; see generate_oneshot_gap_audit below for a
                          # dedicated single-cell run with this turned on.

# --------------------------------------------------------------------------- #
# Controlled inner-solve experiment (modules/federated_optimizing_trigger_policy/inner_solve.py)
# -- these module-level defaults ("joint", the unchanged co-descent) are what
# generate_cell/generate_single_cell/generate_minimal_campaign/generate_all_configs use UNLESS
# overridden per-call. generate_inner_solve_comparison (below) is the dedicated entry point for
# actually comparing modes: it generates one sibling cell PER variant in INNER_SOLVE_VARIANTS,
# temporarily overriding these globals, and does not change what a normal campaign generates.
# --------------------------------------------------------------------------- #
POLICY_INNER_MODE = "joint"
POLICY_INNER_STEPS = 1
POLICY_INNER_ITERS = 200
POLICY_INNER_TOL = 1e-8
POLICY_INNER_MIN_ITERS = 10
POLICY_INNER_RIDGE = 1e-6

# Comparison configurations, extended after the follow-up diagnostics round (B2 >> B2_qp
# persists even with the inner solve; grad_delta_ratio << 1 most batches; span geometry is
# mostly favorable past batch 0): the ORIGINAL 5 (Experiments A/B/C -- baseline, two multi_step
# depths, two qp_pgd rigor levels) plus Experiment D (lambda_b2, counteracting a low
# grad_delta_ratio), Experiment E (two-timescale lr: u fast, delta slow), and the "decisive
# combo" the analysis converged on -- an oracle-quality u (qp_pgd at 1000 iters) SO delta's own
# gradient is evaluated against a near-exact conditional optimum, combined with D+E so delta
# cannot simply out-run it again. (variant_tag, override dict) pairs, applied on top of the
# POLICY_INNER_*/LAMBDA_B2/LR_DELTA/LR_POLICY module defaults above (only the listed keys are
# overridden per variant).
INNER_SOLVE_VARIANTS = [
    ("joint", {"policy_inner_mode": "joint"}),
    ("multi_step5", {"policy_inner_mode": "multi_step", "policy_inner_steps": 5}),
    ("multi_step20", {"policy_inner_mode": "multi_step", "policy_inner_steps": 20}),
    (
        "qp_pgd200",
        {
            "policy_inner_mode": "qp_pgd",
            "policy_inner_iters": 200,
            "policy_inner_tol": 1e-8,
        },
    ),
    (
        "qp_pgd1000",
        {
            "policy_inner_mode": "qp_pgd",
            "policy_inner_iters": 1000,
            "policy_inner_tol": 1e-10,
        },
    ),
    # Experiment D (isolated): joint co-descent, but B2 weighted up so delta's gradient is not
    # dominated by L_bd alone (see grad_delta_ratio in diagnostics.jsonl).
    ("lambda_b2_10", {"policy_inner_mode": "joint", "lambda_b2": 10.0}),
    ("lambda_b2_100", {"policy_inner_mode": "joint", "lambda_b2": 100.0}),
    # Experiment E (isolated): two-timescale learning rates -- u fast, delta slow -- still
    # "joint" co-descent (no inner solve), to isolate the lr effect from the inner-solve one.
    (
        "two_timescale",
        {
            "policy_inner_mode": "joint",
            "lr_delta": 1e-3,
            "lr_policy": 1e-2,
        },
    ),
    (
        "two_timescale_aggressive",
        {
            "policy_inner_mode": "joint",
            "lr_delta": 3e-4,
            "lr_policy": 3e-2,
        },
    ),
    # The decisive combo: near-exact u (qp_pgd, 1000 iters) + B2 upweighted (D) + delta slowed
    # down relative to u (E) -- if the attack STILL fails here, per the analysis, the bottleneck
    # is no longer u's optimization but the geometry of the reachable gradient set itself.
    (
        "decisive_combo",
        {
            "policy_inner_mode": "qp_pgd",
            "policy_inner_iters": 1000,
            "policy_inner_tol": 1e-10,
            "lambda_b2": 10.0,
            "lr_delta": 1e-3,
            "lr_policy": 1e-2,
        },
    ),
]


def _toml_bool(x):
    return "true" if x else "false"


LEARNING_RATE = {"r32p": 0.1, "r18": 0.1, "vgg": 0.01}
WEIGHT_DECAY = {"r32p": 2e-4, "r18": 2e-4, "vgg": 2e-4}
MILESTONE = {"r32p": [75, 125], "r18": [75, 125], "vgg": [125]}

CLUSTER_ROOT = "/shared/data1/Projects/DLWP/j1067582/martin/FLIP"

EXP_BASE = Path(
    "experiments/federated_experiments/threat_model_expert_policy"
).resolve()

MODULE_NAME = "federated_optimizing_trigger_policy"


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


def resolve_beta_gamma(budget_target, dataset, num_poisoned, num_honests):
    """
    Mirrors resolve_beta_and_lambda_poison + the A1 correction exactly (see
    federated_optimizing_trigger_policy/run_module.py): given a TARGET GLOBAL flip budget
    (aligned with the other two generators' BUDGETS), derive this module's own LOCAL beta,
    gamma, beta_global (== the target's implied global beta), and s_beta.
    """
    n_train = N_TRAIN[dataset]
    gamma = num_poisoned / (num_poisoned + num_honests)
    beta_theory_global = budget_target / n_train
    beta_local = beta_theory_global / gamma
    beta_global = (
        gamma * beta_local
    )  # == beta_theory_global, recomputed via the module's own
    # formula rather than reused directly, so this function
    # exercises the SAME algebra run_module.py does (A1/A2).
    return beta_local, gamma, beta_global, n_train


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

POLICY_TEMPLATE = """[federated_optimizing_trigger_policy]
# A1/A2 (docs/policy_module_audit_report.md): beta below is LOCAL (this cell's own
# corrupted-worker shard fraction), NOT the global def:budget beta.
#   gamma            = {gamma:.6f}   (num_poisoned={num_poisoned}, num_honests={num_honests})
#   beta_local       = {beta_local:.6f}
#   beta_global      = {beta_global:.6f}   (= gamma*beta_local -- what lambda_poison="beta" resolves to)
#   s_beta           = {s_beta:.6f}   ({s_beta_regime})
#   lambda_poison    = "{lambda_poison}"   (resolves to {lambda_poison_resolved:.6f} at run time)
dataset = "{dataset}"
model = "{model_flag}"
source_label = {source_label}
target_label = {target_label}
optim_kwargs = {{lr = {lr}, momentum = 0.9, nesterov = true, weight_decay = {wd}}}
scheduler_kwargs = {{milestones = {milestones}, gamma = 0.1}}
output_dir = "{cluster_root}/out/checkpoints/{model_flag}_policy_{cell_tag}/0/"
output_dir_trigger = "{cell_dir}/trigger"
output_dir_policy = "{cell_dir}/policy"
device = "cuda"
lambda_bd = {lambda_bd}
lambda_b2 = {lambda_b2}
lambda_penalty = 0.0
lambda_delta = {lambda_delta}
lambda_tv = 0.0
kappa = 0.0
epsilon = {epsilon}
lr_delta = {lr_delta}
lr_policy = {lr_policy}
n_steps = {n_steps}
epochs = {epochs}
checkpoint_backward = true
beta = {beta_local:.6f}
num_honests = {num_honests}
num_poisoned = {num_poisoned}
lambda_poison = "{lambda_poison}"
lambda_overflow = "clip"
alpha_ckpt = {alpha_ckpt}
num_chckpt = {num_chckpt}
expert_path = "{cluster_root}/out/checkpoints/{model_flag}_1xs/{{}}/model_{{}}_{{}}.pth"
normalization = "{normalization}"
diag_every = {diag_every}
metrics_log_path = "{cell_dir}/metrics.json"

# Diagnostics (see modules/federated_optimizing_trigger_policy/diagnostics.py's module
# docstring for how to read these together) -- diag_path below means diagnostics.jsonl IS
# written for this cell; see gen_configs.py's DIAG_* constants to change what gets logged.
diag_path = "{cell_dir}/diagnostics.jsonl"
diag_qp_iters = {diag_qp_iters}
diag_qp_convergence = {diag_qp_convergence}
diag_qp_check_iters = {diag_qp_check_iters}
diag_policy_nnz_threshold = {diag_policy_nnz_threshold}
diag_policy_topk = {diag_policy_topk}
diag_policy_full_vector = {diag_policy_full_vector}
diag_discretization = {diag_discretization}
diag_gradient_balance = {diag_gradient_balance}
diag_actual_gradient = {diag_actual_gradient}
diag_actual_gradient_every = {diag_actual_gradient_every}
diag_constraint_tol = {diag_constraint_tol}
diag_span_projection = {diag_span_projection}
diag_direction_scaling = {diag_direction_scaling}
diag_oneshot_gap = {diag_oneshot_gap}

# Controlled inner-solve experiment (modules/federated_optimizing_trigger_policy/inner_solve.py)
# -- "joint" (default) is the unchanged co-descent; see that module's docstring for the others.
policy_inner_mode = "{policy_inner_mode}"
policy_inner_steps = {policy_inner_steps}
policy_inner_iters = {policy_inner_iters}
policy_inner_tol = {policy_inner_tol}
policy_inner_min_iters = {policy_inner_min_iters}
policy_inner_ridge = {policy_inner_ridge}

[federated_optimizing_trigger_policy.expert_config]
experts = 1
min = 0
max = 20
trajectories = [50, 100, 150, 200]
"""

POLICY_TO_FLIPS_TEMPLATE = """[federated_policy_to_flips]
policy_path = "{cell_dir}/policy/policy_stripe_{model_flag}_{dataset}_{num_poisoned}vs{num_honests}.npz"
dataset = "{dataset}"
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
optim_kwargs = {{lr = {lr}, momentum = 0.9, nesterov = true, weight_decay = {wd}}}
schedule_kwargs = {{milestones = {milestones}, gamma = 0.1}}
"""


def cell_name(model_flag, dataset, seed, budget_target):
    return f"{model_flag}/{dataset}/{NUM_POISONED}vs{NUM_HONESTS}/budget{budget_target}/seed{seed}"


def generate_cell(
    model_flag,
    dataset,
    seed,
    budget_target,
    dry_run=False,
    overrides=None,
    cell_tag_suffix=None,
):
    """
    overrides: optional dict overriding any of policy_inner_mode/_steps/_iters/_tol/_min_iters/
    _ridge, lambda_b2, lr_delta, lr_policy for THIS cell only (on top of the module-level
    defaults below) -- used by generate_inner_solve_comparison to generate one sibling cell per
    experiment variant (Sections 12/Experiments A-E of the diagnostics task) without touching
    the defaults a normal campaign uses.
    cell_tag_suffix: optional string appended to this cell's directory/tag, so several variants
    of the SAME (model, dataset, seed, budget) don't overwrite each other's config.toml.
    """
    beta_local, gamma, beta_global, n_train = resolve_beta_gamma(
        budget_target,
        dataset,
        NUM_POISONED,
        NUM_HONESTS,
    )
    pi_min_approx = 1.0 / 10  # CIFAR-family: 10 balanced classes: min_y(pi_y) ~= 0.1
    s_beta = beta_global / (gamma * pi_min_approx)
    s_beta_regime = (
        "SATURATED (rem:saturated) -- lambda=beta not theoretically justified"
        if s_beta > 1
        else "unsaturated -- lambda=beta justified (prop:budget-match)"
    )
    lambda_poison_resolved = (
        beta_global if LAMBDA_POISON == "beta" else float(LAMBDA_POISON)
    )

    cell_config = dict(
        policy_inner_mode=POLICY_INNER_MODE,
        policy_inner_steps=POLICY_INNER_STEPS,
        policy_inner_iters=POLICY_INNER_ITERS,
        policy_inner_tol=POLICY_INNER_TOL,
        policy_inner_min_iters=POLICY_INNER_MIN_ITERS,
        policy_inner_ridge=POLICY_INNER_RIDGE,
        lambda_b2=LAMBDA_B2,
        lr_delta=LR_DELTA,
        lr_policy=LR_POLICY,
        diag_oneshot_gap=DIAG_ONESHOT_GAP,
        diag_qp_iters=DIAG_QP_ITERS,
        diag_every=DIAG_EVERY,
    )
    if overrides:
        cell_config.update(overrides)

    cell_tag = f"budget{budget_target}_seed{seed}"
    if cell_tag_suffix:
        cell_tag = f"{cell_tag}_{cell_tag_suffix}"

    if beta_local > 1:
        reason = (
            f"beta_local={beta_local:.6f} > 1 -- infeasible (lem:beta-bar: a single corrupted "
            f"worker cannot flip more than its own whole shard); budget_target={budget_target} "
            f"implies beta_global={beta_global:.6f} > gamma={gamma:.6f}, no allocation exists. "
            "Lower budget_target, raise num_poisoned (increases gamma), or lower num_honests."
        )
        print(
            f"REFUSED [{model_flag}/{dataset}/budget{budget_target}/seed{seed}]: {reason}"
        )
        return [], reason

    if s_beta > 1:
        print(
            f"WARNING [{model_flag}/{dataset}/budget{budget_target}/seed{seed}]: "
            f"s_beta={s_beta:.4f} > 1 -- {s_beta_regime}."
        )

    lr = LEARNING_RATE.get(model_flag, 0.1)
    wd = WEIGHT_DECAY.get(model_flag, 2e-4)
    milestones = MILESTONE.get(model_flag, [75, 125])

    cell_dir = EXP_BASE / cell_name(model_flag, dataset, seed, budget_target)
    if cell_tag_suffix:
        cell_dir = cell_dir / cell_tag_suffix
    train_expert_dir = EXP_BASE / f"train_expert/{model_flag}_1xs"
    policy_dir = cell_dir / "policy_opt"
    flips_dir = cell_dir / "policy_to_flips"

    configs = {
        train_expert_dir / "config.toml": TRAIN_EXPERT_TEMPLATE.format(
            cluster_root=CLUSTER_ROOT,
            model_flag=model_flag,
            dataset=dataset,
            source_label=SOURCE_LABEL,
            target_label=TARGET_LABEL,
            checkpoint_iters=CHECKPOINT_ITERS,
            epochs=EPOCHS_EXPERT,
            lr=lr,
            wd=wd,
            milestones=milestones,
        ),
        policy_dir / "config.toml": POLICY_TEMPLATE.format(
            gamma=gamma,
            num_poisoned=NUM_POISONED,
            num_honests=NUM_HONESTS,
            beta_local=beta_local,
            beta_global=beta_global,
            s_beta=s_beta,
            s_beta_regime=s_beta_regime,
            lambda_poison=LAMBDA_POISON,
            lambda_poison_resolved=lambda_poison_resolved,
            dataset=dataset,
            model_flag=model_flag,
            source_label=SOURCE_LABEL,
            target_label=TARGET_LABEL,
            lr=lr,
            wd=wd,
            milestones=milestones,
            cluster_root=CLUSTER_ROOT,
            cell_tag=cell_tag,
            cell_dir=policy_dir,
            lambda_bd=LAMBDA_BD,
            lambda_b2=cell_config["lambda_b2"],
            lambda_delta=LAMBDA_DELTA,
            epsilon=EPSILON,
            lr_delta=cell_config["lr_delta"],
            lr_policy=cell_config["lr_policy"],
            n_steps=N_STEPS,
            epochs=POLICY_EPOCHS,
            alpha_ckpt=ALPHA_CKPT,
            num_chckpt=NUM_CHCKPT,
            normalization=NORMALIZATION,
            diag_every=cell_config["diag_every"],
            diag_qp_iters=cell_config["diag_qp_iters"],
            diag_qp_convergence=_toml_bool(DIAG_QP_CONVERGENCE),
            diag_qp_check_iters=DIAG_QP_CHECK_ITERS,
            diag_policy_nnz_threshold=DIAG_POLICY_NNZ_THRESHOLD,
            diag_policy_topk=DIAG_POLICY_TOPK,
            diag_policy_full_vector=_toml_bool(DIAG_POLICY_FULL_VECTOR),
            diag_discretization=_toml_bool(DIAG_DISCRETIZATION),
            diag_gradient_balance=_toml_bool(DIAG_GRADIENT_BALANCE),
            diag_actual_gradient=_toml_bool(DIAG_ACTUAL_GRADIENT),
            diag_actual_gradient_every=DIAG_ACTUAL_GRADIENT_EVERY,
            diag_constraint_tol=DIAG_CONSTRAINT_TOL,
            diag_span_projection=_toml_bool(DIAG_SPAN_PROJECTION),
            diag_direction_scaling=_toml_bool(DIAG_DIRECTION_SCALING),
            diag_oneshot_gap=_toml_bool(cell_config["diag_oneshot_gap"]),
            policy_inner_mode=cell_config["policy_inner_mode"],
            policy_inner_steps=cell_config["policy_inner_steps"],
            policy_inner_iters=cell_config["policy_inner_iters"],
            policy_inner_tol=cell_config["policy_inner_tol"],
            policy_inner_min_iters=cell_config["policy_inner_min_iters"],
            policy_inner_ridge=cell_config["policy_inner_ridge"],
        ),
        flips_dir / "config.toml": POLICY_TO_FLIPS_TEMPLATE.format(
            cell_dir=policy_dir,
            model_flag=model_flag,
            dataset=dataset,
            num_poisoned=NUM_POISONED,
            num_honests=NUM_HONESTS,
            flips_dir=flips_dir,
        ),
    }

    # federated_train_user's `budget` is DERIVED (not a free config choice for this module --
    # see the module docstring above): predicted via the SAME formula
    # resolve_beta_and_lambda_poison/materialize_policy_flips's cross-check use
    # (round(beta_local*num_poisoned*n_train/n_w)), matching what federated_policy_to_flips
    # names its own output files with when the policy saturates its budget (the observed
    # regime -- see that module's own cross-check assertion).
    n_w = NUM_POISONED + NUM_HONESTS
    predicted_budget = round(beta_local * NUM_POISONED * n_train / n_w)
    train_user_dir = cell_dir / f"train_user_{predicted_budget}"
    configs[train_user_dir / "config.toml"] = TRAIN_USER_TEMPLATE.format(
        flips_dir=flips_dir,
        train_user_dir=train_user_dir,
        model_flag=model_flag,
        dataset=dataset,
        source_label=SOURCE_LABEL,
        target_label=TARGET_LABEL,
        budget=predicted_budget,
        num_honests=NUM_HONESTS,
        num_poisoned=NUM_POISONED,
        lr=lr,
        wd=wd,
        milestones=milestones,
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
    """Exactly one cell (first model/dataset/seed, first target budget), regularization fixed
    at REGULARIZATION_GRID's defaults -- the minimal preliminary campaign used to sanity check
    the chain before spending a real sweep's compute."""
    paths, reason = generate_cell(
        MODEL_FLAGS[0],
        DATASETS[0],
        SEEDS[0],
        BUDGETS_TARGET[0],
        dry_run=dry_run,
    )
    refused = (
        [(MODEL_FLAGS[0], DATASETS[0], BUDGETS_TARGET[0], SEEDS[0], reason)]
        if reason
        else []
    )
    return paths, refused


def generate_minimal_campaign(dry_run=True):
    """B3: 3 seeds x 3 budgets (as target global budgets, converted to beta_local) x [agg_method
    is logged-only for this module, see AGG_METHODS' comment -- not a distinct axis here]."""
    all_paths, refused = [], []
    for model_flag in MODEL_FLAGS[:1]:
        for dataset in DATASETS[:1]:
            for budget_target in BUDGETS_TARGET[:3]:
                for seed in SEEDS[:3]:
                    paths, reason = generate_cell(
                        model_flag,
                        dataset,
                        seed,
                        budget_target,
                        dry_run=dry_run,
                    )
                    all_paths += paths
                    if reason:
                        refused.append(
                            (model_flag, dataset, budget_target, seed, reason)
                        )
    return all_paths, refused


def generate_all_configs(dry_run=False):
    all_paths, refused = [], []
    for model_flag in MODEL_FLAGS:
        for dataset in DATASETS:
            for budget_target in BUDGETS_TARGET:
                for seed in SEEDS:
                    paths, reason = generate_cell(
                        model_flag,
                        dataset,
                        seed,
                        budget_target,
                        dry_run=dry_run,
                    )
                    all_paths += paths
                    if reason:
                        refused.append(
                            (model_flag, dataset, budget_target, seed, reason)
                        )
    return all_paths, refused


def generate_oneshot_gap_audit(dry_run=True):
    """
    Etape 0 of the "switch to the exact QP solver" task: a single cell (first model/dataset/
    seed/budget, same as generate_single_cell) with diag_oneshot_gap turned ON -- answers,
    directly from the resulting diagnostics.jsonl, whether:
      (1) the existing B2_qp is per-checkpoint or already coupled (it is per-checkpoint --
          established by code inspection, not this run, see the written report);
      (2) how much the independent per-checkpoint QP optima u*_k actually agree
          (u_star_pairwise_cosine_mean/min/max);
      (3) the one-shot gap J(ubar) - mean_k(B2_qp,k) (oneshot_gap_absolute/relative).
    diag_qp_iters is raised (500, vs. the campaign default 50) so the per-checkpoint AND coupled
    solves are both closer to convergence -- this audit's whole point is to measure a genuine
    gap, not solver noise. diag_every is lowered (5, vs. the campaign default 50) so a short
    diagnostic run (see N_STEPS) still produces several audited batches.
    """
    paths, reason = generate_cell(
        MODEL_FLAGS[0], DATASETS[0], SEEDS[0], BUDGETS_TARGET[0], dry_run=dry_run,
        overrides={"diag_oneshot_gap": True, "diag_qp_iters": 500, "diag_every": 5},
        cell_tag_suffix="oneshot_gap_audit",
    )
    refused = [(MODEL_FLAGS[0], DATASETS[0], BUDGETS_TARGET[0], SEEDS[0], reason)] if reason else []
    return paths, refused


def generate_inner_solve_comparison(dry_run=True):
    """
    Generates ONE cell (first model/dataset/seed/budget, same as generate_single_cell) PER
    variant in INNER_SOLVE_VARIANTS -- baseline "joint", multi_step at 5/20 inner steps, qp_pgd
    at 200/1000 iterations (Experiments A/B/C), lambda_b2 at 10/100 (Experiment D),
    two-timescale lr at two aggressiveness levels (Experiment E), and the "decisive combo"
    (qp_pgd 1000 + lambda_b2=10 + slowed-down delta) -- each in its own inner_<tag>
    subdirectory (via cell_tag_suffix) so they can be run and compared side by side without
    overwriting each other. Does NOT touch the module-level POLICY_INNER_*/LAMBDA_B2/LR_DELTA/
    LR_POLICY defaults (those stay at the baseline) -- each variant's overrides are passed in
    per-call via `overrides`, exactly as generate_cell's own docstring describes.
    """
    all_paths, refused = [], []
    for tag, overrides in INNER_SOLVE_VARIANTS:
        paths, reason = generate_cell(
            MODEL_FLAGS[0],
            DATASETS[0],
            SEEDS[0],
            BUDGETS_TARGET[0],
            dry_run=dry_run,
            overrides=overrides,
            cell_tag_suffix=f"inner_{tag}",
        )
        all_paths += paths
        if reason:
            refused.append(
                (MODEL_FLAGS[0], DATASETS[0], BUDGETS_TARGET[0], SEEDS[0], reason)
            )
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
    parser.add_argument(
        "--inner-solve-comparison",
        action="store_true",
        help="one cell per experiment variant (see INNER_SOLVE_VARIANTS: joint/multi_step5/20/"
        "qp_pgd200/1000/lambda_b2_10/100/two_timescale[_aggressive]/decisive_combo), same "
        "(model, dataset, seed, budget) as --single-cell, in sibling inner_<tag> "
        "subdirectories -- for comparing the controlled inner-solve experiment "
        "(modules/federated_optimizing_trigger_policy/inner_solve.py) and the lambda_b2/"
        "two-timescale-lr follow-up experiments.",
    )
    parser.add_argument(
        "--oneshot-gap-audit",
        action="store_true",
        help="Etape 0 of the 'switch to the exact QP solver' task: one cell with "
        "diag_oneshot_gap=true (diag_qp_iters=500, diag_every=5) -- logs, per diagnostic "
        "batch, the pairwise cosine similarity between independent per-checkpoint QP optima "
        "and the one-shot gap between the coupled ubar and the mean per-checkpoint optimum. "
        "See generate_oneshot_gap_audit's docstring.",
    )
    args = parser.parse_args()
    assert sum([
        args.minimal, args.single_cell, args.inner_solve_comparison, args.oneshot_gap_audit,
    ]) <= 1, (
        "pass at most one of --minimal/--single-cell/--inner-solve-comparison/"
        "--oneshot-gap-audit"
    )

    if args.single_cell:
        gen_fn = generate_single_cell
    elif args.minimal:
        gen_fn = generate_minimal_campaign
    elif args.inner_solve_comparison:
        gen_fn = generate_inner_solve_comparison
    elif args.oneshot_gap_audit:
        gen_fn = generate_oneshot_gap_audit
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
        print(f"\n{len(refused)} cell(s) REFUSED (beta_local > 1, lem:beta-bar):")
        for model_flag, dataset, budget_target, seed, reason in refused:
            print(
                f"  [{model_flag}/{dataset}/budget{budget_target}/seed{seed}] {reason}"
            )
