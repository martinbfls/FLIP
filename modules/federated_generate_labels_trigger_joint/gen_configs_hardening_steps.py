"""
gen_configs_hardening_steps.py -- config generator for the P0-P7 robustness-hardening
protocol's Etapes 1-4 (see the accompanying diagnostic writeup's "Protocole experimental").

Each step is ONE cell (single seed=0, BUDGETS=[500, 2000]) -- these are validation runs meant
to check one diagnostic at a time (cos_delta_to_init, mag_active_rate, delta_sign_flip_rate,
expert_asr vs expert_asr_frozen, poison_consistency), not a full campaign sweep. Reuses
gen_configs.py's own templates/helpers (write_config, validate_config, check_delta_min_feasible,
trigger_output_path) exactly like every sibling gen_configs_*.py generator in this module --
only the JOINT_TRIGGER_TEMPLATE is locally extended (same "locally-extended copy" convention
gen_configs_match_lambda_sweep.py already uses for lambda_match) to expose the new P3/P4/P5/P6
fields this campaign needs (metrics_log_path is set here for the FIRST time in this family --
none of the other single-cell generators turn it on by default, but the whole point of this
protocol is watching those diagnostics).

Deployed against TWO branches per step, same convention as _federated_branch.py's
federated_branch_configs (NOT reused directly -- that helper hardcodes agg_method="multikrum";
this file needs a configurable defense per step, see D0's "Tester immediatement contre
trimmed-mean" instruction for Etape 1 specifically):
  - single-user (cell_dir/train_user_{budget}): 1-poisoned/0-honest/mean, undefended --
    where cos_delta_to_init/mag_active_rate/delta_sign_flip_rate/expert_asr are read from
    metrics_log_path during/after GENERATION (no train_user needed for those, but the branch
    is still generated for comparability with the rest of this module's campaigns).
  - defended (cell_dir/{DEFENSE_TAG}/train_user_{budget}): 3-poisoned/7-honest, agg_method
    configurable per step (DEFENSE_AGG_METHOD below) -- what "tester contre trimmed-mean"
    actually measures.

Usage:
    python -m modules.federated_generate_labels_trigger_joint.gen_configs_hardening_steps \\
        --step 1 [--defense-agg trmean] [--dry-run]
"""
import argparse
from pathlib import Path

from modules.federated_generate_labels_trigger_joint.gen_configs import (
    ALPHA_CKPT,
    CHECKPOINT_ITERS,
    CHECKPOINT_SAMPLING,
    CLUSTER_ROOT,
    EPOCHS_EXPERT,
    GAMMA_STEALTH,
    GRADMATCH_EPS,
    GRADMATCH_METRIC,
    JOINT_TRIGGER_TEMPLATE,
    LAMBDA_BD,
    LAMBDA_DELTA,
    LAMBDA_GRADMATCH,
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

MODEL_FLAG = "r32p"
DATASET = "cifar"
SEED = 0
BUDGETS = [500, 2000]

# Single-user (attack-generation) branch axes -- fixed across all 4 steps, matching the rest
# of this module's campaigns (1-poisoned/0-honest/mean generation, deployed against a separate
# defended branch below).
NUM_POISONED = 1
NUM_HONESTS = 0
AGG_METHOD = "mean"

EXP_BASE = Path(
    "experiments/federated_experiments/threat_model_direct_trigger_joint_hardening_steps"
).resolve()
MODULE_NAME = "federated_generate_labels_trigger_joint_hardening_steps"

# Locally-extended copy of JOINT_TRIGGER_TEMPLATE (same splice convention as
# gen_configs_match_lambda_sweep.py's _JOINT_TRIGGER_TEMPLATE_MATCH) -- adds metrics_log_path
# (mandatory for this protocol's own diagnostics) plus every P3/P4/P5/P6 field, right after
# lambda_gradmatch/gradmatch_metric.
_JOINT_TRIGGER_TEMPLATE_HARDENING = JOINT_TRIGGER_TEMPLATE.replace(
    'gradmatch_metric = "{gradmatch_metric}"\n',
    'gradmatch_metric = "{gradmatch_metric}"\n'
    "lambda_margin = {lambda_margin}\n"
    "margin_min = {margin_min}\n"
    "lambda_consistency = {lambda_consistency}\n"
    "lambda_budget = {lambda_budget}\n"
    "z_budget = {z_budget}\n"
    'expert_retrain_agg_method = "{expert_retrain_agg_method}"\n',
).replace(
    "n_checkpoints_per_step = {n_checkpoints_per_step}\n",
    "n_checkpoints_per_step = {n_checkpoints_per_step}\n"
    'metrics_log_path = "{cell_dir}/logs/metrics.json"\n',
)
assert _JOINT_TRIGGER_TEMPLATE_HARDENING != JOINT_TRIGGER_TEMPLATE, (
    "JOINT_TRIGGER_TEMPLATE's lambda_gradmatch/n_checkpoints_per_step lines changed shape -- "
    "update the splice."
)

# --------------------------------------------------------------------------- #
# Per-step overrides (see the diagnostic writeup's "Protocole experimental" for the exact
# rationale of each). Every field not listed here is pinned to gen_configs.py's own current
# defaults (imported above) -- same "everything else pinned" convention as
# gen_configs_match_lambda_sweep.py.
# --------------------------------------------------------------------------- #
STEP_OVERRIDES = {
    1: dict(
        # P0 + P1 alone: epsilon=0.1, delta_min_frac=0.8, lambda_mag=1.0, lambda_align=0.0,
        # trigger_constraint=penalty, lambda_bd=1.0, lambda_delta=0.0, init=stripe,
        # alpha_ckpt=0.1, n_checkpoints_per_step=4.
        tag="step1_p0_p1",
        epsilon=0.1, delta_min_frac=0.8, lambda_mag=1.0, lambda_align=0.0,
        trigger_constraint="penalty", lambda_bd=1.0, lambda_delta=0.0, init="stripe",
        alpha_ckpt=0.1, n_checkpoints_per_step=4,
        expert_retrain_interval=0, expert_retrain_agg_method=AGG_METHOD,
        lambda_margin=0.0, lambda_consistency=0.0, lambda_budget=0.0,
        defense_agg="trmean",
    ),
    2: dict(
        # P2 (honest-client filtering -- code fix, no config knob) + P6 (live retraining under
        # the deployment's own agg_method, exposed but not yet functionally wired -- see P6's
        # own doc): expert_retrain_interval=50, expert_retrain_epochs=20, seed=train_expert's
        # own seed.
        tag="step2_p2_p6",
        epsilon=0.1, delta_min_frac=0.8, lambda_mag=1.0, lambda_align=0.0,
        trigger_constraint="penalty", lambda_bd=1.0, lambda_delta=0.0, init="stripe",
        alpha_ckpt=0.1, n_checkpoints_per_step=4,
        expert_retrain_interval=50, expert_retrain_epochs=20,
        lambda_margin=0.0, lambda_consistency=0.0, lambda_budget=0.0,
        defense_agg="multikrum",
    ),
    3: dict(
        # P5 (coordinate budget) + lambda_match, agg_method stays "mean" in the differentiable
        # path (see P5's own doc: L_budget is the smooth substitute for robustness, not
        # agg_method itself). z_budget here is a PLACEHOLDER (1.0) -- calibrate it first via
        # prelim/calibrate_z_budget.py against the SAME defense_agg before trusting this run.
        tag="step3_p5_match",
        epsilon=0.1, delta_min_frac=0.8, lambda_mag=1.0, lambda_align=0.0,
        trigger_constraint="penalty", lambda_bd=1.0, lambda_delta=0.0, init="stripe",
        alpha_ckpt=0.1, n_checkpoints_per_step=4,
        expert_retrain_interval=50, expert_retrain_epochs=20,
        lambda_margin=0.0, lambda_consistency=0.0, lambda_budget=1.0, z_budget=1.0,
        lambda_match=1.0,
        defense_agg="multikrum",
    ),
    4: dict(
        # P4 (poison consistency) + P3 (margin floor) -- only meaningful with num_poisoned>=2,
        # which the defended branch already provides (3v7); the single-user GENERATION branch
        # stays 1-poisoned/0-honest (gen_configs.py convention), so poison_consistency is
        # expected to read ~0.0 there (vacuous, see poison_consistency's own docstring) -- this
        # step's own diagnostic value comes from the DEFENDED branch's real deployment, not
        # from generation-time poison_consistency in this particular single-cell setup.
        tag="step4_p4_p3",
        epsilon=0.1, delta_min_frac=0.8, lambda_mag=1.0, lambda_align=0.0,
        trigger_constraint="penalty", lambda_bd=1.0, lambda_delta=0.0, init="stripe",
        alpha_ckpt=0.1, n_checkpoints_per_step=4,
        expert_retrain_interval=50, expert_retrain_epochs=20,
        lambda_margin=1.0, margin_min=2.0, lambda_consistency=1.0, lambda_budget=1.0,
        z_budget=1.0, lambda_match=1.0,
        defense_agg="multikrum",
    ),
}
for _step, _cfg in STEP_OVERRIDES.items():
    _cfg.setdefault("margin_min", 2.0)
    _cfg.setdefault("lambda_match", 0.0)
    _cfg.setdefault("expert_retrain_agg_method", AGG_METHOD)
    _cfg.setdefault("expert_retrain_epochs", EPOCHS_EXPERT)


def _defense_dir_tag(agg_method):
    return f"federated_3vs7_{agg_method}"


def generate_step(step, defense_agg=None, budgets=BUDGETS, seed=SEED, dry_run=False):
    if step not in STEP_OVERRIDES:
        raise ValueError(f"Unknown step {step} -- must be one of {sorted(STEP_OVERRIDES)}.")
    cfg = dict(STEP_OVERRIDES[step])
    tag = cfg.pop("tag")
    defense_agg = defense_agg or cfg.pop("defense_agg")
    cfg.pop("defense_agg", None)

    feasible, delta_min, max_reachable = check_delta_min_feasible(
        DATASET, cfg["epsilon"], cfg["delta_min_frac"],
    )
    if not feasible:
        reason = (
            f"delta_min_frac={cfg['delta_min_frac']} -> delta_min={delta_min:.4f} > "
            f"max_reachable={max_reachable:.4f} at epsilon={cfg['epsilon']} -- refusing."
        )
        print(f"REFUSED [{tag}]: {reason}")
        return [], reason

    lr = LEARNING_RATE.get(MODEL_FLAG, 0.1)
    wd = WEIGHT_DECAY.get(MODEL_FLAG, 2e-4)
    milestones = MILESTONE.get(MODEL_FLAG, [75, 125])
    rng_seed = draw_rng_seed(MODEL_FLAG, seed)

    cell_dir = EXP_BASE / tag / MODEL_FLAG / DATASET / f"seed{seed}"
    train_expert_dir = EXP_BASE / f"train_expert/{MODEL_FLAG}_1xs/seed{seed}"
    module_dir = cell_dir / "gen_labels_trigger_joint"
    flips_dir = cell_dir / "select_flips"

    configs = {
        train_expert_dir / "config.toml": TRAIN_EXPERT_TEMPLATE.format(
            cluster_root=CLUSTER_ROOT, model_flag=MODEL_FLAG, dataset=DATASET, seed=seed,
            rng_seed=rng_seed, source_label=SOURCE_LABEL, target_label=TARGET_LABEL,
            checkpoint_iters=CHECKPOINT_ITERS, epochs=EPOCHS_EXPERT, lr=lr, wd=wd,
            milestones=milestones,
            wandb_block_train_expert=wandb_block(
                "train_expert", f"train_expert/{MODEL_FLAG}/{DATASET}/seed{seed}",
                enabled=WANDB_ENABLED, project=WANDB_PROJECT, mode=WANDB_MODE,
                entity=WANDB_ENTITY, group=MODULE_NAME,
            ),
        ),
        module_dir / "config.toml": _JOINT_TRIGGER_TEMPLATE_HARDENING.format(
            cluster_root=CLUSTER_ROOT, model_flag=MODEL_FLAG, dataset=DATASET, seed=seed,
            rng_seed=rng_seed, cell_dir=module_dir, source_label=SOURCE_LABEL,
            target_label=TARGET_LABEL, epsilon=cfg["epsilon"], lr_delta=LR_DELTA,
            lambda_bd=cfg["lambda_bd"], lambda_delta=cfg["lambda_delta"],
            lambda_gradmatch=LAMBDA_GRADMATCH, gradmatch_eps=GRADMATCH_EPS,
            gradmatch_metric=GRADMATCH_METRIC,
            lambda_margin=cfg["lambda_margin"], margin_min=cfg["margin_min"],
            lambda_consistency=cfg["lambda_consistency"], lambda_budget=cfg["lambda_budget"],
            z_budget=cfg.get("z_budget", 1.0),
            expert_retrain_agg_method=cfg["expert_retrain_agg_method"],
            detach_param_dist="false",
            train_pct=TRAIN_PCT, num_honests=NUM_HONESTS, num_poisoned=NUM_POISONED,
            agg_method=AGG_METHOD, gamma_stealth=GAMMA_STEALTH,
            checkpoint_sampling=CHECKPOINT_SAMPLING, alpha_ckpt=cfg["alpha_ckpt"],
            n_iterations=N_ITERATIONS, trigger_constraint=cfg["trigger_constraint"],
            align_kappa=0.6, lambda_align=cfg["lambda_align"], lambda_mag=cfg["lambda_mag"],
            delta_min_frac=cfg["delta_min_frac"], init=cfg["init"],
            expert_retrain_interval=cfg["expert_retrain_interval"],
            expert_retrain_epochs=cfg["expert_retrain_epochs"],
            expert_retrain_checkpoint_iters=CHECKPOINT_ITERS,
            n_checkpoints_per_step=cfg["n_checkpoints_per_step"],
            lr=lr, wd=wd, milestones=milestones,
            wandb_block_module=wandb_block(
                "federated_generate_labels_trigger_joint", f"{MODULE_NAME}/{tag}/seed{seed}",
                enabled=WANDB_ENABLED, project=WANDB_PROJECT, mode=WANDB_MODE,
                entity=WANDB_ENTITY, group=MODULE_NAME,
            ),
        ),
        flips_dir / "config.toml": SELECT_FLIPS_TEMPLATE.format(
            budgets=budgets, module_dir=module_dir, flips_dir=flips_dir,
            num_honests=NUM_HONESTS, num_poisoned=NUM_POISONED,
        ),
    }
    trigger_path = trigger_output_path(
        module_dir, MODEL_FLAG, DATASET, NUM_POISONED, NUM_HONESTS, cfg["init"],
    )
    for budget in budgets:
        train_user_dir = cell_dir / f"train_user_{budget}"
        configs[train_user_dir / "config.toml"] = TRAIN_USER_TEMPLATE.format(
            flips_dir=flips_dir, train_user_dir=train_user_dir, model_flag=MODEL_FLAG,
            dataset=DATASET, source_label=SOURCE_LABEL, target_label=TARGET_LABEL,
            trigger_path=trigger_path, budget=budget, num_honests=NUM_HONESTS,
            num_poisoned=NUM_POISONED, agg_method=AGG_METHOD, lr=lr, wd=wd,
            milestones=milestones,
            wandb_block_train_user=wandb_block(
                "federated_train_user", f"train_user/{MODEL_FLAG}/{DATASET}/{tag}/{budget}/seed{seed}",
                enabled=WANDB_ENABLED, project=WANDB_PROJECT, mode=WANDB_MODE,
                entity=WANDB_ENTITY, group=MODULE_NAME,
            ),
        )

    # Defended branch (3-poisoned/7-honest, agg_method=defense_agg) -- generic, parameterized
    # equivalent of _federated_branch.py's federated_branch_configs (that helper hardcodes
    # agg_method="multikrum" for its own campaigns; this protocol needs trimmed-mean for
    # Etape 1 specifically, see the module docstring above).
    fed_dir = cell_dir / _defense_dir_tag(defense_agg)
    fed_flips_dir = fed_dir / "select_flips"
    configs[fed_flips_dir / "config.toml"] = SELECT_FLIPS_TEMPLATE.format(
        budgets=budgets, module_dir=module_dir, flips_dir=fed_flips_dir,
        num_honests=7, num_poisoned=3,
    )
    for budget in budgets:
        train_user_dir = fed_dir / f"train_user_{budget}"
        configs[train_user_dir / "config.toml"] = TRAIN_USER_TEMPLATE.format(
            flips_dir=fed_flips_dir, train_user_dir=train_user_dir, model_flag=MODEL_FLAG,
            dataset=DATASET, source_label=SOURCE_LABEL, target_label=TARGET_LABEL,
            trigger_path=trigger_path, budget=budget, num_honests=7, num_poisoned=3,
            agg_method=defense_agg, lr=lr, wd=wd, milestones=milestones,
            wandb_block_train_user=wandb_block(
                "federated_train_user",
                f"train_user/{MODEL_FLAG}/{DATASET}/{tag}/{_defense_dir_tag(defense_agg)}/{budget}/seed{seed}",
                enabled=WANDB_ENABLED, project=WANDB_PROJECT, mode=WANDB_MODE,
                entity=WANDB_ENTITY, group=MODULE_NAME,
            ),
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

    return paths, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, required=True, choices=sorted(STEP_OVERRIDES))
    parser.add_argument("--defense-agg", default=None, help="Override the step's own default.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths, refused = generate_step(args.step, defense_agg=args.defense_agg, dry_run=args.dry_run)
    if refused:
        print(f"REFUSED: {refused}")
    elif args.dry_run:
        print(f"[DRY RUN] step {args.step}: {len(paths)} config files would be written.")
        for p in paths:
            print(f"  {p}")
    else:
        print(f"step {args.step}: {len(paths)} config files written and schema-validated.")
