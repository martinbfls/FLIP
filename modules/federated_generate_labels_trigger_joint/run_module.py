"""
Threat model "direct-joint" (P^direct, REAL coupling): jointly optimizes continuous poisoned
logit labels (labels_syn) AND a backdoor trigger (delta) via federated trajectory matching,
extending federated_generate_labels.py -- exactly like federated_generate_labels_trigger, but
with delta's gradient flowing through BOTH the backdoor-efficacy term L_bd AND the MTT
trajectory-alignment term (param_loss), instead of through L_bd alone. See this module's
`run()` docstring for the full derivation and for why the three modules
(federated_generate_labels, federated_generate_labels_trigger, this one) must all keep working
unmodified/side by side.
"""

from pathlib import Path
import random
import sys
import json
import warnings
import torch
import numpy as np

from modules.base_utils.datasets import get_matching_datasets, pick_poisoner, get_n_classes
from modules.base_utils.util import extract_toml, get_module_device, get_mtt_attack_info, \
                                    load_model, either_dataloader_dataset_to_both, make_pbar, \
                                    needs_big_ims, slurmify_path, clf_loss, softmax, \
                                    total_mse_distance, get_train_info, mini_train
from modules.base_utils.experiment_tracker import ExperimentTracker
from modules.federated_generate_labels.utils import coalesce_attack_config, \
                                                     extract_labels, extract_experts, sgd_step, agg
from modules.train_expert.utils import checkpoint_callback
from modules.federated_optimizing_trigger.utils import (
    get_mu, init_delta, raw_to_trigger_preprocess, get_raw_clean_dataset,
    trigger_penalty_hinge, tv_loss,
)
from modules.federated_generate_labels_trigger_joint.utils import (
    TriggerMTTDataset, extract_experts_biased, build_expert_pool,
    directional_floor_penalty, magnitude_floor_penalty, project_trigger_constraints,
    cosine_to, grad_mismatch_penalty, grad_cosine_penalty,
)


def run(experiment_name, module_name, **kwargs):
    """
    Jointly optimizes labels_syn (ell-tilde) and delta -- problem (P^direct), with delta
    REALLY coupled to the MTT alignment term, not just to the isolated backdoor-efficacy term:

        min_ell-tilde,delta  E_t,{(x_i,y_i)}[ L_lambda(ell-tilde, delta, theta_t, {(x_i,y_i)}) ]
        s.t. theta_T in argmin_theta E_(x^adv,y^adv)~D_poisoned(delta)[ L(f_theta(x^adv), y^adv) ]

    where L_lambda = L (the MTT trajectory-alignment loss between U_adv and U_clean, unchanged
    in FORM from federated_generate_labels.py) + lam*||.||_1 (unchanged regularizer) +
    lambda_bd*L_bd (backdoor efficacy) + optional trigger regularizers -- same terms as
    `federated_generate_labels_trigger`. The difference from that module is entirely in HOW
    delta receives gradient through L (see "E2 implementation" below); everything else
    (dataset construction, labels_syn, the regularizers, the output format) is identical.

    Three modules, one family, all must keep running unmodified/side by side:
      - federated_generate_labels          : published baseline. delta is fixed/baked in.
      - federated_generate_labels_trigger  : delta jointly OPTIMIZED but only INDIRECTLY
                                              coupled -- x_t_adv is detached before entering
                                              the MTT branch, so delta's only gradient path is
                                              the isolated L_bd term (corrected docstring now
                                              says this explicitly -- see that module).
      - federated_generate_labels_trigger_joint (THIS module) : delta REALLY coupled -- x_t_adv
                                              is NOT detached, so delta receives gradient
                                              through param_loss too. This is the contribution.
                                              Also fixes federated-aggregation of the expert
                                              gradient (see E2 point 2a below) -- applied here
                                              AND in federated_generate_labels_trigger, but NOT
                                              in the published baseline (out of scope, see
                                              docs/threat_models_audit.md).

    E2 implementation (real coupling):

      1. `x_t_adv[is_poisoned_dev] = x_trig` -- NO `.detach()` (was: `x_trig.detach()` in the
         indirect module). x_t_adv now carries a live dependency on delta into loss_e.

      2. The expert step is made differentiable, mirroring what the student side ALREADY does
         (`grads_s = torch.autograd.grad(loss_s, student_params, create_graph=True)`):

             grads_e = torch.autograd.grad(loss_e, expert_params, create_graph=True,
                                            retain_graph=True)

         instead of `loss_e.backward()`. `expert_params[i].grad` is left UNSET here (never
         assigned anywhere in this module any more -- see point 2b): `expert_params[i]` is
         itself a leaf of `grads_e`'s own create_graph=True graph, so any early `.grad`
         assignment would be at risk of PyTorch's AccumulateGrad mutating it in place the
         moment `grand_loss.backward()` runs later this batch (see
         prelim/tests/test_joint_accumgrad_hazard.py) -- simply never assigning `.grad` at all
         sidesteps the hazard entirely rather than carefully sequencing around it.

      2a. FEDERATED EXPERT AGGREGATION (fix, see docs/threat_models_audit.md): every client's
         expert-side gradient is now collected into `expert_grad_buf` -- honest clients append
         their (non-differentiable) `p.grad.detach().clone()` from the plain `loss.backward()`
         call already used for the student side's honest contribution; poisoned clients append
         their (DIFFERENTIABLE, create_graph=True) `grads_e[i]` directly, undetached. After the
         cid loop, `agg_expert_grads = agg(expert_params, expert_grad_buf, agg_method,
         f=num_poisoned)` aggregates across ALL clients -- honest + poisoned -- exactly
         mirroring `agg_student_grads = agg(student_params, student_grad_buf, ...)` two lines
         below. `sgd_step` (the differentiable-SGD helper shared with the student side,
         `federated_generate_labels.utils.sgd_step`) is then applied to
         (expert_start[i], agg_expert_grads[i]) to get `expert_next_param`, a DIFFERENTIABLE
         function of delta (through every poisoned client's contribution to the aggregate, not
         just one) that serves as `param_loss`/`param_dist`'s comparison target.

         This REPLACES the previous "last-poisoned-client-wins" behavior: `expert_model.
         zero_grad()` was called before EVERY client's expert forward/backward, so only the
         LAST-processed client's raw `.grad` (always a poisoned client's, since poisoned
         clients are processed after honest ones) ever fed the expert-side update -- a single-
         client, non-federated target, while the student side was already properly aggregated.
         With num_honests=num_poisoned=1 (a common sweep configuration) this made the matching
         objective structurally close to trivial regardless of any regularizer: the expert
         "trajectory" being matched was just one client's local gradient, not a federated
         aggregate, undermining exactly what the MTT matching loss is supposed to measure. This
         was identified as the most likely primary driver of the expert_asr collapse observed
         in the P3 anti-collapse sweep, ahead of the delta_min-unreachability and
         regularization-calibration explanations investigated first -- see
         docs/threat_models_audit.md.

         Like `agg(student_params, ...)`, `agg(expert_params, ...)` sets `expert_params[i].grad`
         as a side effect. This is safe for the SAME reason it has always been safe for
         `student_params` in this module (never audited as a separate case before, but the
         identical pattern): nothing downstream ever reads `expert_params[i].grad` again this
         batch (there is no real `optimizer_expert.step()` any more -- see point 2b) or in any
         later batch (expert_model.load_state_dict(checkpoint) fully overwrites expert_model at
         the top of every batch, from a checkpoint reloaded fresh off disk -- see point 2b), so
         any AccumulateGrad in-place mutation of that tensor during `grand_loss.backward()` is
         inert. prelim/tests/test_joint_accumgrad_hazard.py is kept (see its header) as a
         general pin of the underlying PyTorch behavior, but no longer protects any live code
         path in this module.

      2b. NO REAL EXPERT OPTIMIZER STEP (removed): `optimizer_expert.step()` and the
         `optimizer_expert.load_state_dict(...)` call that fed it are both GONE. They served no
         purpose beyond producing the point-4 non-regression check's ground truth (also
         removed, see point 4): `expert_model` is reloaded fresh from disk
         (`expert_model.load_state_dict(checkpoint)`) at the START of every batch already,
         so nothing beyond the current batch's own `expert_next_param` computation (built from
         `expert_start`, a pre-batch clone, and never from `expert_params` post-step) ever
         depended on `optimizer_expert.step()` having run. `optimizer_expert` itself is still
         constructed (via `get_mtt_attack_info`, a shared helper also used -- unmodified -- by
         federated_generate_labels and federated_generate_labels_trigger, both of which DO take
         a real expert step) but is otherwise unused in this module; not worth special-casing
         `get_mtt_attack_info` for.

      3. The isolated L_bd term (`torch.autograd.grad(lambda_bd*L_bd_cid, [delta], ...)`,
         manually accumulated into `delta.grad`) is UNCHANGED and stays -- it is an ADDITIONAL
         contribution, not a duplicate: `optimizer_delta.zero_grad()` runs once per BATCH
         (before the cid loop), so `delta.grad` accumulates the L_bd contributions from every
         poisoned client THEN gets `grand_loss.backward()`'s contribution added on top
         (`.backward()`/autograd always ACCUMULATES into `.grad`, never overwrites, absent an
         intervening `zero_grad()` -- verified: none occurs between the two).

      4. Non-regression check (point 4, REMOVED): the old first-batch numeric assertion
         comparing `expert_next` against `optimizer_expert.step()`'s real result no longer
         applies -- there is no real expert step to compare against any more (point 2b), and
         `sgd_step`'s single-step correctness (the actual invariant that check was protecting)
         is now covered by a proper, standalone, synthetic unit test:
         prelim/tests/test_sgd_step.py (covers plain SGD, momentum, weight_decay, nesterov,
         and a pre-loaded non-zero momentum buffer, all against torch.optim.SGD exactly). That
         test also documents -- as a pin, not a fix -- a separate, currently-dormant bug in
         `sgd_step` itself (momentum_buffer not persisted across successive calls sharing the
         same opt_state dict); harmless here since every call site reloads a fresh opt_state
         per batch and calls sgd_step exactly once per param.

    Instrumentation (mandatory from the first run, anti-collapse): per batch, tracks (see
    `metrics_log_path`) the expert's OWN attack success rate (not just the eventual victim's),
    ||delta||_inf and ||delta||_2, and the matching term (param_loss/param_dist) and L_bd term
    SEPARATELY -- never only their sum. The risk this guards against: delta could minimize the
    alignment term by making the target trivially reachable, i.e. by WEAKENING the backdoor
    (expert ASR collapses while the matching term keeps improving) -- exactly the failure mode
    joint optimization introduces that the indirect module cannot exhibit. Sweep lambda_bd over
    >= 3 values and watch expert_asr alongside param_loss/param_dist to check for this.

    H1 diagnostic (frozen-vs-current expert ASR, Step 3 fix applied): `expert_model` is
    reloaded from disk fresh EVERY batch (`checkpoint = torch.load(expert_starts[it])`) and is
    never retrained against the current delta within this module -- unlike
    federated_optimizing_trigger_policy, which retrains its own `model` every outer step at
    the CURRENT trigger. So a collapsing `expert_asr` may not indicate delta is failing to
    induce a backdoor; it may indicate delta has drifted away from whatever fixed trigger the
    loaded checkpoints' own backdoor was baked against (their own `train_expert` run's
    `poisoner`), a mismatch this module cannot close by construction. `expert_asr_frozen`
    (logged alongside `expert_asr`, same FIXED 256-example source-class evaluation set --
    `asr_eval_raw`, built once before the training loop -- same checkpoint, but triggered with
    the FROZEN `delta_init` instead of the current delta) and `cos_delta_to_init`/
    `delta_drift_l2` (how far delta has moved from `delta_init`) are the diagnostic: if
    `expert_asr_frozen` stays high while `expert_asr` (current) collapses as delta drifts, the
    checkpoints' backdoor is tied to the original trigger, not being actively unlearned -- a
    design difference from the policy module's live-retraining architecture, not a
    regularization-calibration defect.

    `expert_retrain_interval` (2026-08-31, closes the gap above): if set > 0, every that many
    outer iterations a FRESH expert is retrained from scratch against the CURRENT trigger (via
    `pick_poisoner("optimized", ..., delta=delta.detach())` + `get_matching_datasets` +
    `mini_train`, mirroring train_expert's own training call exactly, for
    `expert_retrain_epochs` epochs), and its checkpoints (written under a PRIVATE
    `<output_dir>/../expert_retrain/round<n>/` directory, never overwriting the shared
    train_expert checkpoints this run started from) replace the expert trajectory
    (`expert_starts`/`expert_opt_starts`, and `expert_pool` if pooling is on) for the remaining
    iterations. Default 0 leaves the frozen-trajectory behavior above completely unchanged.
    With retraining on, `expert_asr` should stop drifting away from `expert_asr_frozen` purely
    because of trajectory staleness -- a persistent gap then points at delta/L_bd itself,
    not this design difference.

    `seed` (2026-09-01, isolates the trigger as retraining's only source of difference): if
    set, every retraining round reseeds Python's random/numpy/torch RNGs to this SAME value
    BEFORE building that round's poisoned dataset and initializing the fresh model -- pass the
    ORIGINAL train_expert step's own `seed` (schemas/train_expert.toml) here so a retrained
    expert's weight init and data order replicate the original expert's exactly. Without this,
    each retraining round's fresh model rode on whatever the ambient torch RNG state happened
    to be at that point in the run -- confounding "did retraining change the expert because of
    the poisoned trigger" with "did it change merely because of a different random init/data
    order," which this field removes as a factor. No effect when unset or when
    expert_retrain_interval=0 (both leave the prior, unseeded behavior unchanged).

    Averaging the loss over multiple checkpoints (2026-09-02, `n_checkpoints_per_step`):
    conditioning every outer step's gradient on a SINGLE expert checkpoint (even with P3's
    pool_size random draw, still one draw per step) makes the resulting labels_syn/delta fit to
    whatever quirks that one checkpoint happens to have. If set > 1, every batch-step instead
    draws that many DISTINCT checkpoints from the pool (requires pool_size >=
    n_checkpoints_per_step -- see the ValueError this raises otherwise) and computes THIS step's
    grand_loss as the mean, over those n_checkpoints_per_step checkpoints, of each one's own
    otherwise-unchanged matching term (param_loss/param_dist) and gradient-mismatch penalty --
    reg_term and the trigger/anti-collapse regularizers do NOT depend on the checkpoint, so they
    are still computed/added exactly once, not per checkpoint. The isolated L_bd manual
    delta.grad accumulation (point 3 above) is explicitly scaled by 1/n_checkpoints_per_step per
    checkpoint's poisoned clients, to stay consistent with the mean the rest of grand_loss
    already takes. A SINGLE combined `grand_loss.backward()` call still runs once per step.
    expert_asr/expert_asr_frozen/L_bd_mean/matching_term (both history and tracker.log) are
    likewise averaged over the n_checkpoints_per_step checkpoints; `matching_term_std`/
    `L_bd_mean_std`/`expert_asr_std` (history only) additionally record the SPREAD across them,
    a diagnostic for whether the checkpoints disagree a lot on a given step. Default 1 leaves
    the single-checkpoint-per-step behavior completely, bit-for-bit unchanged (mean of one
    element).

    IMPLEMENTATION NOTE (why n_checkpoints_per_step DISTINCT, PERSISTENT model instances, not
    one model reloaded n_checkpoints_per_step times): `expert_models`/`student_models` below are
    lists of n_checkpoints_per_step separate nn.Module instances, each `load_state_dict`-ed from
    its own checkpoint ONCE per batch-step. Reusing ONE model object and reloading its
    state_dict between checkpoints WITHIN the same step would mutate leaf parameter tensors
    still referenced by an EARLIER checkpoint's not-yet-backwarded create_graph=True graph --
    PyTorch's per-tensor version counter detects this and raises "one of the variables needed
    for gradient computation has been modified by an inplace operation" at the combined
    backward() -- a hard failure, not a silent bug, but one that only manifests with > 1
    checkpoint per step. Verified on a tiny synthetic model in
    prelim/tests/test_multi_checkpoint_step.py (both the hazard and the fix). Memory scales
    roughly linearly with n_checkpoints_per_step (n_checkpoints_per_step DISTINCT sets of model
    parameters, each with its own live create_graph=True computation graph, all held
    simultaneously until the single combined backward() completes) -- size pool_size/GPU memory
    accordingly.

    Two Step-3 corrections relative to the original H1 instrumentation (see
    docs/threat_models_audit.md for the original run's refuted-but-confounded result): (1)
    `delta_init` is now captured INSIDE the training loop, the first time execution reaches
    delta's own post-step clamp_/projection, instead of from the raw pre-clamp init (strength=
    6.0, never guaranteed to respect epsilon) -- the original capture point made
    `cos_delta_to_init`/`delta_drift_l2` meaningless at step 0 (cos != 1.0, drift_l2 exceeding
    the epsilon ball's own max reachable norm), an artifact, not a real finding. (2)
    `expert_asr`/`expert_asr_frozen` are now measured on a FIXED 256-example set instead of
    accumulated over whichever small, noisy poisoned-row subset happened to land in a given
    batch's mini-batches. See docs/threat_models_audit.md for the re-run verdict under this
    corrected instrumentation.

    Gradient-mismatch penalty (2026-08-31, optional, off by default): an additional term,
    L_gradmatch (`lambda_gradmatch * L_gradmatch` added to grand_loss, weight default 0.0 =
    no-op), measuring how distinguishable grad(L_p)(theta_k)(delta) is from grad(L_c)(theta_k)
    at the current checkpoint. `gradmatch_metric` selects the distance used (2026-09-01):
    "relerr" (default, original formulation) is the scale-SENSITIVE relative squared error
    ||grad(L_c)(theta_k) - grad(L_p)(theta_k)(delta)||^2 / ||grad(L_c)(theta_k)||^2
    (grad_mismatch_penalty); "cosine" is the scale-INVARIANT 1 - cos(grad(L_c)(theta_k),
    grad(L_p)(theta_k)(delta)) instead (grad_cosine_penalty) -- unlike "relerr", it does not
    penalize the two gradients merely having different magnitudes, only differing in direction.
    penalizing delta for making the poisoned-example gradient distinguishable from the clean-
    example gradient AT THE SAME checkpoint theta_k -- both gradients flattened/concatenated
    across every expert_model parameter. grad(L_c)(theta_k) is the mean gradient of the
    classification loss over every genuinely CLEAN example seen this batch (an honest client's
    whole local batch, plus the non-poisoned rows of each poisoned client's local batch) --
    computed with no dependency on delta (treated as a constant). grad(L_p)(theta_k)(delta) is
    the mean gradient of the classification loss over every genuinely POISONED example this
    batch (triggered input via T_delta, target label) -- reuses the same forward pass already
    computed for L_bd (`logits_bd = expert_model(x_trig)`, `L_bd_cid`), just differentiated
    w.r.t. expert_params (create_graph=True) INSTEAD OF (isolated torch.autograd.grad calls, in
    the same combined call) w.r.t. [delta] -- so it carries a live dependency on delta and
    contributes to delta.grad via grand_loss.backward(), exactly like param_loss/L_bd already
    do. Both means are simple means-of-per-contributing-chunk (one chunk per honest client /
    per poisoned client's clean subset / per poisoned client's poisoned subset), matching this
    module's existing `agg()` convention rather than weighting by exact example count. Computed
    only when lambda_gradmatch != 0 (`track_gradmatch` below) -- the clean-subset-only forward
    pass this requires for poisoned clients is pure overhead otherwise.

    Comparability across the three arms (prerequisite before any campaign, not just doc):
      - checkpoint sampling: this module draws its expert-trajectory index via
        `extract_experts_biased` (own utils.py), the SAME exponentially-biased distribution
        (`alpha_ckpt`) `federated_optimizing_trigger_policy`'s `sample_checkpoints` uses --
        NOT `federated_generate_labels_trigger`'s (and federated_generate_labels') uniform
        `extract_experts`. A joint-vs-indirect comparison therefore now crosses this same
        checkpoint-sampling factor UNLESS `federated_generate_labels_trigger` is also updated
        (out of scope here -- it must stay unmodified per the three-module contract) or unless
        the comparison is read with that caveat in mind.
      - same flip_budget / expert lambda_target / checkpoint window (expert_config) across all
        three experiment configs -- this module does not enforce it in code (it has no
        visibility into a DIFFERENT module's config), only documents it.
      - agg_method="mean" everywhere (see the WARNING below), matching
        federated_optimizing_trigger_policy's (P^mean) objective, which has no agg_method.

    :param experiment_name: Name of the experiment in configuration.
    :param module_name: Name of the module in configuration.
    :param kwargs: Additional arguments (such as slurm id).
    """

    slurm_id = kwargs.get('slurm_id', None)

    args = extract_toml(experiment_name, module_name)
    tracker = ExperimentTracker(experiment_name, module_name, args, slurm_id=slurm_id)

    input_pths = args["input_pths"]
    opt_pths = args["opt_pths"]
    expert_model_flag = args["expert_model"]
    dataset_flag = args["dataset"]
    clean_label = args["source_label"]
    target_label = args["target_label"]
    lam = args.get("lambda", 0.0)
    train_pct = args.get("train_pct", 1.0)
    batch_size = args.get("batch_size", None)
    epochs = args.get("epochs", None)
    expert_config = args.get('expert_config', {})
    config = coalesce_attack_config(args.get("attack_config", {}))
    num_honests = args.get("num_honests", 2)
    num_poisoned = args.get("num_poisoned", 1)
    output_dir = slurmify_path(args["output_dir"], slurm_id)
    attack = args.get("attack", "backdoor")
    clean_trajectory = args.get("clean_trajectory", False)
    # gamma_stealth: a scalar stealth/backdoor loss weight (multiplies grand_loss below) --
    # UNRELATED to federated_optimizing_trigger_policy's gamma = num_poisoned/(num_poisoned+
    # num_honests), a federated-aggregation fraction. Same name, disjoint concepts (see
    # docs/threat_models_audit.md §1/§7) -- renamed here to remove the collision. 'gamma' is
    # still accepted (deprecated) so existing configs keep working.
    if "gamma_stealth" in args:
        gamma_stealth = args["gamma_stealth"]
        if "gamma" in args:
            raise ValueError(
                "Pass exactly one of gamma_stealth or gamma (deprecated alias), not both."
            )
    elif "gamma" in args:
        warnings.warn(
            "'gamma' is deprecated in federated_generate_labels_trigger_joint -- this is a "
            "stealth/backdoor loss weight, unrelated to federated_optimizing_trigger_policy's "
            "gamma = num_poisoned/(num_poisoned+num_honests). Use 'gamma_stealth' instead.",
            DeprecationWarning, stacklevel=2,
        )
        gamma_stealth = args["gamma"]
    else:
        gamma_stealth = 1.0
    agg_method = args.get("agg_method", "mean")
    if agg_method != "mean":
        print(
            f"WARNING: agg_method={agg_method!r} != 'mean' -- federated_optimizing_trigger_"
            "policy's (P^mean) objective has no agg_method parameter and always assumes mean "
            "aggregation. Comparing this run against a federated_optimizing_trigger_policy "
            "run is no longer a single-factor (direct-vs-policy formulation) comparison: the "
            "aggregator is a second, confounded factor."
        )

    # Trigger hyperparameters -- same names as federated_optimizing_trigger_policy for direct
    # comparability between the two threat models.
    epsilon = args.get("epsilon", 0.1)
    lr_delta = args.get("lr_delta", 1e-2)
    lambda_bd = args.get("lambda_bd", 1.0)
    lambda_penalty = args.get("lambda_penalty", 0.0)
    # lambda_delta defaults to 0 HERE specifically (same default as the other two modules, but
    # load-bearing in THIS one): it penalizes ||delta||, i.e. actively pushes delta's magnitude
    # toward zero -- directly fighting the magnitude floor (L_mag) below, which exists BECAUSE
    # expert_asr was observed collapsing to ~0 while the matching term kept improving (see
    # run() docstring's "Anti-collapse regularizers"). A nonzero lambda_delta here would let the
    # optimizer trade L_mag off against itself instead of genuinely keeping delta's magnitude up.
    lambda_delta = args.get("lambda_delta", 0.0)
    lambda_tv = args.get("lambda_tv", 0.0)
    kappa = args.get("kappa", 0.0)
    init = args.get("init", "stripe")

    # Anti-collapse regularizers (see run() docstring, "Anti-collapse regularizers"): distinct
    # from lambda_penalty/kappa (trigger_penalty_hinge's STEALTH ceiling on
    # cos(delta, mu_target-mu_source), ALSO unchanged and still available above) -- these are a
    # FLOOR on cos(delta, mu_target) alone, plus a floor on ||delta||_2 (without which the
    # directional floor is vacuously satisfiable via delta -> 0). trigger_constraint selects
    # whether these are soft penalties added to grand_loss ("penalty", default, matches the
    # other two modules' existing clamp_-based mechanism) or hard constraints enforced by
    # projection after every optimizer_delta.step() ("projection").
    trigger_constraint = args.get("trigger_constraint", "penalty")
    align_kappa = args.get("align_kappa", 0.6)
    lambda_align = args.get("lambda_align", 1.0)
    lambda_mag = args.get("lambda_mag", 1.0)
    delta_min_frac = args.get("delta_min_frac", 0.5)

    # Gradient-mismatch penalty (see run() docstring): off by default (0.0 = no-op, identical
    # to the module's behavior before this term existed). gradmatch_metric selects the distance
    # used between grad(L_c)(theta_k) and grad(L_p)(theta_k)(delta): "relerr" (default, the
    # original formulation) is the scale-SENSITIVE relative-squared-error ratio
    # ||.-.||^2/||grad(L_c)||^2; "cosine" is the scale-INVARIANT 1 - cos(.,.) instead (see
    # grad_cosine_penalty).
    lambda_gradmatch = args.get("lambda_gradmatch", 0.0)
    gradmatch_eps = args.get("gradmatch_eps", 1e-8)
    gradmatch_metric = args.get("gradmatch_metric", "relerr")
    if gradmatch_metric not in ("relerr", "cosine"):
        raise ValueError(
            f"gradmatch_metric must be 'relerr' or 'cosine', got {gradmatch_metric!r}"
        )
    track_gradmatch = lambda_gradmatch != 0

    output_dir_trigger = slurmify_path(
        args.get("output_dir_trigger", "optimized_trigger"), slurm_id,
    )

    # Checkpoint-sampling comparability (correction F): 'biased' is THIS module's own prior
    # behavior (extract_experts_biased, same alpha_ckpt-driven exponential family
    # federated_optimizing_trigger_policy's sample_checkpoints uses) -- default kept as-is, no
    # regression. federated_generate_labels_trigger defaults to 'uniform' instead (its own
    # prior behavior). Set both to the same value to remove this as a comparison factor.
    checkpoint_sampling = args.get("checkpoint_sampling", "biased")
    alpha_ckpt = args.get("alpha_ckpt", 0.01)

    # Live expert retraining (closes the H1 gap documented above, "design difference from the
    # policy module's live-retraining architecture"): every expert_retrain_interval outer
    # iterations (`it`, each already one full epoch over mtt_dataset), retrain a FRESH expert
    # from scratch against the CURRENT trigger for expert_retrain_epochs epochs (same duration
    # as the ORIGINAL train_expert step -- see _retrain_expert_with_trigger below), and splice
    # its checkpoints in as the expert trajectory for the remaining iterations. 0 (default)
    # disables this entirely -- unchanged behavior, a frozen expert trajectory throughout.
    expert_retrain_interval = args.get("expert_retrain_interval", 0)
    expert_retrain_epochs = args.get("expert_retrain_epochs", 20)
    # Must stay consistent with expert_config's own `trajectories` granularity (same
    # constraint the ORIGINAL train_expert step's own checkpoint_iters already had to satisfy
    # -- see checkpoint_callback: a trajectory value like 150 is only ever hit if it's a
    # multiple of checkpoint_iters) for extract_experts/extract_experts_biased's redraw
    # (below) to find checkpoints at the sampled trajectory positions.
    expert_retrain_checkpoint_iters = args.get("expert_retrain_checkpoint_iters", 50)
    expert_retrain_optim_kwargs = args.get("expert_retrain_optim_kwargs", {})
    expert_retrain_scheduler_kwargs = args.get("expert_retrain_scheduler_kwargs", {})
    # Real RNG seed (see schemas/federated_generate_labels_trigger_joint.toml's `seed` doc) --
    # reused ONLY by _retrain_expert_with_trigger below (when expert_retrain_interval>0), to
    # reseed torch/numpy/random to the SAME value the original train_expert step used (its own
    # `seed`, see schemas/train_expert.toml), so a retrained expert's weight init/data order
    # replicate the ORIGINAL expert's -- isolating the poisoned trigger as the only difference.
    seed = args.get("seed", None)

    # P3: pool_size checkpoints preloaded into RAM once, then drawn from uniformly at random
    # per outer step (`it`), instead of a single checkpoint indexed sequentially by `it` --
    # conditioning every step's gradient on just one point of the expert trajectory made
    # optimization unstable. pool_size=1 keeps the exact previous (sequential, per-step) code
    # path -- see the pool_size==1 branch below.
    pool_size = args.get("pool_size", 15)

    # Multi-checkpoint averaging (2026-09-02, see run() docstring "Averaging the loss over
    # multiple checkpoints"): if > 1, every outer step draws n_checkpoints_per_step DISTINCT
    # checkpoints from the pool (instead of just one) and computes THIS step's grand_loss as
    # the MEAN of each checkpoint's own (otherwise unchanged) per-checkpoint grand_loss, before
    # a SINGLE backward() call. Requires pool_size > 1 (checkpoint pooling enabled) -- validated
    # against the ACTUAL (possibly clamped) pool_size once it's built, below. Default 1 leaves
    # the single-checkpoint-per-step behavior completely unchanged.
    n_checkpoints_per_step = args.get("n_checkpoints_per_step", 1)
    if n_checkpoints_per_step < 1:
        raise ValueError(
            f"n_checkpoints_per_step must be >= 1, got {n_checkpoints_per_step}"
        )
    if n_checkpoints_per_step > 1 and pool_size == 1:
        raise ValueError(
            "n_checkpoints_per_step > 1 requires pool_size > 1 (checkpoint pooling must be "
            "enabled to draw multiple DISTINCT checkpoints per step) -- got pool_size=1."
        )

    # Instrumentation (mandatory, see run() docstring): if set, per-batch metrics (expert_asr,
    # delta_inf, delta_l2, param_loss, param_dist, L_bd_mean, grand_loss) are dumped as JSON.
    metrics_log_path = args.get("metrics_log_path", None)
    if metrics_log_path is not None:
        metrics_log_path = slurmify_path(metrics_log_path, slurm_id)
    history = [] if metrics_log_path else None

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Build datasets and initialize labels
    print("Building datasets...")
    # Placeholder image-poisoner: only its LABEL-flip effect and poison_inds selection are
    # used (see docstring) -- "1xs" (the sinusoidal stripe already used elsewhere in the
    # pipeline) is an arbitrary but consistent choice; its image effect is discarded below.
    placeholder_poisoner = pick_poisoner("1xs", dataset_flag, target_label, delta=None)

    big_ims = needs_big_ims(expert_model_flag)
    *_, mtt_dataset_base = get_matching_datasets(
        dataset_flag, placeholder_poisoner, clean_label, train_pct=train_pct, big=big_ims,
        clean=clean_trajectory,
    )
    mtt_dataset = TriggerMTTDataset.from_mtt_dataset(mtt_dataset_base)

    n_classes = get_n_classes(dataset_flag)
    labels = extract_labels(mtt_dataset.distill, config['one_hot_temp'], n_classes)
    labels_init = torch.stack(extract_labels(mtt_dataset.distill, 1, n_classes))
    labels_syn = torch.stack(labels).requires_grad_(True)

    # Raw ([0,1], unnormalized/unaugmented) images, same index space as mtt_dataset.distill --
    # used to rebuild T_delta(x) for genuinely-poisoned draws. Same convention
    # federated_optimizing_trigger uses for its own trigger-optimization loader.
    raw_train_dataset = get_raw_clean_dataset(dataset_flag, train=True)

    # Load expert trajectories
    print("Loading expert trajectories...")
    if checkpoint_sampling == "uniform":
        expert_starts, expert_opt_starts = extract_experts(
            expert_config, input_pths, config['iterations'], expert_opt_path=opt_pths,
        )
    else:
        expert_starts, expert_opt_starts = extract_experts_biased(
            expert_config,
            input_pths,
            config['iterations'],
            alpha_ckpt,
            expert_opt_path=opt_pths,
        )

    # Optimize labels and trigger jointly
    print("Training...")

    # n_checkpoints_per_step DISTINCT, PERSISTENT model instances (not one reloaded K times --
    # see run() docstring "Averaging the loss over multiple checkpoints" for why this is
    # required, not just a memory/parallelism choice): each expert_models[k]/student_models[k]
    # is loaded from its own checkpoint ONCE per batch-step and only reloaded again at the START
    # of the NEXT batch-step, by which point this step's single, combined grand_loss.backward()
    # has ALREADY run -- exactly mirroring the single-checkpoint invariant this module has
    # always relied on (load_state_dict only ever happens between one graph's construction and
    # ITS OWN eventual backward(), never in between). Reusing ONE model object and reloading it
    # K times per step instead would mutate leaf parameter tensors still referenced by an
    # EARLIER checkpoint's not-yet-backwarded create_graph=True graph -- PyTorch detects this via
    # its per-tensor version counter and raises "modified by an inplace operation" at backward()
    # time (a hard, unambiguous failure, not a silent correctness bug -- verified in
    # prelim/tests/test_multi_checkpoint_step.py; kept as a regression pin for this design
    # choice).
    student_models = [
        load_model(expert_model_flag, n_classes) for _ in range(n_checkpoints_per_step)
    ]
    expert_models = [
        load_model(expert_model_flag, n_classes) for _ in range(n_checkpoints_per_step)
    ]

    device = get_module_device(student_models[0])

    mu = get_mu(dataset_flag, target_label, device, model_flag=expert_model_flag)
    mu_source = mu if clean_label == -1 else get_mu(
        dataset_flag, clean_label, device, model_flag=expert_model_flag,
    )

    delta = init_delta(
        mu.shape, horizontal=True, strength=6.0, freq=16, device=device, init=init,
    )
    # Magnitude floor: a FRACTION of the init trigger's own norm, computed (not hardcoded) --
    # see run() docstring's "Anti-collapse regularizers". Computed BEFORE requires_grad_ so it
    # is a plain float, independent of delta's later optimization.
    delta_min = delta_min_frac * delta.detach().norm().item()
    print(
        f"delta_min={delta_min:.6f} ({delta_min_frac} * ||delta_init||_2="
        f"{delta.detach().norm().item():.6f})"
    )
    # H1 diagnostic (not a correction -- see run() docstring): delta_init is captured LATER,
    # inside the training loop, right after delta's first clamp_/projection (Step 3 fix -- see
    # docs/threat_models_audit.md): capturing it HERE, from the raw init (strength=6.0, never
    # clamped to epsilon), previously made cos_delta_to_init/delta_drift_l2 meaningless at
    # step 0 (cos != 1.0, drift_l2 could exceed the epsilon ball's own max reachable norm) --
    # comparing against an INFEASIBLE reference point. Deferred capture guarantees delta_init
    # is always a genuinely feasible (epsilon-respecting) trigger value.
    delta.requires_grad_(True)
    optimizer_delta = torch.optim.Adam([delta], lr=lr_delta)

    batch_size, epochs, optimizer_expert, optimizer_labels = get_mtt_attack_info(
        expert_models[0].parameters(),
        labels_syn,
        config['expert_kwargs'],
        config['labels_kwargs'],
        batch_size=batch_size,
        epochs=epochs
    )
    batch_size = batch_size // (num_honests + num_poisoned)
    loaders = []
    for _ in range(num_honests + num_poisoned):
        loader, _ = either_dataloader_dataset_to_both(
            mtt_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        loaders.append(loader)

    # Fixed evaluation set for expert_asr / expert_asr_frozen (Step 3, H1 fix): 256 examples
    # from the source class (or, if clean_label == -1 -- no specific source class -- the first
    # 256 images regardless of class), evaluated IDENTICALLY every batch. Replaces the previous
    # per-batch accumulation over whichever (small, noisy -- often just a handful of rows)
    # poisoned subset happened to land in that batch's mini-batches, which made expert_asr's
    # step-to-step variance partly an artifact of sample size rather than a real signal.
    N_ASR_EVAL = 256
    asr_eval_raw = []
    for i in range(len(raw_train_dataset)):
        x_i, y_i = raw_train_dataset[i]
        if clean_label == -1 or y_i == clean_label:
            asr_eval_raw.append(x_i)
            if len(asr_eval_raw) >= N_ASR_EVAL:
                break
    asr_eval_raw = torch.stack(asr_eval_raw).to(device)

    delta_init = None  # H1 diagnostic -- captured on the first batch, see below

    # P3: preload `pool_size` (params, optimizer-state) checkpoint pairs into RAM once, then
    # draw one uniformly at random per outer step below -- instead of a single checkpoint
    # indexed sequentially by `it`. pool_size==1 keeps the ORIGINAL sequential per-step disk
    # load, bit-for-bit -- see prelim/tests/test_expert_checkpoint_pool.py. Weights are cached
    # in float32 on CPU (never half) -- this module's create_graph=True backward (student_grad_
    # buf below) differentiates through expert_start/state_dict-derived quantities, and half
    # precision there risks numerical issues the original code never had to deal with.
    expert_pool = None
    if pool_size != 1:
        print(f"Preloading up to {pool_size} expert checkpoints into RAM (float32, CPU)...")
        expert_pool, pool_size = build_expert_pool(expert_starts, expert_opt_starts, pool_size)

    # n_checkpoints_per_step must not exceed the ACTUAL pool_size -- build_expert_pool above may
    # have clamped the requested pool_size down to the number of distinct checkpoints actually
    # available (see its own docstring), so this is checked here, not at param-parsing time.
    if n_checkpoints_per_step > pool_size:
        raise ValueError(
            f"n_checkpoints_per_step={n_checkpoints_per_step} > pool_size={pool_size} "
            "(possibly already clamped down from a larger request, see build_expert_pool) -- "
            "cannot draw that many DISTINCT checkpoints per step."
        )

    def _opt_state_to_device(opt_state_cpu, device):
        return {
            "state": {
                k: {
                    kk: (vv.to(device) if torch.is_tensor(vv) else vv)
                    for kk, vv in v.items()
                }
                for k, v in opt_state_cpu["state"].items()
            },
            "param_groups": opt_state_cpu["param_groups"],
        }

    def _load_expert_for_step(it):
        """Returns (params_state_dict, opt_state_dict) for outer step `it` -- params are
        load_state_dict-ed into expert_model/student_model by the caller, opt_state_dict is
        read directly (sgd_step, below) into device-resident tensor arithmetic."""
        if pool_size == 1:
            checkpoint = torch.load(expert_starts[it])
            opt_state = torch.load(expert_opt_starts[it])
            return checkpoint, opt_state
        checkpoint, opt_state_cpu = random.choice(expert_pool)
        return checkpoint, _opt_state_to_device(opt_state_cpu, device)

    def _load_experts_for_step(it):
        """Returns a list of n_checkpoints_per_step (params_state_dict, opt_state_dict) pairs
        for outer step `it` (see run() docstring "Averaging the loss over multiple
        checkpoints"). n_checkpoints_per_step==1 (default) just wraps _load_expert_for_step's
        single draw in a list -- bit-for-bit the prior single-checkpoint behavior. > 1 draws
        that many DISTINCT entries from expert_pool (validated >= n_checkpoints_per_step
        above) -- distinct, not with replacement, so the K checkpoints averaged this step are
        genuinely different points on the expert trajectory."""
        if n_checkpoints_per_step == 1:
            return [_load_expert_for_step(it)]
        drawn = random.sample(expert_pool, n_checkpoints_per_step)
        return [
            (checkpoint, _opt_state_to_device(opt_state_cpu, device))
            for checkpoint, opt_state_cpu in drawn
        ]

    # Live expert retraining -- see the run() docstring's H1 diagnostic and
    # expert_retrain_interval's own comment above. Writes to a PRIVATE directory
    # (output_dir/../expert_retrain/round<n>/), never overwriting the shared train_expert
    # checkpoints this run started from (those may be reused by other cells/seeds -- see
    # gen_configs.py's per-seed train_expert dedup). Redraws the remaining outer iterations'
    # expert_starts/expert_opt_starts against this new, trigger-poisoned trajectory via the
    # SAME extract_experts/extract_experts_biased functions used at startup -- no new sampling
    # logic -- and rebuilds expert_pool if pooling is on.
    retrain_base_dir = Path(output_dir).parent / "expert_retrain"

    def _retrain_expert_with_trigger(round_idx, delta_snapshot, it, remaining_iterations):
        nonlocal expert_starts, expert_opt_starts, expert_pool, pool_size

        print(
            f"[expert_retrain] round {round_idx}: retraining a fresh expert for "
            f"{expert_retrain_epochs} epochs against the CURRENT trigger "
            f"(||delta||_inf={delta_snapshot.abs().max().item():.4f})..."
        )

        # Reseed to the ORIGINAL train_expert step's own seed (see schemas/
        # federated_generate_labels_trigger_joint.toml's `seed` doc), BEFORE building this
        # round's poisoned dataset and initializing the fresh model -- same placement (relative
        # to dataset construction/model init) as train_expert/run_module.py's own seeding call,
        # so this round's weight init and data order replicate the ORIGINAL expert's exactly,
        # leaving the poisoned trigger (via `poisoner` below) as the only source of difference.
        # No-op (unseeded, as before this field existed) if `seed` was never set.
        if seed is not None:
            print(f"[expert_retrain] round {round_idx}: reseeding RNGs to seed={seed} "
                  "(matching the original train_expert step) before this round's dataset/model.")
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        poisoner = pick_poisoner(
            "optimized", dataset_flag, target_label, delta=delta_snapshot.detach().cpu(),
        )
        poison_train, _, test, poison_test, _ = get_matching_datasets(
            dataset_flag, poisoner, clean_label, train_pct=train_pct, big=big_ims,
        )

        retrain_model = load_model(expert_model_flag, n_classes)
        retrain_batch_size, retrain_epochs, retrain_opt, retrain_sched = get_train_info(
            retrain_model.parameters(), "sgd",
            batch_size=None, epochs=expert_retrain_epochs,
            optim_kwargs=expert_retrain_optim_kwargs,
            scheduler_kwargs=expert_retrain_scheduler_kwargs,
        )
        # Single expert index (0) per round, matching expert_config['experts']=1's own
        # convention -- see extract_experts/extract_experts_biased's `expert` draw below.
        round_dir = retrain_base_dir / f"round{round_idx}" / "0"
        mini_train(
            model=retrain_model,
            train_data=poison_train,
            test_data=[test, poison_test.poison_dataset],
            batch_size=retrain_batch_size,
            opt=retrain_opt,
            scheduler=retrain_sched,
            epochs=retrain_epochs,
            callback=lambda m, o, e, i: checkpoint_callback(
                m, o, e, i, expert_retrain_checkpoint_iters, str(round_dir),
            ),
        )

        round_input_pths = str(retrain_base_dir / f"round{round_idx}" / "{}" / "model_{}_{}.pth")
        round_opt_pths = str(
            retrain_base_dir / f"round{round_idx}" / "{}" / "model_{}_{}_opt.pth"
        )
        if checkpoint_sampling == "uniform":
            new_starts, new_opt_starts = extract_experts(
                expert_config, round_input_pths, remaining_iterations,
                expert_opt_path=round_opt_pths,
            )
        else:
            new_starts, new_opt_starts = extract_experts_biased(
                expert_config, round_input_pths, remaining_iterations, alpha_ckpt,
                expert_opt_path=round_opt_pths,
            )
        expert_starts = expert_starts[:it] + new_starts
        expert_opt_starts = expert_opt_starts[:it] + new_opt_starts

        if pool_size != 1:
            print(
                f"[expert_retrain] round {round_idx}: refreshing the {pool_size}-checkpoint "
                "pool from the retrained trajectory..."
            )
            expert_pool, pool_size = build_expert_pool(
                new_starts, new_opt_starts, pool_size,
            )
        print(f"[expert_retrain] round {round_idx}: done, trajectory refreshed from it={it}.")

    losses = []
    align_active_window = []  # rolling fraction of steps where L_align > 0 -- see hinge_rate
    mag_active_window = []    # in federated_optimizing_trigger_policy for the same pattern
    ANTI_COLLAPSE_WINDOW = 50

    with make_pbar(total=config['iterations'] * len(mtt_dataset)) as pbar:
        expert_retrain_round = 0
        for it in range(config['iterations']):
            if expert_retrain_interval and it > 0 and it % expert_retrain_interval == 0:
                expert_retrain_round += 1
                _retrain_expert_with_trigger(
                    expert_retrain_round, delta, it, config['iterations'] - it,
                )

            for batches in zip(*loaders):

                # Multi-checkpoint averaging (see run() docstring "Averaging the loss over
                # multiple checkpoints"): n_checkpoints_per_step DISTINCT checkpoints drawn for
                # THIS batch-step (n_checkpoints_per_step==1, the default, draws exactly the
                # ONE checkpoint the pre-existing code always drew -- see _load_experts_for_step).
                # optimizer_delta.zero_grad() runs ONCE per batch-step, BEFORE the k-loop below,
                # so delta.grad accumulates the (1/n_checkpoints_per_step-scaled, see point 3
                # below) L_bd contribution from every poisoned client of EVERY checkpoint k --
                # exactly the single-checkpoint invariant, extended across k.
                checkpoints_k = _load_experts_for_step(it)
                optimizer_delta.zero_grad()

                mtt_term_list = []
                gradmatch_list = []
                matching_term_list = []
                L_bd_mean_list = []

                for k, (checkpoint, state_dict) in enumerate(checkpoints_k):
                    # expert_models[k]/student_models[k] are PERSISTENT, DISTINCT instances
                    # (see their construction site's own comment for why this must not be one
                    # model object reloaded n_checkpoints_per_step times within this step).
                    expert_model = expert_models[k]
                    student_model = student_models[k]
                    expert_model.load_state_dict(checkpoint)
                    student_model.load_state_dict({kk: v.clone() for kk, v in checkpoint.items()})

                    expert_start = [p.clone() for p in expert_model.parameters()]

                    # optimizer_expert itself is never stepped or loaded from disk any more (E2,
                    # point 2b) -- state_dict (loaded directly, independent of optimizer_expert) is
                    # the only source of per-param momentum buffers / SGD hyperparameters used by
                    # both sgd_step calls below.

                    expert_params = list(expert_model.parameters())
                    student_params = list(student_model.parameters())

                    student_grad_buf = [[] for _ in student_params]
                    expert_grad_buf = [[] for _ in expert_params]  # E2, point 2a
                    # Gradient-mismatch penalty buffers (see run() docstring) -- one chunk per
                    # contributing clean/poisoned example source this batch, meaned below.
                    clean_grad_chunks = [[] for _ in expert_params]
                    poison_grad_chunks = [[] for _ in expert_params]

                    L_bd_sum = torch.tensor(0.0, device=device)
                    n_bd_valid = 0

                    for cid, batch in enumerate(batches):

                        # HONEST CLIENTS -- delta plays no role here; the real (non-differentiable)
                        # backward is exactly as before. Feeds BOTH expert_grad_buf and
                        # student_grad_buf (E2, point 2a) -- an honest client's contribution to the
                        # federated aggregate is the same gradient on both sides, same as
                        # federated_generate_labels/federated_generate_labels_trigger.
                        if cid < num_honests:
                            x, y = batch[0].to(device), batch[1].to(device)

                            expert_model.zero_grad()
                            loss = clf_loss(expert_model(x), y)
                            loss.backward()

                            for i, p in enumerate(expert_params):
                                if p.grad is not None:
                                    g = p.grad.detach().clone()
                                    expert_grad_buf[i].append(g)
                                    student_grad_buf[i].append(g)
                                    if track_gradmatch:
                                        # An honest client's local batch is entirely clean by
                                        # construction -- this whole-batch gradient is exactly a
                                        # grad(L_c) contributing chunk (see run() docstring).
                                        clean_grad_chunks[i].append(g)

                        # POISONED CLIENTS
                        else:
                            x_t, y_t, x_d, _, idx, is_poisoned = batch
                            # idx and is_poisoned are kept on CPU (as collated) for indexing idx
                            # itself and for the .any()/.tolist() checks below; is_poisoned_dev is
                            # the copy used to index device-resident tensors (x_t_adv, y_t).
                            x_t, y_t = x_t.to(device), y_t.to(device)
                            x_d = x_d.to(device)
                            is_poisoned_dev = is_poisoned.to(device)
                            y_d = labels_syn[idx].to(device)

                            # Rebuild the genuinely-poisoned rows of x_t from raw pixels via
                            # T_delta (differentiable); the honest fraction of this poisoned
                            # client's local batch (is_poisoned == False) keeps the clean,
                            # unmodified image + true label MTTDataset already returns for it.
                            x_t_adv = x_t
                            if is_poisoned.any():
                                idx_poisoned = idx[is_poisoned].tolist()
                                x_raw_adv = torch.stack(
                                    [raw_train_dataset[i][0] for i in idx_poisoned]
                                ).to(device)
                                x_trig = raw_to_trigger_preprocess(
                                    x_raw_adv, delta, dataset_flag=dataset_flag,
                                    model_flag=expert_model_flag,
                                )
                                x_t_adv = x_t.clone()
                                # NOT detached (E2, point 2): x_t_adv now carries delta's gradient
                                # into loss_e below -- this is the real-coupling change relative to
                                # federated_generate_labels_trigger's x_trig.detach().
                                x_t_adv[is_poisoned_dev] = x_trig

                            # Expert (E2, point 1): differentiable w.r.t. BOTH expert_params AND
                            # delta (via x_t_adv). p.grad is deliberately NEVER set on expert_params
                            # anywhere in this module any more (E2, point 2a/2b): expert_params[i]
                            # is a LEAF that also participates in grads_e's own create_graph=True
                            # graph, so an early `.grad` assignment risks PyTorch's AccumulateGrad
                            # mutating it in place the moment grand_loss.backward() runs later this
                            # batch (see prelim/tests/test_joint_accumgrad_hazard.py) -- simply
                            # never assigning it sidesteps the hazard rather than sequencing around
                            # it. agg(expert_params, ...) below DOES set expert_params[i].grad as a
                            # side effect, but only AFTER this point and harmlessly (see point 2a).
                            expert_model.zero_grad()
                            loss_e = clf_loss(expert_model(x_t_adv), y_t)
                            grads_e = torch.autograd.grad(
                                loss_e, expert_params, create_graph=True, retain_graph=True,
                            )
                            # E2, point 2a: this client's (differentiable, undetached) contribution
                            # to the federated expert-gradient aggregate -- replaces the old
                            # "last-poisoned-client-wins" grads_e_last.
                            for i, g in enumerate(grads_e):
                                expert_grad_buf[i].append(g)

                            # Gradient-mismatch penalty (see run() docstring): the CLEAN subset of
                            # this poisoned client's own local batch (is_poisoned == False rows,
                            # untouched x_t/y_t) is itself a genuinely clean example source -- a
                            # separate, independent forward/backward (not reused elsewhere), no
                            # dependency on delta.
                            if track_gradmatch:
                                clean_mask = ~is_poisoned_dev
                                if clean_mask.any():
                                    loss_clean_local = clf_loss(
                                        expert_model(x_t[clean_mask]), y_t[clean_mask]
                                    )
                                    clean_grads_local = torch.autograd.grad(
                                        loss_clean_local, expert_params, allow_unused=True,
                                    )
                                    for i, g in enumerate(clean_grads_local):
                                        if g is not None:
                                            clean_grad_chunks[i].append(g.detach())

                            # Backdoor efficacy term: CE of the CURRENT expert on the
                            # genuinely-triggered rows. Differentiated w.r.t. delta (isolated,
                            # unscaled -- see below) exactly as in the indirect module (point 3): an
                            # ADDITIONAL contribution to delta.grad, not a duplicate of the
                            # param_loss path. When track_gradmatch, the SAME call also
                            # differentiates w.r.t. expert_params (create_graph=True) -- this is
                            # exactly grad(L_p)(theta_k)(delta) (see run() docstring): the mean
                            # gradient of the classification loss over the genuinely-poisoned rows,
                            # still carrying delta's live dependency via x_trig.
                            if is_poisoned.any():
                                logits_bd = expert_model(x_trig)
                                L_bd_cid = clf_loss(logits_bd, y_t[is_poisoned_dev])
                                grad_targets = [delta] + expert_params if track_gradmatch else [delta]
                                # retain_graph=True (unlike the indirect module): x_trig's node is
                                # shared with x_t_adv (which fed loss_e/grads_e, still needed by
                                # grand_loss.backward() later this batch) -- freeing it here would
                                # break that second, later backward through the same node.
                                bd_grads = torch.autograd.grad(
                                    L_bd_cid, grad_targets,
                                    retain_graph=True, allow_unused=True,
                                    create_graph=track_gradmatch,
                                )
                                delta_grad_raw = bd_grads[0]
                                if delta_grad_raw is not None:
                                    # Scaled by 1/n_checkpoints_per_step (2026-09-02, see run()
                                    # docstring "Averaging the loss over multiple checkpoints"):
                                    # this manual accumulation bypasses grand_loss's own graph
                                    # (point 3 above), so it does NOT automatically get averaged
                                    # by grand_loss's mtt_term_avg/L_gradmatch_avg mean-over-k --
                                    # this explicit division keeps it consistent with those
                                    # (n_checkpoints_per_step==1 makes this a no-op, /1).
                                    delta_grad = (
                                        lambda_bd / n_checkpoints_per_step
                                    ) * delta_grad_raw.detach()
                                    delta.grad = (
                                        delta_grad if delta.grad is None
                                        else delta.grad + delta_grad
                                    )
                                if track_gradmatch:
                                    for i, g in enumerate(bd_grads[1:]):
                                        if g is not None:
                                            poison_grad_chunks[i].append(g)
                                L_bd_sum = L_bd_sum + L_bd_cid.detach()
                                n_bd_valid += 1

                            # Student
                            loss_s = clf_loss(student_model(x_d), softmax(y_d))
                            grads_s = torch.autograd.grad(
                                loss_s, student_params, create_graph=True
                            )

                            for i, g in enumerate(grads_s):
                                student_grad_buf[i].append(g)

                    # Gradient-mismatch penalty for THIS checkpoint (see run() docstring):
                    # means-of-contributing-chunks per expert_model parameter, then
                    # flattened/combined into a single ratio. Falls back to 0.0 whenever
                    # disabled, or whenever this batch happened to draw zero clean or zero
                    # poisoned examples for some parameter (e.g. num_honests==0 and no poisoned
                    # client's batch had any clean rows).
                    if track_gradmatch and all(len(c) > 0 for c in clean_grad_chunks) and all(
                        len(c) > 0 for c in poison_grad_chunks
                    ):
                        clean_grad_mean = [
                            torch.stack(c, dim=0).mean(dim=0) for c in clean_grad_chunks
                        ]
                        poison_grad_mean = [
                            torch.stack(c, dim=0).mean(dim=0) for c in poison_grad_chunks
                        ]
                        penalty_fn = (
                            grad_cosine_penalty if gradmatch_metric == "cosine"
                            else grad_mismatch_penalty
                        )
                        L_gradmatch_k = penalty_fn(
                            clean_grad_mean, poison_grad_mean, eps=gradmatch_eps,
                        )
                    else:
                        L_gradmatch_k = torch.tensor(0.0, device=device)

                    # Aggregate student gradients (DIFFERENTIABLE)
                    agg_student_grads = agg(
                        student_params,
                        student_grad_buf,
                        agg_method,
                        f=num_poisoned
                    )
                    # Aggregate expert gradients across ALL clients (DIFFERENTIABLE w.r.t. delta
                    # through every poisoned client's contribution) -- E2, point 2a. Replaces the
                    # old "last-poisoned-client-wins" grads_e_last. Sets expert_params[i].grad as
                    # a side effect, harmlessly -- see point 2a in the docstring above.
                    agg_expert_grads = agg(
                        expert_params,
                        expert_grad_buf,
                        agg_method,
                        f=num_poisoned
                    )

                    # MTT objective for THIS checkpoint, against the DIFFERENTIABLE
                    # expert_next_param (E2, points 1/2a) instead of the real,
                    # non-differentiable post-step expert_params -- there is no real expert
                    # optimizer step in this module any more (point 2b).
                    param_loss = torch.tensor(0.0, device=device)
                    param_dist = torch.tensor(0.0, device=device)

                    if attack in ["backdoor", "untargeted"]:
                        for init_p, student, grad, grad_e, state in zip(
                            expert_start,
                            student_params,
                            agg_student_grads,
                            agg_expert_grads,
                            state_dict["state"].values(),
                        ):
                            student_update = sgd_step(
                                student, grad, state, state_dict["param_groups"][0]
                            )
                            # Independent copy of this parameter's optimizer state for the
                            # expert-side differentiable step -- sgd_step only WRITES a
                            # momentum_buffer back when one is not already present, so this call
                            # can never perturb `state` (also read by the student's own sgd_step
                            # call above). `init_p` (from expert_start, cloned before this
                            # checkpoint's own cid loop) is BOTH the pre-update value sgd_step
                            # steps from AND param_dist's baseline.
                            state_expert_copy = {
                                kk: (v.clone() if torch.is_tensor(v) else v)
                                for kk, v in state.items()
                            }
                            expert_next_param = sgd_step(
                                init_p, grad_e, state_expert_copy, state_dict["param_groups"][0]
                            )

                            param_loss += total_mse_distance(student_update, expert_next_param)
                            param_dist += total_mse_distance(init_p, expert_next_param)

                        mtt_term_k = param_loss / param_dist

                    # Multi-checkpoint averaging (see run() docstring): collect this
                    # checkpoint's own differentiable contributions (mtt_term_k, L_gradmatch_k)
                    # for averaging OUTSIDE the k-loop -- reg_term/trigger-regularizers/
                    # anti-collapse terms below do NOT depend on the checkpoint, so they are
                    # computed/added exactly ONCE, after the k-loop, not per checkpoint.
                    mtt_term_list.append(mtt_term_k)
                    gradmatch_list.append(L_gradmatch_k)
                    matching_term_list.append(
                        (param_loss / param_dist).item() if param_dist.item() != 0 else 0.0
                    )
                    L_bd_mean_list.append(
                        (L_bd_sum / n_bd_valid).item() if n_bd_valid > 0 else 0.0
                    )

                # Step 5 instrumentation: snapshot of delta.grad from JUST the L_bd path
                # (accumulated across ALL n_checkpoints_per_step checkpoints' cid loops above,
                # point 3, each already scaled by 1/n_checkpoints_per_step -- see that scaling's
                # own comment), taken BEFORE grand_loss.backward() adds the MTT/param_loss-path
                # contribution on top -- lets mtt_delta_grad_norm below isolate each path's
                # magnitude without a separate lambda_bd=0 run.
                L_bd_only_delta_grad = (
                    delta.grad.detach().clone() if delta.grad is not None
                    else torch.zeros_like(delta)
                )

                # reg_term depends only on labels_syn (not on any checkpoint) -- computed once.
                reg_term = lam * torch.linalg.vector_norm(
                    softmax(labels_syn) - labels_init,
                    ord=1,
                    dim=1
                ).mean()

                # Multi-checkpoint averaging (see run() docstring): mean over the
                # n_checkpoints_per_step checkpoints' own mtt_term_k/L_gradmatch_k --
                # n_checkpoints_per_step==1 makes this exactly the single value it always was
                # (mean of one element), so grand_loss is bit-for-bit unchanged in that case.
                mtt_term_avg = torch.stack(mtt_term_list).mean()
                L_gradmatch_avg = torch.stack(gradmatch_list).mean()

                grand_loss = gamma_stealth * (mtt_term_avg + reg_term)

                # Trigger regularizers (optional, default 0) -- same terms/names as
                # federated_optimizing_trigger_policy, reused unchanged. L_pen/kappa is the
                # STEALTH ceiling on cos(delta, mu_target-mu_source) (trigger_penalty_hinge,
                # shared/unmodified file) -- unrelated to the anti-collapse floor below.
                # Checkpoint-independent -- computed/added exactly ONCE.
                L_pen = trigger_penalty_hinge(delta, mu, mu_source, kappa)
                L_tv = tv_loss(delta)
                grand_loss = grand_loss + (
                    lambda_penalty * L_pen + lambda_delta * delta.norm() + lambda_tv * L_tv
                )

                # Anti-collapse regularizers (see run() docstring): L_align is a FLOOR on
                # cos(delta, mu_target) alone (align_kappa, distinct from kappa above), L_mag a
                # floor on ||delta||_2 so L_align cannot be satisfied vacuously via delta -> 0.
                # Always computed (for instrumentation, both branches of trigger_constraint),
                # only ADDED to grand_loss under trigger_constraint=="penalty" -- under
                # "projection" they are enforced as hard constraints after optimizer_delta.step()
                # instead (see below), and adding them here too would double-enforce them.
                # Checkpoint-independent -- computed/added exactly ONCE.
                L_align, _ = directional_floor_penalty(delta, mu, align_kappa)
                L_mag, _ = magnitude_floor_penalty(delta, delta_min)
                if trigger_constraint == "penalty":
                    grand_loss = grand_loss + (lambda_align * L_align + lambda_mag * L_mag)

                grand_loss = grand_loss + lambda_gradmatch * L_gradmatch_avg

                # Optimize labels and trigger. delta.grad already holds the per-client L_bd
                # contributions accumulated above (point 3) -- grand_loss.backward() ADDS the
                # param_loss-path contribution on top (torch always accumulates into .grad;
                # optimizer_delta.zero_grad() ran once, before the cid loop, and nothing zeros
                # delta.grad again in between).
                optimizer_labels.zero_grad()
                grand_loss.backward()

                # Step 5 instrumentation: norm of grand_loss.backward()'s contribution to
                # delta.grad beyond the L_bd-only snapshot taken before it -- i.e. the
                # MTT/param_loss path plus L_pen/delta.norm()/L_tv/L_align/L_mag (whichever of
                # those are active this run). Under the Step 5 reference config (lambda_penalty
                # = lambda_delta = lambda_tv = lambda_align = lambda_mag = 0), this isolates the
                # pure MTT-path contribution exactly, since none of the other regularizers
                # depend on delta when their weights are 0.
                mtt_delta_grad_norm = (
                    (delta.grad.detach() - L_bd_only_delta_grad).norm().item()
                    if delta.grad is not None else 0.0
                )

                optimizer_labels.step()
                optimizer_delta.step()

                with torch.no_grad():
                    if trigger_constraint == "projection":
                        # Hard constraints (§1.4): replaces the plain clamp_ with alternating
                        # projection onto {Linf ball} inter {cone K_align_kappa(mu)} inter
                        # {||delta||_2 >= delta_min} -- see project_trigger_constraints.
                        delta.copy_(
                            project_trigger_constraints(delta, mu, epsilon, align_kappa, delta_min)
                        )
                    else:
                        delta.clamp_(-epsilon, epsilon)

                # No real expert optimizer step any more (E2, point 2b) -- every
                # expert_models[k] is reloaded fresh from disk (or the pool) at the top of the
                # NEXT batch's k-loop regardless, and nothing in THIS batch reads
                # expert_params[i].grad past this point. .eval() is kept on each of the K models
                # for continuity with the prior behavior (expert_model spends most of its life
                # in eval mode, e.g. for BatchNorm running-stats use) even though it no longer
                # follows a real .step() call.
                for m in expert_models:
                    m.eval()

                # Multi-checkpoint averaging (see run() docstring): mean of the
                # n_checkpoints_per_step per-checkpoint values collected during the k-loop above
                # -- n_checkpoints_per_step==1 makes this exactly the single value it always was.
                L_bd_mean = float(np.mean(L_bd_mean_list))
                matching_term = float(np.mean(matching_term_list))

                # Anti-collapse instrumentation (§2): recomputed AFTER the step and the
                # clamp_/projection, on the delta the NEXT batch will actually start from --
                # same convention as delta_l2/delta_linf below, and what the "cos stable
                # exactly at align_kappa, ||delta||_2 collapsing, expert_asr at zero" failure
                # signature (see run() docstring) would actually be observed in.
                with torch.no_grad():
                    delta_l2 = delta.norm().item()
                    delta_linf = delta.abs().max().item()
                    L_align_post, cos_target_post = directional_floor_penalty(delta, mu, align_kappa)
                    cos_target_post, L_align_post = cos_target_post.item(), L_align_post.item()
                    cos_source_post = cosine_to(delta, mu_source).item()
                    L_mag_post, _ = magnitude_floor_penalty(delta, delta_min)
                    L_mag_post = L_mag_post.item()

                    # H1 diagnostic, Step 3 fix (see run() docstring / delta_init comment
                    # above): delta_init is captured HERE, the first time this code runs --
                    # i.e. from delta right after ITS OWN first clamp_/projection -- so it is
                    # always a genuinely feasible reference point, and cos_to_init/delta_drift_l2
                    # are exactly 1.0/0.0 by construction on the batch that captures it.
                    just_captured = delta_init is None
                    if just_captured:
                        delta_init = delta.detach().clone()
                    cos_to_init = cosine_to(delta, delta_init).item()
                    delta_drift_l2 = (delta - delta_init).norm().item()
                    if just_captured:
                        assert abs(cos_to_init - 1.0) < 1e-5 and delta_drift_l2 < 1e-6, (
                            "H1 diagnostic sanity check (Step 3 fix): delta_init was just "
                            f"captured but cos_to_init={cos_to_init}, "
                            f"delta_drift_l2={delta_drift_l2} -- expected exactly 1.0/0.0."
                        )

                    # expert_asr / expert_asr_frozen (Step 3 fix): evaluated on the FIXED
                    # source-class set (asr_eval_raw, built once before the loop) instead of
                    # whichever small poisoned-row subset landed in this batch -- each of the K
                    # expert_models[k] (this batch's checkpoints, in eval mode, weights
                    # UNCHANGED since the k-loop populated them -- nothing between there and here
                    # mutates a model's own parameters, only delta/labels_syn) triggered with
                    # the CURRENT delta vs. the frozen delta_init, same rows both times, then
                    # averaged across the K checkpoints (see run() docstring "Averaging the loss
                    # over multiple checkpoints") -- n_checkpoints_per_step==1 makes this exactly
                    # the single value it always was.
                    x_trig_eval = raw_to_trigger_preprocess(
                        asr_eval_raw, delta, dataset_flag=dataset_flag,
                        model_flag=expert_model_flag,
                    )
                    x_trig_eval_frozen = raw_to_trigger_preprocess(
                        asr_eval_raw, delta_init, dataset_flag=dataset_flag,
                        model_flag=expert_model_flag,
                    )
                    asr_mean_list = [
                        (m(x_trig_eval).argmax(dim=1) == target_label).float().mean().item()
                        for m in expert_models
                    ]
                    asr_frozen_mean_list = [
                        (m(x_trig_eval_frozen).argmax(dim=1) == target_label)
                        .float().mean().item()
                        for m in expert_models
                    ]
                    asr_mean = float(np.mean(asr_mean_list))
                    asr_frozen_mean = float(np.mean(asr_frozen_mean_list))

                align_active_window.append(L_align_post > 0)
                mag_active_window.append(L_mag_post > 0)
                if len(align_active_window) > ANTI_COLLAPSE_WINDOW:
                    align_active_window.pop(0)
                if len(mag_active_window) > ANTI_COLLAPSE_WINDOW:
                    mag_active_window.pop(0)
                align_active_rate = sum(align_active_window) / len(align_active_window)
                mag_active_rate = sum(mag_active_window) / len(mag_active_window)

                losses.append(grand_loss.item())
                if history is not None:
                    history.append({
                        "it": it,
                        "grand_loss": grand_loss.item(),
                        "matching_term": matching_term,
                        "L_bd_mean": L_bd_mean,
                        "expert_asr": asr_mean,
                        "expert_asr_frozen": asr_frozen_mean,
                        "cos_delta_to_init": cos_to_init,
                        "delta_drift_l2": delta_drift_l2,
                        "delta_l2": delta_l2,
                        "delta_linf": delta_linf,
                        "cos_target": cos_target_post,
                        "cos_source": cos_source_post,
                        "L_align": L_align_post,
                        "L_mag": L_mag_post,
                        "align_active_rate": align_active_rate,
                        "mag_active_rate": mag_active_rate,
                        "reg_term": reg_term.item(),
                        "mtt_delta_grad_norm": mtt_delta_grad_norm,
                        "L_gradmatch": L_gradmatch_avg.item(),
                        # Multi-checkpoint spread diagnostics (see run() docstring "Averaging
                        # the loss over multiple checkpoints"): std across the
                        # n_checkpoints_per_step checkpoints used THIS step -- 0.0 by
                        # construction when n_checkpoints_per_step==1. A persistently large
                        # spread means the checkpoints disagree a lot on this step's matching
                        # term / ASR, worth knowing alongside the mean.
                        "matching_term_std": float(np.std(matching_term_list)),
                        "L_bd_mean_std": float(np.std(L_bd_mean_list)),
                        "expert_asr_std": float(np.std(asr_mean_list)),
                    })
                tracker.log(
                    it,
                    grand_loss=grand_loss.item(),
                    L_bd_mean=L_bd_mean,
                    expert_asr=asr_mean,
                    expert_asr_frozen=asr_frozen_mean,
                    delta_l2=delta_l2,
                    delta_linf=delta_linf,
                    L_align=L_align_post,
                    L_mag=L_mag_post,
                    mtt_delta_grad_norm=mtt_delta_grad_norm,
                    L_gradmatch=L_gradmatch_avg.item(),
                )

                pbar.update(batch_size)
                pbar.set_postfix(
                    g_loss=f"{np.mean(losses[-20:]):.4g}",
                    match=f"{matching_term:.4g}" if attack in ["backdoor"] else "N/A",
                    L_bd=f"{L_bd_mean:.4g}",
                    expert_asr=f"{asr_mean:.4g}",
                    asr_frozen=f"{asr_frozen_mean:.4g}",
                    cos_init=f"{cos_to_init:.4g}",
                    delta_linf=f"{delta_linf:.4g}",
                    delta_l2=f"{delta_l2:.4g}",
                    cos_tgt=f"{cos_target_post:.4g}",
                    L_align=f"{L_align_post:.4g}",
                    L_mag=f"{L_mag_post:.4g}",
                    align_rate=f"{align_active_rate:.2f}",
                    mag_rate=f"{mag_active_rate:.2f}",
                    mtt_grad=f"{mtt_delta_grad_norm:.4g}",
                    gradmatch=f"{L_gradmatch_avg.item():.4g}",
                )

    # Save results
    print("Saving results...")
    y_true = torch.stack([mtt_dataset[i][3].detach() for i in range(len(mtt_dataset.distill))])
    np.save(output_dir + "labels.npy", labels_syn.detach().numpy())
    np.save(output_dir + "true.npy", y_true)
    np.save(output_dir + "losses.npy", losses)

    if metrics_log_path:
        Path(metrics_log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_log_path, "w") as f:
            json.dump(history, f, indent=2)

    tracker.finalize()

    run_tag = f"{num_poisoned}vs{num_honests}"
    Path(output_dir_trigger).mkdir(parents=True, exist_ok=True)
    trig_path = (
        Path(output_dir_trigger)
        / f"opt_trig_direct_joint_{init}_{expert_model_flag}_{dataset_flag}_{run_tag}.pt"
    )
    torch.save(delta.detach().cpu(), trig_path)
    print(f"Saved trigger to {trig_path}")


if __name__ == "__main__":
    experiment_name, module_name = sys.argv[1], sys.argv[2]
    run(experiment_name, module_name)
