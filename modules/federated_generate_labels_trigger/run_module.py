"""
Threat model "direct" (P^direct), INDIRECT coupling: optimizes continuous poisoned logit
labels (labels_syn) alongside a backdoor trigger (delta), via federated trajectory matching,
extending federated_generate_labels.py (which optimizes labels_syn alone, against a trigger
fixed and baked into the dataset ahead of time). "Jointly" here means delta and labels_syn are
both trainable parameters in the SAME loop, NOT that both receive gradient from the same loss
term -- see `run()`'s docstring: delta's ONLY gradient path is the isolated backdoor-efficacy
term L_bd; the MTT trajectory-alignment term (param_loss, the bulk of grand_loss) never
backpropagates into delta, because x_t_adv is `.detach()`-ed before it. For the version where
delta is ALSO coupled through param_loss, see federated_generate_labels_trigger_joint.

Expert-side federated aggregation (corrected, see docs/threat_models_audit.md): the expert
model's optimizer step is now driven by `agg(expert_params, expert_grad_buf, agg_method,
f=num_poisoned)` over ALL clients (honest + poisoned), symmetric to how the student side has
always been aggregated via `agg(student_params, student_grad_buf, ...)`. Previously
`expert_grad_buf` was populated by every client but never consumed -- `expert_model.
zero_grad()` before each client's turn meant only the LAST client's raw `.grad` reached
`optimizer_expert.step()`, a single-client (non-federated) expert update. The published
baseline (federated_generate_labels) has this same bug -- with an abandoned, commented-out
attempt at the identical fix already present in its code -- but is deliberately left
unmodified; see docs/threat_models_audit.md for the full analysis and the arbitration that
scoped this fix to this module and federated_generate_labels_trigger_joint only.
"""

from pathlib import Path
import random
import sys
import warnings
import torch
import numpy as np

from modules.base_utils.datasets import get_matching_datasets, pick_poisoner, get_n_classes
from modules.base_utils.util import extract_toml, get_module_device, get_mtt_attack_info, \
                                    load_model, either_dataloader_dataset_to_both, make_pbar, \
                                    needs_big_ims, slurmify_path, clf_loss, softmax, total_mse_distance
from modules.federated_generate_labels.utils import coalesce_attack_config, extract_experts, \
                                                     extract_labels, sgd_step, agg
from modules.federated_optimizing_trigger.utils import (
    get_mu, init_delta, raw_to_trigger_preprocess, get_raw_clean_dataset,
    trigger_penalty_hinge, tv_loss,
)
from modules.federated_generate_labels_trigger.utils import TriggerMTTDataset, \
                                                             extract_experts_biased, build_expert_pool


def run(experiment_name, module_name, **kwargs):
    """
    Optimizes labels_syn (ell-tilde) and delta side by side -- problem (P^direct), INDIRECT
    coupling:

        min_ell-tilde  E_t,{(x_i,y_i)}[ L_lambda(ell-tilde, theta_t, {(x_i,y_i)}) ]
        s.t. theta_T in argmin_theta E_(x^adv,y^adv)~D_poisoned[ L(f_theta(x^adv), y^adv) ]

    where L_lambda = L (the MTT trajectory-alignment loss between U_adv and U_clean, exactly
    as implemented -- unmodified -- in federated_generate_labels.py) + lam*||.||_1 (unchanged
    regularizer), extended here with the SAME two additional terms
    federated_optimizing_trigger_policy adds on top of its own alignment term:

      - the backdoor efficacy term kappa * E[loss_c(f_theta_t(T_delta(X)), y_target)],
        named `lambda_bd` here (consistent with federated_optimizing_trigger_policy's
        naming: `kappa` there names the unrelated trigger-hinge-penalty margin);
      - the trigger's own constraint/regularizers: |delta|_inf <= epsilon (enforced via
        clamp_ after every step) and the optional lambda_penalty/lambda_delta/lambda_tv
        regularizers, reused UNCHANGED from federated_optimizing_trigger.utils.

    IMPORTANT -- delta's gradient path (E1, corrected): despite the "min_ell-tilde" objective
    above being written as if L_lambda depended on delta too, `grand_loss` (== L_lambda in
    code) does NOT actually backpropagate into delta. In the poisoned-client branch below,
    `x_t_adv[is_poisoned_dev] = x_trig.detach()` detaches the trigger-rebuilt images before
    they enter `loss_e`/`loss_s` -- the entire MTT trajectory-alignment term (param_loss,
    which is what `grand_loss` mostly consists of) is therefore constant w.r.t. delta. delta's
    ONLY gradient comes from the separate, isolated `L_bd_cid` term (a plain classification
    loss on the current expert's triggered predictions, backpropagated through its OWN
    `torch.autograd.grad(lambda_bd*L_bd_cid, [delta], ...)` call, never touching `grand_loss`
    or `expert_params.grad`). So: labels_syn is optimized against the full MTT alignment loss,
    delta is optimized against backdoor efficacy alone, and the coupling between the two is
    INDIRECT -- it only happens through the shared, re-retrained expert trajectory that both
    influence in their own way, not through a shared differentiable loss term. This is a
    deliberate simplification kept for stability/cost reasons, not a bug -- for the version
    where delta ALSO receives gradient through param_loss (x_t_adv left undetached, the expert
    step itself made differentiable), see the separate
    `federated_generate_labels_trigger_joint` module: a new module, not a modification of this
    one, since the two threat models are meant to coexist and be compared.

    Architecture note: federated_generate_labels' poisoned-client "adversarial" branch (x_t)
    is baked in at the PIL level by a FIXED poisoner before any training happens -- not
    differentiable w.r.t. a trigger. Here, `TriggerMTTDataset` (see utils.py) additionally
    flags which draws are genuinely from the poison segment; for those, x_t is rebuilt from
    raw pixels via `raw_to_trigger_preprocess(x_raw, delta, ...)` (delta-differentiable,
    matching how federated_optimizing_trigger applies T_delta), while the honest fraction of
    a poisoned client's local shard (the clean pass-through draws) is left untouched -- same
    realistic client-data mixture federated_generate_labels already models. The image-level
    poisoner passed to `get_matching_datasets` is therefore only a placeholder, used solely
    for its LABEL-flip effect and for selecting poison_inds; its image effect is discarded.

    Budget beta: no in-loop projection here -- exactly as in the base federated_generate_labels
    pipeline, the budget is applied downstream by federated_select_flips (its `budgets` list),
    run unchanged against this module's labels.npy/true.npy.

    :param experiment_name: Name of the experiment in configuration.
    :param module_name: Name of the module in configuration.
    :param kwargs: Additional arguments (such as slurm id).
    """

    slurm_id = kwargs.get('slurm_id', None)

    args = extract_toml(experiment_name, module_name)

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
            "'gamma' is deprecated in federated_generate_labels_trigger -- this is a "
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
    lambda_delta = args.get("lambda_delta", 0.0)
    lambda_tv = args.get("lambda_tv", 0.0)
    kappa = args.get("kappa", 0.0)
    init = args.get("init", "stripe")
    output_dir_trigger = slurmify_path(
        args.get("output_dir_trigger", "optimized_trigger"), slurm_id,
    )

    # Checkpoint-sampling comparability (correction F): 'uniform' is THIS module's own prior
    # behavior (extract_experts, np.random.randint) -- default kept as-is, no regression.
    # federated_generate_labels_trigger_joint defaults to 'biased' instead (its own prior
    # behavior). Set both to the same value to remove checkpoint sampling as a comparison factor.
    checkpoint_sampling = args.get("checkpoint_sampling", "uniform")
    alpha_ckpt = args.get("alpha_ckpt", 0.01)

    # P3: pool_size checkpoints preloaded into RAM once, then drawn from uniformly at random
    # per outer step (`it`), instead of a single checkpoint indexed sequentially by `it` --
    # conditioning every step's gradient on just one point of the expert trajectory made
    # optimization unstable. pool_size=1 keeps the exact previous (sequential, per-step) code
    # path -- see the pool_size==1 branch below.
    pool_size = args.get("pool_size", 15)

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
    if checkpoint_sampling == "biased":
        expert_starts, expert_opt_starts = extract_experts_biased(
            expert_config, input_pths, config['iterations'], alpha_ckpt, expert_opt_path=opt_pths,
        )
    else:
        expert_starts, expert_opt_starts = extract_experts(
            expert_config,
            input_pths,
            config['iterations'],
            expert_opt_path=opt_pths
        )

    # Optimize labels and trigger jointly
    print("Training...")

    student_model = load_model(expert_model_flag, n_classes)
    expert_model = load_model(expert_model_flag, n_classes)

    device = get_module_device(student_model)

    mu = get_mu(dataset_flag, target_label, device, model_flag=expert_model_flag)
    mu_source = mu if clean_label == -1 else get_mu(
        dataset_flag, clean_label, device, model_flag=expert_model_flag,
    )

    delta = init_delta(
        mu.shape, horizontal=True, strength=6.0, freq=16, device=device, init=init,
    )
    delta.requires_grad_(True)
    optimizer_delta = torch.optim.Adam([delta], lr=lr_delta)

    batch_size, epochs, optimizer_expert, optimizer_labels = get_mtt_attack_info(
        expert_model.parameters(),
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

    # P3: preload `pool_size` (params, optimizer-state) checkpoint pairs into RAM once, then
    # draw one uniformly at random per outer step below -- instead of a single checkpoint
    # indexed sequentially by `it` (conditioning every step's gradient on just one point of the
    # expert trajectory made optimization unstable). pool_size==1 keeps the ORIGINAL sequential
    # per-step disk load, bit-for-bit -- see prelim/tests/test_expert_checkpoint_pool.py.
    expert_pool = None
    if pool_size != 1:
        print(f"Preloading up to {pool_size} expert checkpoints into RAM (float32, CPU)...")
        expert_pool, pool_size = build_expert_pool(expert_starts, expert_opt_starts, pool_size)

    def _load_expert_for_step(it):
        """Returns (params_state_dict, opt_state_dict) for outer step `it` -- the params are
        load_state_dict-ed into expert_model/student_model by the caller, and opt_state_dict is
        both load_state_dict-ed into optimizer_expert AND read directly (sgd_step, below)."""
        if pool_size == 1:
            checkpoint = torch.load(expert_starts[it])
            optimizer_expert.load_state_dict(torch.load(expert_opt_starts[it]))
            opt_state = torch.load(expert_opt_starts[it])
            return checkpoint, opt_state
        checkpoint, opt_state_cpu = random.choice(expert_pool)
        optimizer_expert.load_state_dict(opt_state_cpu)
        # optimizer_expert.state_dict() (not opt_state_cpu directly): sgd_step below reads
        # state_dict["state"].values() straight into device-resident tensor arithmetic, unlike
        # load_state_dict itself (which casts device/dtype internally) -- reading back through
        # the optimizer gives tensors already on expert_model's own device.
        return checkpoint, optimizer_expert.state_dict()

    losses = []

    with make_pbar(total=config['iterations'] * len(mtt_dataset)) as pbar:
        for it in range(config['iterations']):
            for batches in zip(*loaders):

                # Load expert trajectory
                checkpoint, state_dict = _load_expert_for_step(it)
                expert_model.load_state_dict(checkpoint)
                student_model.load_state_dict({k: v.clone() for k, v in checkpoint.items()})

                expert_start = [p.clone() for p in expert_model.parameters()]

                expert_params = list(expert_model.parameters())
                student_params = list(student_model.parameters())

                expert_grad_buf = [[] for _ in expert_params]
                student_grad_buf = [[] for _ in student_params]

                optimizer_delta.zero_grad()
                L_bd_sum = torch.tensor(0.0, device=device)
                n_bd_valid = 0

                for cid, batch in enumerate(batches):

                    # HONEST CLIENTS -- unchanged from federated_generate_labels.
                    if cid < num_honests:
                        x, y = batch[0].to(device), batch[1].to(device)

                        expert_model.zero_grad()
                        loss = clf_loss(expert_model(x), y)
                        loss.backward()

                        for i, p in enumerate(expert_params):
                            if p.grad is not None:
                                expert_grad_buf[i].append(p.grad.detach().clone())
                                student_grad_buf[i].append(p.grad.detach().clone())

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
                            # Detached copy: loss_e below feeds expert_params.grad (consumed
                            # by optimizer_expert.step()) and must stay unaffected by kappa --
                            # delta gets gradient ONLY through the isolated L_bd_cid term below.
                            x_t_adv[is_poisoned_dev] = x_trig.detach()

                        # Expert
                        expert_model.zero_grad()
                        loss_e = clf_loss(expert_model(x_t_adv), y_t)
                        loss_e.backward()

                        for i, p in enumerate(expert_params):
                            if p.grad is not None:
                                expert_grad_buf[i].append(p.grad.detach().clone())

                        # Backdoor efficacy term: CE of the CURRENT expert on the
                        # genuinely-triggered rows, differentiated ONLY w.r.t. delta (isolated
                        # torch.autograd.grad call -- never touches expert_params.grad).
                        if is_poisoned.any():
                            logits_bd = expert_model(x_trig)
                            L_bd_cid = clf_loss(logits_bd, y_t[is_poisoned_dev])
                            (delta_grad,) = torch.autograd.grad(
                                lambda_bd * L_bd_cid, [delta],
                                retain_graph=False, allow_unused=True,
                            )
                            if delta_grad is not None:
                                delta.grad = (
                                    delta_grad.detach() if delta.grad is None
                                    else delta.grad + delta_grad.detach()
                                )
                            L_bd_sum = L_bd_sum + L_bd_cid.detach()
                            n_bd_valid += 1

                        # Student
                        loss_s = clf_loss(student_model(x_d), softmax(y_d))
                        grads_s = torch.autograd.grad(
                            loss_s, student_params, create_graph=True
                        )

                        for i, g in enumerate(grads_s):
                            student_grad_buf[i].append(g)

                # Aggregate expert gradients across ALL clients (honest + poisoned) before the
                # real optimizer step -- FIX (see docs/threat_models_audit.md): previously
                # `expert_grad_buf` was populated by every client but never consumed;
                # `expert_model.zero_grad()` ran before EACH client's turn, so only the LAST
                # client's raw `.grad` survived to feed `optimizer_expert.step()` below --
                # effectively a single-client (non-federated) expert update, while the student
                # side was already properly aggregated via `agg()`. `agg()` sets `p.grad = g`
                # for each param as a side effect, which is exactly what `optimizer_expert.
                # step()` reads. This mirrors the SAME aggregation already applied to the
                # student side two lines below -- same `agg_method`/`f=num_poisoned`. The
                # published baseline (federated_generate_labels) has this identical bug (with
                # the fix even attempted and then commented out there) but is intentionally
                # NOT touched here -- see docs/threat_models_audit.md.
                agg(
                    expert_params,
                    expert_grad_buf,
                    agg_method,
                    f=num_poisoned
                )
                optimizer_expert.step()
                expert_model.eval()

                # Aggregate student gradients (DIFFERENTIABLE)
                agg_student_grads = agg(
                    student_params,
                    student_grad_buf,
                    agg_method,
                    f=num_poisoned
                )

                # MTT objective (unchanged)
                param_loss = torch.tensor(0.0, device=device)
                param_dist = torch.tensor(0.0, device=device)

                reg_term = lam * torch.linalg.vector_norm(
                    softmax(labels_syn) - labels_init,
                    ord=1,
                    dim=1
                ).mean()

                if attack in ["backdoor", "untargeted"]:
                    for init_p, student, expert, grad, state in zip(
                        expert_start,
                        student_params,
                        expert_params,
                        agg_student_grads,
                        state_dict["state"].values(),
                    ):
                        student_update = sgd_step(
                            student, grad, state, state_dict["param_groups"][0]
                        )

                        param_loss += total_mse_distance(student_update, expert)
                        param_dist += total_mse_distance(init_p, expert)

                    grand_loss = (param_loss / param_dist) + reg_term
                    grand_loss = gamma_stealth * grand_loss

                # Trigger regularizers (optional, default 0) -- same terms/names as
                # federated_optimizing_trigger_policy, reused unchanged.
                L_pen = trigger_penalty_hinge(delta, mu, mu_source, kappa)
                L_tv = tv_loss(delta)
                grand_loss = grand_loss + (
                    lambda_penalty * L_pen + lambda_delta * delta.norm() + lambda_tv * L_tv
                )

                # Optimize labels and trigger
                optimizer_labels.zero_grad()
                grand_loss.backward()
                optimizer_labels.step()
                optimizer_delta.step()

                with torch.no_grad():
                    delta.clamp_(-epsilon, epsilon)

                L_bd_mean = (L_bd_sum / n_bd_valid).item() if n_bd_valid > 0 else 0.0
                losses.append(grand_loss.item())
                pbar.update(batch_size)
                pbar.set_postfix(
                    g_loss=f"{np.mean(losses[-20:]):.4g}",
                    backdoor_loss=f"{(param_loss/param_dist).item():.4g}" if attack in ["backdoor"] else "N/A",
                    reg=f"{reg_term.item():.4g}",
                    L_bd=f"{L_bd_mean:.4g}",
                    delta_norm=f"{delta.detach().norm().item():.4g}"
                )

    # Save results
    print("Saving results...")
    y_true = torch.stack([mtt_dataset[i][3].detach() for i in range(len(mtt_dataset.distill))])
    np.save(output_dir + "labels.npy", labels_syn.detach().numpy())
    np.save(output_dir + "true.npy", y_true)
    np.save(output_dir + "losses.npy", losses)

    run_tag = f"{num_poisoned}vs{num_honests}"
    Path(output_dir_trigger).mkdir(parents=True, exist_ok=True)
    trig_path = (
        Path(output_dir_trigger)
        / f"opt_trig_direct_{init}_{expert_model_flag}_{dataset_flag}_{run_tag}.pt"
    )
    torch.save(delta.detach().cpu(), trig_path)
    print(f"Saved trigger to {trig_path}")


if __name__ == "__main__":
    experiment_name, module_name = sys.argv[1], sys.argv[2]
    run(experiment_name, module_name)
