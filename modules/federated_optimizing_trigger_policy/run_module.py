from modules.base_utils.datasets import get_n_classes
from modules.base_utils.util import (
    get_train_info,
    mini_train,
    load_model,
    extract_toml,
    slurmify_path,
    make_pbar,
    needs_big_ims,
)
from modules.federated_optimizing_trigger.utils import (
    sample_checkpoints,
    compute_batch_gradients,
    trigger_penalty_hinge,
    tv_loss,
    compute_expected_flip_gradients,
    compute_class_frequencies,
    get_class_conditional_samples,
    get_mu,
    extract_experts,
    get_clean_dataset,
    get_poison_dataset,
    move_to_device,
    init_delta,
    raw_to_preprocess,
    raw_to_trigger_preprocess,
    get_raw_clean_dataset,
    resolve_beta_and_lambda_poison,
)
from modules.federated_optimizing_trigger.run_module import build_loader
from modules.federated_optimizing_trigger_policy.utils import init_policy, project_policy_budget
from modules.train_expert.utils import checkpoint_callback
import torch
from torch.utils.data import ConcatDataset, Subset
import numpy as np
from pathlib import Path
import os
import json
import matplotlib.pyplot as plt
import copy

WINDOW_SIZE = 50


def _compute_step_policy(
    batch,
    expert_models,
    sampled_k,
    source_label,
    target_label,
    loss_fn,
    delta,
    u,
    device,
    dataset_flag,
    model_flag,
    lambda_poison,
    n_classes,
    flip_grad_cache,
    class_samples_raw,
    pi,
    checkpoint_backward,
    lambda_bd,
):
    '''Per sampled checkpoint theta_k, the (P^mean) objective's two terms:

        v = mu_p - g_c                                     poisoning-induced gradient shift
                                                             (identical to `_compute_step` in
                                                             federated_optimizing_trigger)
        Gu = Gbar_k @ u                                     policy-reachable gradient shift,
                                                             Gbar_k = compute_expected_flip_
                                                             gradients' G (already the mean-
                                                             aggregation-consistent, pi_y-
                                                             weighted expected direction a
                                                             label-flipping policy induces)

        B2_k = ||Gu - v||^2 / (||v||^2 + eps)               alignment between what the trigger
                                                             does and what the learned policy u
                                                             can reproduce -- same structure as
                                                             federated_optimizing_trigger's B2
                                                             (compute_v_polytope_distance), but
                                                             u is a jointly-learned parameter
                                                             instead of the QP optimum w*.
        L_bd_k = CE(f_theta_k(T_delta(x)), y_target)        backdoor loss on triggered examples

    Both are averaged over sampled_k, matching `_compute_step`'s convention (same keys: B2,
    L_bd, lambda_effective) so the two threat models' per-step metrics stay directly
    comparable.
    '''
    eps_den = 1e-8
    n_exp = len(sampled_k)

    x_raw, y = move_to_device(batch, device)
    n_b = x_raw.shape[0]
    x_clean = raw_to_preprocess(x_raw, dataset_flag=dataset_flag, model_flag=model_flag)

    mask = y == source_label
    y_poison = y.clone()
    has_poison = bool(mask.sum().item() > 0)
    if has_poison:
        idx_source = mask.nonzero(as_tuple=True)[0]
        target_count = min(int(round(lambda_poison * n_b)), idx_source.numel())
        perm = torch.randperm(idx_source.numel(), device=idx_source.device)[:target_count]
        keep = idx_source[perm]
        mask = torch.zeros_like(mask)
        mask[keep] = True
        y_poison[mask] = target_label

    lambda_effective = mask.sum().item() / n_b if n_b > 0 else 0.0

    B2_sum, bd_loss_sum = None, None
    n_valid = 0

    for k in sampled_k:
        M = expert_models[k].to(device).eval()
        params = list(M.parameters())

        # x_poisoned rebuilt fresh per checkpoint -- see federated_optimizing_trigger's
        # `_compute_step` docstring for why (checkpoint_backward frees delta's subgraph).
        x_poisoned = x_clean.clone()
        if has_poison:
            x_poisoned[mask] = raw_to_trigger_preprocess(
                x_raw[mask], delta, dataset_flag=dataset_flag, model_flag=model_flag,
            )

        grads_c, _ = compute_batch_gradients(
            M, loss_fn, (x_clean, y), create_graph=False, retain_graph=False,
        )
        g_c = torch.cat([g.reshape(-1) for g in grads_c]).detach()

        grads_p, logits_p = compute_batch_gradients(
            M, loss_fn, (x_poisoned, y_poison), create_graph=True, retain_graph=True,
        )
        mu_p = torch.cat([g.reshape(-1) for g in grads_p])

        v = mu_p - g_c

        if k in flip_grad_cache:
            G_k, _, _ = flip_grad_cache[k]
        else:
            G_k, Q_k, pairs_k = compute_expected_flip_gradients(
                M, loss_fn, class_samples_raw, n_classes, pi,
                dataset_flag=dataset_flag, model_flag=model_flag, params=params,
            )
            flip_grad_cache[k] = (G_k, Q_k, pairs_k)

        Gu = G_k @ u.to(dtype=G_k.dtype)
        den = v.detach().norm() ** 2 + eps_den
        B2_k = ((Gu - v) ** 2).sum() / den

        # L_bd: CE restricted to the actually-triggered examples only (see
        # federated_optimizing_trigger's `_compute_step` docstring).
        L_bd_k = (
            loss_fn(logits_p[mask], y_poison[mask])
            if mask.sum() > 0 else torch.tensor(0.0, device=device)
        )

        if checkpoint_backward:
            step_loss = (B2_k + lambda_bd * L_bd_k) / n_exp
            step_loss.backward()
            B2_k, L_bd_k = B2_k.detach(), L_bd_k.detach()

        if B2_sum is None:
            B2_sum, bd_loss_sum = B2_k, L_bd_k
        else:
            B2_sum = B2_sum + B2_k
            bd_loss_sum = bd_loss_sum + L_bd_k
        n_valid += 1

    B2 = B2_sum / n_valid
    L_bd = bd_loss_sum / n_valid

    return {"B2": B2, "L_bd": L_bd, "lambda_effective": lambda_effective}


def optimize_trigger_policy_step(
    expert_models,
    loader,
    source_label,
    target_label,
    loss_fn,
    delta,
    u,
    mu,
    mu_source,
    optimizer_delta,
    optimizer_policy,
    lambda_bd,
    lambda_penalty,
    lambda_delta,
    lambda_tv,
    kappa,
    alpha_ckpt,
    num_chckpt,
    epsilon,
    beta,
    lambda_poison,
    n_classes,
    class_samples_raw,
    pi,
    run_tag,
    device="cuda",
    dataset_flag="cifar",
    init="stripe",
    model_flag="r32p",
    checkpoint_backward=True,
):
    '''Runs one outer step's worth of trigger+policy-optimization batches against a fixed set
    of expert checkpoints (see `_compute_step_policy` for the per-checkpoint objective).
    Mirrors federated_optimizing_trigger.run_module.optimize_trigger_step, but replaces its
    QP-projected w* (compute_v_polytope_distance) with the jointly-learned policy u, stepped
    by its own Adam optimizer and projected onto U_beta = {u>=0, sum(u)<=beta} after every
    batch: the discrete/QP label-flip feasibility check becomes an explicit, differentiable
    attack policy, co-trained with delta instead of solved in closed form at each step.
    '''
    sampled_k = sample_checkpoints(
        len(expert_models), num_chckpt, alpha=alpha_ckpt, device=device,
    )

    # G/Q depend only on (checkpoint, dataset), not on delta/u: fresh cache reused across every
    # batch of this call, discarded once these checkpoints are replaced -- same convention as
    # federated_optimizing_trigger's optimize_trigger_step.
    flip_grad_cache = {}

    total_steps = len(loader)
    pbar = make_pbar(
        loader,
        total=total_steps,
        desc="Optimizing trigger+policy",
        leave=False,
    )

    hinge_window = []
    metrics_history = {"B2": [], "L_bd": [], "lambda_effective": [], "beta_used": []}

    for batch in pbar:
        optimizer_delta.zero_grad()
        optimizer_policy.zero_grad()

        result = _compute_step_policy(
            batch, expert_models, sampled_k, source_label, target_label, loss_fn, delta, u,
            device, dataset_flag, model_flag, lambda_poison, n_classes,
            flip_grad_cache, class_samples_raw, pi, checkpoint_backward, lambda_bd,
        )
        B2, L_bd = result["B2"], result["L_bd"]

        L_pen = trigger_penalty_hinge(delta, mu, mu_source, kappa)
        L_tv = tv_loss(delta)

        if checkpoint_backward:
            L_reg = (
                lambda_penalty * L_pen
                + lambda_delta * delta.norm()
                + lambda_tv * L_tv
            )
            L_reg.backward()
        else:
            L_tot = (
                B2
                + lambda_bd * L_bd
                + lambda_penalty * L_pen
                + lambda_delta * delta.norm()
                + lambda_tv * L_tv
            )
            L_tot.backward()

        optimizer_delta.step()
        optimizer_policy.step()

        with torch.no_grad():
            delta.clamp_(-epsilon, epsilon)
            u.copy_(project_policy_budget(u, beta))

        hinge_window.append(L_pen.item() > 0)
        if len(hinge_window) > WINDOW_SIZE:
            hinge_window.pop(0)

        beta_used = u.detach().sum().item()
        metrics_history["B2"].append(B2.item())
        metrics_history["L_bd"].append(L_bd.item())
        metrics_history["lambda_effective"].append(result["lambda_effective"])
        metrics_history["beta_used"].append(beta_used)

        pbar.set_postfix({
            "B2": f"{B2.item():.6f}",
            "L_bd": f"{L_bd.item():.4f}",
            "lambda_eff": f"{result['lambda_effective']:.4f}",
            "L_pen": f"{L_pen.item():.4f}",
            "hinge_rate": f"{sum(hinge_window) / len(hinge_window):.2f}",
            "||delta||": f"{delta.norm().item():.4f}",
            "beta_used": f"{beta_used:.4f}",
        })

    delta_img = delta.detach().cpu().numpy().transpose(1, 2, 0)
    delta_img = (delta_img - delta_img.min()) / (
        delta_img.max() - delta_img.min() + 1e-8
    )
    plt.imshow(delta_img)
    plt.title("Optimized Trigger (Delta) -- policy threat model")
    plt.axis("off")
    os.makedirs("out/optimizing_trigger_policy", exist_ok=True)
    plt.savefig(
        f"out/optimizing_trigger_policy/opt_trig_{init}_{model_flag}_{dataset_flag}_{run_tag}.png"
    )
    plt.close()

    step_summary = {
        key: (sum(vals) / len(vals) if vals else None)
        for key, vals in metrics_history.items()
    }
    return delta, u, step_summary


def optimize_trigger_policy(
    model,
    loss_fn,
    dataset_flag,
    mu,
    mu_source,
    source_label,
    target_label,
    lambda_bd=1.0,
    lambda_penalty=0.0,
    lambda_delta=0.0,
    lambda_tv=0.0,
    kappa=0.0,
    alpha_ckpt=0.1,
    num_chckpt=4,
    epsilon=0.03,
    lr_delta=1e-2,
    lr_policy=1e-2,
    n_steps=100,
    device="cuda",
    train_flag="sgd",
    batch_size=None,
    epochs=20,
    optim_kwargs={},
    scheduler_kwargs={},
    policy_optim_kwargs={},
    expert_config={},
    expert_path=None,
    chkpt_iters=50,
    output_dir=None,
    init="stripe",
    num_honests=5,
    num_poisoned=5,
    model_flag="r32p",
    output_dir_trigger="optimized_trigger",
    output_dir_policy="optimized_policy",
    restart=False,
    beta=None,
    flip_budget=None,
    lambda_poison="beta",
    lambda_overflow="clip",
    source_duplication=False,
    checkpoint_backward=True,
    batch_size_trigger=256,
    flip_gradient_samples_per_class=64,
    metrics_log_path=None,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir_trigger).mkdir(parents=True, exist_ok=True)
    Path(output_dir_policy).mkdir(parents=True, exist_ok=True)

    run_tag = f"{num_poisoned}vs{num_honests}"

    trig_path = Path(output_dir_trigger).joinpath(
        f"opt_trig_policy_{init}_{model_flag}_{dataset_flag}_{run_tag}.pt"
    )
    if restart and trig_path.exists():
        delta = torch.load(trig_path, map_location=device)
    else:
        delta = init_delta(
            mu.shape, horizontal=True, strength=6.0, freq=16, device=device, init=init
        )
    delta.requires_grad_(True)
    optimizer_delta = torch.optim.Adam([delta], lr=lr_delta)

    raw_train_dataset = get_raw_clean_dataset(dataset_flag, train=True)
    n_train = len(raw_train_dataset)
    n_classes = get_n_classes(dataset_flag)

    # beta / lambda_poison resolution, shared with federated_optimizing_trigger so that
    # lambda == beta (the (P^mean) constraint) is enforced identically in both pipelines.
    beta, flip_budget, lambda_poison = resolve_beta_and_lambda_poison(
        beta, flip_budget, lambda_poison, num_poisoned, num_honests, n_train,
    )

    # pi (class frequencies) depends only on the dataset -- computed once, reused below and
    # for the B2-analog objective.
    pi = compute_class_frequencies(dataset_flag, n_classes)
    pi_source = pi[source_label]

    trigger_opt_dataset = raw_train_dataset
    if source_duplication and lambda_poison > pi_source:
        n_add = round(lambda_poison * n_train / (1 - lambda_poison))
        labels = np.array([y for _, y in raw_train_dataset.dataset])
        source_indices = np.where(labels == source_label)[0]
        dup_rng = np.random.RandomState(0)
        dup_indices = dup_rng.choice(source_indices, size=n_add, replace=True)
        trigger_opt_dataset = ConcatDataset(
            [raw_train_dataset, Subset(raw_train_dataset, dup_indices)]
        )

    loader = build_loader(trigger_opt_dataset, batch_size=batch_size_trigger)

    class_samples_raw = get_class_conditional_samples(
        dataset_flag, n_classes, flip_gradient_samples_per_class, device
    )
    # pairs (ordered (y, c) class pairs spanning u) depend only on which classes have samples
    # and n_classes -- same ordering compute_expected_flip_gradients uses for G's columns.
    classes_present = sorted(class_samples_raw.keys())
    pairs = [(y, c) for y in classes_present for c in range(n_classes) if c != y]
    n_pairs = len(pairs)

    u = init_policy(n_pairs, device=device)
    optimizer_policy = torch.optim.Adam([u], lr=lr_policy, **policy_optim_kwargs)

    checkpoints_start = extract_experts(expert_config, expert_path)

    big_ims = needs_big_ims(model_flag)

    history = [] if metrics_log_path else None

    for step in range(n_steps):
        print(f"\n=== Trigger+policy optimization step {step + 1}/{n_steps} ===")

        delta_eval = delta.clone().detach().cpu()

        batch_size_, epochs_, opt, lr_scheduler = get_train_info(
            model.parameters(),
            train_flag,
            batch_size=batch_size,
            epochs=epochs,
            optim_kwargs=optim_kwargs,
            scheduler_kwargs=scheduler_kwargs,
        )

        # lambda_target=lambda_poison=beta: theta_bar_{k+1} = theta_bar_k - eta_k grad(theta_bar_k)
        # is trained at the SAME poison rate the (P^mean) objective assumes -- this is what
        # realizes lambda=beta at the expert-retraining level.
        poison_train_dataset = get_poison_dataset(
            dataset_flag,
            source_label,
            target_label,
            delta_eval,
            train=True,
            big=big_ims,
            lambda_target=lambda_poison,
            lambda_overflow=lambda_overflow,
        )

        clean_test_dataset = get_clean_dataset(dataset_flag, train=False, big=big_ims)
        poison_test_dataset = get_poison_dataset(
            dataset_flag,
            source_label,
            target_label,
            delta_eval,
            train=False,
            big=big_ims,
            include_clean=False,
        )

        mini_train_out = mini_train(
            model=model,
            train_data=poison_train_dataset,
            test_data=[clean_test_dataset, poison_test_dataset],
            batch_size=batch_size_,
            opt=opt,
            scheduler=lr_scheduler,
            epochs=epochs_,
            callback=lambda m, o, e, i: checkpoint_callback(
                m, o, e, i, chkpt_iters, output_dir
            ),
            record=history is not None,
        )
        clean_acc, poison_acc = None, None
        if history is not None:
            _, clean_hist, poison_hist = mini_train_out
            clean_acc = clean_hist[-1][0] if clean_hist else None
            poison_acc = poison_hist[-1][0] if poison_hist else None

        expert_models = []
        for ckpt_path in checkpoints_start:
            M = copy.deepcopy(model).to(device)
            state = torch.load(ckpt_path, map_location=device)
            M.load_state_dict(state)
            M.eval()
            expert_models.append(M)

        delta, u, step_summary = optimize_trigger_policy_step(
            expert_models=expert_models,
            loader=loader,
            source_label=source_label,
            target_label=target_label,
            loss_fn=loss_fn,
            delta=delta,
            u=u,
            mu=mu,
            mu_source=mu_source,
            optimizer_delta=optimizer_delta,
            optimizer_policy=optimizer_policy,
            lambda_bd=lambda_bd,
            lambda_penalty=lambda_penalty,
            lambda_delta=lambda_delta,
            lambda_tv=lambda_tv,
            kappa=kappa,
            alpha_ckpt=alpha_ckpt,
            num_chckpt=num_chckpt,
            epsilon=epsilon,
            beta=beta,
            lambda_poison=lambda_poison,
            n_classes=n_classes,
            class_samples_raw=class_samples_raw,
            pi=pi,
            run_tag=run_tag,
            device=device,
            dataset_flag=dataset_flag,
            init=init,
            model_flag=model_flag,
            checkpoint_backward=checkpoint_backward,
        )

        del expert_models
        torch.cuda.empty_cache() if device == "cuda" else None

        if history is not None:
            history.append({
                "step": step,
                "clean_acc": clean_acc,
                "poison_acc": poison_acc,
                **step_summary,
            })

    if metrics_log_path:
        Path(metrics_log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_log_path, "w") as f:
            json.dump(history, f, indent=2)

    return delta.detach(), u.detach(), pairs, beta, n_train, run_tag


def run(experiment_name, module_name, **kwargs):
    """
    Jointly optimizes a backdoor trigger (delta) and an explicit label-flipping attack
    policy (u), under mean aggregation -- problem (P^mean):

        min_{|delta|_inf<=eps, u in U_beta}
            E_k[ ||Gbar(theta_bar_k) u - v_k(delta)||^2 / rho_k^2 ]
            + kappa * E_k[ E_X[ loss_c(f_theta_bar_k(T_delta(X)), y_target) ] ]
        s.t.  theta_bar_{k+1} = theta_bar_k - eta_k grad(theta_bar_k),  lambda = beta

    Gbar(theta_bar_k) is `compute_expected_flip_gradients`'s G: the pi_y-weighted expected
    per-class-pair gradient direction reachable by a label-flipping policy on the attacker's
    own shard -- exactly the mean-aggregation-consistent reachable set already used by
    federated_optimizing_trigger's B2 term, except here u is a learned policy rather than an
    implicit QP optimum. v_k(delta) is the poisoning-induced gradient shift mu_p - g_c. The
    kappa*L_bd term is the config's `lambda_bd` (kept distinct from `kappa`, which -- as in
    federated_optimizing_trigger -- names the *hinge margin* of the (optional) trigger-vs-mu
    penalty, not this loss weight).

    lambda=beta is enforced exactly as in federated_optimizing_trigger: lambda_poison
    resolves to beta by default (`resolve_beta_and_lambda_poison`), and couples both the
    per-batch poisoning rate in the objective AND the expert's actual retraining poison rate
    (get_poison_dataset's lambda_target).

    Threat model: same as federated_optimizing_trigger (see its `run` docstring) -- the
    attacker needs only the model architecture, a sample from the training distribution, its
    own budget beta, and (y_source, y_target). num_honests/num_poisoned are logging-only.

    Outputs: the optimized trigger (.pt, same naming convention as
    federated_optimizing_trigger) and the optimized policy (u, pairs, beta, n_train -- .npz),
    consumed downstream by `federated_policy_to_flips` to materialize concrete per-worker
    label flips, then by (unmodified) `federated_train_user` to train and evaluate the victim.
    """
    slurm_id = kwargs.get("slurm_id", None)
    args = extract_toml(experiment_name, module_name)

    dataset_flag = args["dataset"]
    model_flag = args["model"]
    y_source = args["source_label"]
    y_target = args["target_label"]

    num_honests = args.get("num_honests", 5)
    num_poisoned = args.get("num_poisoned", 5)

    lambda_bd = args.get("lambda_bd", 1.0)
    lambda_penalty = args.get("lambda_penalty", 0.0)
    lambda_delta = args.get("lambda_delta", 0.0)
    lambda_tv = args.get("lambda_tv", 0.0)
    kappa = args.get("kappa", 0.0)

    epsilon = args.get("epsilon", 0.1)
    lr_delta = args.get("lr_delta", 1e-2)
    lr_policy = args.get("lr_policy", 1e-2)
    n_steps = args.get("n_steps", 100)

    alpha_ckpt = args.get("alpha_ckpt", 0.01)
    num_chckpt = args.get("num_chckpt", 15)
    restart = args.get("restart", False)
    expert_config = args.get("expert_config", {})
    expert_path = args.get("expert_path", None)

    device = args.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    init = args.get("init", "stripe")

    beta = args.get("beta", None)
    flip_budget = args.get("flip_budget", None)
    lambda_poison = args.get("lambda_poison", "beta")
    lambda_overflow = args.get("lambda_overflow", "clip")
    source_duplication = args.get("source_duplication", False)
    checkpoint_backward = args.get("checkpoint_backward", True)
    batch_size_trigger = args.get("batch_size_trigger", 256)
    flip_gradient_samples_per_class = args.get("flip_gradient_samples_per_class", 64)
    epochs = args.get("epochs", 20)
    metrics_log_path = args.get("metrics_log_path", None)
    if metrics_log_path is not None:
        metrics_log_path = slurmify_path(metrics_log_path, slurm_id)

    optim_kwargs = args.get("optim_kwargs", {})
    scheduler_kwargs = args.get("scheduler_kwargs", {})
    policy_optim_kwargs = args.get("policy_optim_kwargs", {})

    output_dir = slurmify_path(args["output_dir"], slurm_id)
    output_dir_trigger = slurmify_path(
        args.get("output_dir_trigger", "optimized_trigger"), slurm_id,
    )
    output_dir_policy = slurmify_path(
        args.get("output_dir_policy", "optimized_policy"), slurm_id,
    )

    if epochs < 5 and "out/checkpoints" in Path(output_dir).as_posix():
        raise ValueError(
            f"epochs={epochs} < 5 with output_dir={output_dir!r} pointing under "
            "out/checkpoints/: this looks like a smoke/debug run about to overwrite real "
            "training checkpoints. Use a temporary output_dir for short/debug runs, or set "
            "epochs >= 5 for a real training run."
        )

    print(f"Output directory: {output_dir}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    n_classes = get_n_classes(dataset_flag)
    model = load_model(model_flag, n_classes).to(device)
    model.eval()

    mu = get_mu(dataset_flag, y_target, device, model_flag=model_flag)
    mu_source = get_mu(dataset_flag, y_source, device, model_flag=model_flag)

    print("Jointly optimizing trigger and attack policy...")
    loss_fn = torch.nn.CrossEntropyLoss()
    delta, u, pairs, beta_resolved, n_train, run_tag = optimize_trigger_policy(
        model=model,
        dataset_flag=dataset_flag,
        loss_fn=loss_fn,
        num_honests=num_honests,
        num_poisoned=num_poisoned,
        mu=mu,
        mu_source=mu_source,
        source_label=y_source,
        target_label=y_target,
        lambda_bd=lambda_bd,
        lambda_penalty=lambda_penalty,
        lambda_delta=lambda_delta,
        lambda_tv=lambda_tv,
        kappa=kappa,
        alpha_ckpt=alpha_ckpt,
        num_chckpt=num_chckpt,
        epsilon=epsilon,
        lr_delta=lr_delta,
        lr_policy=lr_policy,
        n_steps=n_steps,
        epochs=epochs,
        device=device,
        expert_config=expert_config,
        expert_path=expert_path,
        output_dir=output_dir,
        output_dir_trigger=output_dir_trigger,
        output_dir_policy=output_dir_policy,
        init=init,
        optim_kwargs=optim_kwargs,
        scheduler_kwargs=scheduler_kwargs,
        policy_optim_kwargs=policy_optim_kwargs,
        model_flag=model_flag,
        restart=restart,
        beta=beta,
        flip_budget=flip_budget,
        lambda_poison=lambda_poison,
        lambda_overflow=lambda_overflow,
        source_duplication=source_duplication,
        checkpoint_backward=checkpoint_backward,
        batch_size_trigger=batch_size_trigger,
        flip_gradient_samples_per_class=flip_gradient_samples_per_class,
        metrics_log_path=metrics_log_path,
    )

    print("Optimized trigger and policy obtained.")

    os.makedirs("out/optimizing_trigger_policy", exist_ok=True)
    delta_img = delta.detach().cpu().numpy().transpose(1, 2, 0)
    delta_img = (delta_img - delta_img.min()) / (
        delta_img.max() - delta_img.min() + 1e-8
    )
    plt.imshow(delta_img)
    plt.title("Optimized Trigger (Delta) -- policy threat model")
    plt.axis("off")
    plt.savefig(
        f"out/optimizing_trigger_policy/opt_trig_policy_{init}_{model_flag}_{dataset_flag}_{run_tag}.png"
    )
    plt.close()

    Path(output_dir_trigger).mkdir(parents=True, exist_ok=True)
    trig_path = (
        Path(output_dir_trigger)
        / f"opt_trig_policy_{init}_{model_flag}_{dataset_flag}_{run_tag}.pt"
    )
    torch.save(delta.detach().cpu(), trig_path)

    Path(output_dir_policy).mkdir(parents=True, exist_ok=True)
    policy_path = (
        Path(output_dir_policy)
        / f"policy_{init}_{model_flag}_{dataset_flag}_{run_tag}.npz"
    )
    pairs_arr = np.array(pairs, dtype=np.int64)  # (P, 2): columns [y, c]
    np.savez(
        policy_path,
        u=u.detach().cpu().numpy(),
        pairs_y=pairs_arr[:, 0],
        pairs_c=pairs_arr[:, 1],
        beta=np.array(beta_resolved, dtype=np.float64),
        n_train=np.array(n_train, dtype=np.int64),
        source_label=np.array(y_source, dtype=np.int64),
        target_label=np.array(y_target, dtype=np.int64),
    )
    print(f"Saved trigger to {trig_path}")
    print(f"Saved policy to {policy_path}")


if __name__ == "__main__":
    run("optimizing_trigger_policy_example", "federated_optimizing_trigger_policy")
