from modules.base_utils.datasets import get_n_classes
from modules.base_utils.util import (
    get_train_info,
    mini_train,
    load_model,
    either_dataloader_dataset_to_both,
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
    compute_v_polytope_distance,
    compute_beta_star,
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
from modules.train_expert.utils import checkpoint_callback
from modules.base_utils.experiment_tracker import ExperimentTracker
import torch
from torch.utils.data import ConcatDataset, Subset
import numpy as np
from pathlib import Path
import os
import json
import matplotlib.pyplot as plt
import copy

WINDOW_SIZE = 50


def build_loader(dataset, batch_size):
    loader, _ = either_dataloader_dataset_to_both(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    return loader


def _compute_step(
    batch,
    expert_models,
    sampled_k,
    source_label,
    target_label,
    loss_fn,
    delta,
    device,
    dataset_flag,
    model_flag,
    lambda_poison,
    n_classes,
    flip_grad_cache,
    class_samples_raw,
    pi,
    beta,
    flip_qp_ridge,
    checkpoint_backward,
    lambda_b1,
    lambda_b2,
    lambda_bd,
    beta_star_grid=None,
    beta_star_k=None,
):
    '''Per sampled checkpoint theta_k:

        g_c  = grad_theta loss(x, y)                          detached
        mu_p = grad_theta loss(x_poisoned, y_poisoned)         create_graph=True
               (a lambda_poison fraction of the batch's source-class examples
               are triggered and relabeled to target_label)
        v = mu_p - g_c

        B1_k = ||v||^2 / (||mu_p||^2 + eps)                    magnitude of the
                                                                 poisoning-induced
                                                                 gradient shift
        B2_k = dist(v, W_beta)^2 / (||v||^2 + eps)              feasibility of v
                                                                 as a label-flip
                                                                 mixture (in [0,1])

    Both are averaged over sampled_k. B1/B2 are purely a per-checkpoint
    gradient-geometry comparison between "what the trigger does" (mu_p) and
    "what it would take to get there by flipping labels" (the feasible
    polytope W_beta).

    If beta_star_grid is non-empty, additionally computes (once per batch, from
    ONLY the `beta_star_k` checkpoint's v -- not averaged over sampled_k, which
    would multiply the cost by num_chckpt and stop being negligible) the
    normalized-B2-vs-beta curve and beta_star (see `compute_beta_star`).
    `beta_star_k` is rotated round-robin across sampled_k from one batch to
    the next by the caller (`optimize_trigger_step`), rather than always
    being the last checkpoint sampled -- same per-batch cost, but yields a
    distribution over checkpoints instead of a single noisy one.
    '''
    eps_den = 1e-8
    n_exp = len(sampled_k)

    x_raw, y = move_to_device(batch, device)
    n_b = x_raw.shape[0]
    x_clean = raw_to_preprocess(x_raw, dataset_flag=dataset_flag, model_flag=model_flag)

    # mask / y_poison (which examples get poisoned, and to which label) are
    # fixed for the whole batch/step -- shared across every sampled checkpoint.
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

    # Actually-realized poison fraction of this batch, post idx_source.numel()
    # capping above -- can fall short of lambda_poison whenever lambda_poison
    # > the batch's naturally-source-labeled fraction (~= pi_source); see
    # optimize_trigger's source_duplication and its 5%-deviation warning.
    lambda_effective = mask.sum().item() / n_b if n_b > 0 else 0.0

    B1_sum, B2_sum, bd_loss_sum = None, None, None
    n_valid = 0
    beta_star_v, beta_star_G, beta_star_Q, beta_star_pairs = None, None, None, None

    for k in sampled_k:
        M = expert_models[k].to(device).eval()
        params = list(M.parameters())

        # x_poisoned is rebuilt fresh per checkpoint (cheap: clone + masked
        # elementwise add, not a forward pass) rather than shared across the
        # sampled_k loop: under checkpoint_backward=True, freeing checkpoint
        # k's graph would otherwise free the delta -> x_poisoned subgraph that
        # every other checkpoint's forward pass also depends on, since it's
        # the same tensor object -- causing "backward through the graph a
        # second time" from the second checkpoint onward.
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

        den1 = mu_p.detach().norm() ** 2 + eps_den
        B1_k = (v ** 2).sum() / den1

        if k in flip_grad_cache:
            G_k, Q_k, pairs_k = flip_grad_cache[k]
        else:
            G_k, Q_k, pairs_k = compute_expected_flip_gradients(
                M, loss_fn, class_samples_raw, n_classes, pi,
                dataset_flag=dataset_flag, model_flag=model_flag, params=params,
            )
            flip_grad_cache[k] = (G_k, Q_k, pairs_k)

        dist2, _ = compute_v_polytope_distance(
            v, G_k, Q_k, pairs_k, beta, ridge=flip_qp_ridge,
        )
        den2 = v.detach().norm() ** 2 + eps_den
        B2_k = dist2 / den2

        if beta_star_grid and k == beta_star_k:
            beta_star_v, beta_star_G = v.detach(), G_k
            beta_star_Q, beta_star_pairs = Q_k, pairs_k

        # L_bd (backdoor loss): CE restricted to the actually-triggered examples
        # (mask, post lambda_poison subsampling) -- NOT the whole batch, whose
        # untriggered examples keep y_poison == y and would just add ordinary
        # clean-classification loss unrelated to the backdoor objective.
        L_bd_k = (
            loss_fn(logits_p[mask], y_poison[mask])
            if mask.sum() > 0 else torch.tensor(0.0, device=device)
        )

        if checkpoint_backward:
            step_loss = (lambda_b1 * B1_k + lambda_b2 * B2_k + lambda_bd * L_bd_k) / n_exp
            step_loss.backward()
            B1_k, B2_k, L_bd_k = B1_k.detach(), B2_k.detach(), L_bd_k.detach()

        if B1_sum is None:
            B1_sum, B2_sum, bd_loss_sum = B1_k, B2_k, L_bd_k
        else:
            B1_sum = B1_sum + B1_k
            B2_sum = B2_sum + B2_k
            bd_loss_sum = bd_loss_sum + L_bd_k
        n_valid += 1

    B1 = B1_sum / n_valid
    B2 = B2_sum / n_valid
    L_bd = bd_loss_sum / n_valid

    beta_star_curve, beta_star = None, None
    if beta_star_grid:
        # beta_star_v/G/Q/pairs are the single beta_star_k checkpoint's
        # values, captured during the loop above -- intentional (see
        # docstring): a single checkpoint's worth of QP solves per batch
        # keeps this negligible relative to the num_chckpt-checkpoint B1/B2
        # computation above.
        beta_star_curve, beta_star = compute_beta_star(
            beta_star_v, beta_star_G, beta_star_Q, beta_star_pairs,
            beta_star_grid, ridge=flip_qp_ridge,
        )

    return {
        "B1": B1, "B2": B2, "L_bd": L_bd, "lambda_effective": lambda_effective,
        "beta_star_curve": beta_star_curve, "beta_star": beta_star,
    }


def optimize_trigger_step(
    expert_models,
    loader,
    source_label,
    target_label,
    loss_fn,
    delta,
    mu,
    mu_source,
    optimizer_delta,
    lambda_bd,
    lambda_penalty,
    lambda_delta,
    lambda_tv,
    kappa,
    alpha_ckpt,
    num_chckpt,
    epsilon,
    lambda_poison,
    n_classes,
    class_samples_raw,
    pi,
    beta,
    flip_qp_ridge,
    lambda_b1,
    lambda_b2,
    run_tag,
    device="cuda",
    dataset_flag="cifar",
    init="stripe",
    model_flag="r32p",
    checkpoint_backward=True,
    beta_star_grid=None,
):
    '''Runs one outer step's worth of trigger-optimization batches against a
    fixed set of expert checkpoints (see `_compute_step` for the objective).
    `beta` and `run_tag` (used only for the output filename) are both derived
    upstream, in `optimize_trigger`.
    '''
    sampled_k = sample_checkpoints(
        len(expert_models),
        num_chckpt,
        alpha=alpha_ckpt,
        device=device,
    )

    # G/Q depend only on (checkpoint, dataset), not on the trigger: expert_models
    # is fixed for the whole call, so a fresh cache here is reused across every
    # batch of this call and discarded once these checkpoints (this outer
    # training step's experts) are replaced.
    flip_grad_cache = {}

    total_steps = len(loader)
    pbar = make_pbar(
        loader,
        total=total_steps,
        desc="Optimizing trigger",
        leave=False,
    )

    hinge_window = []
    metrics_history = {"B1": [], "B2": [], "L_bd": [], "lambda_effective": []}
    last_beta_star_curve, last_beta_star = None, None
    beta_star_history = {"beta_star": [], "beta_star_curve": []} if beta_star_grid else None
    # Round-robin cursor into sampled_k for compute_beta_star: batch i uses
    # sampled_k[i % len(sampled_k)], instead of always the last checkpoint --
    # same per-batch cost, but sweeps every sampled checkpoint over enough
    # batches instead of always reading off one (noisy) checkpoint.
    beta_star_cursor = 0

    for batch in pbar:
        optimizer_delta.zero_grad()

        beta_star_k = None
        if beta_star_grid:
            beta_star_k = sampled_k[beta_star_cursor % len(sampled_k)]
            beta_star_cursor += 1

        result = _compute_step(
            batch, expert_models, sampled_k, source_label, target_label, loss_fn, delta,
            device, dataset_flag, model_flag, lambda_poison, n_classes,
            flip_grad_cache, class_samples_raw, pi, beta, flip_qp_ridge,
            checkpoint_backward, lambda_b1, lambda_b2, lambda_bd,
            beta_star_grid=beta_star_grid, beta_star_k=beta_star_k,
        )
        B1, B2, L_bd = result["B1"], result["B2"], result["L_bd"]
        beta_star_curve, beta_star = result["beta_star_curve"], result["beta_star"]

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
                lambda_b1 * B1
                + lambda_b2 * B2
                + lambda_bd * L_bd
                + lambda_penalty * L_pen
                + lambda_delta * delta.norm()
                + lambda_tv * L_tv
            )
            L_tot.backward()

        optimizer_delta.step()

        with torch.no_grad():
            delta.clamp_(-epsilon, epsilon)

        hinge_window.append(L_pen.item() > 0)
        if len(hinge_window) > WINDOW_SIZE:
            hinge_window.pop(0)

        metrics_history["B1"].append(B1.item())
        metrics_history["B2"].append(B2.item())
        metrics_history["L_bd"].append(L_bd.item())
        metrics_history["lambda_effective"].append(result["lambda_effective"])
        if beta_star_grid:
            last_beta_star_curve, last_beta_star = beta_star_curve, beta_star
            beta_star_history["beta_star"].append(beta_star)
            beta_star_history["beta_star_curve"].append(beta_star_curve)

        postfix = {
            "B1": f"{B1.item():.6f}",
            "B2": f"{B2.item():.6f}",
            "L_bd": f"{L_bd.item():.4f}",
            "lambda_eff": f"{result['lambda_effective']:.4f}",
            "L_pen": f"{L_pen.item():.4f}",
            "hinge_rate": f"{sum(hinge_window) / len(hinge_window):.2f}",
            "||delta||": f"{delta.norm().item():.4f}",
        }
        if beta_star_grid:
            postfix["beta*"] = f"{beta_star:.4g}" if beta_star is not None else "None"
        pbar.set_postfix(postfix)

    delta_img = delta.detach().cpu().numpy().transpose(1, 2, 0)
    delta_img = (delta_img - delta_img.min()) / (
        delta_img.max() - delta_img.min() + 1e-8
    )
    plt.imshow(delta_img)
    plt.title("Optimized Trigger (Delta)")
    plt.axis("off")
    os.makedirs("out/optimizing_trigger", exist_ok=True)
    plt.savefig(
        f"out/optimizing_trigger/opt_trig_{init}_{model_flag}_{dataset_flag}_{run_tag}.png"
    )
    plt.close()

    delta_save = delta.detach().cpu()
    os.makedirs("optimized_trigger", exist_ok=True)
    torch.save(
        delta_save,
        f"optimized_trigger/opt_trig_{init}_{model_flag}_{dataset_flag}_{run_tag}.pt",
    )

    step_summary = {
        key: (sum(vals) / len(vals) if vals else None)
        for key, vals in metrics_history.items()
    }
    if beta_star_grid:
        # Last batch's curve/beta_star, not averaged across batches (matches
        # the existing clean_acc/poison_acc "last value of the step" convention).
        step_summary["beta_star_curve"] = last_beta_star_curve
        step_summary["beta_star"] = last_beta_star
        # Full per-batch history (one entry per batch, round-robin over
        # sampled_k -- see beta_star_cursor above): a distribution over
        # checkpoints instead of a single noisy last-checkpoint reading.
        step_summary["beta_star_distribution"] = beta_star_history["beta_star"]
        curves = beta_star_history["beta_star_curve"]
        step_summary["beta_star_curve_mean"] = [
            sum(vals) / len(vals) for vals in zip(*curves)
        ] if curves else None
    return delta, step_summary


def optimize_trigger(
    model,
    loss_fn,
    dataset_flag,
    mu,
    mu_source,
    source_label,
    target_label,
    lambda_bd=0.0,
    lambda_penalty=0.1,
    lambda_delta=0.01,
    lambda_b1=0.0,
    lambda_b2=0.0,
    lambda_tv=0.0,
    kappa=0.0,
    alpha_ckpt=0.1,
    num_chckpt=4,
    epsilon=0.03,
    lr_delta=1e-2,
    n_steps=100,
    device="cuda",
    train_flag="sgd",
    batch_size=None,
    epochs=20,
    optim_kwargs={},
    scheduler_kwargs={},
    expert_config={},
    expert_path="/shared/data1/Projects/DLWP/j1067582/martin/FLIP/out/checkpoints/r32p_1xs/{}/model_{}_{}.pth",
    chkpt_iters=50,
    output_dir="/shared/data1/Projects/DLWP/j1067582/martin/FLIP/out/checkpoints/r32p_1xs/0/",
    init="stripe",
    num_honests=5,
    num_poisoned=5,
    model_flag="r32p",
    output_dir_trigger="/shared/data1/Projects/DLWP/j1067582/martin/FLIP/optimized_trigger",
    restart=False,
    beta=None,
    flip_budget=None,
    lambda_poison=None,
    lambda_overflow="clip",
    source_duplication=False,
    checkpoint_backward=True,
    batch_size_trigger=256,
    flip_qp_ridge=1e-6,
    flip_gradient_samples_per_class=64,
    metrics_log_path=None,
    beta_star_grid=None,
    tracker=None,
):

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(output_dir_trigger).mkdir(parents=True, exist_ok=True)

    # run_tag: only used for output filenames, computed here (not inside
    # optimize_trigger_step, which no longer knows about num_honests/num_poisoned).
    run_tag = f"{num_poisoned}vs{num_honests}"

    trig_path = Path(output_dir_trigger).joinpath(
        f"opt_trig_{init}_{model_flag}_{dataset_flag}_{run_tag}.pt"
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

    n_w = num_honests + num_poisoned

    # beta -- the fraction of the attacker's OWN shard it can afford to flip
    # -- is the primary parameter: it is a property of the attacker alone,
    # not of the federated deployment it will eventually be used against
    # (see `run`'s docstring). num_honests/num_poisoned/flip_budget below
    # exist only to translate beta into a "number of flips per round" for
    # human-readable logging/run naming under one particular assumed
    # deployment size; they play no role in the trigger objective itself.
    beta, flip_budget, lambda_poison = resolve_beta_and_lambda_poison(
        beta, flip_budget, lambda_poison, num_poisoned, num_honests, n_train,
    )

    # pi (class frequencies) depends only on the dataset (not on the
    # checkpoint or the trigger): computed once and reused both below (for
    # source_duplication) and further down for the B2 objective.
    pi = compute_class_frequencies(dataset_flag, n_classes)
    pi_source = pi[source_label]

    trigger_opt_dataset = raw_train_dataset
    if source_duplication and lambda_poison > pi_source:
        # _compute_step's per-batch masking caps target_count at
        # idx_source.numel() (the batch's naturally-source-labeled examples,
        # ~= pi_source * n_b): lambda_poison above pi_source can never
        # actually be realized per batch no matter how high it's set. Fix:
        # duplicate (with replacement) the same n_add extra source-class
        # examples get_poison_dataset's lambda_overflow="duplicate" would add
        # to the expert's train set at lambda_target=lambda_poison (same
        # ratio), so idx_source is large enough for target_count to reach
        # round(lambda_poison * n_b) instead of being capped.
        n_add = round(lambda_poison * n_train / (1 - lambda_poison))
        labels = np.array([y for _, y in raw_train_dataset.dataset])
        source_indices = np.where(labels == source_label)[0]
        dup_rng = np.random.RandomState(0)
        dup_indices = dup_rng.choice(source_indices, size=n_add, replace=True)
        trigger_opt_dataset = ConcatDataset(
            [raw_train_dataset, Subset(raw_train_dataset, dup_indices)]
        )
        print(
            f"[optimize_trigger] source_duplication: lambda_poison={lambda_poison:.6f} "
            f"> pi_source={pi_source:.6f}; duplicated {n_add} extra source-class "
            "examples into the trigger-optimization loader (same n_add ratio as the "
            "expert's lambda_overflow='duplicate' train set)."
        )

    loader = build_loader(trigger_opt_dataset, batch_size=batch_size_trigger)

    # class_samples_raw depends only on the dataset -- computed once and
    # reused across every step/checkpoint.
    class_samples_raw = get_class_conditional_samples(
        dataset_flag, n_classes, flip_gradient_samples_per_class, device
    )

    checkpoints_start = extract_experts(expert_config, expert_path)

    big_ims = needs_big_ims(model_flag)

    history = [] if metrics_log_path else None

    for step in range(n_steps):
        print(f"\n=== Trigger optimization step {step + 1}/{n_steps} ===")

        delta_eval = delta.clone().detach().cpu()

        batch_size_, epochs_, opt, lr_scheduler = get_train_info(
            model.parameters(),
            train_flag,
            batch_size=batch_size,
            epochs=epochs,
            optim_kwargs=optim_kwargs,
            scheduler_kwargs=scheduler_kwargs,
        )

        # lambda_target=lambda_poison couples the expert's actual retraining
        # poison rate to the rate the trigger objective assumes (both equal
        # beta by default) -- see get_poison_dataset's docstring.
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

        # include_clean=False, lambda_target=None: every source-class test
        # example is triggered and relabeled, none are left clean. This is
        # the strict ASR test set -- with include_clean=True (the default),
        # poison_acc would be measured on a ConcatDataset dominated by
        # untriggered clean examples and would mostly track clean accuracy
        # instead of attack success. lambda_target is deliberately NOT
        # applied here (unlike poison_train_dataset above): ASR should
        # reflect success on the whole source class, not on whatever
        # fraction the training regime happened to poison.
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

        delta, step_summary = optimize_trigger_step(
            expert_models=expert_models,
            loader=loader,
            source_label=source_label,
            target_label=target_label,
            loss_fn=loss_fn,
            delta=delta,
            mu=mu,
            mu_source=mu_source,
            optimizer_delta=optimizer_delta,
            lambda_bd=lambda_bd,
            lambda_penalty=lambda_penalty,
            lambda_delta=lambda_delta,
            lambda_tv=lambda_tv,
            kappa=kappa,
            alpha_ckpt=alpha_ckpt,
            num_chckpt=num_chckpt,
            epsilon=epsilon,
            lambda_poison=lambda_poison,
            n_classes=n_classes,
            class_samples_raw=class_samples_raw,
            pi=pi,
            beta=beta,
            flip_qp_ridge=flip_qp_ridge,
            lambda_b1=lambda_b1,
            lambda_b2=lambda_b2,
            run_tag=run_tag,
            device=device,
            dataset_flag=dataset_flag,
            init=init,
            model_flag=model_flag,
            checkpoint_backward=checkpoint_backward,
            beta_star_grid=beta_star_grid,
        )

        del expert_models
        torch.cuda.empty_cache()

        # lambda_effective: mean, over this step's batches, of mask.sum()/n_b
        # -- the poison fraction the objective actually realized per batch,
        # as opposed to lambda_poison, the fraction it targeted. These can
        # diverge whenever lambda_poison > pi_source (see source_duplication
        # above): _compute_step caps target_count at the batch's naturally-
        # source-labeled count.
        lambda_effective = step_summary.get("lambda_effective")
        if lambda_effective is not None:
            rel_dev = (
                abs(lambda_effective - lambda_poison) / lambda_poison
                if lambda_poison > 0 else float("inf")
            )
            print(
                f"lambda_effective={lambda_effective:.6f} (target lambda_poison="
                f"{lambda_poison:.6f}, relative deviation={rel_dev:.2%})"
            )
            if rel_dev > 0.05:
                print(
                    f"WARNING: lambda_effective deviates from lambda_poison by "
                    f"{rel_dev:.2%} (> 5%). pi_source={pi_source:.6f} -- if "
                    f"lambda_poison > pi_source, set source_duplication=True to "
                    "close this gap."
                )

        if history is not None:
            history.append({
                "step": step,
                "clean_acc": clean_acc,
                "poison_acc": poison_acc,
                **step_summary,
            })
        if tracker is not None:
            tracker.log(
                step,
                clean_acc=clean_acc,
                poison_acc=poison_acc,
                **{
                    k: v for k, v in step_summary.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                },
            )

    if metrics_log_path:
        Path(metrics_log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_log_path, "w") as f:
            json.dump(history, f, indent=2)

    return delta.detach()


def run(experiment_name, module_name, **kwargs):
    """
    Optimizes and saves a backdoor trigger (delta): per sampled expert
    checkpoint, compares the gradient shift a poisoned batch induces to the
    feasible gradient polytope spanned by expected label-flip directions.

    Threat model -- exactly what the attacker needs to know to run this:
      - the model architecture (model/model_flag)
      - a sample from the training distribution (used to build class-
        conditional gradient estimates and to retrain the expert; it does
        NOT need to be the actual data any deployment's honest participants
        hold)
      - beta: its own capacity, the fraction of its own data shard it is
        able/willing to flip
      - y_source, y_target: the backdoor's source and target labels

    What this module does NOT use, and the attacker does not need to know:
      - any aggregation rule a downstream federated deployment might use
      - n_w / n_mal, the number of honest/malicious participants in such a
        deployment -- num_honests/num_poisoned below exist only to convert
        beta into a human-readable flip_budget for logging/run naming under
        one assumed deployment size; they do not affect the objective
      - the honest participants' data
    """

    slurm_id = kwargs.get("slurm_id", None)
    args = extract_toml(experiment_name, module_name)
    tracker = ExperimentTracker(experiment_name, module_name, args, slurm_id=slurm_id)

    dataset_flag = args["dataset"]
    model_flag = args["model"]
    y_source = args["source_label"]
    y_target = args["target_label"]

    num_honests = args.get("num_honests", 5)
    num_poisoned = args.get("num_poisoned", 5)

    # L_bd (backdoor CE loss on triggered examples) is satisfied by
    # construction: mini_train retrains the expert on poison_train_dataset
    # against the CURRENT delta every outer step, so pushing further on L_bd
    # w.r.t. delta yields no useful shaping signal. Defaults to 0.0; still
    # computed every batch for logging/diagnostics (see _compute_step).
    lambda_bd = args.get("lambda_bd", 0.0)
    lambda_penalty = args.get("lambda_penalty", 0.0)
    lambda_delta = args.get("lambda_delta", 0.0)

    epsilon = args.get("epsilon", 0.1)
    lr_delta = args.get("lr_delta", 1e-2)
    n_steps = args.get("n_steps", 100)

    alpha_ckpt = args.get("alpha_ckpt", 0.01)
    num_chckpt = args.get("num_chckpt", 15)
    restart = args.get("restart", False)
    expert_config = args.get("expert_config", {})
    expert_path = args.get("expert_path", None)

    device = args.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    init = args.get("init", "stripe")

    lambda_b1 = args.get("lambda_b1", 0.0)
    lambda_b2 = args.get("lambda_b2", 0.0)
    # beta (primary) vs flip_budget (legacy, derives beta): optimize_trigger
    # raises if both or neither are given -- see its docstring/resolution.
    beta = args.get("beta", None)
    flip_budget = args.get("flip_budget", None)
    lambda_poison = args.get("lambda_poison", "beta")
    lambda_overflow = args.get("lambda_overflow", "clip")
    source_duplication = args.get("source_duplication", False)
    kappa = args.get("kappa", 0.0)
    lambda_tv = args.get("lambda_tv", 0.0)
    checkpoint_backward = args.get("checkpoint_backward", True)
    batch_size_trigger = args.get("batch_size_trigger", 256)
    flip_qp_ridge = args.get("flip_qp_ridge", 1e-6)
    flip_gradient_samples_per_class = args.get("flip_gradient_samples_per_class", 64)
    epochs = args.get("epochs", 20)
    metrics_log_path = args.get("metrics_log_path", None)
    if metrics_log_path is not None:
        metrics_log_path = slurmify_path(metrics_log_path, slurm_id)
    beta_star_grid = args.get("beta_star_grid", [])

    optim_kwargs = args.get("optim_kwargs", {})
    scheduler_kwargs = args.get("scheduler_kwargs", {})

    output_dir = slurmify_path(args["output_dir"], slurm_id)
    output_dir_trigger = slurmify_path(
        args.get(
            "output_dir_trigger",
            "/shared/data1/Projects/DLWP/j1067582/martin/FLIP/optimized_trigger",
        ),
        slurm_id,
    )

    if epochs < 5 and "out/checkpoints" in Path(output_dir).as_posix():
        raise ValueError(
            f"epochs={epochs} < 5 with output_dir={output_dir!r} pointing under "
            "out/checkpoints/: this looks like a smoke/debug run about to overwrite "
            "real training checkpoints (mini_train's checkpoint_callback writes into "
            "output_dir every step). Use a temporary output_dir "
            "(e.g. tempfile.mkdtemp()) for short/debug runs, or set epochs >= 5 for "
            "a real training run."
        )

    print(f"Output directory: {output_dir}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    n_classes = get_n_classes(dataset_flag)
    model = load_model(model_flag, n_classes).to(device)

    print("Model loaded on device:", device)

    model.eval()

    mu = get_mu(dataset_flag, y_target, device, model_flag=model_flag)
    mu_source = get_mu(dataset_flag, y_source, device, model_flag=model_flag)

    print("Optimizing trigger...")
    loss_fn = torch.nn.CrossEntropyLoss()
    optimized_delta = optimize_trigger(
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
        alpha_ckpt=alpha_ckpt,
        num_chckpt=num_chckpt,
        epsilon=epsilon,
        lr_delta=lr_delta,
        n_steps=n_steps,
        epochs=epochs,
        device=device,
        expert_config=expert_config,
        expert_path=expert_path,
        output_dir=output_dir,
        output_dir_trigger=output_dir_trigger,
        init=init,
        optim_kwargs=optim_kwargs,
        scheduler_kwargs=scheduler_kwargs,
        model_flag=model_flag,
        restart=restart,
        lambda_b1=lambda_b1,
        lambda_b2=lambda_b2,
        beta=beta,
        flip_budget=flip_budget,
        lambda_poison=lambda_poison,
        lambda_overflow=lambda_overflow,
        source_duplication=source_duplication,
        kappa=kappa,
        lambda_tv=lambda_tv,
        checkpoint_backward=checkpoint_backward,
        batch_size_trigger=batch_size_trigger,
        flip_qp_ridge=flip_qp_ridge,
        flip_gradient_samples_per_class=flip_gradient_samples_per_class,
        metrics_log_path=metrics_log_path,
        tracker=tracker,
        beta_star_grid=beta_star_grid,
    )

    print("Optimized trigger obtained.")

    os.makedirs("out/optimizing_trigger", exist_ok=True)
    delta_img = optimized_delta.detach().cpu().numpy().transpose(1, 2, 0)
    delta_img = (delta_img - delta_img.min()) / (
        delta_img.max() - delta_img.min() + 1e-8
    )
    plt.imshow(delta_img)
    plt.title("Optimized Trigger (Delta)")
    plt.axis("off")
    plt.savefig(
        f"out/optimizing_trigger/opt_trig_{init}_{model_flag}_{dataset_flag}_{num_poisoned}vs{num_honests}.png"
    )

    Path(output_dir_trigger).mkdir(parents=True, exist_ok=True)
    torch.save(
        optimized_delta.detach().cpu(),
        Path(output_dir_trigger)
        / f"opt_trig_{init}_{model_flag}_{dataset_flag}_{num_poisoned}vs{num_honests}.pt",
    )

    tracker.finalize()


if __name__ == "__main__":
    run("optimizing_trigger_example", "optimizing_trigger")
