"""
scripts/check_b2_null_trigger.py -- Task 0 of the federated_optimizing_trigger_policy review.

Empirically checks whether B2 admits a trivial minimum at the null trigger (delta=0). In this
module's convention (rem:no-pi / PIEGE 1, see run_module.py's `_compute_step_policy` docstring),
v = mu_p - g_c reduces to lambda*(g_hat^delta_{s,t} - g_hat_{s,s}); at delta=0 this is exactly
lambda*Gbar_{s,t}, one column of Gbar, reachable by u_triv = beta_local * e_{s,t} -- a policy
that flips exactly beta_local's worth of source-class examples straight to the target class
(feasible iff beta_local <= pi_source, task 1's own A3 condition). If B2 at (delta=0, u_triv) is
already ~=0, the objective has an easy escape hatch that pushes delta toward zero (a genuinely
useless trigger) rather than toward one that induces a hard-to-reach gradient shift.

READING THE RESULT (do not automate a reaction to this -- see above, this script only measures):
    B2_null_triv << B2_current  => the trivial minimum is confirmed: B2, evaluated with the
        run's current co-descended (delta, u), measures mostly what a well-chosen LABEL-FLIP
        policy alone (delta=0) can already reach on the source->target gradient shift -- the
        (P^mean) objective rewards delta for making that shift easier to hit, not necessarily
        for being an effective trigger per se.
    B2_null_triv ~= B2_current or higher => the trigger IS doing real work at this point: no
        LOCAL label-flip policy alone reaches v as well as the current (delta, u) does.
    B2_null_qp <= B2_null_triv (expected, u_triv is just one point in U_loc; B2_null_qp is the
        QP optimum over the whole set) -- a large gap between them means e_{s,t} alone is a
        poor proxy for what U_loc can really do at delta=0, and B2_null_triv understates how
        trivial the delta=0 minimum can get.

Usage:
    python scripts/check_b2_null_trigger.py <experiment_name> [--n-batches 20]

<experiment_name> is the same argument extract_toml expects (experiments/<experiment_name>/
config.toml), pointing at a federated_optimizing_trigger_policy config.
"""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if not torch.cuda.is_available():
    # load_model() calls .cuda() unconditionally -- same portability shim as
    # prelim/check_policy_consistency.py, harmless since real CUDA would never be used here
    # anyway when torch.cuda.is_available() is already False.
    torch.nn.Module.cuda = lambda self, device=None: self.to("cpu")
    torch.Tensor.cuda = lambda self, device=None, non_blocking=False: self.to("cpu")

from modules.base_utils.datasets import get_n_classes
from modules.base_utils.util import extract_toml, load_model
from modules.federated_optimizing_trigger.run_module import build_loader
from modules.federated_optimizing_trigger.utils import (
    compute_batch_gradients,
    compute_class_frequencies,
    compute_expected_flip_gradients,
    extract_experts,
    get_class_conditional_samples,
    get_mu,
    get_raw_clean_dataset,
    init_delta,
    move_to_device,
    raw_to_preprocess,
    resolve_beta_and_lambda_poison,
    sample_checkpoints,
)
from modules.federated_optimizing_trigger_policy import inner_solve
from modules.federated_optimizing_trigger_policy.run_module import _build_poison_mask


# --------------------------------------------------------------------------- #
# Reconstruction -- mirrors optimize_trigger_policy's own startup exactly (same functions,
# same order), loading checkpoints only (no expert retraining) and the trigger/policy from
# their saved outputs if present, per the module's own naming conventions.
# --------------------------------------------------------------------------- #
def build_context(experiment_name, module_name, device):
    args = extract_toml(experiment_name, module_name)

    dataset_flag = args["dataset"]
    model_flag = args["model"]
    source_label = args["source_label"]
    target_label = args["target_label"]
    num_honests = args.get("num_honests", 5)
    num_poisoned = args.get("num_poisoned", 5)
    batch_size_trigger = args.get("batch_size_trigger", 256)
    flip_gradient_samples_per_class = args.get("flip_gradient_samples_per_class", 64)
    expert_config = args.get("expert_config", {})
    expert_path = args["expert_path"]
    output_dir_trigger = args.get("output_dir_trigger", "optimized_trigger")
    output_dir_policy = args.get("output_dir_policy", "optimized_policy")
    init = args.get("init", "stripe")
    num_chckpt = args.get("num_chckpt", 15)
    alpha_ckpt = args.get("alpha_ckpt", 0.01)
    normalization = args.get("normalization", "rho")

    n_classes = get_n_classes(dataset_flag)
    model = load_model(model_flag, n_classes).to(device)
    model.eval()

    raw_train_dataset = get_raw_clean_dataset(dataset_flag, train=True)
    n_train = len(raw_train_dataset)

    beta, _flip_budget, lambda_poison = resolve_beta_and_lambda_poison(
        args.get("beta", None), args.get("flip_budget", None),
        args.get("lambda_poison", "beta"), num_poisoned, num_honests, n_train,
    )
    gamma = num_poisoned / (num_poisoned + num_honests)
    beta_global = gamma * beta
    if args.get("lambda_poison", "beta") == "beta":
        lambda_poison = beta_global

    pi = compute_class_frequencies(dataset_flag, n_classes)
    pi_source = pi[source_label]

    class_samples_raw = get_class_conditional_samples(
        dataset_flag, n_classes, flip_gradient_samples_per_class, device,
    )
    classes_present = sorted(class_samples_raw.keys())
    pairs = [(y, c) for y in classes_present for c in range(n_classes) if c != y]

    loader = build_loader(raw_train_dataset, batch_size=batch_size_trigger)

    checkpoint_paths = extract_experts(expert_config, expert_path)
    expert_models = []
    for ckpt_path in checkpoint_paths:
        M = copy.deepcopy(model).to(device)
        state = torch.load(ckpt_path, map_location=device)
        M.load_state_dict(state)
        M.eval()
        expert_models.append(M)
    print(f"Loaded {len(expert_models)} expert checkpoint(s) from expert_path={expert_path!r}.")

    mu_target = get_mu(dataset_flag, target_label, device, model_flag=model_flag)
    run_tag = f"{num_poisoned}vs{num_honests}"

    trig_path = (
        Path(output_dir_trigger) / f"opt_trig_policy_{init}_{model_flag}_{dataset_flag}_{run_tag}.pt"
    )
    if trig_path.exists():
        delta = torch.load(trig_path, map_location=device).detach()
        print(f"Loaded trigger from {trig_path} (||delta||_inf={delta.abs().max().item():.4f}).")
    else:
        delta = init_delta(
            mu_target.shape, horizontal=True, strength=6.0, freq=16, device=device, init=init,
        ).detach()
        print(f"No saved trigger at {trig_path} -- using a fresh init_delta({init!r}) instead.")

    policy_path = (
        Path(output_dir_policy) / f"policy_{init}_{model_flag}_{dataset_flag}_{run_tag}.npz"
    )
    if policy_path.exists():
        saved = np.load(policy_path)
        saved_pairs = list(zip(saved["pairs_y"].tolist(), saved["pairs_c"].tolist()))
        if saved_pairs != pairs:
            print(
                f"WARNING: saved policy at {policy_path} was optimized against a different "
                "`pairs` ordering than this run's -- ignoring it, using u=0 instead."
            )
            u_current = torch.zeros(len(pairs), device=device)
        else:
            u_current = torch.as_tensor(saved["u"], device=device, dtype=torch.float32)
            print(f"Loaded saved policy from {policy_path} (sum(u)={u_current.sum().item():.6f}).")
    else:
        u_current = torch.zeros(len(pairs), device=device)
        print(f"No saved policy at {policy_path} -- using u=0 instead.")

    return dict(
        dataset_flag=dataset_flag, model_flag=model_flag, source_label=source_label,
        target_label=target_label, n_classes=n_classes, gamma=gamma, beta=beta,
        beta_global=beta_global, lambda_poison=lambda_poison, pi=pi, pi_source=pi_source,
        pairs=pairs, class_samples_raw=class_samples_raw, loader=loader,
        expert_models=expert_models, delta=delta, u_current=u_current,
        num_chckpt=num_chckpt, alpha_ckpt=alpha_ckpt, normalization=normalization,
        loss_fn=torch.nn.CrossEntropyLoss(), device=device,
    )


# --------------------------------------------------------------------------- #
# Step 2 -- B2_current / B2_null_triv / B2_null_qp, reusing inner_solve.build_inner_context +
# inner_solve.aggregate_b2/aggregate_qp/qp_pgd_solve exactly (no reimplementation of B2 itself).
# --------------------------------------------------------------------------- #
def measure_batch(ctx, batch, flip_grad_cache):
    pairs = ctx["pairs"]
    source_label, target_label = ctx["source_label"], ctx["target_label"]

    sampled_k = sample_checkpoints(
        len(ctx["expert_models"]), ctx["num_chckpt"], alpha=ctx["alpha_ckpt"],
        device=ctx["device"],
    )

    x_raw, y = move_to_device(batch, ctx["device"])
    x_clean = raw_to_preprocess(x_raw, dataset_flag=ctx["dataset_flag"], model_flag=ctx["model_flag"])
    mask, y_poison, has_poison, lam_eff, lam_eff_ratio = _build_poison_mask(
        y, x_raw.shape[0], source_label, target_label, ctx["lambda_poison"],
    )
    if not has_poison:
        return None

    contexts_current = inner_solve.build_inner_context(
        ctx["expert_models"], sampled_k, x_clean, y, x_raw, mask, y_poison, has_poison,
        ctx["delta"], ctx["loss_fn"], ctx["dataset_flag"], ctx["model_flag"], ctx["n_classes"],
        flip_grad_cache, ctx["class_samples_raw"], ctx["pi"], ctx["gamma"], ctx["beta"],
        ctx["beta_global"], pairs, ctx["normalization"], ctx["device"],
    )
    delta_zero = torch.zeros_like(ctx["delta"])
    contexts_null = inner_solve.build_inner_context(
        ctx["expert_models"], sampled_k, x_clean, y, x_raw, mask, y_poison, has_poison,
        delta_zero, ctx["loss_fn"], ctx["dataset_flag"], ctx["model_flag"], ctx["n_classes"],
        flip_grad_cache, ctx["class_samples_raw"], ctx["pi"], ctx["gamma"], ctx["beta"],
        ctx["beta_global"], pairs, ctx["normalization"], ctx["device"],
    )

    B2_current = inner_solve.aggregate_b2(ctx["u_current"], contexts_current)

    idx_st = pairs.index((source_label, target_label))
    u_triv = torch.zeros(len(pairs), device=ctx["device"])
    u_triv[idx_st] = ctx["beta"]
    triv_feasible = inner_solve.check_feasible(u_triv, ctx["beta"], pairs, ctx["pi"])
    B2_null_triv = inner_solve.aggregate_b2(u_triv, contexts_null)

    Q_agg, c_agg, pairs_agg = inner_solve.aggregate_qp(contexts_null)
    u_qp, _iters, _obj_start, _obj_end, _converged = inner_solve.qp_pgd_solve(
        Q_agg, c_agg, u_triv, ctx["beta"], pairs_agg, ctx["pi"],
    )
    B2_null_qp = inner_solve.aggregate_b2(u_qp, contexts_null)

    v_over_rho = [
        (c["v"].norm().item() / max(c["rho_k"], 1e-8)) for c in contexts_current
    ]

    return {
        "B2_current": B2_current, "B2_null_triv": B2_null_triv, "B2_null_qp": B2_null_qp,
        "triv_feasible": triv_feasible, "v_over_rho_per_checkpoint": v_over_rho,
    }


# --------------------------------------------------------------------------- #
# Step 3 -- independent recomputation of pi[y]*(g_{y,c}-g_{y,y}), checked against
# compute_expected_flip_gradients's own G_k column for the SAME checkpoint. Written
# independently of that function's own internals (a fresh torch.autograd.grad/
# compute_batch_gradients call, not a copy of its loop) so a future drift in either the
# pi_y-scaling convention or the loss/preprocessing pipeline would show up as a nonzero
# relative error here.
# --------------------------------------------------------------------------- #
def check_pi_convention(ctx, model, check_pairs):
    dataset_flag, model_flag = ctx["dataset_flag"], ctx["model_flag"]
    pi = ctx["pi"]
    class_samples_raw = ctx["class_samples_raw"]
    loss_fn = ctx["loss_fn"]
    params = list(model.parameters())

    G_k, _Q_k, pairs_k = compute_expected_flip_gradients(
        model, loss_fn, class_samples_raw, ctx["n_classes"], pi,
        dataset_flag=dataset_flag, model_flag=model_flag, params=params,
    )

    results = []
    for (y, c) in check_pairs:
        x = raw_to_preprocess(class_samples_raw[y], dataset_flag=dataset_flag, model_flag=model_flag)
        n_y = x.shape[0]
        y_lab = torch.full((n_y,), y, dtype=torch.long, device=x.device)
        c_lab = torch.full((n_y,), c, dtype=torch.long, device=x.device)

        grads_y, _ = compute_batch_gradients(model, loss_fn, (x, y_lab), create_graph=False)
        g_y = torch.cat([g.reshape(-1) for g in grads_y]).detach()
        grads_c, _ = compute_batch_gradients(model, loss_fn, (x, c_lab), create_graph=False)
        g_c = torch.cat([g.reshape(-1) for g in grads_c]).detach()

        manual_col = pi[y] * (g_c - g_y)
        ref_col = G_k[:, pairs_k.index((y, c))]
        rel_err = (
            (manual_col - ref_col).norm().item() / max(ref_col.norm().item(), 1e-12)
        )
        results.append((y, c, rel_err))
    return results


def fmt(x):
    return f"{x:.6e}" if x is not None else "None"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name")
    parser.add_argument("--module-name", default="federated_optimizing_trigger_policy")
    parser.add_argument("--n-batches", type=int, default=20)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ctx = build_context(args.experiment_name, args.module_name, device)

    print(
        f"\nbeta_local={ctx['beta']:.6f}, pi_source={ctx['pi_source']:.6f}, "
        f"gamma={ctx['gamma']:.6f}, lambda_poison={ctx['lambda_poison']:.6f}"
    )

    # --- Step 3: pi_y convention check, on the FIRST expert checkpoint, for the
    # (source, target) pair plus up to two others. ---
    check_pairs = [(ctx["source_label"], ctx["target_label"])]
    for (y, c) in ctx["pairs"]:
        if len(check_pairs) >= 3:
            break
        if (y, c) != check_pairs[0]:
            check_pairs.append((y, c))
    pi_check_results = check_pi_convention(ctx, ctx["expert_models"][0], check_pairs)
    print("\n=== Step 3: pi_y convention check (G_k[:, (y,c)] == pi[y]*(g_{y,c}-g_{y,y})) ===")
    for (y, c, rel_err) in pi_check_results:
        print(f"  (y={y}, c={c}): relative error = {rel_err:.3e}")

    # --- Step 2: B2_current / B2_null_triv / B2_null_qp over --n-batches batches. ---
    flip_grad_cache = {}
    records = []
    loader_iter = iter(ctx["loader"])
    for i in range(args.n_batches):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(ctx["loader"])
            batch = next(loader_iter)
        rec = measure_batch(ctx, batch, flip_grad_cache)
        if rec is None:
            print(f"  batch {i}: skipped (empty poison mask)")
            continue
        records.append(rec)
        print(
            f"  batch {i}: B2_current={fmt(rec['B2_current'])} "
            f"B2_null_triv={fmt(rec['B2_null_triv'])} B2_null_qp={fmt(rec['B2_null_qp'])} "
            f"u_triv_feasible={rec['triv_feasible']} "
            f"||v||/rho per checkpoint={[f'{x:.3f}' for x in rec['v_over_rho_per_checkpoint']]}"
        )

    if not records:
        print("\nNo non-empty batches measured -- nothing to summarize.")
        sys.exit(0)

    print("\n=== Step 4: summary (mean +/- std over measured batches) ===")
    for key in ["B2_current", "B2_null_triv", "B2_null_qp"]:
        vals = np.array([r[key] for r in records], dtype=np.float64)
        print(f"  {key}: {vals.mean():.6e} +/- {vals.std():.6e}")

    b2_current_vals = np.array([r["B2_current"] for r in records], dtype=np.float64)
    b2_null_triv_vals = np.array([r["B2_null_triv"] for r in records], dtype=np.float64)
    ratio = b2_null_triv_vals / np.maximum(np.abs(b2_current_vals), 1e-12)
    print(f"  B2_null_triv / B2_current: mean={ratio.mean():.6e}, std={ratio.std():.6e}")

    n_infeasible = sum(1 for r in records if not r["triv_feasible"])
    if n_infeasible:
        print(
            f"\nWARNING: u_triv was INFEASIBLE on {n_infeasible}/{len(records)} measured "
            "batches -- this means beta_local > pi_source (task 1's A3 condition); "
            "B2_null_triv above is then the value of an INFEASIBLE point, not a valid lower "
            "bound, and should not be trusted as evidence either way."
        )
