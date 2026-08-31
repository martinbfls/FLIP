"""
scripts/solve_policy_frozen_trigger.py -- Task 10 of the federated_optimizing_trigger_policy
review: isolates "is the per-class label-flip policy expressive enough?" by FREEZING the
trigger at the classic 1xs stripe (the same one train_expert's checkpoints were trained
against) and solving ONLY for u -- no delta optimization, no L_bd, no trigger regularizer, no
per-step expert retraining (delta constant => a single expert trajectory, the one
train_expert already produced against the 1xs poisoner, is exactly what this needs).

Does NOT modify run_module.py: reuses `optimize_trigger_policy` itself (called with
`n_steps=0`, see build_context below) to resolve beta/beta_global/gamma/pairs through the
SAME A3 (docs/policy_module_audit_report.md) and 5b (task 5b: round(lambda_poison*
batch_size_trigger)>=1) guards `run()` would apply -- n_steps=0 means the guards and the
setup they gate run exactly as they would for a real training run, but the (never reached)
training loop itself never executes, so nothing is retrained or written beyond the
config's own output directories. Everything else (`inner_solve.build_inner_context`,
`aggregate_qp`/`aggregate_qp_from_history`/`push_context_history`, `qp_pgd_solve`,
`check_feasible`, `four_term_decomposition`, `nnls_cone_projection`, `get_or_build_flip_grad_
cache_entry` via build_inner_context) is reused unchanged, no reimplementation of the actual
solve.

Task 10a (trigger consistency, done first -- blocking): v_k must be computed against EXACTLY
the trigger the loaded expert checkpoints were trained against. train_expert poisons via
`pick_poisoner("1xs", ...)` -> `StripePoisoner` (uint8, [0,255]-scale arithmetic, base_utils/
datasets.py); the policy module poisons via `raw_to_trigger_preprocess(x_raw, delta, ...)`
([0,1]-scale, delta a plain additive tensor). `check_trigger_consistency` below builds
delta_stripe = init_delta(shape, horizontal=True, strength=6.0, freq=16, init="stripe") (the
SAME call run_module.py's own init_delta(..., init="stripe") makes) and compares
(x_raw+delta_stripe).clamp(0,1) against StripePoisoner's actual output on the SAME images, in
RAW [0,1] space. If they do not coincide (they do not, by construction -- see the function's
own docstring for why), delta_frozen is instead the EMPIRICAL difference (triggered - clean)
StripePoisoner actually produces, extracted directly from real images -- no fudge factor.

Usage:
    python scripts/solve_policy_frozen_trigger.py <experiment_name> \
        --lambda-ratios 0.25,0.5,0.75,1.0 --flip-gradient-samples 256 \
        --n-batches 50 --num-chckpt 15 --qp-ridge 1e-3
"""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if not torch.cuda.is_available():
    torch.nn.Module.cuda = lambda self, device=None: self.to("cpu")
    torch.Tensor.cuda = lambda self, device=None, non_blocking=False: self.to("cpu")

from modules.base_utils.datasets import get_n_classes, StripePoisoner
from modules.base_utils.util import extract_toml, load_model
from modules.federated_generate_labels.utils import extract_experts
from modules.federated_optimizing_trigger.run_module import build_loader
from modules.federated_optimizing_trigger.utils import (
    compute_class_frequencies,
    get_class_conditional_samples,
    get_mu,
    get_raw_clean_dataset,
    init_delta,
    move_to_device,
    raw_to_preprocess,
    sample_checkpoints,
)
from modules.federated_optimizing_trigger_policy import diagnostics as diag
from modules.federated_optimizing_trigger_policy import inner_solve
from modules.federated_optimizing_trigger_policy.run_module import (
    _build_poison_mask,
    optimize_trigger_policy,
)


# --------------------------------------------------------------------------- #
# Task 10a
# --------------------------------------------------------------------------- #
def check_trigger_consistency(dataset_flag, device, n_check=32, tol=1e-3):
    '''
    Compares, on `n_check` real training images, (x_raw + delta_stripe).clamp(0,1) (delta_
    stripe = init_delta(..., init="stripe"), the SAME formula run_module.py's own trigger init
    uses) against StripePoisoner's actual output on the SAME images, both in RAW [0,1] space.

    These are NOT expected to match: init_delta's "stripe" branch scales by `strength=6.0`
    directly in [0,1]-space (a sine wave in [-6,6] added to a [0,1] image and clamped is nearly
    a square wave saturated at 0/1 almost everywhere), while StripePoisoner adds the SAME
    `strength=6` in [0,255]-space (a ~2.4%-amplitude perturbation, imperceptible) -- the two
    "strength=6" values live on different scales and are not interchangeable.

    Returns (delta_frozen: (C,H,W) tensor on `device`, path_used: "init_delta" or "empirical",
    max_err: float, rel_err: float) -- max_err/rel_err are always computed against
    init_delta's own construction (so the gap is visible even when the empirical path is
    used); path_used records which one was actually returned as delta_frozen.
    '''
    raw_dataset = get_raw_clean_dataset(dataset_flag, train=True)
    x_raws = torch.stack([raw_dataset[i][0] for i in range(n_check)])  # (N,C,H,W), [0,1]

    stripe = StripePoisoner(strength=6, freq=16)  # horizontal=True default, matches "1xs"
    stripe_outs = []
    for i in range(n_check):
        pil_img = transforms.ToPILImage()(x_raws[i])
        stripe_outs.append(transforms.ToTensor()(stripe.poison(pil_img)))
    x_via_stripe = torch.stack(stripe_outs)  # (N,C,H,W), [0,1]

    shape = tuple(x_raws.shape[1:])
    delta_init_delta = init_delta(
        shape, horizontal=True, strength=6.0, freq=16, device="cpu", init="stripe",
    ).detach()
    x_via_init_delta = (x_raws + delta_init_delta.unsqueeze(0)).clamp(0, 1)

    diff = x_via_stripe - x_via_init_delta
    max_err = diff.abs().max().item()
    v_norm = x_via_stripe.norm().item()
    rel_err = (diff.norm().item() / v_norm) if v_norm > 1e-12 else float("nan")

    print(
        f"[10a] init_delta('stripe', strength=6.0) vs StripePoisoner(strength=6): "
        f"max_err={max_err:.6f} rel_err={rel_err:.6f} (tol={tol})"
    )

    if max_err <= tol:
        print("[10a] MATCH -- using init_delta's construction as delta_frozen.")
        return delta_init_delta.to(device), "init_delta", max_err, rel_err

    print(
        "[10a] MISMATCH -- init_delta('stripe', strength=6.0) operates in [0,1]-scale while "
        "StripePoisoner's strength=6 is a [0,255]-scale amplitude; NOT using init_delta. "
        "Extracting delta_frozen empirically as (triggered - clean) instead."
    )
    # Empirical extraction, from an image unlikely to saturate at either bound. Cross-checked
    # against two more images to confirm the additive mask is consistent (StripePoisoner's own
    # perturbation is content-independent EXCEPT at the [0,255] clip boundary) -- reported, not
    # silently trusted.
    ref_idx = 0
    delta_frozen = (x_via_stripe[ref_idx] - x_raws[ref_idx]).clone()
    for other_idx in (1, 2):
        other_delta = x_via_stripe[other_idx] - x_raws[other_idx]
        unsaturated = (x_via_stripe[other_idx] > 1e-3) & (x_via_stripe[other_idx] < 1 - 1e-3)
        if unsaturated.any():
            disagreement = (
                (other_delta[unsaturated] - delta_frozen[unsaturated]).abs().max().item()
            )
            print(
                f"[10a] cross-check vs image {other_idx} (unsaturated pixels only): "
                f"max disagreement={disagreement:.6f}"
            )
    return delta_frozen.to(device), "empirical", max_err, rel_err


# --------------------------------------------------------------------------- #
# Task 10b, step 1 -- resolve beta/beta_global/gamma/pi/pairs via optimize_trigger_policy's OWN
# A3/5b guards (n_steps=0: the guarded setup runs, the training loop never does).
# --------------------------------------------------------------------------- #
def resolve_via_optimize_trigger_policy(args, dataset_flag, model_flag, n_classes, device,
                                         lambda_poison_value):
    model = load_model(model_flag, n_classes).to(device)
    model.eval()
    source_label = args["source_label"]
    target_label = args["target_label"]
    mu = get_mu(dataset_flag, target_label, device, model_flag=model_flag)
    mu_source = mu if source_label == -1 else get_mu(
        dataset_flag, source_label, device, model_flag=model_flag,
    )
    delta_stub, u_zero, pairs, beta, n_train, run_tag = optimize_trigger_policy(
        model=model,
        loss_fn=torch.nn.CrossEntropyLoss(),
        dataset_flag=dataset_flag,
        mu=mu,
        mu_source=mu_source,
        source_label=source_label,
        target_label=target_label,
        device=device,
        n_steps=0,
        expert_config=args.get("expert_config", {}),
        expert_path=args["expert_path"],
        output_dir=args["output_dir"],
        output_dir_trigger=args.get("output_dir_trigger", "optimized_trigger"),
        output_dir_policy=args.get("output_dir_policy", "optimized_policy"),
        num_honests=args.get("num_honests", 5),
        num_poisoned=args.get("num_poisoned", 5),
        model_flag=model_flag,
        beta=args.get("beta", None),
        flip_budget=args.get("flip_budget", None),
        lambda_poison=lambda_poison_value,
        batch_size_trigger=args.get("batch_size_trigger", 256),
        flip_gradient_samples_per_class=args.get("flip_gradient_samples_per_class", 64),
        init=args.get("init", "stripe"),
    )
    gamma = args.get("num_poisoned", 5) / (args.get("num_poisoned", 5) + args.get("num_honests", 5))
    return pairs, beta, gamma, n_train, run_tag


# --------------------------------------------------------------------------- #
# Task 10b, steps 2-4
# --------------------------------------------------------------------------- #
def solve_for_ratio(args, dataset_flag, model_flag, n_classes, device, expert_models, loader,
                     pi, class_samples_raw, pairs, beta, gamma, ratio, n_batches, num_chckpt,
                     alpha_ckpt, qp_ridge, delta_frozen, source_label, target_label,
                     normalization="rho"):
    # Re-resolves (and re-validates via A3/5b) for THIS ratio's explicit numeric lambda_poison
    # -- same pattern task 7's sweep uses (lambda_poison bypasses the "beta" auto-coupling).
    # Both calls can raise ValueError from A3/5b (reused, not duplicated) if this ratio is
    # infeasible -- caught here and reported as a REFUSED row, not a crash.
    try:
        _pairs, beta_local, gamma_, n_train, _run_tag = resolve_via_optimize_trigger_policy(
            args, dataset_flag, model_flag, n_classes, device, lambda_poison_value="beta",
        )
        beta_global = gamma_ * beta_local
        lambda_poison = ratio * beta_global
        resolve_via_optimize_trigger_policy(
            args, dataset_flag, model_flag, n_classes, device, lambda_poison_value=lambda_poison,
        )
    except ValueError as e:
        return {"refused": f"A3/5b guard: {e}"}

    sampled_k = sample_checkpoints(len(expert_models), num_chckpt, alpha=alpha_ckpt, device=device)
    flip_grad_cache = {}
    history = {}
    v_sum, v_count = {}, {}
    loader_iter = iter(loader)
    contexts = None
    loss_fn = torch.nn.CrossEntropyLoss()

    n_used = 0
    for _ in range(n_batches):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)
        x_raw, y = move_to_device(batch, device)
        x_clean = raw_to_preprocess(x_raw, dataset_flag=dataset_flag, model_flag=model_flag)
        mask, y_poison, has_poison, lam_eff, _lam_eff_ratio = _build_poison_mask(
            y, x_raw.shape[0], source_label, target_label, lambda_poison,
        )
        if not has_poison:
            continue
        n_used += 1
        contexts = inner_solve.build_inner_context(
            expert_models, sampled_k, x_clean, y, x_raw, mask, y_poison, has_poison,
            delta_frozen, loss_fn, dataset_flag, model_flag, n_classes, flip_grad_cache,
            class_samples_raw, pi, gamma, beta_local, beta_global, pairs, normalization,
            device, v_estimator="analytic", source_label=source_label,
            target_label=target_label, lambda_eff=lam_eff,
        )
        inner_solve.push_context_history(history, contexts, n_batches)
        for ctx in contexts:
            v_sum[ctx["k"]] = ctx["v"].detach().clone() + v_sum.get(ctx["k"], 0.0)
            v_count[ctx["k"]] = v_count.get(ctx["k"], 0) + 1

    if contexts is None:
        return {"refused": f"no non-empty batch out of {n_batches} at lambda_poison={lambda_poison:.6f}"}

    Q_agg, c_agg, pairs_agg = inner_solve.aggregate_qp_from_history(contexts, history, m=n_batches)
    u_star, iters, obj_start, obj_end, converged = inner_solve.qp_pgd_solve(
        Q_agg, c_agg, np.zeros(len(pairs_agg)), beta_local, pairs_agg, pi,
        max_iters=2000, tol=1e-10, min_iters=50, ridge=qp_ridge,
    )
    feasible = inner_solve.check_feasible(u_star, beta_local, pairs_agg, pi)

    contexts_avg = [
        {**ctx, "v": v_sum[ctx["k"]] / v_count[ctx["k"]]} for ctx in contexts
    ]

    idx_st = pairs_agg.index((source_label, target_label))
    u_np = diag.as_numpy(u_star)
    u_triv = np.zeros(len(pairs_agg))
    u_triv[idx_st] = beta_local

    b2_qp_final = inner_solve.aggregate_b2(u_star, contexts_avg)
    b2_triv = inner_solve.aggregate_b2(u_triv, contexts_avg)

    four_terms = [
        inner_solve.four_term_decomposition(
            ctx["G_obj"], ctx["v"], ctx["den"], u_triv, u_star, ctx["rho_k"],
        )
        for ctx in contexts_avg
    ]
    four_term_mean = {
        key: float(np.mean([ft[key] for ft in four_terms])) for key in four_terms[0]
    }

    cosines = []
    for ctx in contexts_avg:
        Gu = (ctx["G_obj"] @ torch.as_tensor(u_np, dtype=ctx["G_obj"].dtype, device=ctx["G_obj"].device))
        cos = torch.nn.functional.cosine_similarity(
            Gu.reshape(1, -1), ctx["v"].reshape(1, -1).to(Gu.dtype), eps=1e-8,
        ).item()
        cosines.append(cos)

    topk_idx = np.argsort(-np.abs(u_np))[:10]
    topk = [(pairs_agg[i], float(u_np[i])) for i in topk_idx]

    return {
        "refused": None,
        "lambda_poison": lambda_poison,
        "beta_local": beta_local,
        "beta_global": beta_global,
        "n_batches_used": n_used,
        "B2_qp_final": b2_qp_final,
        "B2_triv": b2_triv,
        "four_term_decomposition": four_term_mean,
        "u_star_mass_on_st_fraction": float(u_np[idx_st] / max(u_np.sum(), 1e-12)),
        "sum_u_over_beta_local": float(u_np.sum() / max(beta_local, 1e-12)),
        "nnz": int((np.abs(u_np) > 1e-8).sum()),
        "top10_pairs": topk,
        "cosine_Gu_v_mean": float(np.mean(cosines)),
        "feasible": bool(feasible),
        "qp_actual_iters": iters,
        "qp_converged": bool(converged) if converged is not None else None,
        "u_star": u_np,
        "pairs": pairs_agg,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name")
    parser.add_argument("--module-name", default="federated_optimizing_trigger_policy")
    parser.add_argument("--lambda-ratios", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--flip-gradient-samples", type=int, default=256)
    parser.add_argument("--n-batches", type=int, default=50)
    parser.add_argument("--num-chckpt", type=int, default=15)
    parser.add_argument("--alpha-ckpt", type=float, default=0.01)
    parser.add_argument("--qp-ridge", type=float, default=1e-3)
    parser.add_argument("--out-dir", default="out/solve_policy_frozen_trigger")
    cli_args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = extract_toml(cli_args.experiment_name, cli_args.module_name)
    args["flip_gradient_samples_per_class"] = cli_args.flip_gradient_samples

    dataset_flag = args["dataset"]
    model_flag = args["model"]
    source_label = args["source_label"]
    target_label = args["target_label"]
    n_classes = get_n_classes(dataset_flag)
    num_honests = args.get("num_honests", 5)
    num_poisoned = args.get("num_poisoned", 5)

    delta_frozen, path_used, max_err, rel_err = check_trigger_consistency(dataset_flag, device)
    print(f"[10a] delta_frozen source: {path_used!r}")

    print("Loading expert checkpoints (no mini_train)...")
    checkpoint_paths = extract_experts(args.get("expert_config", {}), args["expert_path"])
    base_model = load_model(model_flag, n_classes).to(device)
    expert_models = []
    for ckpt_path in checkpoint_paths:
        M = copy.deepcopy(base_model).to(device)
        M.load_state_dict(torch.load(ckpt_path, map_location=device))
        M.eval()
        expert_models.append(M)
    print(f"Loaded {len(expert_models)} expert checkpoint(s).")

    pi = compute_class_frequencies(dataset_flag, n_classes)
    class_samples_raw = get_class_conditional_samples(
        dataset_flag, n_classes, cli_args.flip_gradient_samples, device,
    )
    raw_train_dataset = get_raw_clean_dataset(dataset_flag, train=True)
    loader = build_loader(raw_train_dataset, batch_size=args.get("batch_size_trigger", 256))

    pairs0, beta0, gamma0, n_train0, _run_tag = resolve_via_optimize_trigger_policy(
        args, dataset_flag, model_flag, n_classes, device, lambda_poison_value="beta",
    )
    beta_global0 = gamma0 * beta0
    print(f"beta_local={beta0:.6f} beta_global={beta_global0:.6f} gamma={gamma0:.6f}")

    ratios = [float(r) for r in cli_args.lambda_ratios.split(",")]
    out_dir = Path(cli_args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(delta_frozen.detach().cpu(), out_dir / "delta_frozen.pt")

    print(f"\n{'ratio':>8} {'B2_qp':>12} {'B2_triv':>12} {'mass_st_frac':>14} {'sum_u/beta':>12} {'nnz':>6}")
    results = {}
    for ratio in ratios:
        result = solve_for_ratio(
            args, dataset_flag, model_flag, n_classes, device, expert_models, loader,
            pi, class_samples_raw, pairs0, beta0, gamma0, ratio, cli_args.n_batches,
            cli_args.num_chckpt, cli_args.alpha_ckpt, cli_args.qp_ridge, delta_frozen,
            source_label, target_label,
        )
        results[ratio] = result
        if result["refused"]:
            print(f"{ratio:>8.2f}  REFUSED: {result['refused']}")
            continue
        print(
            f"{ratio:>8.2f} {result['B2_qp_final']:>12.4e} {result['B2_triv']:>12.4e} "
            f"{result['u_star_mass_on_st_fraction']:>14.4f} "
            f"{result['sum_u_over_beta_local']:>12.4f} {result['nnz']:>6d}"
        )
        print(f"  four_term_decomposition: {result['four_term_decomposition']}")
        print(f"  cos(G_obj@u*, v) mean: {result['cosine_Gu_v_mean']:.4f}")
        print(f"  feasible={result['feasible']} qp_iters={result['qp_actual_iters']} "
              f"qp_converged={result['qp_converged']}")
        print(f"  top10 |u*| pairs: {result['top10_pairs']}")

        u_star = result["u_star"]
        pairs_agg = result["pairs"]
        pairs_arr = np.array(pairs_agg, dtype=np.int64)
        npz_path = out_dir / f"policy_frozen_stripe_ratio{ratio}.npz"
        np.savez(
            npz_path,
            u=u_star.astype(np.float32),
            pairs_y=pairs_arr[:, 0],
            pairs_c=pairs_arr[:, 1],
            beta=np.array(beta0, dtype=np.float64),
            n_train=np.array(n_train0, dtype=np.int64),
            source_label=np.array(source_label, dtype=np.int64),
            target_label=np.array(target_label, dtype=np.int64),
            num_honests=np.array(num_honests, dtype=np.int64),
            num_poisoned=np.array(num_poisoned, dtype=np.int64),
            gamma=np.array(gamma0, dtype=np.float64),
        )
        print(f"  saved: {npz_path}")

    print(
        "\nReading the first column above: if u_star_mass_on_st_fraction ~= 1, the QP just "
        "rediscovered the naive source->target flip and the per-class policy has no content "
        "of its own at this ratio; substantially below 1 means it is exploiting directions "
        "the naive flip cannot reach (see top10_pairs for which)."
    )
