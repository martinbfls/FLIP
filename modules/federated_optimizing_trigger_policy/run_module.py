"""
Theory: sec:attacker-problem (eq:P) -- (P^mean), the mean-aggregation attacker's problem:

    min_{||delta||_inf<=eps, (u^i) in U_beta(fraktur)}
        E_k[ ||Gbar(theta_bar_k) ubar - v_k(delta)||^2 / rho_k^2 ]
        + kappa * E_k[ E_X[ loss_c(f_theta_bar_k(T_delta(X)), y_target)] ]
    s.t. theta_bar_{k+1} = theta_bar_k - eta_k grad(theta_bar_k),  lambda = beta

This module jointly optimizes the trigger `delta` and an explicit LOCAL label-flipping policy
`u` (see rem:units -- u in U_loc, NOT the aggregate ubar in U_beta) against this objective,
under `rem:solver`'s solver (b) (simultaneous descent on delta and u, not solver (a)'s exact
inner QP + Danskin step -- see `optimize_trigger_policy_step`'s docstring for why, and the
`B2_qp` diagnostic that quantifies the gap to solver (a)).

Diagnostics: this module can optionally emit a diagnostics.jsonl (one JSON record per
diagnostic event, `diag_path`/`diag_*` config options below) instrumenting WHERE the attack is
failing -- optimization of u vs. the QP optimum, geometric feasibility of v(delta) under
G_obj@U_loc, the loss from discretizing u into integer flip counts, analytic-vs-actual gradient
mismatch, and delta's own gradient balance between B2 and lambda_bd*L_bd. See
`diagnostics.py`'s module docstring for the full field list and how to read them together, and
this module's `run()` docstring for the individual `diag_*` options. All default to `diag_path
= null` (no file written) -- disabled diagnostics change nothing about the run.

Scope conventions adopted throughout this module (see docs/policy_module_audit_report.md
Section 2.2 for the full derivation):
  - `beta` (the module's own parameter) is the theory's LOCAL corruption rate -- the fraction
    of a SINGLE corrupted worker's own shard that worker may flip -- NOT the theory's global
    `beta := N_flip/n` (def:budget). Concretely `beta` (code) == `beta_theory / gamma`.
  - `u` is correspondingly LOCAL: `u in U_loc = {u>=0, sum(u)<=beta, sum_c u_{y,c}<=pi_y}`
    (eq:Uloc, `project_policy_budget`), one shared policy vector deployed identically by every
    corrupted worker (the homogeneous configuration of lem:hom-wlog).
  - A1 correction (docs/policy_module_audit_report.md Section 2.6): `lambda_poison="beta"`
    (the default) resolves to `beta_global := gamma*beta` (== `beta_theory`, def:budget) as
    `eq:P`'s constraint `lambda=beta` requires -- NOT the LOCAL `beta` directly (the pre-A1
    behavior). This coupling is only theoretically justified when `s_beta :=
    beta_global/(gamma*min_y(pi_y)) <= 1` (prop:budget-match's unsaturated-regime hypothesis;
    `rem:saturated` lists `prop:budget-match` among the results lost once it fails) -- a
    warning is printed at startup when `s_beta > 1`, and an explicit numeric `lambda_poison`
    (used verbatim, unscaled) is the way to sweep lambda independently of beta in that regime.
    See `optimize_trigger_policy`'s docstring.

Chain position: consumes expert/trigger checkpoints (`expert_config`/`expert_path`, real
weights, no theory correspondence of their own) and a raw dataset (for pi_y, the shift
gradients G, and mu/mu_source). Produces (a) an optimized trigger `delta` (.pt, same naming
convention as the sibling federated_optimizing_trigger module) and (b) an optimized LOCAL
policy `u` alongside its metadata (pairs, beta, n_train, num_honests/num_poisoned/gamma --
.npz). Both are consumed downstream by `federated_policy_to_flips`, which materializes `u` into
concrete per-worker label flips for the (unmodified) `federated_train_user` victim-training
pipeline.
"""
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
    project_gradient,
)
from modules.federated_optimizing_trigger.run_module import build_loader
from modules.federated_optimizing_trigger_policy.utils import (
    init_policy, project_policy_budget, project_gradient_descent_local,
)
from modules.federated_optimizing_trigger_policy import diagnostics as diag
from modules.train_expert.utils import checkpoint_callback
import torch
from torch.utils.data import ConcatDataset, Subset
import numpy as np
from pathlib import Path
import os
import json
import matplotlib.pyplot as plt
import copy
import time as _time

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
    gamma,
    beta,
    beta_global,
    pairs,
    normalization,
    checkpoint_backward,
    lambda_bd,
    run_diag,
    diag_qp_iters=50,
    diag_qp_convergence=False,
    diag_qp_check_iters=(50, 200, 1000),
    diag_policy_nnz_threshold=1e-8,
    diag_policy_topk=10,
    diag_policy_full_vector=False,
    diag_discretization=True,
    diag_gradient_balance=True,
    diag_actual_gradient_run=False,
    n_train=None,
    class_counts=None,
    diag_constraint_tol=1e-8,
    diag_span_projection=True,
    diag_direction_scaling=True,
):
    '''Per sampled checkpoint theta_k, the (P^mean) objective's two terms:

        v = mu_p - g_c  poisoning-induced gradient shift (identical to `_compute_step` in federated_optimizing_trigger)
        Gu = G_obj_k @ u policy-reachable gradient shift
        under MEAN aggregation, u being a LOCAL policy (fraction of a SINGLE
        corrupted worker's own shard -- same scope as `beta` and
        `project_policy_budget`, see prelim/SPEC.md's U_loc). G_obj_k is
        built from `compute_expected_flip_ gradients`' G_k (columns
        pi_y*(g_{y,c}-g_{y,y})) by two per-column rescalings, done ONCE
        when the cache is filled below:
             (i)  divide by pi_y -- G_k's pi_y factor makes u AN
              AGGREGATE/population rate (materialize_policy_flips
              computes round(u*n_train), i.e. treats u as exactly
              that); undo it to get H[:,(y,c)] = g_{y,c}-g_{y,y}
            for a LOCAL, per-worker u.
            (ii) multiply by gamma = num_poisoned / (num_poisoned + num_honests) -- n_p corrupted workers, EACH deploying the same local u, contribute a gamma*H@u shift to the MEAN-aggregated
             gradient (n_p/n_b of them carry the shift, the rest
            contribute ~0), not H@u.

        Net: G_obj_k = (gamma/pi_y) * G_k column-wise. `u` itself is
        projected onto its LOCAL feasible set {u>=0, sum(u)<=beta,
        sum_c u_{y,c}<=pi_y} by `project_policy_budget` in the
        caller -- NOT gamma*pi_y: a single worker's own capacity has nothing
        to do with how many other corrupted workers exist.

        rho_k = beta_global * varsigma_k       (eq:rho, A2) radius of the reachable set under
        the GLOBAL beta (def:budget) -- varsigma_k = max_{y,c}||G_k[:,(y,c)]/pi_y|| (eq:varsigma,
        named explicitly, not fused into this formula -- A2). beta_global = gamma*beta (this
        module's own, LOCAL `beta` -- see the module header docstring's scope section and A1);
        NOT `beta` directly, unlike before A1's correction. Depends only on the checkpoint and
        beta_global, so cached alongside G_obj_k/Q_obj_k.

        B2_k = ||Gu - v||^2 / rho_k^2   ("rho", default)   alignment between what the trigger  or ||Gu - v||^2 / (||v||^2+eps)  ("v")        does and what the learned policy u can reproduce. Both variants are always computed and logged (see `normalization`); "rho" is a delta-independent, non-saturating denominator (unlike "v", which saturates at exactly 1 whenever u=0 is outside the reachable set, and whose gradient signal to delta decays as delta moves further out of reach).

        L_bd_k = CE(f_theta_k(T_delta(x)), y_target) backdoor loss on triggered examples

    If `run_diag` (see `diag_every` in the caller), also solves for w*_k on U_loc ITSELF (A4:
    via `project_gradient_descent_local`, projected gradient descent with the exact
    `project_policy_budget` projection, warm-started from the current `u` -- NOT the shared
    `federated_optimizing_trigger.utils.project_gradient`, which enforces only the global
    sum(w)<=beta constraint and would solve over a strictly LARGER polytope than u's own U_loc)
    and reports B2(w*_k) as `B2_qp` alongside B2(u) under the SAME normalization -- the
    Danskin-gap diagnostic (see optimize_trigger_policy_step's docstring): if u is tracking
    delta well, B2_qp should sit close to B2. The PRE-A4 (global-budget-only) diagnostic value
    is kept under `B2_qp_relaxed` for one transition period.

    All terms are averaged over sampled_k, matching the original convention (same keys: B2,
    L_bd, lambda_effective) so the two threat models' per-step metrics stay directly
    comparable.
    '''
    eps_den = 1e-8
    n_exp = len(sampled_k)

    x_raw, y = move_to_device(batch, device)
    n_b = x_raw.shape[0]
    x_clean = raw_to_preprocess(x_raw, dataset_flag=dataset_flag, model_flag=model_flag)

    # A3 (docs/policy_module_audit_report.md): the mask below can select AT MOST
    # idx_source.numel() rows -- the source-class examples actually present in THIS batch
    # (~pi_source*n_b on average) -- regardless of how large lambda_poison is. Unlike
    # get_poison_dataset (ADD-based, used for theta_bar_k's retraining), this per-batch mask
    # never duplicates rows to reach target_count, so lambda_effective is capped near
    # pi_source whenever lambda_poison > pi_source. See lambda_effective_ratio below and the
    # startup guard in optimize_trigger_policy (raises if lambda_poison > pi_source, since the
    # cap is then structurally guaranteed, not just possible on an unlucky batch).
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
    # A3: ratio of what was actually realized to what was requested -- 1.0 means uncapped this
    # batch, < 1.0 means the source-class supply in this batch fell short of target_count.
    lambda_effective_ratio = lambda_effective / lambda_poison if lambda_poison > 0 else 0.0

    B2_sum, bd_loss_sum = None, None
    B2_rho_sum, B2_v_sum, B2_qp_sum = None, None, None
    B2_qp_relaxed_sum, pg_iters_sum, pg_obj_decrement_sum = None, None, None
    n_valid = 0

    # Diagnostic G (gradient balance) accumulators -- only populated on run_diag batches.
    grad_B2_norm_sum, grad_BD_norm_sum, grad_balance_n = None, None, 0
    # Context stashed at the FIRST sampled checkpoint only, for the representative-checkpoint
    # diagnostics (A/B/C/D below) -- diag_record stays a single-checkpoint snapshot rather than
    # an average over sampled_k, so it is directly attributable to one (checkpoint, batch).
    diag_ctx = None

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

        # Theory: sec:trigger-vk (eq:vk-delta) -- v_k(delta) = grad_Lp[delta](theta_k) -
        # grad_Lc(theta_k), the trigger-induced gradient shift (mu_p, g_c standing in for the
        # two gradients under the batch's poisoned/clean labels respectively).
        v = mu_p - g_c

        if k in flip_grad_cache:
            G_obj, Q_obj, pairs_k, rho_k = flip_grad_cache[k]
        else:
            # Theory: def:shifts (eq:shift) -- Gbar_{y,c} = g_{y,c} - g_{y,y}, WITH NO pi_y
            # factor (rem:no-pi). The shared compute_expected_flip_gradients still returns the
            # PRE-rem:no-pi convention (G_k[:,(y,c)] = pi_y*(g_{y,c}-g_{y,y}), an earlier draft's
            # definition -- see federated_optimizing_trigger/utils.py's own docstring) and is
            # not modified here (shared, read-only): the two-factor rescaling below is where
            # this module corrects it back to the def:shifts convention. PIEGE 1 (pi_y
            # convention): do not double-correct -- compute_expected_flip_gradients' OWN
            # docstring calls its pi_y factor deliberate for ITS internal budget-constraint
            # convention (a DIFFERENT, simpler scheme used elsewhere); it is this module's job,
            # not that shared function's, to strip it back out for eq:P's Gbar.
            G_k, Q_k, pairs_k = compute_expected_flip_gradients(
                M, loss_fn, class_samples_raw, n_classes, pi,
                dataset_flag=dataset_flag, model_flag=model_flag, params=params,
            )
            # See this function's docstring: G_k's columns are pi_y*(g_{y,c}-g_{y,y})
            # (aggregate-rate convention); u is local, so rescale by gamma/pi_y once here and
            # cache the result instead of G_k -- never rescaled again per-batch.
            #   (i)  /pi_y : undoes compute_expected_flip_gradients' pi_y factor -> Gbar itself
            #        (def:shifts, rem:no-pi).
            #   (ii) *gamma: theory's LOCAL reading of eq:P (the paragraph right after the
            #        lambda=beta constraint) -- b_k = gamma*Gbar(theta_bar_k)*u^i
            #        (eq:local-scope) when the decision variable is the per-worker u^i rather
            #        than the aggregate ubar. PIEGE 2 (local/aggregate scope): these are TWO
            #        SEPARATE corrections that compose multiplicatively (gamma/pi_y), not one --
            #        see docs/policy_module_audit_report.md Section 2.1/2.2.
            scale = torch.tensor(
                [gamma / pi[y] for (y, c) in pairs_k], device=G_k.device, dtype=G_k.dtype,
            )
            G_obj = G_k * scale
            scale_np = scale.detach().cpu().numpy().astype(np.float64)
            Q_obj = np.outer(scale_np, scale_np) * Q_k

            # A2 (docs/policy_module_audit_report.md Section "rayon de shift"): varsigma_k
            # exposed as its OWN named variable -- Theory: eq:varsigma, varsigma_k =
            # max_{y,c}||Gbar_{y,c}||, Gbar_{y,c} = G_k[:,(y,c)]/pi_y (the pi_y factor stripped
            # back out, def:shifts/rem:no-pi -- PIEGE 1) -- rather than folded into a single
            # fused norm call, so `rho_k = beta_global * varsigma_k` (eq:rho) can be read off
            # directly without re-deriving it from G_obj's own gamma/pi_y scaling.
            pi_col = torch.tensor(
                [pi[y] for (y, c) in pairs_k], device=G_k.device, dtype=G_k.dtype,
            )
            varsigma_k = (G_k.detach() / pi_col).norm(dim=0).max().item()
            rho_k = beta_global * varsigma_k

            # A2 self-check (run once, first checkpoint cached this call): rho_k above must
            # equal beta_local * max_col_norm(G_obj) exactly, since G_obj's own gamma factor
            # (PIEGE 2) cancels against beta_local's local scope (beta_local*gamma ==
            # beta_global by construction, A1) -- NOT a coincidence, but fragile: this assertion
            # catches a future change that drops G_obj's gamma factor or switches beta to
            # global scope without updating rho_k to match.
            if len(flip_grad_cache) == 0:
                rho_k_check = beta * G_obj.detach().norm(dim=0).max().item()
                # rtol=1e-2 (not float32 machine precision): the two paths take DIFFERENT
                # float32 multiply/reduction orders over a D~10^5-10^6 parameter vector (D =
                # number of model params), so a ~1e-4 RELATIVE gap is normal accumulation
                # noise, empirically observed (~1.4e-4 on r32p/CIFAR -- see
                # prelim/verify_A1_A2_A5.py), not a sign of an algebraic error. A REAL
                # regression (dropped gamma factor, wrong beta scope) would be off by a
                # constant factor like gamma or 1/gamma -- orders of magnitude past this
                # tolerance, not a rounding-sized gap.
                assert abs(rho_k_check - rho_k) < 1e-2 * max(abs(rho_k), 1e-8), (
                    f"A2 self-check failed: rho_k={rho_k:.6f} (beta_global*varsigma_k) vs "
                    f"rho_k_check={rho_k_check:.6f} (beta_local*max_col_norm(G_obj)) -- these "
                    "should match to within float32 rounding (beta_local*gamma == "
                    "beta_global by construction). A mismatch this large means the gamma/pi_y "
                    "scaling changed inconsistently -- see docs/theory/threat_model.tex eq:rho "
                    "and this module's header docstring."
                )

            flip_grad_cache[k] = (G_obj, Q_obj, pairs_k, rho_k)

        # Theory: eq:P's first term (local reading, the paragraph right after the lambda=beta
        # constraint) -- ||gamma*Gbar(theta_bar_k)*u^i - v_k(delta)||^2 / rho_k^2. G_obj already
        # carries the gamma factor (see above), so Gu below IS gamma*Gbar@u^i directly.
        Gu = G_obj @ u.to(dtype=G_obj.dtype)
        sq_err = ((Gu - v) ** 2).sum()
        den_v = v.detach().norm() ** 2 + eps_den
        den_rho = rho_k ** 2 + eps_den
        B2_v_k = sq_err / den_v
        B2_rho_k = sq_err / den_rho
        B2_k = B2_rho_k if normalization == "rho" else B2_v_k

        # Theory: eq:P's second term, kappa*E_k[E_X[loss_c(f(T_delta(X)), y_target)]] -- this
        # module names the weight `lambda_bd` (kept distinct from `kappa`, which here names
        # trigger_penalty_hinge's STEALTH margin instead, see that function). L_bd: CE
        # restricted to the actually-triggered examples only (see federated_optimizing_trigger's
        # `_compute_step` docstring).
        L_bd_k = (
            loss_fn(logits_p[mask], y_poison[mask])
            if mask.sum() > 0 else torch.tensor(0.0, device=device)
        )

        # A4 (docs/policy_module_audit_report.md Section 2.5/A4) -- Theory: rem:solver's
        # diagnostic, now solving on U_loc ITSELF (eq:Uloc, the same feasible set u is
        # actually optimized over) via projected gradient descent
        # (`project_gradient_descent_local`, warm-started from the current u), replacing the
        # old QP over a strictly LARGER polytope (global sum(w)<=beta only, via the shared
        # project_gradient -- no per-class pi_y caps). Reported as `B2_qp` (the corrected
        # value); the OLD (relaxed-polytope) value is kept under `B2_qp_relaxed` for one
        # transition period so the two can be compared on the same run.
        B2_qp_k, B2_qp_relaxed_k = None, None
        pg_iters_k, pg_obj_decrement_k = None, None
        if run_diag:
            c_np = (G_obj.T @ v).detach().cpu().numpy().astype(np.float64)
            den_qp = den_rho if normalization == "rho" else den_v

            w_pg, pg_iters_k, pg_obj_decrement_k = project_gradient_descent_local(
                Q_obj, c_np, u.detach(), beta, pairs_k, pi, n_iters=diag_qp_iters,
            )
            w_pg = w_pg.to(device=v.device, dtype=v.dtype)
            sq_err_pg = ((G_obj @ w_pg - v.detach()) ** 2).sum()
            B2_qp_k = (sq_err_pg / den_qp).detach()

            # B2_qp_relaxed: the PRE-A4 diagnostic (global budget only, strict superset of
            # U_loc) -- see the docstring note above.
            w_star = project_gradient(Q_obj, c_np, beta, pairs_k)
            w_star = w_star.to(device=v.device, dtype=v.dtype)
            sq_err_qp = ((G_obj @ w_star - v.detach()) ** 2).sum()
            B2_qp_relaxed_k = (sq_err_qp / den_qp).detach()

        # Diagnostic G (Section 8): grad_delta B2_k and grad_delta (lambda_bd*L_bd_k),
        # measured SEPARATELY via non-accumulating, graph-preserving `torch.autograd.grad`
        # calls -- BEFORE the real `step_loss.backward()` below, which is the (only) call
        # allowed to touch delta.grad and, under checkpoint_backward, frees the graph.
        if run_diag and diag_gradient_balance and has_poison:
            g_B2_norm, g_BD_norm = diag.gradient_balance(B2_k, L_bd_k, lambda_bd, delta)
            if g_B2_norm is not None:
                grad_B2_norm_sum = g_B2_norm if grad_B2_norm_sum is None else grad_B2_norm_sum + g_B2_norm
                grad_BD_norm_sum = g_BD_norm if grad_BD_norm_sum is None else grad_BD_norm_sum + g_BD_norm
                grad_balance_n += 1

        # Stash the FIRST sampled checkpoint's diagnostic-relevant tensors for the
        # representative-checkpoint diagnostics (A/B/C/D) computed once after this loop --
        # captured BEFORE checkpoint_backward's in-place `.detach()` below.
        if run_diag and diag_ctx is None:
            diag_ctx = {
                "k": k, "model": M, "v": v.detach(), "G_obj": G_obj, "Q_obj": Q_obj,
                "pairs_k": pairs_k, "rho_k": rho_k,
                "den": float(den_rho if normalization == "rho" else den_v),
                "Gu_current": Gu.detach(), "B2_current": B2_k.detach().item(),
                "w_pg": w_pg.detach() if run_diag else None,
                "B2_qp": B2_qp_k.item() if B2_qp_k is not None else None,
                "c_np": c_np if run_diag else None,
            }

        if checkpoint_backward:
            step_loss = (B2_k + lambda_bd * L_bd_k) / n_exp
            step_loss.backward()
            B2_k, L_bd_k = B2_k.detach(), L_bd_k.detach()
            B2_rho_k, B2_v_k = B2_rho_k.detach(), B2_v_k.detach()

        if B2_sum is None:
            B2_sum, bd_loss_sum = B2_k, L_bd_k
            B2_rho_sum, B2_v_sum = B2_rho_k, B2_v_k
        else:
            B2_sum = B2_sum + B2_k
            bd_loss_sum = bd_loss_sum + L_bd_k
            B2_rho_sum = B2_rho_sum + B2_rho_k
            B2_v_sum = B2_v_sum + B2_v_k
        if B2_qp_k is not None:
            B2_qp_sum = B2_qp_k if B2_qp_sum is None else B2_qp_sum + B2_qp_k
            B2_qp_relaxed_sum = (
                B2_qp_relaxed_k if B2_qp_relaxed_sum is None
                else B2_qp_relaxed_sum + B2_qp_relaxed_k
            )
            pg_iters_sum = pg_iters_k if pg_iters_sum is None else pg_iters_sum + pg_iters_k
            pg_obj_decrement_sum = (
                pg_obj_decrement_k if pg_obj_decrement_sum is None
                else pg_obj_decrement_sum + pg_obj_decrement_k
            )
        n_valid += 1

    B2 = B2_sum / n_valid
    L_bd = bd_loss_sum / n_valid
    B2_rho = (B2_rho_sum / n_valid).item()
    B2_v = (B2_v_sum / n_valid).item()
    B2_qp = (B2_qp_sum / n_valid).item() if B2_qp_sum is not None else None
    B2_qp_relaxed = (
        (B2_qp_relaxed_sum / n_valid).item() if B2_qp_relaxed_sum is not None else None
    )
    pg_iters = (pg_iters_sum / n_valid) if pg_iters_sum is not None else None
    pg_obj_decrement = (
        (pg_obj_decrement_sum / n_valid) if pg_obj_decrement_sum is not None else None
    )

    grad_delta_B2_norm = grad_B2_norm_sum / grad_balance_n if grad_balance_n else None
    grad_delta_BD_norm = grad_BD_norm_sum / grad_balance_n if grad_balance_n else None
    grad_delta_ratio = (
        grad_delta_B2_norm / max(grad_delta_BD_norm, 1e-8)
        if grad_delta_B2_norm is not None else None
    )

    # Representative-checkpoint diagnostics (A/B/C/D) -- a single-checkpoint snapshot (the
    # first entry of sampled_k, stashed as diag_ctx above), NOT averaged over sampled_k like
    # B2/B2_qp above: these diagnostics are meant to be read alongside a specific checkpoint,
    # not blurred across several. None of this touches B2/B2_qp/u/delta themselves.
    diag_record = None
    if run_diag and diag_ctx is not None:
        diag_record = _build_diag_record(
            diag_ctx, u, beta, gamma, n_train, class_counts, pi, loss_fn, class_samples_raw,
            dataset_flag, model_flag, diag_qp_convergence, diag_qp_check_iters,
            diag_policy_nnz_threshold, diag_policy_topk, diag_policy_full_vector,
            diag_discretization, diag_actual_gradient_run,
            diag_constraint_tol, diag_span_projection, diag_direction_scaling,
        )

    return {
        "B2": B2, "L_bd": L_bd, "lambda_effective": lambda_effective,
        "lambda_effective_ratio": lambda_effective_ratio,
        "B2_rho": B2_rho, "B2_v": B2_v, "B2_qp": B2_qp,
        "B2_qp_relaxed": B2_qp_relaxed, "pg_iters": pg_iters,
        "pg_obj_decrement": pg_obj_decrement,
        "grad_delta_B2_norm": grad_delta_B2_norm, "grad_delta_BD_norm": grad_delta_BD_norm,
        "grad_delta_ratio": grad_delta_ratio,
        "diag_record": diag_record, "diag_checkpoint": diag_ctx["k"] if diag_ctx else None,
    }


def _build_diag_record(
    diag_ctx, u, beta, gamma, n_train, class_counts, pi, loss_fn, class_samples_raw,
    dataset_flag, model_flag, diag_qp_convergence, diag_qp_check_iters,
    diag_policy_nnz_threshold, diag_policy_topk, diag_policy_full_vector,
    diag_discretization, diag_actual_gradient_run,
    diag_constraint_tol=1e-8, diag_span_projection=True, diag_direction_scaling=True,
):
    '''
    Assembles one diagnostics.jsonl record's worth of Diagnostic A/B/C/D/F metrics for a single
    representative checkpoint (diag_ctx, stashed by `_compute_step_policy` at the first sampled
    checkpoint of the batch). See diagnostics.py's module docstring for how to read the
    resulting fields together. Pure w.r.t. u/delta -- everything here reads existing tensors or
    solves an independent reference problem (QP, discretization, an actual gradient), never
    feeds back into the optimizer.
    '''
    v = diag_ctx["v"]
    G_obj = diag_ctx["G_obj"]
    rho_k = diag_ctx["rho_k"]
    den = diag_ctx["den"]
    pairs_k = diag_ctx["pairs_k"]
    w_pg = diag_ctx["w_pg"]

    record = {
        "B2_current_continuous": diag_ctx["B2_current"],
        "B2_qp_continuous": diag_ctx["B2_qp"],
    }
    if diag_ctx["B2_qp"] is not None:
        gap_abs, gap_rel = diag.qp_gap(diag_ctx["B2_current"], diag_ctx["B2_qp"])
        record["qp_gap_absolute"] = gap_abs
        record["qp_gap_relative"] = gap_rel

    u_np = diag.as_numpy(u)
    record.update({f"current_{k}": v_ for k, v_ in diag.policy_stats(
        u_np, pairs_k, beta, diag_policy_nnz_threshold, diag_policy_topk,
    ).items()})
    if diag_policy_full_vector:
        record["current_u_full"] = u_np.tolist()

    # Diagnostic B: geometric feasibility of v(delta) under G_obj @ U_loc, for both the
    # co-descended u and (if available) the QP reference w_pg.
    record.update(diag.geometric_feasibility(diag_ctx["Gu_current"], v, rho_k, "current"))
    record["v_norm"] = v.detach().norm().item()
    if diag_direction_scaling:
        record.update({f"{k}_current": v_ for k, v_ in diag.direction_amplitude_scaling(
            diag_ctx["Gu_current"], v,
        ).items()})

    # current-side constraint activity: cheap (u is already in hand), logged for symmetry with
    # the qp_* fields below, even though Section 4/7's main question is about the QP reference.
    activity_current = diag.constraint_activity(u_np, beta, pairs_k, pi, diag_constraint_tol)
    record["current_global_budget_active"] = activity_current["global_budget_active"]
    record["current_any_class_cap_active"] = activity_current["any_class_cap_active"]
    record["current_num_active_class_caps"] = activity_current["num_active_class_caps"]

    if w_pg is not None:
        Gu_qp = (G_obj @ w_pg).detach()
        record.update(diag.geometric_feasibility(Gu_qp, v, rho_k, "qp"))
        record.update({f"qp_{k}": v_ for k, v_ in diag.policy_stats(
            w_pg, pairs_k, beta, diag_policy_nnz_threshold, diag_policy_topk,
        ).items()})
        if diag_policy_full_vector:
            record["qp_u_full"] = diag.as_numpy(w_pg).tolist()

        # Section 3: is u_current close to u_qp (a lag problem) or far from it while B2 is
        # already close to B2_qp (an already-near-optimal-but-still-infeasible u)?
        l1_dist, l2_dist = diag.policy_distance(u_np, w_pg)
        record["policy_qp_l1_distance"] = l1_dist
        record["policy_qp_l2_distance"] = l2_dist

        # Section 4/7: which constraint of U_loc (if any) is binding at the QP reference --
        # tells whether sweeping beta could plausibly help (global budget active) or not
        # (a per-class cap is active instead -- that source class is simply out of examples).
        activity_qp = diag.constraint_activity(w_pg, beta, pairs_k, pi, diag_constraint_tol)
        record["qp_global_budget_active"] = activity_qp["global_budget_active"]
        record["qp_any_class_cap_active"] = activity_qp["any_class_cap_active"]
        record["qp_num_active_class_caps"] = activity_qp["num_active_class_caps"]
        record["qp_active_class_caps"] = activity_qp["active_class_caps"]

        # Section 6: `project_gradient_descent_local` is ALWAYS warm-started from the current
        # (co-descended) u -- both here (diag_qp_iters) and in the convergence sweep below (same
        # u_init passed to every check_iters value) -- so B2_qp_50/200/1000 are directly
        # comparable to each other and to B2_qp_continuous, never confounded by a different
        # starting point. See project_gradient_descent_local's own docstring.
        record["qp_warm_start"] = True

        if diag_direction_scaling:
            record.update({f"{k}_qp": v_ for k, v_ in diag.direction_amplitude_scaling(
                Gu_qp, v,
            ).items()})

    # Diagnostic C: discretization gap, for both the continuous current policy and the QP
    # reference, materialized via the EXACT federated_policy_to_flips counting rule.
    if diag_discretization and n_train is not None and class_counts is not None:
        u_discrete, n_realized_current = diag.discretize_policy(
            u_np, pairs_k, gamma, n_train, class_counts,
        )
        B2_current_discrete, Gu_current_discrete = diag.b2_value(G_obj, u_discrete, v, den)
        record["B2_current_discrete"] = B2_current_discrete
        gap_abs, gap_rel = diag.discretization_gap(diag_ctx["B2_current"], B2_current_discrete)
        record["discretization_gap_current_absolute"] = gap_abs
        record["discretization_gap_current_relative"] = gap_rel
        record["current_u_minus_u_discrete_l1"] = float(np.abs(u_np - u_discrete).sum())
        record["current_u_minus_u_discrete_l2"] = float(np.linalg.norm(u_np - u_discrete))

        if w_pg is not None and diag_ctx["B2_qp"] is not None:
            u_qp_discrete, n_realized_qp = diag.discretize_policy(
                w_pg, pairs_k, gamma, n_train, class_counts,
            )
            B2_qp_discrete, _ = diag.b2_value(G_obj, u_qp_discrete, v, den)
            record["B2_qp_discrete"] = B2_qp_discrete
            gap_abs, gap_rel = diag.discretization_gap(diag_ctx["B2_qp"], B2_qp_discrete)
            record["discretization_gap_qp_absolute"] = gap_abs
            record["discretization_gap_qp_relative"] = gap_rel
            record["qp_u_minus_u_discrete_l1"] = float(np.abs(w_pg.detach().cpu().numpy() - u_qp_discrete).sum())
            record["qp_u_minus_u_discrete_l2"] = float(
                np.linalg.norm(w_pg.detach().cpu().numpy() - u_qp_discrete)
            )

    # Diagnostic A.2.1: solver-convergence sweep -- only when explicitly enabled (expensive at
    # the largest check_iters value), only on this already-diagnostic batch. Every check_iters
    # value is warm-started from the SAME u.detach() (see project_gradient_descent_local's own
    # docstring and the qp_warm_start note above), so the B2_qp_<n> values below are directly
    # comparable to each other.
    if diag_qp_convergence and diag_ctx["c_np"] is not None:
        sweep = diag.qp_convergence_sweep(
            diag_ctx["Q_obj"], diag_ctx["c_np"], u.detach(), beta, pairs_k, pi,
            diag_qp_check_iters,
        )
        b2_by_iters = {}
        for n_iters, w in sweep.items():
            b2_val, _ = diag.b2_value(G_obj, w, v, den)
            b2_by_iters[n_iters] = b2_val
            record[f"B2_qp_{n_iters}"] = b2_val
        for (a, b), rel_improvement in diag.qp_convergence_relative_improvements(b2_by_iters).items():
            record[f"qp_{a}_vs_{b}_relative_improvement"] = rel_improvement

    # Section 7: unconstrained projection of v onto span(G_obj) -- separates a subspace-coverage
    # limitation (B2_span itself high) from a CONSTRAINT limitation (B2_span low, B2_qp high --
    # see qp_global_budget_active/qp_any_class_cap_active above for which constraint).
    if diag_span_projection and diag_ctx["c_np"] is not None:
        record.update(diag.span_projection(G_obj, diag_ctx["Q_obj"], diag_ctx["c_np"], v, den))

    # Diagnostic D: actual vs. predicted gradient shift -- expensive (materializes flips and
    # runs real forward/backward passes), only when explicitly requested for this batch.
    if diag_actual_gradient_run and w_pg is not None:
        actual_shift = diag.compute_actual_gradient_shift(
            diag_ctx["model"], loss_fn, class_samples_raw, pairs_k, pi, gamma, w_pg,
            dataset_flag, model_flag,
        )
        predicted_shift = (G_obj @ w_pg).detach()
        record.update(diag.actual_vs_predicted(actual_shift, predicted_shift))

    return record


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
    beta_global,
    lambda_poison,
    n_classes,
    class_samples_raw,
    pi,
    gamma,
    pairs,
    run_tag,
    device="cuda",
    dataset_flag="cifar",
    init="stripe",
    model_flag="r32p",
    checkpoint_backward=True,
    normalization="rho",
    diag_every=50,
    outer_step=0,
    diagnostics_writer=None,
    n_train=None,
    class_counts=None,
    diag_qp_iters=50,
    diag_qp_convergence=False,
    diag_qp_check_iters=(50, 200, 1000),
    diag_policy_nnz_threshold=1e-8,
    diag_policy_topk=10,
    diag_policy_full_vector=False,
    diag_discretization=True,
    diag_gradient_balance=True,
    diag_actual_gradient=False,
    diag_actual_gradient_every=0,
    diag_constraint_tol=1e-8,
    diag_span_projection=True,
    diag_direction_scaling=True,
):
    '''Runs one outer step's worth of trigger+policy-optimization batches against a fixed set
    of expert checkpoints (see `_compute_step_policy` for the per-checkpoint objective).
    Mirrors federated_optimizing_trigger.run_module.optimize_trigger_step, but replaces its
    QP-projected w* (compute_v_polytope_distance) with the jointly-learned policy u, stepped
    by its own Adam optimizer and projected onto u's LOCAL feasible set
    {u>=0, sum(u)<=beta, sum_c u_{y,c}<=pi_y} (`project_policy_budget`) after every batch: the
    discrete/QP label-flip feasibility check becomes an explicit, differentiable attack
    policy, co-trained with delta instead of solved in closed form at each step.

    Theory: prop:structure(iii) states min_delta min_u == min_{(delta,u)} -- a genuinely joint
    problem, not a decoupling; rem:solver then admits TWO solvers consistent with that: (a)
    exact inner QP solve for u*(delta) each outer step, detached, plus a Danskin
    projected-gradient step on delta (prop:danskin) -- the reported objective is then EXACTLY
    E_k[a_k/rho_k^2] and the delta-gradient is the true hypergradient; (b) simultaneous descent
    on delta AND u together, projecting u after each step -- cheaper, still a valid descent
    method on the joint objective, but u LAGS delta.

    PIEGE 3 (which solver, and what it costs): this function implements solver (b) -- u is
    co-descended with delta (two Adam optimizers, `optimizer_delta`/`optimizer_policy`, each
    stepped once per batch after a SINGLE shared `.backward()` -- see `_compute_step_policy`'s
    B2_k/L_bd_k, neither ever detaches u), NOT solver (a)'s resolved-and-detached optimum. Per
    rem:solver this has two consequences that must stay visible rather than being silently
    absorbed into "B2": the reported `B2` (B2_rho/B2_v) OVERSTATES `E_k[a_k/rho_k^2]` by
    whatever amount u lags delta, and prop:danskin does NOT apply -- delta's gradient here is
    NOT the hypergradient of V(delta):=min_u J(delta,u). `diag_every` controls the periodic
    diagnostic rem:solver itself prescribes as the reconciliation: solve the QP optimum on the
    SAME v_k (see `_compute_step_policy`) and report it (`B2_qp`) alongside the co-descended
    `B2`, so the gap -- the quantity to monitor -- is measured rather than assumed small.
    '''
    # G/Q depend only on (checkpoint, dataset), not on delta/u: fresh cache reused across every
    # batch of this call, discarded once these checkpoints are replaced -- same convention as
    # federated_optimizing_trigger's optimize_trigger_step. P3 (checkpoint-pool stability fix):
    # `sampled_k` is now redrawn PER BATCH (below), not once for the whole call -- conditioning
    # every batch's B2/B2_qp on the SAME fixed num_chckpt checkpoints for the entire call was
    # the same single-point-of-the-trajectory instability the sibling _trigger/_joint modules'
    # checkpoint pool addresses. flip_grad_cache needs no structural change to support this: it
    # already does a lazy per-k lookup (cache hit -> reuse G_obj/Q_obj; miss -> compute once and
    # store), so it already IS the pool -- redrawing sampled_k more often just lets it cover
    # more of `expert_models` over the course of this call (up to len(expert_models), each
    # entry computed at most once) instead of being capped at num_chckpt for the whole call.
    flip_grad_cache = {}

    total_steps = len(loader)
    pbar = make_pbar(
        loader,
        total=total_steps,
        desc="Optimizing trigger+policy",
        leave=False,
    )

    hinge_window = []
    metrics_history = {
        "B2": [], "L_bd": [], "lambda_effective": [], "lambda_effective_ratio": [],
        "beta_used": [],
        "B2_rho": [], "B2_v": [], "B2_qp": [],
        "B2_qp_relaxed": [], "pg_iters": [], "pg_obj_decrement": [],
    }

    diag_event_counter = 0

    for batch_idx, batch in enumerate(pbar):
        optimizer_delta.zero_grad()
        optimizer_policy.zero_grad()

        sampled_k = sample_checkpoints(
            len(expert_models), num_chckpt, alpha=alpha_ckpt, device=device,
        )

        run_diag = (batch_idx % diag_every == 0)
        diag_actual_gradient_run = False
        if run_diag:
            diag_actual_gradient_run = diag_actual_gradient and (
                diag_actual_gradient_every <= 0
                or diag_event_counter % diag_actual_gradient_every == 0
            )
            diag_event_counter += 1

        result = _compute_step_policy(
            batch, expert_models, sampled_k, source_label, target_label, loss_fn, delta, u,
            device, dataset_flag, model_flag, lambda_poison, n_classes,
            flip_grad_cache, class_samples_raw, pi, gamma, beta, beta_global, pairs,
            normalization, checkpoint_backward, lambda_bd, run_diag,
            diag_qp_iters=diag_qp_iters, diag_qp_convergence=diag_qp_convergence,
            diag_qp_check_iters=diag_qp_check_iters,
            diag_policy_nnz_threshold=diag_policy_nnz_threshold,
            diag_policy_topk=diag_policy_topk,
            diag_policy_full_vector=diag_policy_full_vector,
            diag_discretization=diag_discretization,
            diag_gradient_balance=diag_gradient_balance,
            diag_actual_gradient_run=diag_actual_gradient_run,
            n_train=n_train, class_counts=class_counts,
            diag_constraint_tol=diag_constraint_tol,
            diag_span_projection=diag_span_projection,
            diag_direction_scaling=diag_direction_scaling,
        )
        B2, L_bd = result["B2"], result["L_bd"]

        if run_diag and diagnostics_writer is not None and result["diag_record"] is not None:
            diagnostics_writer.write({
                "outer_step": outer_step, "batch_idx": batch_idx,
                "checkpoint": result["diag_checkpoint"],
                **result["diag_record"],
                "grad_delta_B2_norm": result["grad_delta_B2_norm"],
                "grad_delta_BD_norm": result["grad_delta_BD_norm"],
                "grad_delta_ratio": result["grad_delta_ratio"],
            })

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
            u.copy_(project_policy_budget(u, beta, pairs, pi))

        hinge_window.append(L_pen.item() > 0)
        if len(hinge_window) > WINDOW_SIZE:
            hinge_window.pop(0)

        beta_used = u.detach().sum().item()
        metrics_history["B2"].append(B2.item())
        metrics_history["L_bd"].append(L_bd.item())
        metrics_history["lambda_effective"].append(result["lambda_effective"])
        metrics_history["lambda_effective_ratio"].append(result["lambda_effective_ratio"])
        metrics_history["beta_used"].append(beta_used)
        metrics_history["B2_rho"].append(result["B2_rho"])
        metrics_history["B2_v"].append(result["B2_v"])
        if result["B2_qp"] is not None:
            metrics_history["B2_qp"].append(result["B2_qp"])
            metrics_history["B2_qp_relaxed"].append(result["B2_qp_relaxed"])
            metrics_history["pg_iters"].append(result["pg_iters"])
            metrics_history["pg_obj_decrement"].append(result["pg_obj_decrement"])

        postfix = {
            "B2": f"{B2.item():.6f}",
            "L_bd": f"{L_bd.item():.4f}",
            "lambda_eff": f"{result['lambda_effective']:.4f}",
            "lambda_eff_ratio": f"{result['lambda_effective_ratio']:.3f}",
            "L_pen": f"{L_pen.item():.4f}",
            "hinge_rate": f"{sum(hinge_window) / len(hinge_window):.2f}",
            "||delta||": f"{delta.norm().item():.4f}",
            "beta_used": f"{beta_used:.4f}",
        }
        if result["B2_qp"] is not None:
            # A4: B2_qp is now the U_loc-correct projected-gradient value; B2_qp_relaxed is
            # the old, looser-polytope QP value, kept for transition-period comparison.
            postfix["B2_qp"] = f"{result['B2_qp']:.6f}"
            postfix["B2_qp_relaxed"] = f"{result['B2_qp_relaxed']:.6f}"
            postfix["pg_decr"] = f"{result['pg_obj_decrement']:.3e}"
        pbar.set_postfix(postfix)

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
    normalization="rho",
    diag_every=50,
    expert_budget=None,
    diag_path=None,
    diag_qp_iters=50,
    diag_qp_convergence=False,
    diag_qp_check_iters=(50, 200, 1000),
    diag_policy_nnz_threshold=1e-8,
    diag_policy_topk=10,
    diag_policy_full_vector=False,
    diag_discretization=True,
    diag_gradient_balance=True,
    diag_actual_gradient=False,
    diag_actual_gradient_every=0,
    diag_constraint_tol=1e-8,
    diag_span_projection=True,
    diag_direction_scaling=True,
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

    # beta / lambda_poison resolution -- shared with federated_optimizing_trigger (unmodified,
    # see federated_optimizing_trigger/utils.py:resolve_beta_and_lambda_poison).
    # NOTE: unlike in federated_optimizing_trigger (where resolve_beta_and_lambda_poison's
    # docstring is accurate), num_honests/num_poisoned are NOT logging-only here: gamma =
    # num_poisoned / (num_poisoned + num_honests), derived from them below, enters the
    # (P^mean) objective directly (see `_compute_step_policy`) and the flip materialization
    # downstream (federated_policy_to_flips). beta remains the LOCAL rate (fraction of a
    # single corrupted worker's own shard) exactly as resolve_beta_and_lambda_poison computes
    # it -- resolve_beta_and_lambda_poison's own flip_budget formula
    # (beta * num_poisoned * n_train / n_w) already assumes this scope.
    lambda_poison_requested = lambda_poison  # A1: captured BEFORE resolution, see below
    beta, flip_budget, lambda_poison = resolve_beta_and_lambda_poison(
        beta, flip_budget, lambda_poison, num_poisoned, num_honests, n_train,
    )
    gamma = num_poisoned / (num_poisoned + num_honests)

    # pi (class frequencies) depends only on the dataset -- computed once, reused below and
    # for the B2-analog objective. Deliberately decoupled from train_expert's own `budget`
    # (which, if set, grows that module's OWN training set by an ADDED, not substituted,
    # poisoned segment -- see modules/base_utils/datasets.get_matching_datasets): pi_y here
    # describes the composition of the attacker's shard (what u operates on), not the
    # population the expert checkpoints were pretrained on. If `expert_budget` is given (the
    # `budget` value used for the train_expert run that produced these checkpoints, purely for
    # this log line -- NOT threaded into pi/n_train/beta), report how far the expert's actual
    # realized poison rate drifts from lambda_poison because of that addition-not-replacement:
    # lambda_expert_realise = expert_budget / (n_train + expert_budget) is typically a few
    # percentage points below lambda_poison for the same nominal budget.
    pi = compute_class_frequencies(dataset_flag, n_classes)
    pi_source = pi[source_label]

    # Diagnostic C (discretization): approximated per-class example counts, used ONLY to
    # replay federated_policy_to_flips's exact rounding+clipping rule (compute_flip_counts)
    # without re-scanning the training set's label array a second time -- pi is already an
    # empirical frequency over that same training set, computed once above. Not necessarily
    # bit-identical to the real per-class count federated_policy_to_flips sees (see
    # diagnostics.discretize_policy's docstring).
    diag_class_counts = {y: int(round(pi[y] * n_train)) for y in pi}

    # A1 correction (docs/policy_module_audit_report.md Section 2.6): resolve_beta_and_
    # lambda_poison (shared) resolves lambda_poison="beta" to `beta` itself -- mechanically
    # true, but `beta` above is the LOCAL rate (beta_theory/gamma), while eq:P's constraint
    # `lambda=beta` (sec:attacker-problem) names the GLOBAL beta of def:budget. Only the
    # "beta"-coupled case is corrected here: an EXPLICIT numeric lambda_poison in the config
    # (lambda_poison_requested is a float, not the string "beta") is used VERBATIM, unscaled --
    # that is the whole point of exposing it (see the s_beta warning below): it lets a campaign
    # sweep lambda independently of beta when the coupling isn't theoretically justified.
    beta_global = gamma * beta  # = beta_theory (def:budget) -- rem:units' aggregate-units beta
    if lambda_poison_requested == "beta":
        lambda_poison = beta_global
    print(
        f"beta_local={beta:.6f} (this module's own scope) -> beta_global={beta_global:.6f} "
        f"(= gamma*beta_local, def:budget) -- lambda_poison resolved to "
        f"{lambda_poison:.6f} ({'beta_global, A1-corrected' if lambda_poison_requested == 'beta' else 'explicit numeric override, unscaled'})."
    )

    # rem:saturated / prop:budget-match: eq:P's `lambda=beta` constraint (and the budget-
    # matching argument behind it, prop:budget-match) is only derived under
    # beta<=gamma*min_y(pi_y) (the UNSATURATED regime); prop:budget-match is explicitly listed
    # among the results "genuinely lost" once capacities bind (rem:saturated). s_beta > 1 means
    # this run's beta exceeds that threshold -- the lambda=beta coupling is not theoretically
    # justified here, and lambda should be swept explicitly rather than locked to beta.
    pi_min = min(pi.values())
    s_beta = beta_global / (gamma * pi_min)
    if s_beta > 1:
        print(
            f"WARNING: s_beta={s_beta:.4f} > 1 (beta_global={beta_global:.6f}, gamma={gamma:.4f}, "
            f"min_y(pi_y)={pi_min:.6f}) -- SATURATED regime (rem:saturated): "
            "prop:budget-match's hypothesis beta<=gamma*min_y(pi_y) does not hold, so eq:P's "
            "lambda=beta constraint is not theoretically justified for this configuration. "
            "Consider passing an explicit numeric lambda_poison (rather than the default "
            "\"beta\") and sweeping it independently."
        )
    else:
        print(f"s_beta={s_beta:.4f} <= 1 -- unsaturated regime, lambda=beta is justified (prop:budget-match).")

    # A3 (docs/policy_module_audit_report.md, option (ii)): _compute_step_policy's per-batch
    # mask can select at most idx_source.numel() rows (~pi_source*n_b), so lambda_effective is
    # structurally capped near pi_source whenever lambda_poison exceeds it -- this is
    # detectable a priori, not just observable on an unlucky batch, so it is a startup error
    # rather than a per-batch warning. theta_bar_k's retraining (get_poison_dataset, ADD-based,
    # below) does NOT have this cap -- it reaches lambda_poison exactly (mod overflow, already
    # logged) -- so leaving this uncaught would train theta_bar_k and compute v_k at two
    # DIFFERENT, silently-diverging rates. Chosen over lifting the cap via duplication (the
    # other option presented for arbitration): duplicating rows in the objective's own batch
    # would change what v_k measures, which is undesirable while A1 has just changed
    # lambda_poison's own scale -- see docs/policy_module_audit_report.md for the two options.
    if lambda_poison > pi_source:
        raise ValueError(
            f"A3: lambda_poison={lambda_poison:.6f} > pi_source={pi_source:.6f} -- "
            "_compute_step_policy's per-batch mask cannot realize this rate (it selects from "
            "the source-class rows actually present in a batch, ~pi_source*n_b, never "
            "duplicating), so lambda_effective would be structurally capped near pi_source "
            "while theta_bar_k's retraining (get_poison_dataset, ADD-based) reaches "
            "lambda_poison exactly -- the two would silently diverge. Lower beta (hence "
            "lambda_poison, if lambda_poison=\"beta\") or pass an explicit numeric "
            "lambda_poison already <= pi_source."
        )

    if expert_budget is not None:
        lambda_expert_realise = expert_budget / (n_train + expert_budget)
        print(
            f"lambda_poison={lambda_poison:.6f} (objective/expert-retraining target rate) vs "
            f"lambda_expert_realise={lambda_expert_realise:.6f} (train_expert's actual "
            f"realized rate with expert_budget={expert_budget}, n_train={n_train} -- "
            "get_matching_datasets ADDS poisoned examples rather than replacing clean ones)."
        )

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
    diagnostics_writer = diag.DiagnosticsWriter(diag_path)

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
        # get_train_info (modules/base_utils/util.py) falls back to DEFAULT_SGD_EPOCHS=200
        # whenever the passed `epochs` is falsy (0 or None) -- the same value federated_
        # train_user typically trains for. A config bug resolving `epochs` to a falsy value
        # would therefore silently make this per-step expert retraining run as long as
        # federated_train_user's own schedule instead of this module's configured `epochs`,
        # with no error. Fail loudly instead.
        expert_epochs = epochs_
        assert expert_epochs == epochs, (
            f"expert_epochs resolved to {expert_epochs}, expected config epochs={epochs} -- "
            "get_train_info silently substitutes DEFAULT_SGD_EPOCHS for a falsy `epochs`."
        )
        if step == 0:
            print(
                f"[hyperparams] expert_epochs={expert_epochs} (per outer step), "
                f"n_steps={n_steps}, batch_size={batch_size_}, train_flag={train_flag}, "
                f"lr_delta={lr_delta}, lr_policy={lr_policy}"
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

        print(
            f"[DIAG policy] calling mini_train: epochs={epochs_}, "
            f"n_train={len(poison_train_dataset)}, "
            f"n_test={len(clean_test_dataset) if clean_test_dataset else None}",
            flush=True,
        )
        _t_expert = _time.time()

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

        print(f"[DIAG policy] mini_train done in {_time.time()-_t_expert:.1f}s", flush=True)
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
            beta_global=beta_global,
            lambda_poison=lambda_poison,
            n_classes=n_classes,
            class_samples_raw=class_samples_raw,
            pi=pi,
            gamma=gamma,
            pairs=pairs,
            run_tag=run_tag,
            device=device,
            dataset_flag=dataset_flag,
            init=init,
            model_flag=model_flag,
            checkpoint_backward=checkpoint_backward,
            normalization=normalization,
            diag_every=diag_every,
            outer_step=step,
            diagnostics_writer=diagnostics_writer,
            n_train=n_train,
            class_counts=diag_class_counts,
            diag_qp_iters=diag_qp_iters,
            diag_qp_convergence=diag_qp_convergence,
            diag_qp_check_iters=diag_qp_check_iters,
            diag_policy_nnz_threshold=diag_policy_nnz_threshold,
            diag_policy_topk=diag_policy_topk,
            diag_policy_full_vector=diag_policy_full_vector,
            diag_discretization=diag_discretization,
            diag_gradient_balance=diag_gradient_balance,
            diag_actual_gradient=diag_actual_gradient,
            diag_actual_gradient_every=diag_actual_gradient_every,
            diag_constraint_tol=diag_constraint_tol,
            diag_span_projection=diag_span_projection,
            diag_direction_scaling=diag_direction_scaling,
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

    diagnostics_writer.close()

    return delta.detach(), u.detach(), pairs, beta, n_train, run_tag


def run(experiment_name, module_name, **kwargs):
    """
    Theory: sec:attacker-problem (eq:P) -- jointly optimizes a backdoor trigger (delta) and an
    explicit label-flipping attack policy (u), under mean aggregation -- problem (P^mean):

        min_{|delta|_inf<=eps, u in U_loc}
            E_k[ ||gamma * H(theta_bar_k) u - v_k(delta)||^2 / rho_k^2 ]
            + kappa * E_k[ E_X[ loss_c(f_theta_bar_k(T_delta(X)), y_target) ] ]
        s.t.  theta_bar_{k+1} = theta_bar_k - eta_k grad(theta_bar_k),  lambda = beta

    This is eq:P's LOCAL reading (the note right after its `lambda=beta` constraint): the
    decision variable here is the per-worker u^i (named `u`), not the aggregate ubar, so the
    first term is ||gamma*Gbar(theta_bar_k)*u^i - v_k(delta)||^2/rho_k^2 rather than
    ||Gbar(theta_bar_k)*ubar - v_k(delta)||^2/rho_k^2.

    u is LOCAL (rem:units): u_{y,c} is the fraction of a SINGLE corrupted worker's own shard
    flipped from y to c, u in U_loc = {u>=0, sum(u)<=beta, sum_c u_{y,c}<=pi_y} (eq:Uloc,
    `project_policy_budget`); beta is that same worker's own flip budget (its fraction of ITS
    shard, NOT a global fraction of the whole federated dataset -- NOT the `beta` of def:budget
    directly, see the A1 note below and `optimize_trigger_policy`'s docstring).
    H(theta_bar_k)[:,(y,c)] = g_{y,c}-g_{y,y} (def:shifts' Gbar, `compute_expected_flip_
    gradients`'s G with its pi_y factor divided back out per rem:no-pi -- PIEGE 1, see
    `_compute_step_policy`) is what ONE corrupted worker deploying u contributes to its own
    gradient message; under mean aggregation, gamma = num_poisoned/(num_poisoned+num_honests)
    corrupted workers (all deploying the SAME u, the homogeneous configuration of
    lem:hom-wlog -- PIEGE 2) contribute gamma*H(theta_bar_k)@u to the aggregated shift -- see
    `_compute_step_policy`'s docstring for the full derivation. v_k(delta) is the
    poisoning-induced gradient shift mu_p - g_c (eq:vk-delta). The kappa*L_bd term is the
    config's `lambda_bd` (kept distinct from `kappa`, which -- as in federated_optimizing_trigger
    -- names the *hinge margin* of the (optional) trigger-vs-mu penalty, not this loss weight).

    A1 (docs/policy_module_audit_report.md Section 2.6 -- CORRECTED): `lambda_poison="beta"`
    (the default) now resolves to `beta_global := gamma*beta` (== `beta_theory`, def:budget) as
    eq:P's constraint `lambda=beta` requires -- NOT this module's own LOCAL `beta` directly, as
    it did before this correction (which silently over-poisoned by `1/gamma`, gamma<=1). This
    coupling is only theoretically justified when `s_beta := beta_global/(gamma*min_y(pi_y))
    <= 1` (prop:budget-match's unsaturated-regime hypothesis; `rem:saturated` lists
    `prop:budget-match` among the results lost once it fails) -- a warning prints at startup
    when `s_beta > 1`. An explicit numeric `lambda_poison` (used verbatim, unscaled) is the way
    to sweep lambda independently of beta in that regime. See `optimize_trigger_policy`'s
    docstring for the full derivation.

    Threat model: same as federated_optimizing_trigger (see its `run` docstring) -- the
    attacker needs only the model architecture, a sample from the training distribution, its
    own budget beta, and (y_source, y_target). UNLIKE federated_optimizing_trigger,
    num_honests/num_poisoned are NOT logging-only here: gamma, derived from them, enters the
    (P^mean) objective and (via `federated_policy_to_flips`) the flip materialization -- see
    `optimize_trigger_policy`'s docstring.

    Outputs: the optimized trigger (.pt, same naming convention as
    federated_optimizing_trigger) and the optimized policy (u, pairs, beta, n_train -- .npz),
    consumed downstream by `federated_policy_to_flips` to materialize concrete per-worker
    label flips, then by (unmodified) `federated_train_user` to train and evaluate the victim.
    """
    slurm_id = kwargs.get("slurm_id", None)
    args = extract_toml(experiment_name, module_name)

    print(f"[DIAG policy] config epochs={args.get('epochs', 'NOT SET')}", flush=True)
    print(f"[DIAG policy] config n_steps={args.get('n_steps', 'NOT SET')}", flush=True)

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
    normalization = args.get("normalization", "rho")
    diag_every = args.get("diag_every", 50)
    expert_budget = args.get("expert_budget", None)

    diag_path = args.get("diag_path", None)
    if diag_path is not None:
        diag_path = slurmify_path(diag_path, slurm_id)
    diag_qp_iters = args.get("diag_qp_iters", 50)
    diag_qp_convergence = args.get("diag_qp_convergence", False)
    diag_qp_check_iters = args.get("diag_qp_check_iters", [50, 200, 1000])
    diag_policy_nnz_threshold = args.get("diag_policy_nnz_threshold", 1e-8)
    diag_policy_topk = args.get("diag_policy_topk", 10)
    diag_policy_full_vector = args.get("diag_policy_full_vector", False)
    diag_discretization = args.get("diag_discretization", True)
    diag_gradient_balance = args.get("diag_gradient_balance", True)
    diag_actual_gradient = args.get("diag_actual_gradient", False)
    diag_actual_gradient_every = args.get("diag_actual_gradient_every", 0)
    diag_constraint_tol = args.get("diag_constraint_tol", 1e-8)
    diag_span_projection = args.get("diag_span_projection", True)
    diag_direction_scaling = args.get("diag_direction_scaling", True)

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
        normalization=normalization,
        diag_every=diag_every,
        expert_budget=expert_budget,
        diag_path=diag_path,
        diag_qp_iters=diag_qp_iters,
        diag_qp_convergence=diag_qp_convergence,
        diag_qp_check_iters=diag_qp_check_iters,
        diag_policy_nnz_threshold=diag_policy_nnz_threshold,
        diag_policy_topk=diag_policy_topk,
        diag_policy_full_vector=diag_policy_full_vector,
        diag_discretization=diag_discretization,
        diag_gradient_balance=diag_gradient_balance,
        diag_actual_gradient=diag_actual_gradient,
        diag_actual_gradient_every=diag_actual_gradient_every,
        diag_constraint_tol=diag_constraint_tol,
        diag_span_projection=diag_span_projection,
        diag_direction_scaling=diag_direction_scaling,
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
    # num_honests/num_poisoned/gamma: saved so federated_policy_to_flips can cross-check its
    # OWN config against what this policy was actually optimized against, instead of each
    # module silently recomputing gamma from its own (possibly divergent) num_honests/
    # num_poisoned -- see federated_policy_to_flips/run_module.py's cross-check.
    np.savez(
        policy_path,
        u=u.detach().cpu().numpy(),
        pairs_y=pairs_arr[:, 0],
        pairs_c=pairs_arr[:, 1],
        beta=np.array(beta_resolved, dtype=np.float64),
        n_train=np.array(n_train, dtype=np.int64),
        source_label=np.array(y_source, dtype=np.int64),
        target_label=np.array(y_target, dtype=np.int64),
        num_honests=np.array(num_honests, dtype=np.int64),
        num_poisoned=np.array(num_poisoned, dtype=np.int64),
        gamma=np.array(num_poisoned / (num_poisoned + num_honests), dtype=np.float64),
    )
    print(f"Saved trigger to {trig_path}")
    print(f"Saved policy to {policy_path}")


if __name__ == "__main__":
    run("optimizing_trigger_policy_example", "federated_optimizing_trigger_policy")
