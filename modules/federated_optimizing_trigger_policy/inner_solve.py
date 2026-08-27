"""
Controlled inner-solve experiment for federated_optimizing_trigger_policy's co-descended
(delta, u) attack. Implements `policy_inner_mode` (see run_module.py's `run()` docstring and
schemas/federated_optimizing_trigger_policy.toml): a way to compare the CURRENT joint co-descent
against alternatives that push u closer to its conditional optimum u*(delta) BEFORE delta is
updated, to test whether policy lag (u trailing delta, quantified by the B2 >> B2_qp diagnostic
in diagnostics.py) is responsible for part of the attack's failure.

Modes ("joint" is the ONLY one exercised by default -- see run_module.py's `policy_inner_mode`
config option):
  - "joint": unchanged co-descent (this module is not even imported on that path for the
    per-batch update itself -- only `get_or_build_flip_grad_cache_entry` below is shared, a
    pure extraction of run_module.py's existing cache-fill logic, not a behavior change).
  - "multi_step": K plain (Adam, via the EXISTING optimizer_policy) gradient steps on u against
    the AGGREGATE quadratic B2(u) for the CURRENT delta held fixed, then ONE delta update.
  - "qp_pgd" / "qp_pgd_reset": an aggressive projected-gradient QP solve (this module's own
    `qp_pgd_solve`, distinct from diagnostics.py's `project_gradient_descent_local` -- that
    function is a DIAGNOSTIC over a single representative checkpoint, warm-started from u,
    fixed at n_iters; this one solves the actual MULTI-CHECKPOINT aggregate problem the batch's
    B2 objective is built from, with an early-stop tolerance) against the same aggregate
    problem, replacing u directly (`u.copy_`, no `optimizer_policy.step()` in this mode -- see
    run_module.py's batch loop). "_reset" cold-starts from u=0 instead of warm-starting from the
    current u, to separate a warm-start bias from genuine under-iteration.

In every non-"joint" mode, delta's own update uses u AFTER the inner solve, held constant (a
plain tensor value, not requiring grad) -- run_module.py achieves this by passing `u.detach()`
into `_compute_step_policy` for the delta-update pass, so NO gradient flows through the inner
solve into delta (an alternating-minimization scheme, not yet a bilevel/implicit-gradient one --
see run_module.py's docstring for why this is a deliberate first step).

The AGGREGATE QP (Section 3 of the task): for sampled checkpoints k=1..K, this module's B2 is
    B2(delta, u) = mean_k [ ||G_k @ u - v_k(delta)||^2 / den_k ]
(den_k = rho_k^2+eps under normalization="rho", or ||v_k||^2+eps under "v" -- EXACTLY
run_module.py's own den_rho/den_v, never re-derived here). Minimizing over u (delta fixed, so
v_k/den_k are constants) is the quadratic
    min_u  0.5 u^T Q u - c^T u,   Q = mean_k(G_k^T G_k / den_k),   c = mean_k(G_k^T v_k / den_k)
-- note the 1/den_k weight is INSIDE the mean: with different checkpoints generally having
different den_k, this is NOT the same problem as summing the K checkpoints unweighted (that
would silently up-weight whichever checkpoint happens to have the smallest den_k).
"""
import numpy as np
import torch

from modules.federated_optimizing_trigger.utils import (
    compute_batch_gradients, compute_expected_flip_gradients,
    raw_to_preprocess, raw_to_trigger_preprocess,
)
from modules.federated_optimizing_trigger_policy.utils import (
    project_policy_budget, project_gradient_descent_local,
)
from modules.federated_optimizing_trigger_policy import diagnostics as diag

_EPS_DEN = 1e-8


def get_or_build_flip_grad_cache_entry(
    k, flip_grad_cache, M, loss_fn, class_samples_raw, n_classes, pi, dataset_flag, model_flag,
    params, gamma, beta, beta_global,
):
    '''
    Returns (G_obj, Q_obj, pairs_k, rho_k) for checkpoint k, filling `flip_grad_cache[k]` on a
    miss. Pure extraction of `_compute_step_policy`'s own cache-fill block (run_module.py) --
    same formulas, same A2 self-check, NOT a new convention -- factored out here so the new
    inner-solve path (`build_inner_context` below) can reuse the EXACT SAME (G_obj, Q_obj,
    rho_k) the joint path uses for each checkpoint, from the SAME shared `flip_grad_cache`
    dict, instead of a second, possibly-drifting implementation. See run_module.py's
    `_compute_step_policy` docstring for the full derivation (PIEGE 1/PIEGE 2, eq:rho/eq:varsigma).
    '''
    if k in flip_grad_cache:
        return flip_grad_cache[k]

    G_k, Q_k, pairs_k = compute_expected_flip_gradients(
        M, loss_fn, class_samples_raw, n_classes, pi,
        dataset_flag=dataset_flag, model_flag=model_flag, params=params,
    )
    scale = torch.tensor(
        [gamma / pi[y] for (y, c) in pairs_k], device=G_k.device, dtype=G_k.dtype,
    )
    G_obj = G_k * scale
    scale_np = scale.detach().cpu().numpy().astype(np.float64)
    Q_obj = np.outer(scale_np, scale_np) * Q_k

    pi_col = torch.tensor([pi[y] for (y, c) in pairs_k], device=G_k.device, dtype=G_k.dtype)
    varsigma_k = (G_k.detach() / pi_col).norm(dim=0).max().item()
    rho_k = beta_global * varsigma_k

    if len(flip_grad_cache) == 0:
        rho_k_check = beta * G_obj.detach().norm(dim=0).max().item()
        assert abs(rho_k_check - rho_k) < 1e-2 * max(abs(rho_k), 1e-8), (
            f"A2 self-check failed: rho_k={rho_k:.6f} (beta_global*varsigma_k) vs "
            f"rho_k_check={rho_k_check:.6f} (beta_local*max_col_norm(G_obj)) -- see "
            "run_module.py's _compute_step_policy docstring."
        )

    flip_grad_cache[k] = (G_obj, Q_obj, pairs_k, rho_k)
    return flip_grad_cache[k]


def compute_frozen_v(M, loss_fn, x_clean, y, x_raw, mask, y_poison, has_poison, delta_frozen,
                      dataset_flag, model_flag):
    '''
    v_k(delta) = grad_Lp[delta](theta_k) - grad_Lc(theta_k), computed with delta HELD FIXED
    (`delta_frozen`, expected to be a plain detached tensor -- e.g. `delta.detach()`): no
    create_graph/retain_graph needed since we never need a delta-gradient out of this call (the
    inner solve only ever needs v_k's VALUE), unlike `_compute_step_policy`'s own v computation
    (create_graph=True there, so mu_p keeps a graph back to delta for the OUTER, delta-update
    pass). Returns a detached (D,) tensor.
    '''
    x_poisoned = x_clean.clone()
    if has_poison:
        x_poisoned[mask] = raw_to_trigger_preprocess(
            x_raw[mask], delta_frozen, dataset_flag=dataset_flag, model_flag=model_flag,
        )

    grads_c, _ = compute_batch_gradients(M, loss_fn, (x_clean, y), create_graph=False)
    g_c = torch.cat([g.reshape(-1) for g in grads_c]).detach()

    grads_p, _ = compute_batch_gradients(M, loss_fn, (x_poisoned, y_poison), create_graph=False)
    mu_p = torch.cat([g.reshape(-1) for g in grads_p]).detach()

    return mu_p - g_c


def build_inner_context(
    expert_models, sampled_k, x_clean, y, x_raw, mask, y_poison, has_poison, delta_frozen,
    loss_fn, dataset_flag, model_flag, n_classes, flip_grad_cache, class_samples_raw, pi, gamma,
    beta, beta_global, normalization, device,
):
    '''
    Builds, for EACH sampled checkpoint, the (G_obj, v, den) triple the aggregate QP is built
    from -- delta held fixed at `delta_frozen` throughout (no forward pass here depends on a
    delta that could still change mid-way through this list, since it is the SAME snapshot for
    every k). Reuses `flip_grad_cache` (shared with `_compute_step_policy`'s own per-batch
    cache) so a checkpoint computed by one path is not recomputed by the other within the same
    batch.

    Returns: list of dicts {k, G_obj, Q_obj, pairs_k, rho_k, v, den}.
    '''
    contexts = []
    for k in sampled_k:
        M = expert_models[k].to(device).eval()
        params = list(M.parameters())
        G_obj, Q_obj, pairs_k, rho_k = get_or_build_flip_grad_cache_entry(
            k, flip_grad_cache, M, loss_fn, class_samples_raw, n_classes, pi, dataset_flag,
            model_flag, params, gamma, beta, beta_global,
        )
        v = compute_frozen_v(
            M, loss_fn, x_clean, y, x_raw, mask, y_poison, has_poison, delta_frozen,
            dataset_flag, model_flag,
        )
        den_v = v.norm() ** 2 + _EPS_DEN
        den_rho = rho_k ** 2 + _EPS_DEN
        den = float(den_rho if normalization == "rho" else den_v)
        contexts.append({
            "k": k, "G_obj": G_obj, "Q_obj": Q_obj, "pairs_k": pairs_k, "rho_k": rho_k,
            "v": v, "den": den,
        })
    return contexts


def aggregate_qp(contexts):
    '''
    Q = mean_k(Q_obj_k / den_k), c = mean_k(G_obj_k^T @ v_k / den_k) -- see this module's
    docstring for why the 1/den_k weight must be INSIDE the mean. Assumes every context shares
    the same `pairs_k` ordering (same assumption `_compute_step_policy`'s own B2_qp diagnostic
    already makes for a single checkpoint).

    Returns (Q: (P,P) float64 ndarray, c: (P,) float64 ndarray, pairs: the shared pairs list).
    '''
    K = len(contexts)
    pairs = contexts[0]["pairs_k"]
    Q_sum = np.zeros_like(contexts[0]["Q_obj"], dtype=np.float64)
    c_sum = np.zeros(Q_sum.shape[0], dtype=np.float64)
    for ctx in contexts:
        Q_sum += ctx["Q_obj"] / ctx["den"]
        c_k = (ctx["G_obj"].T @ ctx["v"]).detach().cpu().numpy().astype(np.float64)
        c_sum += c_k / ctx["den"]
    return Q_sum / K, c_sum / K, pairs


def aggregate_b2(u, contexts):
    '''mean_k ||G_obj_k @ u - v_k||^2 / den_k -- the SAME quantity `_compute_step_policy`'s own
    B2 (averaged over sampled_k) computes, evaluated here for an arbitrary `u` (e.g. before/
    after an inner solve) without needing a fresh forward/backward pass.'''
    return float(np.mean([diag.b2_value(ctx["G_obj"], u, ctx["v"], ctx["den"])[0] for ctx in contexts]))


def qp_pgd_solve(Q, c, u_init, beta, pairs, pi, max_iters=200, tol=1e-8, min_iters=10, ridge=1e-6):
    '''
    Aggressive projected-gradient solve of min_{u in U_loc} 0.5 u^T Q u - c^T u (Q, c the
    AGGREGATE multi-checkpoint problem from `aggregate_qp`) -- fixed step size 1/L (L = Q's
    largest eigenvalue, ridge-regularized, same non-expansive-step convention as
    diagnostics.py's `project_gradient_descent_local`, but this is a SEPARATE function: that one
    is a diagnostic fixed at n_iters over a single checkpoint; this one is the actual algorithm
    path for policy_inner_mode="qp_pgd"/"qp_pgd_reset", with early stopping.

    Stops early once `actual_iters >= min_iters` AND the relative objective improvement over the
    last step drops below `tol` -- an EMPIRICAL plateau criterion (the iterate need not be within
    `tol` of the true optimum, just no longer improving fast under this exact step schedule), not
    a certified optimality gap. Always respects min_iters <= actual_iters <= max_iters.

    Returns: (u_star: (P,) torch.float64 tensor, actual_iters: int, obj_start: float,
              obj_end: float, converged: bool).
    '''
    P = Q.shape[0]
    Q_reg = Q + ridge * np.eye(P)
    L = float(np.linalg.eigvalsh(Q_reg).max())
    lr = 1.0 / max(L, ridge)

    w_np = np.asarray(
        u_init.detach().cpu().numpy() if torch.is_tensor(u_init) else u_init, dtype=np.float64,
    ).copy()

    def obj(w):
        return 0.5 * float(w @ Q_reg @ w) - float(c @ w)

    obj_start = obj(w_np)
    prev_obj = obj_start
    converged = False
    actual_iters = 0
    w_t = torch.as_tensor(w_np)

    for _ in range(max_iters):
        grad = Q_reg @ w_np - c
        w_np = w_np - lr * grad
        w_t = project_policy_budget(torch.as_tensor(w_np), beta, pairs, pi)
        w_np = w_t.detach().cpu().numpy().astype(np.float64)
        actual_iters += 1
        cur_obj = obj(w_np)

        if actual_iters >= min_iters:
            rel_improve = (prev_obj - cur_obj) / max(abs(prev_obj), 1e-12)
            prev_obj = cur_obj
            if rel_improve < tol:
                converged = True
                break
        else:
            prev_obj = cur_obj

    return w_t, actual_iters, obj_start, prev_obj, converged


def pairwise_cosine_stats(u_list, eps=1e-8):
    '''
    cos(u_i, u_j) for every pair i<j in `u_list` -- Etape 0 (one-shot coupling audit), Question
    2: how much do the independent per-checkpoint QP optima u*_k actually agree with each
    other? A low mean/min cosine here means a single shared ubar necessarily compromises badly
    on SOME checkpoints (they disagree on which direction to flip), independent of how well any
    per-checkpoint solver converges.

    Returns {"mean", "min", "max", "pairs": [((i, j), cos), ...]} -- None fields if fewer than
    2 policies are given (nothing to pair).
    '''
    n = len(u_list)
    if n < 2:
        return {"mean": None, "min": None, "max": None, "pairs": []}
    arrs = [diag.as_numpy(u) for u in u_list]
    pairs_out = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = arrs[i], arrs[j]
            denom = (np.linalg.norm(a) * np.linalg.norm(b))
            cos = float(np.dot(a, b) / denom) if denom > eps else None
            pairs_out.append(((i, j), cos))
    vals = [c for _, c in pairs_out if c is not None]
    return {
        "mean": float(np.mean(vals)) if vals else None,
        "min": float(np.min(vals)) if vals else None,
        "max": float(np.max(vals)) if vals else None,
        "pairs": pairs_out,
    }


def oneshot_coupling_diagnostic(contexts, u_init, beta, pairs, pi, n_iters=200, ridge=1e-6):
    '''
    Etape 0 of the "switch to the exact QP solver" task -- audits whether a SINGLE, coupled
    ubar (eq:qp: Q = mean_k(Q_k/den_k), c = mean_k(c_k/den_k), exactly `aggregate_qp` below)
    can serve the whole `contexts` window at once, or whether the checkpoints disagree enough
    that per-checkpoint optima and the coupled optimum diverge materially.

    For each checkpoint in `contexts`, solves ITS OWN independent QP via
    `project_gradient_descent_local` -- the SAME function `_compute_step_policy`'s existing
    B2_qp diagnostic already uses (not a new solver, so this is a fair, apples-to-apples
    per-checkpoint baseline) -- giving u*_k. Then:
      1. pairwise_cosine_stats(u*_k list) -- Question 2: how much do the u*_k actually agree?
      2. solves the COUPLED one-shot QP (`aggregate_qp` + `qp_pgd_solve`) -> ubar.
      3. oneshot_gap = J(ubar) - mean_k(B2 at u*_k) -- prop:oneshot-gap (Question 3): how much
         WORSE the single shared policy is, on this window, than the (unrealistic, since it
         cannot be materialized as one policy) average of each checkpoint's own best response.

    NOTE ON SCOPE: `contexts` is whatever window is passed in -- in run_module.py's usage this
    is `sampled_k`, ONE BATCH's small (num_chckpt-sized) sample of checkpoints, redrawn every
    batch (see optimize_trigger_policy_step's P3 comment) -- NOT eq:qp's full [k_0, K]
    trajectory window. Solving the coupled QP over every checkpoint in the run at every
    diagnostic batch would be prohibitively expensive and defeats the point of the existing
    checkpoint subsampling; this diagnostic answers "do checkpoints already disagree within a
    single small sample", a necessary (not sufficient) condition for full-trajectory coupling
    to make sense. A stronger, full-window version would need aggregating this same
    computation across many diagnostic batches/checkpoints outside this function's scope.

    Returns a dict: oneshot_window_size, u_star_pairwise_cosine_mean/min/max,
    B2_per_checkpoint_mean (== what B2_qp already reports, recomputed here independently as a
    consistency check), B2_coupled, oneshot_gap_absolute, oneshot_gap_relative,
    coupled_actual_iters, coupled_converged.

    Etape 0bis.1 (follow-up task -- "the decisive comparison that's missing"): the three
    quantities the decision hinges on -- J(ubar) [B2_coupled], B2_current (the CO-DESCENDED
    policy, `u_init`), and mean_k(B2_QP,k) [B2_per_checkpoint_mean] -- reported on the EXACT
    SAME window, in TWO units: RAW (mean_k ||G_k@u - v_k||^2, no normalization at all -- what
    the existing den-based B2/B2_qp/B2_per_checkpoint_mean/B2_coupled fields are NOT: those
    divide by each checkpoint's OWN den, rho^2 or ||v_k||^2 depending on `normalization`, so
    B2_current_continuous elsewhere is evaluated on a SINGLE representative checkpoint only,
    not this whole window -- an apples-to-oranges comparison this fixes) and ||v||^2-WINDOW-
    normalized (divided by the SAME shared denominator, v_sq_window_mean = mean_k(||v_k||^2),
    for all three -- so they sit on a directly comparable scale, independent of `normalization`
    and of which single checkpoint a diagnostic happens to have picked as representative).

    Etape 0bis.2 (follow-up task -- "is the gap drift or noise?"): per_checkpoint_u_star tags
    each u*_k by its checkpoint id (contexts' own "k"), so an offline reader (see
    prelim/analyze_oneshot_gap_diagnostics.py) can group u*_k observations by checkpoint id
    ACROSS MANY diagnostic batches and separate the pairwise cosine into INTRA-checkpoint (same
    theta_k, different minibatch -- estimation noise on G/v) vs. INTER-checkpoint (different
    theta_k -- genuine trajectory drift). This function itself only ever sees ONE batch's
    window, so it cannot do that separation -- it only supplies the tagged raw material.
    '''
    u_star_per_k = []
    b2_per_k = []
    sq_err_ustar_per_k = []
    sq_err_current_per_k = []
    v_sq_per_k = []
    for ctx in contexts:
        c_k = (ctx["G_obj"].T @ ctx["v"]).detach().cpu().numpy().astype(np.float64)
        u_star_k, _, _ = project_gradient_descent_local(
            ctx["Q_obj"], c_k, u_init, beta, ctx["pairs_k"], pi, n_iters=n_iters, ridge=ridge,
        )
        u_star_per_k.append(u_star_k)
        b2_val, Gu_star = diag.b2_value(ctx["G_obj"], u_star_k, ctx["v"], ctx["den"])
        b2_per_k.append(b2_val)
        sq_err_ustar_per_k.append(((Gu_star - ctx["v"]) ** 2).sum().item())

        u_current_t = torch.as_tensor(
            diag.as_numpy(u_init), dtype=ctx["G_obj"].dtype, device=ctx["G_obj"].device,
        )
        Gu_current = ctx["G_obj"] @ u_current_t
        sq_err_current_per_k.append(((Gu_current - ctx["v"]) ** 2).sum().item())
        v_sq_per_k.append(ctx["v"].norm().item() ** 2)

    cosine_stats = pairwise_cosine_stats(u_star_per_k)
    b2_per_checkpoint_mean = float(np.mean(b2_per_k))

    Q_agg, c_agg, pairs_agg = aggregate_qp(contexts)
    ubar, actual_iters, _, _, converged = qp_pgd_solve(
        Q_agg, c_agg, u_init, beta, pairs_agg, pi,
        max_iters=n_iters, tol=1e-12, min_iters=n_iters, ridge=ridge,
    )
    b2_coupled = aggregate_b2(ubar, contexts)

    eps = 1e-8
    gap_abs = b2_coupled - b2_per_checkpoint_mean
    gap_rel = gap_abs / max(abs(b2_per_checkpoint_mean), eps)

    # 0bis.1: raw + shared-||v||^2-window-normalized versions of the three quantities.
    v_sq_window_mean = float(np.mean(v_sq_per_k))
    sq_err_coupled_per_k = [
        ((ctx["G_obj"] @ torch.as_tensor(diag.as_numpy(ubar), dtype=ctx["G_obj"].dtype,
                                          device=ctx["G_obj"].device) - ctx["v"]) ** 2).sum().item()
        for ctx in contexts
    ]
    b2_current_raw = float(np.mean(sq_err_current_per_k))
    b2_per_checkpoint_mean_raw = float(np.mean(sq_err_ustar_per_k))
    b2_coupled_raw = float(np.mean(sq_err_coupled_per_k))

    return {
        "oneshot_window_size": len(contexts),
        "u_star_pairwise_cosine_mean": cosine_stats["mean"],
        "u_star_pairwise_cosine_min": cosine_stats["min"],
        "u_star_pairwise_cosine_max": cosine_stats["max"],
        "B2_per_checkpoint_mean": b2_per_checkpoint_mean,
        "B2_coupled": b2_coupled,
        "oneshot_gap_absolute": gap_abs,
        "oneshot_gap_relative": gap_rel,
        "coupled_actual_iters": actual_iters,
        "coupled_converged": converged,
        # 0bis.1
        "v_sq_window_mean": v_sq_window_mean,
        "B2_current_window_raw": b2_current_raw,
        "B2_current_window_v2norm": b2_current_raw / max(v_sq_window_mean, eps),
        "B2_per_checkpoint_mean_raw": b2_per_checkpoint_mean_raw,
        "B2_per_checkpoint_mean_v2norm": b2_per_checkpoint_mean_raw / max(v_sq_window_mean, eps),
        "B2_coupled_raw": b2_coupled_raw,
        "B2_coupled_v2norm": b2_coupled_raw / max(v_sq_window_mean, eps),
        # 0bis.2
        "per_checkpoint_u_star": {
            str(ctx["k"]): diag.as_numpy(u_star_k).tolist()
            for ctx, u_star_k in zip(contexts, u_star_per_k)
        },
    }


def check_feasible(u, beta, pairs, pi, tol=1e-8):
    '''u in U_loc = {u>=0, sum(u)<=beta, sum_c u_{y,c}<=pi_y}, within `tol`. Used after an inner
    solve (Section 8) to confirm the replacement policy is actually feasible -- both
    `qp_pgd_solve` and `multi_step_update` project onto U_loc themselves, so this should always
    be true; a failure here would point at a projection or dtype/device bug, not the algorithm.'''
    u_np = diag.as_numpy(u)
    if (u_np < -tol).any():
        return False
    if u_np.sum() > beta + tol:
        return False
    ys = sorted(set(y for y, _ in pairs))
    for y in ys:
        idx = [i for i, (yy, _) in enumerate(pairs) if yy == y]
        if u_np[idx].sum() > pi[y] + tol:
            return False
    return True


def multi_step_update(u, Q_agg, c_agg, beta, pairs, pi, optimizer_policy, k_steps):
    '''
    K plain gradient steps of the EXISTING `optimizer_policy` (Adam) against the aggregate
    quadratic 0.5 u^T Q_agg u - c_agg^T u (delta held fixed -- Q_agg/c_agg were built from a
    frozen-delta context, see `build_inner_context`/`aggregate_qp`), projecting onto U_loc after
    every step. `u` is updated IN PLACE (leaf tensor, `requires_grad` untouched); no delta
    gradient is ever computed here (Q_agg/c_agg are plain numpy arrays, no graph to delta).

    Returns a metrics dict: actual_iters (== k_steps, always -- no early stop in this mode,
    unlike qp_pgd), obj_start, obj_end (0.5 u^T Q_agg u - c_agg^T u, NOT the full B2 with its
    constant v^T v/den term -- comparable across the K steps of THIS call, not directly to
    B2/B2_qp; use `aggregate_b2` before/after for that).
    '''
    Q_t = torch.as_tensor(Q_agg, dtype=u.dtype, device=u.device)
    c_t = torch.as_tensor(c_agg, dtype=u.dtype, device=u.device)

    def qp_obj(u_t):
        return (0.5 * (u_t @ Q_t @ u_t) - (c_t @ u_t)).item()

    obj_start = qp_obj(u.detach())
    for _ in range(k_steps):
        optimizer_policy.zero_grad()
        loss_u = 0.5 * (u @ Q_t @ u) - (c_t @ u)
        loss_u.backward()
        optimizer_policy.step()
        with torch.no_grad():
            u.copy_(project_policy_budget(u, beta, pairs, pi))
    obj_end = qp_obj(u.detach())

    return {
        "actual_iters": k_steps, "obj_start": obj_start, "obj_end": obj_end,
        "obj_decrement": obj_start - obj_end, "converged": None,
    }
