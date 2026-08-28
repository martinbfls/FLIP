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


def estimate_v_analytic(
    M, loss_fn, x_raw, x_clean, y, source_label, target_label, delta, lambda_eff,
    dataset_flag, model_flag, create_graph, need_L_bd=False,
):
    '''
    Task 3 (variance-reduced v estimator, v_estimator="analytic"): the "subsample" estimator
    (see `_compute_step_policy` docstring / this module's other v computation below) estimates
    v_k over only `m = round(lambda_poison*n_b)` examples (the per-batch subsampled mask,
    typically a handful), while G's columns are estimated over `flip_gradient_samples_per_class`
    examples -- Var(v) then dominates E[v]^2 in the ~4.6e5-dimensional gradient space, making B2
    and its delta-gradient mostly noise.

    Let S = ALL source-class rows of this batch (not subsampled) and lambda_eff = the
    "subsample" estimator's OWN target_count/n_b (unchanged -- this is a scale factor, not a
    sample-size choice):

        g_bd = grad_theta[ mean_{i in S} CE(f_theta(T_delta(x_i)), y_target) ]
        g_ss = grad_theta[ mean_{i in S} CE(f_theta(x_i),          y_source) ]
        v    = lambda_eff * (g_bd - g_ss)

    Same expectation as the "subsample" estimator conditionally on the batch (both are
    lambda_eff times a mean gradient difference over an unbiased sample of the source class),
    variance divided by roughly |S|/m ~= pi_source/lambda_poison. Also cheaper: two passes over
    the source-class sub-batch only, instead of two passes over the full batch.

    `create_graph` controls whether g_bd keeps a graph back to `delta` (True for the
    delta-update pass in `_compute_step_policy`, so B2/L_bd can backprop into delta; False for
    the inner-solve path in `compute_frozen_v`, which only ever needs v's VALUE against a frozen
    delta). g_ss never depends on delta and is always detached.

    Returns (v, L_bd) -- L_bd (CE on ALL of S, the direct eq:P reading of E_{X~D_s}, same
    variance reduction as v itself) only computed when `need_L_bd=True`, else None. Caller is
    responsible for ensuring the batch actually has source-class rows (task 2's has_poison
    check, upstream of every caller here, already guarantees this).
    '''
    source_mask = y == source_label
    n_s = int(source_mask.sum().item())

    x_s_raw = x_raw[source_mask]
    x_s_clean = x_clean[source_mask]
    y_target_s = torch.full((n_s,), target_label, dtype=torch.long, device=x_s_raw.device)
    y_source_s = torch.full((n_s,), source_label, dtype=torch.long, device=x_s_raw.device)

    x_s_poisoned = raw_to_trigger_preprocess(
        x_s_raw, delta, dataset_flag=dataset_flag, model_flag=model_flag,
    )
    grads_bd, logits_bd = compute_batch_gradients(
        M, loss_fn, (x_s_poisoned, y_target_s),
        create_graph=create_graph, retain_graph=create_graph,
    )
    g_bd = torch.cat([g.reshape(-1) for g in grads_bd])
    if not create_graph:
        g_bd = g_bd.detach()

    grads_ss, _ = compute_batch_gradients(M, loss_fn, (x_s_clean, y_source_s), create_graph=False)
    g_ss = torch.cat([g.reshape(-1) for g in grads_ss]).detach()

    v = lambda_eff * (g_bd - g_ss)
    L_bd = loss_fn(logits_bd, y_target_s) if need_L_bd else None
    return v, L_bd


def compute_frozen_v(M, loss_fn, x_clean, y, x_raw, mask, y_poison, has_poison, delta_frozen,
                      dataset_flag, model_flag, v_estimator="subsample", source_label=None,
                      target_label=None, lambda_eff=None):
    '''
    v_k(delta) = grad_Lp[delta](theta_k) - grad_Lc(theta_k), computed with delta HELD FIXED
    (`delta_frozen`, expected to be a plain detached tensor -- e.g. `delta.detach()`): no
    create_graph/retain_graph needed since we never need a delta-gradient out of this call (the
    inner solve only ever needs v_k's VALUE), unlike `_compute_step_policy`'s own v computation
    (create_graph=True there, so mu_p keeps a graph back to delta for the OUTER, delta-update
    pass). Returns a detached (D,) tensor.

    Task 3: `v_estimator="analytic"` (see `estimate_v_analytic`) computes v over ALL source-class
    rows instead of `mask`'s subsampled few -- same estimator the delta-update pass uses when
    `v_estimator="analytic"` there too, so both paths share the SAME reduced-variance v. Requires
    `source_label`/`target_label`/`lambda_eff` in that mode.
    '''
    if v_estimator == "analytic":
        v, _L_bd = estimate_v_analytic(
            M, loss_fn, x_raw, x_clean, y, source_label, target_label, delta_frozen, lambda_eff,
            dataset_flag, model_flag, create_graph=False, need_L_bd=False,
        )
        return v.detach()

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
    beta, beta_global, normalization, device, v_estimator="subsample", source_label=None,
    target_label=None, lambda_eff=None,
):
    '''
    Builds, for EACH sampled checkpoint, the (G_obj, v, den) triple the aggregate QP is built
    from -- delta held fixed at `delta_frozen` throughout (no forward pass here depends on a
    delta that could still change mid-way through this list, since it is the SAME snapshot for
    every k). Reuses `flip_grad_cache` (shared with `_compute_step_policy`'s own per-batch
    cache) so a checkpoint computed by one path is not recomputed by the other within the same
    batch.

    Task 3: `v_estimator`/`source_label`/`target_label`/`lambda_eff` are forwarded to
    `compute_frozen_v` unchanged -- same estimator (and, when "analytic", the SAME reduced-
    variance v) as `_compute_step_policy`'s own delta-update pass uses for this batch.

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
            dataset_flag, model_flag, v_estimator=v_estimator, source_label=source_label,
            target_label=target_label, lambda_eff=lambda_eff,
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


# --------------------------------------------------------------------------- #
# Etape 1a/1b (follow-up task: "solveur QP couple, etapes 1 a 5") -- per-checkpoint history of
# (c_k, den_k), used both to answer 1a.1 (how much does aggregating over m batches change u*,
# and where does it plateau) and to drive policy_solver="qp" in production (1b: qp_batches_per_
# checkpoint = the m to use, chosen from 1a.1's measurement).
#
# G_k/Q_k are CHECKPOINT-FIXED: `get_or_build_flip_grad_cache_entry` builds them once per
# checkpoint from `class_samples_raw`, itself fixed for the whole run -- they never vary batch
# to batch for the same checkpoint. All of the batch-to-batch noise 0bis.2 measured therefore
# lives in v_k (hence c_k = G_k^T @ v_k) and, under normalization="v", in den_k too. Averaging
# these SMALL (P-dim / scalar) quantities over m observations is mathematically IDENTICAL to
# averaging v_k itself first (linearity: G_k^T @ mean_b(v_b) == mean_b(G_k^T @ v_b), since G_k
# does not depend on b) but far cheaper: v_k is (D,), D = model parameter count (~1e5-1e6); c_k
# is (P,), P = number of (y,c) pairs (~90). Q itself never needs aggregating -- it already has
# no batch-to-batch noise to average out.
# --------------------------------------------------------------------------- #

def push_context_history(history, contexts, m):
    '''
    Mutates `history` (dict checkpoint_id -> list of (c_k: (P,) ndarray, den_k: float)) in
    place: appends this batch's (c_k, den_k) for every checkpoint in `contexts`, capping each
    checkpoint's list at the last `m` entries (oldest dropped first). Call once per batch,
    before solving, so the solve below sees the just-updated history.
    '''
    for ctx in contexts:
        c_k = (ctx["G_obj"].T @ ctx["v"]).detach().cpu().numpy().astype(np.float64)
        buf = history.setdefault(ctx["k"], [])
        buf.append((c_k, ctx["den"]))
        if len(buf) > m:
            del buf[0]


def aggregate_qp_from_history(contexts, history, m=None):
    '''
    Etape 1b: the coupled QP's (Q, c) exactly as `aggregate_qp`, EXCEPT c (and den) are the
    MEAN over each checkpoint's history (`push_context_history`) instead of this single batch's
    observation. Q needs no such averaging (see this section's docstring). Falls back to the
    current batch's own (c, den) for a checkpoint with empty history (should not happen if
    `push_context_history` was already called on `contexts` this batch, but kept as a safety
    net for m=0/misuse).

    `m`: if given, only the LAST `m` entries of each checkpoint's history are averaged (the
    buffer itself, filled by `push_context_history`, may be kept LONGER than the production
    `qp_batches_per_checkpoint` so the same history can also serve `m_sweep_cosine`'s diagnostic
    sweep over larger m values -- pass the production m here to use only that many, regardless
    of how much longer history happens to be retained). None (default) uses the WHOLE buffer.
    '''
    K = len(contexts)
    pairs = contexts[0]["pairs_k"]
    Q_sum = np.zeros_like(contexts[0]["Q_obj"], dtype=np.float64)
    c_sum = np.zeros(Q_sum.shape[0], dtype=np.float64)
    for ctx in contexts:
        buf = history.get(ctx["k"])
        if buf:
            entries = buf[-m:] if m else buf
            c_bar = np.mean([e[0] for e in entries], axis=0)
            den_bar = float(np.mean([e[1] for e in entries]))
        else:
            c_bar = (ctx["G_obj"].T @ ctx["v"]).detach().cpu().numpy().astype(np.float64)
            den_bar = ctx["den"]
        Q_sum += ctx["Q_obj"] / den_bar
        c_sum += c_bar / den_bar
    return Q_sum / K, c_sum / K, pairs


def m_sweep_cosine(c_history, Q, beta, pairs, pi, m_values, ridge=1e-6, n_iters=500):
    '''
    Etape 1a.1: for ONE checkpoint's Q (fixed, see this section's docstring) and a
    chronological list of its per-batch c observations, solves the QP at c averaged over the
    LAST m observations, for each m in `m_values`, and reports each u*(m)'s cosine similarity
    to u*(max(m_values)) (the most-averaged, best available reference).

    Read the resulting curve as follows: the m where the cosine plateaus is the m to actually
    use in production (qp_batches_per_checkpoint) -- averaging further buys nothing. The
    plateau's cosine value separates the two hypotheses `intra ~ inter` (Diagnostic 0bis.2)
    could not distinguish alone: if it tends to 1, ALL of the residual disagreement was
    estimation noise on v_k, now averaged away; if it plateaus strictly below 1, that residual
    IS trajectory drift (prop:oneshot-gap net of noise), not an artifact of insufficient
    averaging.

    Args:
        c_history: list of (P,) arrays, one per batch this checkpoint appeared in, in
            chronological order. Must have length >= max(m_values).
        Q: (P,P) float64 ndarray, this checkpoint's own Q_obj (unaggregated -- see docstring).
        m_values: iterable of ints, e.g. [1, 2, 5, 10, 20].

    Returns (u_star_by_m: {m: (P,) ndarray}, cosine_by_m: {m: float or None}).
    '''
    m_values = sorted(set(m_values))
    max_m = max(m_values)
    if len(c_history) < max_m:
        raise ValueError(
            f"m_sweep_cosine needs >= max(m_values)={max_m} observations, got {len(c_history)}."
        )
    u_star_by_m = {}
    for m in m_values:
        c_agg = np.mean(c_history[-m:], axis=0)
        u_star, _, _ = project_gradient_descent_local(
            Q, c_agg, np.zeros(Q.shape[0]), beta, pairs, pi, n_iters=n_iters, ridge=ridge,
        )
        u_star_by_m[m] = diag.as_numpy(u_star)

    reference = u_star_by_m[max_m]
    ref_norm = np.linalg.norm(reference)
    cosine_by_m = {}
    for m in m_values:
        a = u_star_by_m[m]
        denom = np.linalg.norm(a) * ref_norm
        cosine_by_m[m] = float(np.dot(a, reference) / denom) if denom > 1e-12 else None
    return u_star_by_m, cosine_by_m


def qp_conditioning_diagnostic(Q, ridge=1e-6):
    '''Etape 1a.2: lambda_min/lambda_max/condition number of Q+ridge*I -- checks whether a
    non-converging coupled solve (qp_pgd_solve's `converged=False`) is a conditioning problem,
    as opposed to simply needing more iterations.'''
    P = Q.shape[0]
    Q_reg = Q + ridge * np.eye(P)
    eigvals = np.linalg.eigvalsh(Q_reg)
    lambda_min = float(eigvals.min())
    lambda_max = float(eigvals.max())
    return {
        "lambda_min": lambda_min, "lambda_max": lambda_max,
        "condition_number": lambda_max / max(lambda_min, 1e-300),
    }


def qp_pgd_solve_with_trace(Q, c, u_init, beta, pairs, pi, max_iters=200, ridge=1e-6, trace_last_n=100):
    '''
    Etape 1a.2: runs `qp_pgd_solve`'s exact algorithm for the FULL `max_iters` budget (no early
    stop -- unlike `qp_pgd_solve`, this is for measuring convergence itself, so stopping early
    would hide the answer), recording the objective value at each of the LAST `trace_last_n`
    iterations. Returns (u_star, obj_decrement_last_n, obj_trace_last_n) -- obj_decrement_last_n
    = obj[-trace_last_n] - obj[-1]; small relative to obj[-1] means the solve had already
    plateaued well before max_iters (non-convergence, if any, is then a conditioning/step-size
    issue, not merely "not enough iterations"); still shrinking means max_iters itself was the
    binding constraint.
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

    trace = []
    w_t = torch.as_tensor(w_np)
    for i in range(max_iters):
        grad = Q_reg @ w_np - c
        w_np = w_np - lr * grad
        w_t = project_policy_budget(torch.as_tensor(w_np), beta, pairs, pi)
        w_np = w_t.detach().cpu().numpy().astype(np.float64)
        if i >= max_iters - trace_last_n:
            trace.append(obj(w_np))

    obj_decrement_last_n = (trace[0] - trace[-1]) if len(trace) >= 2 else 0.0
    return w_t, obj_decrement_last_n, trace


# --------------------------------------------------------------------------- #
# Etape 2 (follow-up task) -- split-half overfitting check: does the QP exploit estimation
# error in G_k (built from a FINITE class_samples_raw sample) rather than genuine signal?
# --------------------------------------------------------------------------- #

def split_half_overfitting_check(
    model, loss_fn, class_samples_raw, n_classes, pi, dataset_flag, model_flag, gamma, beta,
    beta_global, v, beta_local_for_solve, pairs_hint, ridge_values, n_iters=500, seed=0,
):
    '''
    Etape 2: splits `class_samples_raw` (the per-class sample compute_expected_flip_gradients
    is built from) into two INDEPENDENT halves, builds G_1/Q_1 (train) and G_2/Q_2 (holdout)
    for the SAME checkpoint from each half, solves the QP on (Q_1, c_1) -> u*_1, and evaluates
    B2 of u*_1 under BOTH (Q_1, c_1) (train) and (Q_2, c_2) (holdout) -- a QP that is only
    exploiting estimation noise in G_1 shows B2_holdout >> B2_train; a QP that has found a real
    signal shows B2_holdout ~= B2_train.

    Swept over `ridge_values`: returns {ridge: (b2_train, b2_holdout)}, so the caller can find
    the ridge minimizing B2_holdout -- that value is what `qp_ridge` should default to under
    policy_solver="qp" once measured (see the module docstring / run_module.py's `run()`).

    NOTE: needs `class_samples_raw` to have >= 2 samples per class (halved evenly, remainder
    dropped) -- raises if smaller than that.
    '''
    from modules.federated_optimizing_trigger.utils import compute_expected_flip_gradients

    rng = np.random.RandomState(seed)
    half_1, half_2 = {}, {}
    for y, x in class_samples_raw.items():
        n = x.shape[0]
        if n < 2:
            raise ValueError(f"class {y} has only {n} sample(s) -- split_half needs >= 2.")
        idx = rng.permutation(n)
        half = n // 2
        half_1[y] = x[idx[:half]]
        half_2[y] = x[idx[half:2 * half]]

    params = list(model.parameters())
    G_1, Q_1, pairs_1 = compute_expected_flip_gradients(
        model, loss_fn, half_1, n_classes, pi, dataset_flag=dataset_flag, model_flag=model_flag,
        params=params,
    )
    G_2, Q_2, pairs_2 = compute_expected_flip_gradients(
        model, loss_fn, half_2, n_classes, pi, dataset_flag=dataset_flag, model_flag=model_flag,
        params=params,
    )
    assert pairs_1 == pairs_2 == pairs_hint, "split halves disagree on which pairs are present"

    # Same gamma/pi_y rescaling _compute_step_policy/get_or_build_flip_grad_cache_entry apply --
    # duplicated here (not imported) because that helper also fills flip_grad_cache as a
    # side effect, which this diagnostic must NOT touch.
    scale = torch.tensor(
        [gamma / pi[y] for (y, c) in pairs_1], device=G_1.device, dtype=G_1.dtype,
    )
    G_obj_1 = G_1 * scale
    G_obj_2 = G_2 * scale
    scale_np = scale.detach().cpu().numpy().astype(np.float64)
    Q_obj_1 = np.outer(scale_np, scale_np) * Q_1
    Q_obj_2 = np.outer(scale_np, scale_np) * Q_2

    c_1 = (G_obj_1.T @ v).detach().cpu().numpy().astype(np.float64)
    den = float(v.detach().norm().item() ** 2 + 1e-8)

    results = {}
    for ridge in ridge_values:
        u_star, _, _ = project_gradient_descent_local(
            Q_obj_1, c_1, np.zeros(len(pairs_1)), beta, pairs_1, pi, n_iters=n_iters, ridge=ridge,
        )
        b2_train, _ = diag.b2_value(G_obj_1, u_star, v, den)
        b2_holdout, _ = diag.b2_value(G_obj_2, u_star, v, den)
        results[ridge] = (b2_train, b2_holdout)
    return results


# --------------------------------------------------------------------------- #
# Etape 4 (follow-up task) -- four-term decomposition: B2_span (subspace-only), alpha_tilde^2
# (subspace + positivity, NNLS without a budget), B2_QP (subspace + positivity + budget),
# B2_current (what the running algorithm actually achieves). Separates the cost of the CONE
# (u>=0) from the cost of the BUDGET, which the existing span/QP diagnostics (diagnostics.py)
# conflate.
# --------------------------------------------------------------------------- #

def nnls_cone_projection(G_obj, v, den, max_iters=500, ridge=1e-6, tol=1e-10):
    '''
    Etape 4 -- alpha_tilde^2 = min_{u>=0} ||G_obj @ u - v||^2 / den: the SAME projected-gradient
    scheme as `qp_pgd_solve`, but projecting only onto the nonnegative orthant (u>=0, NO budget,
    NO per-class caps) -- separates the cost of requiring u>=0 (the CONE) from the cost of the
    budget/class-cap constraints (already measured by B2_qp/B2_QP relative to this).

    Returns (alpha_tilde_sq: float, u_nnls: (P,) ndarray).
    '''
    P = G_obj.shape[1]
    Q = (G_obj.T @ G_obj).detach().cpu().numpy().astype(np.float64)
    c = (G_obj.T @ v).detach().cpu().numpy().astype(np.float64)
    Q_reg = Q + ridge * np.eye(P)
    L = float(np.linalg.eigvalsh(Q_reg).max())
    lr = 1.0 / max(L, ridge)

    w = np.zeros(P, dtype=np.float64)
    prev_obj = 0.5 * float(w @ Q_reg @ w) - float(c @ w)
    for _ in range(max_iters):
        grad = Q_reg @ w - c
        w = np.clip(w - lr * grad, 0.0, None)
        cur_obj = 0.5 * float(w @ Q_reg @ w) - float(c @ w)
        if abs(prev_obj - cur_obj) < tol * max(abs(prev_obj), 1e-12):
            prev_obj = cur_obj
            break
        prev_obj = cur_obj

    alpha_tilde_sq, _ = diag.b2_value(G_obj, w, v, den)
    return alpha_tilde_sq, w


def four_term_decomposition(G_obj, v, den, u_current, u_qp, rho, span_result=None):
    '''
    Etape 4: assembles the four-row table (all in the SAME /den units already used elsewhere --
    den = rho^2+eps under normalization="rho", the default):
      B2_span   / den  -- geometric ceiling, no constraint at all (span(G_obj) only)
      alpha_tilde^2     -- + positivity (u>=0), still no budget
      B2_QP     / den  -- + budget/class caps (U_loc) -- what B2_qp already reports
      B2_current/ den  -- what the running algorithm actually achieves
    `span_result`, if given, reuses an already-computed diagnostics.span_projection(...) dict
    (same G_obj/v/den) instead of recomputing it.
    '''
    if span_result is None:
        span_result = diag.span_projection(G_obj, (G_obj.T @ G_obj).detach().cpu().numpy(),
                                            (G_obj.T @ v).detach().cpu().numpy(), v, den)
    alpha_tilde_sq, u_nnls = nnls_cone_projection(G_obj, v, den)
    b2_qp, _ = diag.b2_value(G_obj, u_qp, v, den)
    b2_current, _ = diag.b2_value(G_obj, u_current, v, den)
    return {
        "B2_span_over_den": span_result["B2_span"],
        "alpha_tilde_sq": alpha_tilde_sq,
        "B2_QP_over_den": b2_qp,
        "B2_current_over_den": b2_current,
    }
