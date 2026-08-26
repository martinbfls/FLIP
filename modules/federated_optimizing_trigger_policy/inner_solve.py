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
from modules.federated_optimizing_trigger_policy.utils import project_policy_budget
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
