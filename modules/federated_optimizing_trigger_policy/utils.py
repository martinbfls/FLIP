"""
Theory: helpers for the LOCAL flip-mass policy u (rem:units) used by
federated_optimizing_trigger_policy/run_module.py to implement eq:P (P^mean)'s local reading.
See that module's header docstring for the full chain position and scope conventions (`beta`
here is the theory's LOCAL corruption rate, `beta_theory/gamma`, not the global `beta` of
def:budget). `init_policy` creates the raw parameter u; `project_policy_budget` (with its
`_project_nonneg_capped_sum` helper) is the exact Euclidean projection onto U_loc (eq:Uloc).
"""
import numpy as np
import torch


def init_policy(n_pairs, device, dtype=torch.float32):
    '''
    Initializes the attack policy u in R^P (P = n_pairs, one weight per ordered class pair
    (y, c), y != c -- same ordering as `pairs` returned by
    `federated_optimizing_trigger.utils.compute_expected_flip_gradients`). Starts at the
    origin (u = 0), the interior of the feasible set for any beta > 0, pi > 0.
    '''
    u = torch.zeros(n_pairs, device=device, dtype=dtype)
    u.requires_grad_(True)
    return u


def _project_nonneg_capped_sum(u, cap):
    '''
    Euclidean projection of u (any real vector, positive or negative entries) onto
    {x >= 0, sum(x) <= cap}, cap > 0: the standard simplex-projection algorithm (Duchi et
    al., 2008), via a single sort. If u is already feasible after clipping to the nonnegative
    orthant (sum(max(u, 0)) <= cap), that clipped point IS the projection -- the sum
    constraint is inactive. Otherwise the projection lies on the truncated simplex
    {x >= 0, sum(x) = cap}.

    Verified against an independent bisection reference on vectors with negative
    components (see federated_optimizing_trigger_policy's test suite): Duchi's algorithm
    sorts the RAW input (not a pre-clipped copy) -- the final `.clamp(min=0.0)` is what
    handles negative entries, not a preliminary clip.

    Returns a new tensor (does not modify u in place).
    '''
    u_pos = u.clamp(min=0.0)
    total = u_pos.sum()
    if total <= cap:
        return u_pos

    P = u.shape[0]
    u_sorted, _ = torch.sort(u, descending=True)
    cssum = torch.cumsum(u_sorted, dim=0)
    j = torch.arange(1, P + 1, device=u.device, dtype=u.dtype)
    cond = u_sorted - (cssum - cap) / j > 0
    rho = torch.nonzero(cond, as_tuple=False).max()
    theta = (cssum[rho] - cap) / (rho + 1).to(u.dtype)
    return (u - theta).clamp(min=0.0)


def project_policy_budget(u, beta, pairs, pi, tol=1e-9, max_iter=60):
    '''
    Theory: def:config (eq:Uloc) -- U_loc = {u>=0, sum_c u_{y,c}<=pi_y for every y,
    ||u||_1<=beta/gamma}. This function's `beta` argument IS that local cap directly (the
    module's own `beta`, already scoped to beta_theory/gamma -- see the run_module.py header
    docstring's scope conventions), so the set below is exactly U_loc, not a rescaled version
    of it.

    Euclidean projection of u onto

        U_{beta,pi} = { u >= 0, sum(u) <= beta, sum_c u_{y,c} <= pi[y] for every source y }

    the feasible set for the LOCAL attack policy u: u_{y,c} is the fraction of a SINGLE
    corrupted worker's OWN shard flipped from class y to class c (beta is the fraction of
    that worker's whole shard it may flip; see federated_optimizing_trigger_policy's
    beta/gamma docstring for why u is local rather than an aggregate/global rate). A single
    worker cannot flip more of its own class-y examples than it holds -- a pi[y] fraction of
    its shard -- regardless of the global per-worker budget beta. (NOT gamma * pi[y]: gamma
    only enters the *objective*, converting u's local rate into the aggregate gradient shift
    every corrupted worker's copy of u induces under mean aggregation -- see
    `federated_optimizing_trigger_policy/run_module.py`'s `_compute_step_policy`. The
    feasible set for u itself has nothing to do with how many other corrupted workers exist.)

    Solved via the KKT structure of this box-simplex-per-block-plus-one-linear-coupling
    problem, exactly (not an approximate alternating-projection scheme): for a shift s >= 0,

        x(s) = concat_y[ _project_nonneg_capped_sum(u_y - s, pi[y]) ]

    satisfies every per-block constraint for ANY s, and sum(x(s)) is continuous and
    non-increasing in s (each block's contribution is). x(0) is the projection onto the
    per-block constraints alone; if sum(x(0)) <= beta already, the global constraint is
    inactive and x(0) IS the projection (strong duality: with the coupling constraint
    inactive at the optimum, the problem separates exactly into independent per-block
    projections). Otherwise the unique s* with sum(x(s*)) == beta is the coupling
    constraint's KKT multiplier (up to a factor of 2), found by bisection, and x(s*) is the
    projection (again by strong duality -- a convex QP with box-simplex blocks coupled by one
    linear inequality has zero duality gap).

    Args:
        u: (P,) tensor, P = number of ordered (y, c) pairs.
        beta: global (per-worker) budget, beta > 0.
        pairs: list of P (y, c) int tuples, same ordering as u.
        pi: dict y -> pi_y (empirical class frequency), pi_y > 0 for every y in pairs.
        tol: bisection tolerance on the shift s.
        max_iter: bisection iteration cap.

    Returns a new tensor (does not modify u in place).
    '''
    ys = sorted(set(y for y, _ in pairs))
    block_idx = {
        y: torch.tensor([p for p, (yy, _) in enumerate(pairs) if yy == y], device=u.device)
        for y in ys
    }

    def blocks_at(shift):
        out = torch.empty_like(u)
        for y in ys:
            idx = block_idx[y]
            out[idx] = _project_nonneg_capped_sum(u[idx] - shift, pi[y])
        return out

    x0 = blocks_at(0.0)
    if x0.sum() <= beta:
        return x0

    lo, hi = 0.0, float(u.max().clamp(min=0.0).item()) + 1.0
    while blocks_at(hi).sum() > beta and hi < 1e6:
        hi *= 2.0

    for _ in range(max_iter):
        mid = (lo + hi) / 2
        if blocks_at(mid).sum() > beta:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break

    return blocks_at(hi)


def project_gradient_descent_local(Q, c, u_init, beta, pairs, pi, n_iters=50, ridge=1e-6):
    '''
    A4 (docs/policy_module_audit_report.md Section 2.5/A4) -- Theory: sec:solution's QP
    (eq:qp form, specialized to a single checkpoint's own Q, c here), solved by projected
    gradient descent onto U_loc (eq:Uloc, via `project_policy_budget`, the EXACT projection)
    instead of `federated_optimizing_trigger.utils.project_gradient` (shared, read-only): that
    function enforces only the GLOBAL budget constraint sum(w)<=beta, not U_loc's per-class
    pi_y caps -- it solves over a strict SUPERSET of U_loc, so its optimum under-estimates the
    true min_{u in U_loc} residual rem:solver's diagnostic is meant to report. This function
    fixes that: same feasible set as u itself (U_loc), so the reported value is directly
    comparable to what a fully-converged solver (a) would report for u.

        min_w  0.5 w^T Q w - c^T w   s.t.  w in U_loc

    Fixed step size 1/L, L the largest eigenvalue of Q (computed once from the checkpoint's own
    Q -- cheap, P is O(C^2)) -- guarantees the unprojected gradient step is non-expansive for
    this convex quadratic; projection is exact each iteration (`project_policy_budget`), so the
    iterate sequence's objective is expected to decrease (near-)monotonically, logged below as
    `obj_decrement` so a caller can see whether it actually did.

    Warm-started from `u_init` (the CURRENT co-descended u, per A4's instruction) rather than
    from the origin: `n_iters` (a few dozen, not run to full convergence) is enough to make
    genuine progress from a warm start without paying a fresh QP solve's cost every diag step.

    Args:
        Q: (P,P) numpy float64 -- G_obj^T G_obj (already gamma/pi_y-rescaled, see
           `_compute_step_policy`).
        c: (P,) numpy float64 -- G_obj^T v.
        u_init: (P,) array-like (numpy or torch) -- warm-start point, typically the current u.
        beta: LOCAL budget (this module's own `beta`, NOT beta_global -- same scope
              `project_policy_budget` and u itself use).
        pairs, pi: as `project_policy_budget`.
        n_iters: number of projected-gradient steps.
        ridge: diagonal regularization added to Q before computing its largest eigenvalue
            (numerical safety, same convention as `federated_optimizing_trigger.utils.
            project_gradient`'s own ridge).

    Returns:
        w: (P,) torch float64 tensor, the iterate after `n_iters` steps (NOT necessarily the
           exact optimum -- see `obj_decrement` to gauge convergence).
        n_iters: the argument, echoed back for logging symmetry with `obj_decrement`.
        obj_decrement: float, objective value at the warm start minus at the final iterate
            (>=0 expected; a caller logging this over many batches can spot non-convergence or
            a step-size problem if it stops being non-negative).
    '''
    P = Q.shape[0]
    Q_reg = Q + ridge * np.eye(P)
    L = float(np.linalg.eigvalsh(Q_reg).max())
    lr = 1.0 / max(L, ridge)

    w_np = np.asarray(
        u_init.detach().cpu().numpy() if torch.is_tensor(u_init) else u_init,
        dtype=np.float64,
    ).copy()

    def obj(w):
        return 0.5 * float(w @ Q_reg @ w) - float(c @ w)

    obj0 = obj(w_np)
    w_t = torch.as_tensor(w_np)
    for _ in range(n_iters):
        grad = Q_reg @ w_np - c
        w_np = w_np - lr * grad
        w_t = project_policy_budget(torch.as_tensor(w_np), beta, pairs, pi)
        w_np = w_t.detach().cpu().numpy().astype(np.float64)

    obj_decrement = obj0 - obj(w_np)
    return w_t, n_iters, obj_decrement
