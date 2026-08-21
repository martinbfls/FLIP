import torch


def init_policy(n_pairs, device, dtype=torch.float32):
    '''
    Initializes the attack policy u in R^P (P = n_pairs, one weight per ordered class pair
    (y, c), y != c -- same ordering as `pairs` returned by
    `federated_optimizing_trigger.utils.compute_expected_flip_gradients`). Starts at the
    origin (u = 0), the interior of U_beta = {u >= 0, sum(u) <= beta} for any beta > 0.
    '''
    u = torch.zeros(n_pairs, device=device, dtype=dtype)
    u.requires_grad_(True)
    return u


def project_policy_budget(u, beta):
    '''
    Euclidean projection of u onto U_beta = {x : x >= 0, sum(x) <= beta} (beta > 0), i.e. the
    feasible set for the attack policy under an inversion budget of beta: at most a beta
    fraction of the attacker's own shard gets relabeled, split arbitrarily (non-negatively)
    across the ordered class pairs (y, c).

    If u is already feasible after clipping to the nonnegative orthant (sum(max(u, 0)) <=
    beta), that clipped point IS the projection -- the budget constraint is inactive.
    Otherwise the budget constraint is active and the projection lies on the truncated
    simplex {x >= 0, sum(x) = beta}: the standard simplex-projection algorithm (Duchi et al.,
    2008), via a single sort -- no QP solve needed (unlike `project_gradient` in
    `federated_optimizing_trigger.utils`, which projects a *displacement* v onto the image
    G @ U_beta; this instead projects the policy u itself onto U_beta directly).

    Returns a new tensor (does not modify u in place). Callers apply this under
    `torch.no_grad()`, the same way `delta.clamp_(-epsilon, epsilon)` enforces the trigger's
    L_infinity constraint.
    '''
    u_pos = u.clamp(min=0.0)
    total = u_pos.sum()
    if total <= beta:
        return u_pos

    P = u.shape[0]
    u_sorted, _ = torch.sort(u, descending=True)
    cssum = torch.cumsum(u_sorted, dim=0)
    j = torch.arange(1, P + 1, device=u.device, dtype=u.dtype)
    cond = u_sorted - (cssum - beta) / j > 0
    rho = torch.nonzero(cond, as_tuple=False).max()
    theta = (cssum[rho] - beta) / (rho + 1).to(u.dtype)
    return (u - theta).clamp(min=0.0)
