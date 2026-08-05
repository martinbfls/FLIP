import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.federated_optimizing_trigger.utils import (
    project_gradient,
    compute_v_polytope_distance,
)

torch.manual_seed(0)

results = []


def report(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    results.append((name, ok))
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""))


def make_toy_polytope(D=40, n_classes=4, seed=0):
    '''
    G's columns are pi_y-weighted (G[:, (y,c)] = pi_y * g_{y,c}, matching
    compute_expected_flip_gradients), so the budget constraint on w is the
    plain unweighted sum_{y,c} w_{y,c} <= beta (_build_global_budget_constraint).
    '''
    g = torch.Generator().manual_seed(seed)
    pairs = [(y, c) for y in range(n_classes) for c in range(n_classes) if c != y]
    P = len(pairs)
    pi = {y: (y + 1) / sum(range(1, n_classes + 1)) for y in range(n_classes)}
    G_raw = torch.randn(D, P, dtype=torch.float64, generator=g)
    pi_vec = torch.tensor([pi[y] for y, _ in pairs], dtype=torch.float64)
    G = G_raw * pi_vec.unsqueeze(0)
    Q = (G.T @ G).numpy()
    return G, Q, pairs, pi


print("=== (i) B2 == 0 when v = G @ w for an admissible w ===")

D, n_classes = 40, 4
G, Q, pairs, pi = make_toy_polytope(D, n_classes, seed=1)
P = len(pairs)
beta = 0.4

# Admissible w: nonnegative, respecting sum_{y,c} w_{y,c} <= beta (unweighted:
# pi_y already lives inside G's columns, not in the constraint).
w_raw = torch.rand(P, dtype=torch.float64, generator=torch.Generator().manual_seed(2))
scale = beta / w_raw.sum().item()
w_admissible = w_raw * scale * 0.9  # strictly inside the budget, avoid boundary noise

v = (G @ w_admissible).clone().requires_grad_(True)
dist2, w_star = compute_v_polytope_distance(v, G, Q, pairs, beta)
den = v.detach().norm() ** 2 + 1e-8
B2 = (dist2 / den).item()

report(
    "B2 == 0 for v = G @ w_admissible",
    B2 < 1e-6,
    f"B2={B2:.3e}",
)


print("\n=== (ii) B2 == 1 when v is orthogonal to span(G) ===")

v0 = torch.randn(D, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
G_pinv = torch.linalg.pinv(G)
v_orth = (v0 - G @ (G_pinv @ v0)).requires_grad_(True)
ortho_check = (G.T @ v_orth.detach()).abs().max().item()

dist2, w_star = compute_v_polytope_distance(v_orth, G, Q, pairs, beta)
den = v_orth.detach().norm() ** 2 + 1e-8
B2_orth = (dist2 / den).item()

report(
    "B2 == 1 for v orthogonal to span(G)",
    abs(B2_orth - 1.0) < 1e-6,
    f"B2={B2_orth:.8f}, max|G^T v|={ortho_check:.2e}, ||w*||={w_star.norm().item():.2e}",
)


print("\n=== (iii) scale invariance of B1 and B2 ===")

# B1 = ||v||^2 / (||mu_p||^2 + eps): exactly scale invariant when g_c is held
# fixed at 0 (so v == mu_p), since numerator and denominator scale identically.
mu_p_base = torch.randn(D, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
eps_den = 1e-8


def B1_of(mu_p):
    v = mu_p  # g_c == 0
    den = mu_p.norm() ** 2 + eps_den
    return ((v ** 2).sum() / den).item()


b1_vals = [B1_of(alpha * mu_p_base) for alpha in (0.1, 1.0, 5.0, 50.0)]
report(
    "B1 invariant under scaling of mu_p",
    # eps_den doesn't scale with alpha, so exact invariance only holds as
    # eps_den/den -> 0; 1e-6 comfortably clears that residual at alpha=0.1.
    max(b1_vals) - min(b1_vals) < 1e-6,
    f"B1 values across scales: {['%.8f' % x for x in b1_vals]}",
)

# B2 = dist(v, W_beta)^2 / (||v||^2+eps): W_beta is only a CONE (positively
# homogeneous) when the budget constraint never binds, i.e. beta effectively
# unconstrained (only w >= 0 remains active). Projection onto a cone commutes
# with positive scaling, so B2 is then exactly scale invariant in v -- this is
# the regime that decorrelates B2 (direction/feasibility) from B1 (magnitude).
beta_unconstrained = 1e6
v_generic = torch.randn(D, dtype=torch.float64, generator=torch.Generator().manual_seed(5))

b2_vals = []
for alpha in (0.1, 1.0, 5.0, 50.0):
    dist2, _ = compute_v_polytope_distance(alpha * v_generic, G, Q, pairs, beta_unconstrained)
    den = (alpha * v_generic).norm() ** 2 + eps_den
    b2_vals.append((dist2 / den).item())

report(
    "B2 invariant under scaling of v (unconstrained-budget / cone regime)",
    max(b2_vals) - min(b2_vals) < 1e-4,
    f"B2 values across scales: {['%.8f' % x for x in b2_vals]}",
)


print("\n=== (iv) B2 gradient w.r.t. v matches the explicit ((v - G@w*)**2).sum() form ===")

v_check = torch.randn(D, dtype=torch.float64, generator=torch.Generator().manual_seed(6), requires_grad=True)
dist2, w_star = compute_v_polytope_distance(v_check, G, Q, pairs, beta)
dist2.backward()
grad_closed = v_check.grad.clone()

v_check2 = v_check.detach().clone().requires_grad_(True)
c_vec2 = G.T @ v_check2
w_star2 = project_gradient(Q, c_vec2.detach().numpy(), beta, pairs).to(v_check2.dtype)
g_proj2 = (G @ w_star2).detach()
dist2_explicit = ((v_check2 - g_proj2) ** 2).sum()
dist2_explicit.backward()
grad_explicit = v_check2.grad.clone()

report(
    "closed-form B2 grad matches explicit ((v - G@w*)**2).sum() grad to 1e-6",
    (grad_closed - grad_explicit).abs().max().item() < 1e-6
    and abs(dist2.item() - dist2_explicit.item()) < 1e-6,
    f"value diff={abs(dist2.item() - dist2_explicit.item()):.2e}, "
    f"grad max abs diff={(grad_closed - grad_explicit).abs().max().item():.2e}",
)


print("\n=== (v) B2 increases (or stays equal) as beta decreases ===")

# Shrinking beta shrinks the feasible set W_beta (nested: W_beta1 subset
# W_beta2 for beta1 <= beta2), so the distance from a fixed v to W_beta can
# only increase or stay the same -- never decrease -- as beta decreases.
v_mono = torch.randn(D, dtype=torch.float64, generator=torch.Generator().manual_seed(7))
betas_desc = [2.0, 1.0, 0.5, 0.2, 0.05, 0.0]
b2_mono = []
for b in betas_desc:
    dist2, _ = compute_v_polytope_distance(v_mono, G, Q, pairs, b)
    den = v_mono.norm() ** 2 + eps_den
    b2_mono.append((dist2 / den).item())

is_monotonic = all(b2_mono[i] <= b2_mono[i + 1] + 1e-9 for i in range(len(b2_mono) - 1))
report(
    "B2 is non-decreasing as beta decreases",
    is_monotonic,
    f"betas={betas_desc} -> B2={['%.6f' % x for x in b2_mono]}",
)


print("\n=== summary ===")
n_pass = sum(1 for _, ok in results if ok)
print(f"{n_pass}/{len(results)} checks passed")
if n_pass != len(results):
    sys.exit(1)
