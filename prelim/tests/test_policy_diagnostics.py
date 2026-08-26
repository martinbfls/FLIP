"""
prelim/tests/test_policy_diagnostics.py -- non-regression checks on SYNTHETIC tensors only for
the federated_optimizing_trigger_policy diagnostics module (diagnostics.py) added to
instrument WHERE the joint (delta, u) attack fails (Section 11 of the diagnostics task): the
discretization rule shared with federated_policy_to_flips, the QP (Diagnostic A) reference
solver, the discretization gap it can expose (Diagnostic C), and the delta-gradient-balance
diagnostic (Diagnostic G). No dataset, no model, no training -- same convention as
prelim/tests/test_policy_module_fixes.py.

Run:  python prelim/tests/test_policy_diagnostics.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.federated_optimizing_trigger_policy import diagnostics as diag
from modules.federated_policy_to_flips.utils import compute_flip_counts, materialize_policy_flips

_results = []


def check(name, ok, detail=""):
    _results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


# --------------------------------------------------------------------------#
# A. Discretization: compute_flip_counts / discretize_policy vs. the actual downstream
# materialization (materialize_policy_flips) -- same convention, not two.
# --------------------------------------------------------------------------#

def test_discretize_policy_matches_materialize_policy_flips():
    n_classes = 4
    n_train = 1000
    rng = np.random.RandomState(0)
    labels = rng.randint(0, n_classes, size=n_train)
    class_counts = {y: int((labels == y).sum()) for y in range(n_classes)}

    pairs = [(0, 1), (0, 2), (1, 0)]
    u = np.array([0.05, 0.03, 0.02])
    gamma = 0.4

    n_realized = compute_flip_counts(u, pairs, gamma, n_train, class_counts)
    idx_flipped, _targets = materialize_policy_flips(u, pairs, n_train, labels, n_classes, gamma, seed=0)

    check(
        "discretize_policy's realized flip count == materialize_policy_flips' actual flip count",
        int(n_realized.sum()) == len(idx_flipped),
        f"n_realized.sum()={int(n_realized.sum())}, len(idx_flipped)={len(idx_flipped)}",
    )

    u_discrete, n_realized2 = diag.discretize_policy(u, pairs, gamma, n_train, class_counts)
    check("discretize_policy is deterministic / matches compute_flip_counts directly",
          np.array_equal(n_realized, n_realized2))
    check("u_discrete == n_realized / (gamma * n_train)",
          np.allclose(u_discrete, n_realized2 / (gamma * n_train)))


def test_discretize_policy_respects_budget_and_caps_no_negatives():
    n_classes = 3
    n_train = 200
    class_counts = {0: 50, 1: 70, 2: 80}
    # pair (0,1) requests far more than class_counts[0] allows -- must clip, never go negative.
    pairs = [(0, 1), (0, 2), (1, 2)]
    gamma = 0.5
    u = np.array([1.0, 1.0, 0.01])  # requests round(1.0*0.5*200)=100 for EACH of (0,1) and (0,2),
                                    # but class 0 only has 50 examples total -- must clip, and
                                    # split across the two (0,*) pairs via the shared cursor.

    n_realized = compute_flip_counts(u, pairs, gamma, n_train, class_counts)

    check("no negative realized counts", bool((n_realized >= 0).all()), f"n_realized={n_realized}")
    check("class-0 pairs never realize more than class_counts[0] together",
          int(n_realized[0] + n_realized[1]) <= class_counts[0],
          f"realized=({n_realized[0]}, {n_realized[1]}), cap={class_counts[0]}")
    check("class-1 pair respects class_counts[1]",
          int(n_realized[2]) <= class_counts[1])

    u_discrete, _ = diag.discretize_policy(u, pairs, gamma, n_train, class_counts)
    check("u_discrete has no negative entries", bool((u_discrete >= 0).all()))


def test_discretize_policy_small_deterministic_case():
    # A fully deterministic, hand-checkable case: 1 pair, exact division, no clipping.
    n_classes = 2
    n_train = 100
    class_counts = {0: 60, 1: 40}
    pairs = [(0, 1)]
    gamma = 1.0
    u = np.array([0.1])  # n_yc = round(0.1 * 1.0 * 100) = 10, class_counts[0]=60 -- no clipping

    n_realized = compute_flip_counts(u, pairs, gamma, n_train, class_counts)
    check("exact deterministic count matches round(u*gamma*n_train)", int(n_realized[0]) == 10,
          f"n_realized={n_realized}")

    rng = np.random.RandomState(1)
    labels = np.array([0] * 60 + [1] * 40)
    rng.shuffle(labels)
    idx_flipped, targets = materialize_policy_flips(u, pairs, n_train, labels, n_classes, gamma, seed=2)
    check("materialize_policy_flips realizes exactly the same count in this deterministic case",
          len(idx_flipped) == 10, f"len(idx_flipped)={len(idx_flipped)}")
    check("all realized targets are class 1 (the only pair's target)",
          bool((targets == 1).all()))
    check("all flipped indices were originally class 0",
          bool((labels[idx_flipped] == 0).all()))


# --------------------------------------------------------------------------#
# B. QP diagnostics (Diagnostic A): B2_qp <= B2_current, more iterations doesn't degrade,
# budget active/inactive, per-class caps.
# --------------------------------------------------------------------------#

def _synthetic_qp_problem(seed=0, n_pairs=6):
    '''A small synthetic (Q, c, G_obj, v, pairs, pi) consistent with _compute_step_policy's own
    B2 = ||G_obj @ w - v||^2 / den convention (G_obj^T @ G_obj == Q, G_obj^T @ v == c).'''
    rng = np.random.RandomState(seed)
    D = 12
    G_np = rng.normal(size=(D, n_pairs))
    v_np = rng.normal(size=D)
    G = torch.tensor(G_np, dtype=torch.float64)
    v = torch.tensor(v_np, dtype=torch.float64)
    Q = (G_np.T @ G_np).astype(np.float64)
    c = (G_np.T @ v_np).astype(np.float64)
    pairs = [(0, i + 1) for i in range(n_pairs)]  # single source class y=0, C targets
    pi = {0: 1.0}
    return G, v, Q, c, pairs, pi


def test_b2_qp_leq_b2_current():
    G, v, Q, c, pairs, pi = _synthetic_qp_problem(seed=0)
    beta = 0.3
    den = 1.0

    rng = np.random.RandomState(1)
    u_init = torch.as_tensor(rng.uniform(0, beta / len(pairs), size=len(pairs)))

    b2_current, _ = diag.b2_value(G, u_init, v, den)
    w_pg, _, _ = diag.project_gradient_descent_local(Q, c, u_init, beta, pairs, pi, n_iters=200)
    b2_qp, _ = diag.b2_value(G, w_pg, v, den)

    check("B2_qp <= B2_current (projected gradient descent only improves on the warm start)",
          b2_qp <= b2_current + 1e-9, f"b2_current={b2_current:.6f}, b2_qp={b2_qp:.6f}")


def test_more_qp_iterations_does_not_degrade():
    G, v, Q, c, pairs, pi = _synthetic_qp_problem(seed=2)
    beta = 0.3
    den = 1.0
    u_init = torch.zeros(len(pairs), dtype=torch.float64)

    sweep = diag.qp_convergence_sweep(Q, c, u_init, beta, pairs, pi, [10, 50, 200, 1000])
    b2s = {n: diag.b2_value(G, w, v, den)[0] for n, w in sweep.items()}

    ordered = [b2s[n] for n in sorted(b2s)]
    non_increasing = all(ordered[i] >= ordered[i + 1] - 1e-6 for i in range(len(ordered) - 1))
    check("B2_qp is (near-)monotonically non-increasing as n_iters grows",
          non_increasing, f"B2 by n_iters={b2s}")


def test_qp_budget_active_and_inactive():
    G, v, Q, c, pairs, pi = _synthetic_qp_problem(seed=3)
    u_init = torch.zeros(len(pairs), dtype=torch.float64)

    # Inactive: a huge beta should not bind -- unconstrained-ish optimum, sum(w) well under beta.
    w_loose, _, _ = diag.project_gradient_descent_local(Q, c, u_init, 1000.0, pairs, pi, n_iters=300)
    check("loose budget: sum(w) << beta (global constraint inactive)",
          float(diag.policy_l1(w_loose)) < 1000.0 * 0.5)

    # Active: a tiny beta must bind -- sum(w) close to beta (mass pushed to the cap).
    beta_tight = 0.01
    w_tight, _, _ = diag.project_gradient_descent_local(Q, c, u_init, beta_tight, pairs, pi, n_iters=300)
    check("tight budget: sum(w) close to beta (global constraint active)",
          abs(float(diag.policy_l1(w_tight)) - beta_tight) < 1e-3,
          f"sum(w)={float(diag.policy_l1(w_tight)):.6f}, beta={beta_tight}")


def test_qp_respects_per_class_caps():
    G, v, Q, c, pairs, pi = _synthetic_qp_problem(seed=4)
    pi_capped = {0: 0.02}  # much smaller than beta -- the per-class cap should bind first
    beta = 10.0
    u_init = torch.zeros(len(pairs), dtype=torch.float64)

    w, _, _ = diag.project_gradient_descent_local(Q, c, u_init, beta, pairs, pi_capped, n_iters=300)
    check("per-class cap respected (sum(w) <= pi[0])",
          float(diag.policy_l1(w)) <= pi_capped[0] + 1e-6,
          f"sum(w)={float(diag.policy_l1(w)):.6f}, pi[0]={pi_capped[0]}")
    check("per-class cap actually binds (not budget)",
          abs(float(diag.policy_l1(w)) - pi_capped[0]) < 1e-3)


# --------------------------------------------------------------------------#
# C. Discretization gap: a case engineered so the continuous solution is (near-)perfect but
# discretization destroys it (B2_discrete > B2_continuous).
# --------------------------------------------------------------------------#

def test_discretization_gap_destroys_perfect_continuous_solution():
    D = 5
    n_classes = 3
    pairs = [(0, 1), (0, 2)]
    pi = {0: 1.0}
    gamma = 0.3
    n_train = 1000  # gamma*n_train = 300 -- small u values round to 0.

    rng = np.random.RandomState(5)
    g01 = rng.normal(size=D)
    g02 = rng.normal(size=D)
    G = torch.tensor(np.stack([g01, g02], axis=1), dtype=torch.float64)

    # u tiny enough that round(u * gamma * n_train) == 0 for both pairs -- discretization
    # collapses u to exactly zero, while v is built to require u01=u02=u_small exactly.
    u_small = 0.3 / (gamma * n_train)  # rounds to 0 flips (0.5 threshold not reached alone)
    u_continuous = np.array([u_small, u_small])
    v = torch.tensor(u_small * (g01 + g02), dtype=torch.float64)  # so Gu_continuous == v exactly

    den = 1.0
    b2_continuous, _ = diag.b2_value(G, u_continuous, v, den)
    check("continuous solution is (near-)perfect by construction", b2_continuous < 1e-12,
          f"B2_continuous={b2_continuous:.3e}")

    class_counts = {0: n_train}
    u_discrete, n_realized = diag.discretize_policy(u_continuous, pairs, gamma, n_train, class_counts)
    check("discretization rounds both pairs to zero flips", bool((n_realized == 0).all()),
          f"n_realized={n_realized}")

    b2_discrete, _ = diag.b2_value(G, u_discrete, v, den)
    check("B2_discrete > B2_continuous once rounding destroys the (near-)perfect alignment",
          b2_discrete > b2_continuous, f"B2_continuous={b2_continuous:.3e}, B2_discrete={b2_discrete:.3e}")

    gap_abs, gap_rel = diag.discretization_gap(b2_continuous, b2_discrete)
    check("discretization_gap reports a positive absolute and relative gap",
          gap_abs > 0 and gap_rel > 0, f"gap_abs={gap_abs:.3e}, gap_rel={gap_rel:.3e}")


# --------------------------------------------------------------------------#
# D. Gradient balance (Diagnostic G): separate grad norms computable without disturbing the
# real backward pass / delta.grad.
# --------------------------------------------------------------------------#

def test_gradient_balance_does_not_touch_delta_grad_and_preserves_graph():
    delta = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    B2_k = (delta.sum()) ** 2          # grad = 2*sum(delta)*[1,1,1] = 2*6*[1,1,1] = [12,12,12]
    L_bd_k = (delta ** 2).sum()        # grad = 2*delta = [2,4,6]
    lambda_bd = 2.0

    norm_B2, norm_BD = diag.gradient_balance(B2_k, L_bd_k, lambda_bd, delta)

    expected_B2 = float(torch.tensor([12.0, 12.0, 12.0]).norm())
    expected_BD = float((lambda_bd * torch.tensor([2.0, 4.0, 6.0])).norm())
    check("grad_delta_B2_norm matches the analytic gradient norm",
          abs(norm_B2 - expected_B2) < 1e-5, f"got {norm_B2}, expected {expected_B2}")
    check("grad_delta_BD_norm matches the analytic (lambda_bd-scaled) gradient norm",
          abs(norm_BD - expected_BD) < 1e-5, f"got {norm_BD}, expected {expected_BD}")

    check("delta.grad is untouched (still None) after the diagnostic calls -- no accumulation",
          delta.grad is None)

    # The graph must still be usable afterward -- a real backward call (as
    # optimize_trigger_policy_step's checkpoint_backward path does) must still succeed and
    # produce the CORRECT combined gradient, unaffected by the two prior autograd.grad calls.
    step_loss = B2_k + lambda_bd * L_bd_k
    step_loss.backward()
    expected_total = torch.tensor([12.0, 12.0, 12.0]) + lambda_bd * torch.tensor([2.0, 4.0, 6.0])
    check("the real backward() afterward still produces the correct, undisturbed delta.grad",
          torch.allclose(delta.grad, expected_total, atol=1e-5),
          f"delta.grad={delta.grad}, expected={expected_total}")


if __name__ == "__main__":
    test_discretize_policy_matches_materialize_policy_flips()
    test_discretize_policy_respects_budget_and_caps_no_negatives()
    test_discretize_policy_small_deterministic_case()
    test_b2_qp_leq_b2_current()
    test_more_qp_iterations_does_not_degrade()
    test_qp_budget_active_and_inactive()
    test_qp_respects_per_class_caps()
    test_discretization_gap_destroys_perfect_continuous_solution()
    test_gradient_balance_does_not_touch_delta_grad_and_preserves_graph()

    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} checks passed.")
    sys.exit(1 if n_fail else 0)
