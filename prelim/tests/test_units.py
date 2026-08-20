"""
prelim/tests/test_units.py -- non-regression checks on SYNTHETIC tensors only.

These are not experiments: no dataset is downloaded, no model is trained, no
Gbar is estimated on real data. Everything runs on random tensors with d=50,
C=4, n_b=6 and finishes in a couple of seconds, which is what makes it safe to
run in a code-writing session (see prelim/SPEC.md section 1: no unrequested
robustness layers, and section 9: the ordering of the real work).

Run:  python prelim/tests/test_units.py
"""
import os
import sys
import math

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import prelim_lib as pl

D = 50
C = 4
N_B = 6
F = 2
GAMMA = 0.3
BETA = 0.10

PAIRS = [(y, z) for y in range(C) for z in range(C) if z != y]
PI = {y: 1.0 / C for y in range(C)}

_results = []


def check(name, ok, detail=""):
    _results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def synth(seed=0):
    """A synthetic (Gbar, Q, pairs) triple with the shapes of the real thing."""
    rng = np.random.RandomState(seed)
    Gbar = torch.as_tensor(rng.randn(D, len(PAIRS)).astype(np.float32))
    Q = (Gbar.T @ Gbar).numpy().astype(np.float64)
    return Gbar, Q


# --------------------------------------------------------------------------#
# solve_qp
# --------------------------------------------------------------------------#
def test_solve_qp_delegation():
    Gbar, Q = synth(0)
    v = torch.as_tensor(np.random.RandomState(1).randn(D).astype(np.float32))
    c = (Gbar.T @ v).numpy().astype(np.float64)
    w_ours = pl.solve_qp(Q, c, BETA, PI, GAMMA, PAIRS, scope="aggregate", capacity=False).numpy()
    w_repo = pl._project_gradient_reused(Q, c, BETA, PAIRS).numpy()
    gap = float(np.max(np.abs(w_ours - w_repo)))
    check("solve_qp(capacity=False, aggregate) == repo project_gradient", gap <= 1e-6,
          f"max|diff| = {gap:.3e} (threshold 1e-6)")


def test_solve_qp_capacity():
    # Orthonormal columns, so the QP decouples across (y,z) and the effect of the
    # per-class caps is readable without a least-squares coupling term muddying it.
    rng = np.random.RandomState(0)
    M = np.linalg.qr(rng.randn(D, len(PAIRS)))[0]
    Gbar = torch.as_tensor(M.astype(np.float32))
    Q = (Gbar.T @ Gbar).numpy().astype(np.float64)

    # A target reachable only through class 0: gamma*pi[0] = 0.075 < beta = 0.10,
    # so the class cap must bind and stop the budget short of beta.
    v = Gbar[:, PAIRS.index((0, 1))] * 1.0
    c = (Gbar.T @ v).numpy().astype(np.float64)

    w_free = pl.solve_qp(Q, c, BETA, PI, GAMMA, PAIRS, scope="aggregate", capacity=False).numpy()
    w_cap = pl.solve_qp(Q, c, BETA, PI, GAMMA, PAIRS, scope="aggregate", capacity=True).numpy()

    def class_load(w, yy):
        return sum(w[i] for i, (y, _) in enumerate(PAIRS) if y == yy)

    caps_ok = all(class_load(w_cap, yy) <= GAMMA * PI[yy] + 1e-6 for yy in range(C))
    check("solve_qp(capacity=True): per-class caps sum_z u[y,z] <= gamma*pi[y] hold",
          caps_ok, f"max class load = {max(class_load(w_cap, yy) for yy in range(C)):.4g}"
                   f" vs cap {GAMMA * min(PI.values()):.4g}")
    check("solve_qp: budget sum(u) <= beta holds in both variants",
          w_cap.sum() <= BETA + 1e-6 and w_free.sum() <= BETA + 1e-6,
          f"||u||_1 = {w_cap.sum():.4g} (cap) / {w_free.sum():.4g} (free), beta = {BETA}")
    # SPEC section 8/E3: with s_beta > 1 and a single-source objective, capacity=True
    # must stop strictly below beta while capacity=False spends the whole budget.
    check("solve_qp: caps strictly bind for a single-source objective (||u*||_1 < beta)",
          w_cap.sum() < BETA - 1e-4 and abs(w_cap.sum() - GAMMA * PI[0]) < 1e-4,
          f"||u*||_1(capacity=True) = {w_cap.sum():.4g}, cap gamma*pi[0] = {GAMMA * PI[0]:.4g}, "
          f"beta = {BETA}; capacity=False reaches {w_free.sum():.4g}")


def test_solve_qp_scope():
    Gbar, Q = synth(0)
    v = torch.as_tensor(np.random.RandomState(2).randn(D).astype(np.float32))
    c = (Gbar.T @ v).numpy().astype(np.float64)
    w_loc = pl.solve_qp(Q, c, BETA, PI, GAMMA, PAIRS, scope="local", capacity=True).numpy()
    ok_budget = w_loc.sum() <= BETA / GAMMA + 1e-6
    ok_caps = all(sum(w_loc[i] for i, (y, _) in enumerate(PAIRS) if y == yy) <= PI[yy] + 1e-6
                  for yy in range(C))
    check("solve_qp(scope='local'): budget beta/gamma and caps pi[y] hold",
          ok_budget and ok_caps, f"||u_i||_1 = {w_loc.sum():.4g} <= {BETA / GAMMA:.4g}")

    # The budget-scope hazard of SPEC section 2: a deployed u_i = ubar/gamma must
    # satisfy ||u_i||_1 * gamma == ||ubar||_1 exactly.
    w_agg = pl.solve_qp(Q, c, BETA, PI, GAMMA, PAIRS, scope="aggregate", capacity=True).numpy()
    gap = abs((w_agg / GAMMA).sum() * GAMMA - w_agg.sum())
    check("budget-scope relation ||u_i||_1 * gamma == ||ubar||_1", gap <= 1e-9,
          f"|gap| = {gap:.3e}")


# --------------------------------------------------------------------------#
# dist_to_cone / rank_ratio
# --------------------------------------------------------------------------#
def test_dist_to_cone():
    Gbar, Q = synth(0)
    rng = np.random.RandomState(3)

    # v built inside the cone {Gbar u : u >= 0}: distance must vanish, alpha = 1.
    u_in = np.abs(rng.randn(len(PAIRS)))
    v_in = torch.as_tensor((Gbar.numpy() @ u_in).astype(np.float32))
    c_in = (Gbar.T @ v_in).numpy().astype(np.float64)
    d2, alpha, w = pl.dist_to_cone(Q, c_in, float(v_in.norm() ** 2), PAIRS)
    rel = d2 / float(v_in.norm() ** 2)
    check("dist_to_cone: v inside the cone -> dist^2 ~ 0 and alpha_tilde_star ~ 1",
          rel < 1e-6 and abs(alpha - 1.0) < 1e-3, f"dist^2/||v||^2 = {rel:.3e}, alpha = {alpha:.6f}")

    # Generic v: alpha in [0,1] and the "unbounded" budget must stay inactive.
    v = torch.as_tensor(rng.randn(D).astype(np.float32))
    c = (Gbar.T @ v).numpy().astype(np.float64)
    d2, alpha, w = pl.dist_to_cone(Q, c, float(v.norm() ** 2), PAIRS)
    frac = float(w.numpy().sum()) / 1e6
    check("dist_to_cone: alpha_tilde_star in [0,1] for a generic target",
          0.0 <= alpha <= 1.0, f"alpha = {alpha:.6f}")
    check("dist_to_cone: budget inactive at the optimum (sum(w*) << big_beta)",
          frac < 1e-2, f"sum(w*)/big_beta = {frac:.3e} (threshold 1e-2)")


def test_rank_ratio():
    Gbar, Q = synth(0)
    rng = np.random.RandomState(4)
    # v in range(Gbar): the rank ratio must be exactly 1.
    w = rng.randn(len(PAIRS))
    v = torch.as_tensor((Gbar.numpy() @ w).astype(np.float32))
    c = (Gbar.T @ v).numpy().astype(np.float64)
    varpi = pl.rank_ratio(Q, c, float(v.norm() ** 2))
    check("rank_ratio: varpi == 1 when v lies in range(Gbar)", abs(varpi - 1.0) < 1e-3,
          f"varpi = {varpi:.6f}")

    # d=50 > P=12 columns, so a generic v is mostly outside range(Gbar).
    v2 = torch.as_tensor(rng.randn(D).astype(np.float32))
    c2 = (Gbar.T @ v2).numpy().astype(np.float64)
    varpi2 = pl.rank_ratio(Q, c2, float(v2.norm() ** 2))
    check("rank_ratio: varpi in [0,1] for a generic target and below 1",
          0.0 <= varpi2 <= 1.0 + 1e-6 and varpi2 < 1.0, f"varpi = {varpi2:.4f}")

    er = pl.effective_rank(Q)
    check("effective_rank(Q) in (0, P]", 0 < er <= len(PAIRS) + 1e-6, f"eff_rank = {er:.4f} (P = {len(PAIRS)})")


# --------------------------------------------------------------------------#
# support_function / reachable_radius
# --------------------------------------------------------------------------#
def test_support_function():
    Gbar, Q = synth(0)
    col_index = {p: i for i, p in enumerate(PAIRS)}
    rng = np.random.RandomState(5)
    worst = 0.0
    for t in range(5):
        p = torch.as_tensor(rng.randn(D).astype(np.float32))
        h, u_star = pl.support_function(Gbar, p, BETA, PI, GAMMA, PAIRS, col_index)

        # The greedy value must be attained by its own u*, and must dominate any
        # feasible u -- checked against a linear program over U_beta.
        u_vec = np.array([u_star[q] for q in PAIRS])
        attained = float(p.numpy() @ (Gbar.numpy() @ u_vec))
        assert abs(attained - h) < 1e-4 * max(1.0, abs(h)), (attained, h)

        from scipy.optimize import linprog
        obj = -(Gbar.numpy().T @ p.numpy()).astype(np.float64)
        A_ub = [np.ones(len(PAIRS))]
        b_ub = [BETA]
        for yy in range(C):
            A_ub.append(np.array([1.0 if y == yy else 0.0 for y, _ in PAIRS]))
            b_ub.append(GAMMA * PI[yy])
        res = linprog(obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                      bounds=[(0, None)] * len(PAIRS), method="highs")
        h_lp = -res.fun
        worst = max(worst, abs(h - h_lp) / max(1e-9, abs(h_lp)))
    check("support_function: greedy water-filling matches the LP optimum over U_beta",
          worst < 1e-6, f"max relative gap over 5 random p = {worst:.3e}")


def test_reachable_radius():
    Gbar, Q = synth(0)
    br = pl.reachable_radius(Gbar, BETA, PI, GAMMA, PAIRS, n_restarts=4, n_steps=15, seed=0)
    ordered = br["lower_simple"] <= br["lower_ascent"] + 1e-9 and br["lower_ascent"] <= br["upper"] + 1e-9
    check("reachable_radius: lower_simple <= lower_ascent <= upper (bracket, not a value)",
          ordered, f"{br['lower_simple']:.4g} <= {br['lower_ascent']:.4g} <= {br['upper']:.4g}")
    check("reachable_radius: upper == beta * varsigma",
          abs(br["upper"] - BETA * br["varsigma"]) < 1e-9,
          f"upper = {br['upper']:.4g}, beta*varsigma = {BETA * br['varsigma']:.4g}")


# --------------------------------------------------------------------------#
# masses_to_labels
# --------------------------------------------------------------------------#
def test_masses_to_labels():
    rng_t = np.random.RandomState(6)
    n = 600
    targets = rng_t.randint(0, C, size=n)
    u = {p: 0.0 for p in PAIRS}
    u[(0, 1)] = 0.05
    u[(0, 2)] = 0.03
    u[(2, 3)] = 0.04
    rng = np.random.RandomState(7)
    poisoned, u_real = pl.masses_to_labels(targets, u, PAIRS, rng)

    gaps = [abs(u_real[p] - u[p]) for p in PAIRS if u[p] > 0]
    check("masses_to_labels: realised masses within one example of the request",
          max(gaps) <= 1.0 / n + 1e-12, f"max|u_real - u| = {max(gaps):.3e} <= 1/n = {1.0/n:.3e}")

    changed = np.where(poisoned != targets)[0]
    n_expected = sum(int(round(u[p] * n)) for p in PAIRS if u[p] > 0)
    check("masses_to_labels: exactly the requested number of labels moved, no overlap",
          len(changed) == n_expected, f"{len(changed)} changed, {n_expected} requested")

    # Every moved example must have been a true member of its source class, and
    # must now carry the requested target label.
    ok = True
    for (y, z) in [(0, 1), (0, 2), (2, 3)]:
        sel = np.where((targets == y) & (poisoned == z))[0]
        ok = ok and len(sel) == int(round(u[(y, z)] * n))
    check("masses_to_labels: each (y,z) block draws only from true class y", ok)


# --------------------------------------------------------------------------#
# Instrumented aggregation rules (SPEC section 7)
# --------------------------------------------------------------------------#
def _stack(seed=0, n_b=N_B, d=D, n_p=2, shift=3.0):
    rng = np.random.RandomState(seed)
    G = rng.randn(n_b, d).astype(np.float32)
    G[:n_p] += shift * rng.randn(1, d).astype(np.float32)   # perturbed workers
    mal = torch.zeros(n_b, dtype=torch.bool)
    mal[:n_p] = True
    return torch.as_tensor(G), mal


def test_aggregators_flat_vs_repo():
    G, mal = _stack(10)
    worst = {}
    for rule in pl.AGG_RULES:
        agg, sel = pl.aggregate_instrumented(G, rule, F, variant="flat")
        ref = pl.repo_aggregate(G, rule, F)
        worst[rule] = float((agg - ref).abs().max())
        assert sel.ell == pl.agg_ell(rule, N_B, F), (rule, sel.ell)
    check("flat instrumented rule == repo rule, all 5 rules",
          max(worst.values()) <= 1e-6,
          "max|diff| " + ", ".join(f"{k}={v:.1e}" for k, v in worst.items()))


def test_aggregators_single_tensor_concordance():
    """SPEC section 2: flat and per_tensor must coincide on a single-tensor model."""
    G, mal = _stack(11)
    blocks = ((0, D),)                       # a model with exactly one parameter tensor
    worst = {}
    for rule in pl.AGG_RULES:
        a_flat, s_flat = pl.aggregate_instrumented(G, rule, F, variant="flat", check_repo=True)
        a_pt, s_pt = pl.aggregate_instrumented(G, rule, F, variant="per_tensor",
                                               blocks=blocks, check_repo=True)
        worst[rule] = float((a_flat - a_pt).abs().max())
    check("single-tensor model: flat == per_tensor == repo, all 5 rules",
          max(worst.values()) <= 1e-6,
          "max|diff| " + ", ".join(f"{k}={v:.1e}" for k, v in worst.items()))


def test_selection_algebra():
    G, mal = _stack(12)
    ok_pn, ok_A, ok_chi = True, True, True
    detail = []
    mean = G.mean(dim=0)
    for rule in pl.AGG_RULES:
        agg, sel = pl.aggregate_instrumented(G, rule, F, variant="flat")
        A = sel.A(mal)
        ok_A = ok_A and bool(((A >= -1e-6) & (A <= 1 + 1e-6)).all())
        P, N = sel.split_PN(G, mal)
        resid = float(((P + N) - (agg - mean)).abs().max())
        ok_pn = ok_pn and resid <= 1e-4
        detail.append(f"{rule}:{resid:.1e}")
        chi_ref = (sel.n_b - sel.ell) / (sel.ell * sel.n_b)
        ok_chi = ok_chi and abs(sel.chi_ell - chi_ref) < 1e-12 \
                        and abs(sel.lam - 2 * sel.ell * sel.chi_ell) < 1e-12
    check("selection: P + N == b_Agg - b_mean (exact decomposition)", ok_pn,
          "max|resid| " + ", ".join(detail))
    check("selection: A_j in [0,1] for every rule", ok_A)
    check("selection: chi_ell = (n_b-ell)/(ell*n_b) and Lambda = 2*ell*chi_ell", ok_chi)


def _simplex_block(n_b, width, centre, far):
    """
    One parameter tensor on which Krum provably elects `centre`: `centre` sits at
    the origin, three workers sit one unit away along distinct orthogonal axes
    (so their mutual distances are sqrt(2) and their scores are strictly worse),
    and the remaining workers sit `far` away. Deterministic, so the per_tensor
    oscillation below is a property of the rule, not of a lucky draw.
    """
    B = np.zeros((n_b, width), dtype=np.float32)
    others = [i for i in range(n_b) if i != centre]
    for axis, i in enumerate(others[:3]):
        B[i, axis] = 1.0
    for axis, i in enumerate(others[3:], start=3):
        B[i, axis] = far
    return B


def test_osc_flat_vs_per_tensor():
    """
    SPEC section 7's falsifiable structural claim: under `flat`, krum and
    multikrum select one set for every coordinate, so osc(Abar) = 0 EXACTLY;
    under `per_tensor` the set is constant within a tensor but varies between
    tensors, so osc(Abar) > 0.
    """
    blocks = ((0, 20), (20, 45), (45, 60))
    centres = (1, 4, 1)          # worker 1 is perturbed, worker 4 is honest
    parts = [_simplex_block(N_B, e_ - s_, k, far=50.0)
             for (s_, e_), k in zip(blocks, centres)]
    G = torch.as_tensor(np.concatenate(parts, axis=1))
    mal = torch.zeros(N_B, dtype=torch.bool)
    mal[:2] = True                                  # workers 0 and 1 are perturbed

    flat_osc, pt_osc = {}, {}
    for rule in ("krum", "multikrum"):
        _, s_flat = pl.aggregate_instrumented(G, rule, F, variant="flat")
        _, s_pt = pl.aggregate_instrumented(G, rule, F, variant="per_tensor", blocks=blocks)
        flat_osc[rule] = pl.osc(s_flat.A(mal))
        pt_osc[rule] = pl.osc(s_pt.A(mal))
    check("krum/multikrum: osc(A) == 0 exactly under flat",
          all(v == 0.0 for v in flat_osc.values()),
          ", ".join(f"{k}={v:g}" for k, v in flat_osc.items()))
    check("krum/multikrum: osc(A) > 0 under per_tensor (set varies between tensors)",
          all(v > 0.0 for v in pt_osc.values()),
          ", ".join(f"{k}={v:g}" for k, v in pt_osc.items()))


def test_alpha_tilde():
    rng = np.random.RandomState(14)
    v = torch.as_tensor(rng.randn(D).astype(np.float32))
    check("alpha_tilde(v, v) == 1 and alpha_tilde(-v, v) == 0",
          abs(pl.alpha_tilde(v, v) - 1.0) < 1e-6 and pl.alpha_tilde(-v, v) == 0.0)


# --------------------------------------------------------------------------#
def main():
    tests = [
        test_solve_qp_delegation, test_solve_qp_capacity, test_solve_qp_scope,
        test_dist_to_cone, test_rank_ratio,
        test_support_function, test_reachable_radius,
        test_masses_to_labels,
        test_aggregators_flat_vs_repo, test_aggregators_single_tensor_concordance,
        test_selection_algebra, test_osc_flat_vs_per_tensor, test_alpha_tilde,
    ]
    for t in tests:
        try:
            t()
        except Exception as exc:  # a raised assertion is itself a failed check
            check(t.__name__, False, f"raised {type(exc).__name__}: {exc}")
    n_ok = sum(1 for _, ok, _ in _results if ok)
    print(f"\n{n_ok}/{len(_results)} checks passed")
    return 0 if n_ok == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
