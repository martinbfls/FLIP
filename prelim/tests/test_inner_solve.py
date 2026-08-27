"""
prelim/tests/test_inner_solve.py -- non-regression checks on SYNTHETIC tensors only for
modules/federated_optimizing_trigger_policy/inner_solve.py, the controlled inner-solve
experiment (policy_inner_mode="joint"/"multi_step"/"qp_pgd"/"qp_pgd_reset") added to test
whether the co-descended policy u LAGS its conditional optimum u*(delta). No dataset, no model,
no training -- same convention as prelim/tests/test_policy_module_fixes.py /
test_policy_diagnostics.py.

Run:  python prelim/tests/test_inner_solve.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.federated_optimizing_trigger_policy import inner_solve as isolve

_results = []


def check(name, ok, detail=""):
    _results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


PAIRS = [(0, 1), (0, 2), (1, 0), (1, 2)]
PI = {0: 0.5, 1: 0.5}
BETA = 0.3


def _make_context(seed, D=10, den=1.0):
    rng = np.random.RandomState(seed)
    G_np = rng.normal(size=(D, len(PAIRS)))
    v_np = rng.normal(size=D)
    G = torch.tensor(G_np, dtype=torch.float64)
    v = torch.tensor(v_np, dtype=torch.float64)
    Q_obj = (G_np.T @ G_np).astype(np.float64)
    return {
        "k": seed, "G_obj": G, "Q_obj": Q_obj, "pairs_k": PAIRS, "rho_k": 1.0, "v": v,
        "den": den,
    }


def _directional_context(pair_idx, seed, D=6, strength=5.0, den=1.0):
    '''A context deliberately built so its OWN QP optimum concentrates mass on `pair_idx`
    (v is aligned almost entirely with that one column of G) -- used to build checkpoints that
    genuinely DISAGREE on which pair to flip, for test_oneshot_coupling_diagnostic_disagreement.'''
    rng = np.random.RandomState(seed)
    G_np = rng.normal(scale=0.01, size=(D, len(PAIRS)))
    v_np = np.zeros(D)
    v_np[0] = 1.0
    G_np[:, pair_idx] = strength * v_np
    G = torch.tensor(G_np, dtype=torch.float64)
    v = torch.tensor(v_np, dtype=torch.float64)
    Q_obj = (G_np.T @ G_np).astype(np.float64)
    return {
        "k": seed, "G_obj": G, "Q_obj": Q_obj, "pairs_k": PAIRS, "rho_k": 1.0, "v": v,
        "den": den,
    }


# --------------------------------------------------------------------------#
# E. Multi-checkpoint aggregation: the QP built from >1 checkpoints must be the DEN-weighted
# mean over ALL of them, not an arbitrary single one.
# --------------------------------------------------------------------------#

def test_aggregate_qp_is_weighted_mean_not_single_checkpoint():
    ctx1 = _make_context(seed=1, den=2.0)
    ctx2 = _make_context(seed=2, den=5.0)

    Q_agg, c_agg, pairs_agg = isolve.aggregate_qp([ctx1, ctx2])

    c1 = (ctx1["G_obj"].T @ ctx1["v"]).detach().cpu().numpy().astype(np.float64)
    c2 = (ctx2["G_obj"].T @ ctx2["v"]).detach().cpu().numpy().astype(np.float64)
    Q_expected = (ctx1["Q_obj"] / ctx1["den"] + ctx2["Q_obj"] / ctx2["den"]) / 2.0
    c_expected = (c1 / ctx1["den"] + c2 / ctx2["den"]) / 2.0

    check("aggregate_qp's Q matches the den-weighted mean over BOTH checkpoints",
          np.allclose(Q_agg, Q_expected), f"max|diff|={np.abs(Q_agg - Q_expected).max():.3e}")
    check("aggregate_qp's c matches the den-weighted mean over BOTH checkpoints",
          np.allclose(c_agg, c_expected), f"max|diff|={np.abs(c_agg - c_expected).max():.3e}")

    Q_single = ctx1["Q_obj"] / ctx1["den"]
    check("aggregate_qp is NOT just checkpoint 1's own (den-scaled) Q",
          not np.allclose(Q_agg, Q_single), "aggregate should differ from a single checkpoint")
    check("pairs are passed through unchanged", pairs_agg == PAIRS)


def test_aggregate_b2_matches_mean_of_per_checkpoint_b2():
    ctx1 = _make_context(seed=3, den=1.5)
    ctx2 = _make_context(seed=4, den=0.7)
    u = np.array([0.05, 0.03, 0.02, 0.01])

    b2_agg = isolve.aggregate_b2(u, [ctx1, ctx2])
    b2_1 = ((ctx1["G_obj"].detach().cpu().numpy() @ u - ctx1["v"].detach().cpu().numpy()) ** 2).sum() / ctx1["den"]
    b2_2 = ((ctx2["G_obj"].detach().cpu().numpy() @ u - ctx2["v"].detach().cpu().numpy()) ** 2).sum() / ctx2["den"]
    expected = (b2_1 + b2_2) / 2.0

    check("aggregate_b2 matches the hand-computed mean of the two per-checkpoint B2 values",
          abs(b2_agg - expected) < 1e-8, f"got {b2_agg}, expected {expected}")


# --------------------------------------------------------------------------#
# B. Inner QP improves the objective and stays feasible.
# --------------------------------------------------------------------------#

def test_qp_pgd_solve_improves_objective_and_feasible():
    ctx1 = _make_context(seed=5, den=1.0)
    ctx2 = _make_context(seed=6, den=2.0)
    contexts = [ctx1, ctx2]
    Q_agg, c_agg, pairs_agg = isolve.aggregate_qp(contexts)

    u_before = np.array([0.01, 0.0, 0.0, 0.0])
    b2_before = isolve.aggregate_b2(u_before, contexts)

    u_star, actual_iters, obj_start, obj_end, converged = isolve.qp_pgd_solve(
        Q_agg, c_agg, u_before, BETA, pairs_agg, PI, max_iters=300, tol=1e-10, min_iters=50,
    )
    b2_after = isolve.aggregate_b2(u_star, contexts)

    check("B2(u_after) <= B2(u_before) after the inner QP solve",
          b2_after <= b2_before + 1e-9, f"before={b2_before:.6f}, after={b2_after:.6f}")
    check("u_star is feasible (u in U_loc)", isolve.check_feasible(u_star, BETA, pairs_agg, PI))
    check("min_iters <= actual_iters <= max_iters", 50 <= actual_iters <= 300,
          f"actual_iters={actual_iters}")
    check("obj_end <= obj_start (the QP's own internal objective, not B2)",
          obj_end <= obj_start + 1e-9, f"obj_start={obj_start:.6f}, obj_end={obj_end:.6f}")


# --------------------------------------------------------------------------#
# C. More iterations -> B2 does not get worse (near-monotonic convergence).
# --------------------------------------------------------------------------#

def test_qp_pgd_solve_more_iterations_does_not_hurt():
    ctx1 = _make_context(seed=7, den=1.0)
    ctx2 = _make_context(seed=8, den=3.0)
    contexts = [ctx1, ctx2]
    Q_agg, c_agg, pairs_agg = isolve.aggregate_qp(contexts)
    u_init = np.zeros(len(PAIRS))

    # min_iters == max_iters in both calls -- forces the FULL iteration budget to run (no early
    # stop), so the two counts are directly comparable.
    u_50, _, _, _, _ = isolve.qp_pgd_solve(
        Q_agg, c_agg, u_init, BETA, pairs_agg, PI, max_iters=50, tol=1e-12, min_iters=50,
    )
    u_200, _, _, _, _ = isolve.qp_pgd_solve(
        Q_agg, c_agg, u_init, BETA, pairs_agg, PI, max_iters=200, tol=1e-12, min_iters=200,
    )
    b2_50 = isolve.aggregate_b2(u_50, contexts)
    b2_200 = isolve.aggregate_b2(u_200, contexts)

    check("B2 after 200 iterations <= B2 after 50 iterations",
          b2_200 <= b2_50 + 1e-6, f"B2_50={b2_50:.8f}, B2_200={b2_200:.8f}")


# --------------------------------------------------------------------------#
# D. Warm start vs. cold start converge close together given enough iterations (convex QP ->
# unique global optimum regardless of starting point).
# --------------------------------------------------------------------------#

def test_warm_start_and_cold_start_converge_to_the_same_point():
    ctx1 = _make_context(seed=9, den=1.0)
    ctx2 = _make_context(seed=10, den=1.5)
    contexts = [ctx1, ctx2]
    Q_agg, c_agg, pairs_agg = isolve.aggregate_qp(contexts)

    u_current = np.array([0.02, 0.01, 0.0, 0.0])
    u_warm, _, _, _, _ = isolve.qp_pgd_solve(
        Q_agg, c_agg, u_current, BETA, pairs_agg, PI, max_iters=2000, tol=0.0, min_iters=2000,
    )
    u_cold, _, _, _, _ = isolve.qp_pgd_solve(
        Q_agg, c_agg, np.zeros(len(PAIRS)), BETA, pairs_agg, PI, max_iters=2000, tol=0.0,
        min_iters=2000,
    )

    diff = np.abs(isolve.diag.as_numpy(u_warm) - isolve.diag.as_numpy(u_cold)).max()
    check("warm-start and cold-start solutions are close after enough iterations",
          diff < 1e-3, f"max|u_warm - u_cold|={diff:.3e}")


# --------------------------------------------------------------------------#
# check_feasible
# --------------------------------------------------------------------------#

def test_check_feasible():
    feasible_u = np.array([0.1, 0.1, 0.05, 0.05])  # sum=0.3==BETA, class sums 0.2/0.1 <= PI
    check("a feasible policy is flagged feasible",
          isolve.check_feasible(feasible_u, BETA, PAIRS, PI))

    over_budget_u = np.array([0.2, 0.2, 0.0, 0.0])  # sum=0.4 > BETA=0.3
    check("a policy exceeding the global budget is flagged infeasible",
          not isolve.check_feasible(over_budget_u, BETA, PAIRS, PI))

    negative_u = np.array([-0.01, 0.05, 0.0, 0.0])
    check("a policy with a negative entry is flagged infeasible",
          not isolve.check_feasible(negative_u, BETA, PAIRS, PI))

    over_class_cap_u = np.array([0.6, 0.0, 0.0, 0.0])  # class-0 sum=0.6 > pi[0]=0.5
    check("a policy exceeding a per-class cap is flagged infeasible",
          not isolve.check_feasible(over_class_cap_u, 10.0, PAIRS, PI))


# --------------------------------------------------------------------------#
# multi_step_update: K plain gradient steps via a real torch optimizer improve the objective
# and stay feasible.
# --------------------------------------------------------------------------#

def test_multi_step_update_improves_and_feasible():
    ctx1 = _make_context(seed=11, den=1.0)
    ctx2 = _make_context(seed=12, den=2.0)
    contexts = [ctx1, ctx2]
    Q_agg, c_agg, pairs_agg = isolve.aggregate_qp(contexts)

    u = torch.zeros(len(PAIRS), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([u], lr=0.05)

    b2_before = isolve.aggregate_b2(u.detach(), contexts)
    metrics = isolve.multi_step_update(u, Q_agg, c_agg, BETA, pairs_agg, PI, optimizer, k_steps=100)
    b2_after = isolve.aggregate_b2(u.detach(), contexts)

    check("multi_step_update's actual_iters == k_steps (no early stop in this mode)",
          metrics["actual_iters"] == 100)
    check("B2 improves after multi_step_update's K gradient steps",
          b2_after <= b2_before + 1e-9, f"before={b2_before:.6f}, after={b2_after:.6f}")
    check("u remains a leaf tensor requiring grad after multi_step_update (in-place updates only)",
          u.requires_grad and u.is_leaf)
    check("u is feasible after multi_step_update", isolve.check_feasible(u.detach(), BETA, pairs_agg, PI))


# --------------------------------------------------------------------------#
# Etape 0 (one-shot coupling audit): pairwise_cosine_stats and oneshot_coupling_diagnostic.
# --------------------------------------------------------------------------#

def test_pairwise_cosine_stats_identical_and_orthogonal():
    a = np.array([1.0, 2.0, 3.0])
    check("identical vectors -> cosine 1.0",
          abs(isolve.pairwise_cosine_stats([a, a.copy()])["mean"] - 1.0) < 1e-9)

    b = np.array([1.0, 0.0])
    c = np.array([0.0, 1.0])
    stats = isolve.pairwise_cosine_stats([b, c])
    check("orthogonal vectors -> cosine 0.0", abs(stats["mean"]) < 1e-9, f"got {stats['mean']}")

    stats_single = isolve.pairwise_cosine_stats([a])
    check("fewer than 2 policies -> None fields, no crash", stats_single["mean"] is None)


def test_oneshot_coupling_diagnostic_identical_checkpoints_zero_gap():
    # _directional_context (not _make_context) -- guarantees a non-degenerate (nonzero) QP
    # optimum: a random G/v pair can easily have c = G^T v with no positive component, in which
    # case the correct optimum IS u*=0 and cosine is undefined (nothing to compare direction
    # against) -- not a bug, just not what this test wants to exercise.
    ctx = _directional_context(pair_idx=0, seed=20)
    # Two independent context objects but with the IDENTICAL (G, v, den) -- checkpoints that
    # agree perfectly should show cosine ~= 1 and ~zero one-shot gap (the coupled ubar and the
    # per-checkpoint optima coincide when there is nothing to disagree about).
    ctx_copy = {**ctx, "k": 21}
    result = isolve.oneshot_coupling_diagnostic(
        [ctx, ctx_copy], np.zeros(len(PAIRS)), BETA, PAIRS, PI, n_iters=300,
    )
    check("identical checkpoints: u*_k pairwise cosine ~= 1",
          abs(result["u_star_pairwise_cosine_mean"] - 1.0) < 1e-3,
          f"cosine_mean={result['u_star_pairwise_cosine_mean']}")
    check("identical checkpoints: one-shot gap ~= 0",
          abs(result["oneshot_gap_absolute"]) < 1e-3,
          f"gap_absolute={result['oneshot_gap_absolute']}")
    check("oneshot_window_size reports the number of contexts passed in",
          result["oneshot_window_size"] == 2)


def test_oneshot_coupling_diagnostic_disagreeing_checkpoints_positive_gap():
    # Two checkpoints deliberately built to WANT different pairs flipped (pair 0 vs pair 2,
    # different source classes so neither is capped by the other's per-class budget) -- a
    # single shared ubar must compromise, so the gap should be clearly positive and the
    # per-checkpoint optima should NOT agree (cosine well below 1).
    ctx_a = _directional_context(pair_idx=0, seed=30)
    ctx_b = _directional_context(pair_idx=2, seed=31)
    result = isolve.oneshot_coupling_diagnostic(
        [ctx_a, ctx_b], np.zeros(len(PAIRS)), BETA, PAIRS, PI, n_iters=300,
    )
    check("disagreeing checkpoints: u*_k pairwise cosine well below 1",
          result["u_star_pairwise_cosine_mean"] < 0.5,
          f"cosine_mean={result['u_star_pairwise_cosine_mean']}")
    check("disagreeing checkpoints: one-shot gap is clearly positive (a shared ubar must compromise)",
          result["oneshot_gap_absolute"] > 1e-3,
          f"gap_absolute={result['oneshot_gap_absolute']:.6f}")
    check("disagreeing checkpoints: B2_coupled >= B2_per_checkpoint_mean (coupling can only hurt, "
          "never help, relative to each checkpoint's own independent optimum)",
          result["B2_coupled"] >= result["B2_per_checkpoint_mean"] - 1e-9)


# --------------------------------------------------------------------------#
# Etape 0bis.1/0bis.2 (follow-up task): raw/||v||^2-window-normalized B2, and per-checkpoint
# u* tagging for offline intra/inter-checkpoint cosine decomposition.
# --------------------------------------------------------------------------#

def test_oneshot_coupling_diagnostic_window_normalized_fields():
    ctx_a = _directional_context(pair_idx=0, seed=40)
    ctx_b = _directional_context(pair_idx=2, seed=41)
    u_current = np.array([0.01, 0.0, 0.0, 0.0])
    result = isolve.oneshot_coupling_diagnostic(
        [ctx_a, ctx_b], u_current, BETA, PAIRS, PI, n_iters=300,
    )

    check("v_sq_window_mean == mean of the two contexts' own ||v_k||^2",
          abs(result["v_sq_window_mean"]
              - np.mean([ctx_a["v"].norm().item() ** 2, ctx_b["v"].norm().item() ** 2])) < 1e-9)

    for prefix in ("B2_current_window", "B2_per_checkpoint_mean", "B2_coupled"):
        raw_key = f"{prefix}_raw" if prefix != "B2_current_window" else "B2_current_window_raw"
        norm_key = f"{prefix}_v2norm" if prefix != "B2_current_window" else "B2_current_window_v2norm"
        check(f"{norm_key} == {raw_key} / v_sq_window_mean (shared denominator, not per-checkpoint)",
              abs(result[norm_key] - result[raw_key] / result["v_sq_window_mean"]) < 1e-9,
              f"{norm_key}={result[norm_key]}, {raw_key}={result[raw_key]}")

    check("all three raw B2 quantities are non-negative (they are mean squared residuals)",
          result["B2_current_window_raw"] >= 0
          and result["B2_per_checkpoint_mean_raw"] >= 0
          and result["B2_coupled_raw"] >= 0)


def test_oneshot_coupling_diagnostic_per_checkpoint_u_star_tagging():
    ctx_a = _directional_context(pair_idx=0, seed=50)
    ctx_b = _directional_context(pair_idx=2, seed=51)
    ctx_a["k"], ctx_b["k"] = 7, 12  # distinct, arbitrary checkpoint ids
    result = isolve.oneshot_coupling_diagnostic(
        [ctx_a, ctx_b], np.zeros(len(PAIRS)), BETA, PAIRS, PI, n_iters=300,
    )
    tagged = result["per_checkpoint_u_star"]
    check("per_checkpoint_u_star has one entry per context, keyed by checkpoint id",
          set(tagged.keys()) == {"7", "12"}, f"keys={list(tagged.keys())}")
    check("each tagged u* has the right dimensionality (P pairs)",
          all(len(v) == len(PAIRS) for v in tagged.values()))
    # ctx_a favors pair_idx=0, ctx_b favors pair_idx=2 -- the tagged u*_k should reflect that.
    check("checkpoint 7's u* concentrates on pair 0 (its own directional context)",
          tagged["7"][0] > tagged["7"][2])
    check("checkpoint 12's u* concentrates on pair 2 (its own directional context)",
          tagged["12"][2] > tagged["12"][0])


# --------------------------------------------------------------------------#
# Etape 1a/1b (follow-up task): history-averaged (Q,c), m-sweep cosine, conditioning,
# convergence trace, NNLS cone projection, four-term decomposition.
# --------------------------------------------------------------------------#

def test_push_context_history_caps_at_m_and_averages():
    ctx = _directional_context(pair_idx=0, seed=60)
    history = {}
    # 3 observations with DIFFERENT v (so c differs each time), m=2 -- only the last 2 kept.
    for i in range(3):
        ctx_i = dict(ctx)
        ctx_i["v"] = ctx["v"] * (i + 1)  # v1, 2v1, 3v1
        isolve.push_context_history(history, [ctx_i], m=2)

    buf = history[ctx["k"]]
    check("history capped at m=2 entries", len(buf) == 2)
    c_expected_last_two_mean = np.mean([
        (ctx["G_obj"].T @ (ctx["v"] * 2)).detach().cpu().numpy(),
        (ctx["G_obj"].T @ (ctx["v"] * 3)).detach().cpu().numpy(),
    ], axis=0)
    c_actual_mean = np.mean([e[0] for e in buf], axis=0)
    check("averaged c matches the mean of the LAST 2 (not all 3) observations",
          np.allclose(c_actual_mean, c_expected_last_two_mean, atol=1e-8))


def test_aggregate_qp_from_history_matches_aggregate_qp_with_single_observation():
    ctx1 = _make_context(seed=61, den=1.0)
    ctx2 = _make_context(seed=62, den=2.0)
    contexts = [ctx1, ctx2]
    history = {}
    isolve.push_context_history(history, contexts, m=5)

    Q_a, c_a, pairs_a = isolve.aggregate_qp(contexts)
    Q_b, c_b, pairs_b = isolve.aggregate_qp_from_history(contexts, history)
    check("Q unaffected by history (checkpoint-fixed)", np.allclose(Q_a, Q_b))
    check("c matches aggregate_qp with only one observation in history",
          np.allclose(c_a, c_b, atol=1e-8))


def test_m_sweep_cosine_plateaus_at_1_for_noise_free_history():
    ctx = _directional_context(pair_idx=0, seed=63)
    # Identical c every batch (no noise at all) -- cosine should be exactly 1 for every m.
    c_k = (ctx["G_obj"].T @ ctx["v"]).detach().cpu().numpy().astype(np.float64)
    c_history = [c_k.copy() for _ in range(25)]
    u_star_by_m, cosine_by_m = isolve.m_sweep_cosine(
        c_history, ctx["Q_obj"], BETA, PAIRS, PI, [1, 2, 5, 10, 20], n_iters=300,
    )
    check("noise-free history: cosine is ~1 for every m",
          all(abs(c - 1.0) < 1e-6 for c in cosine_by_m.values()),
          f"cosine_by_m={cosine_by_m}")


def test_m_sweep_cosine_improves_toward_reference_with_noisy_history():
    ctx = _directional_context(pair_idx=0, seed=64)
    c_true = (ctx["G_obj"].T @ ctx["v"]).detach().cpu().numpy().astype(np.float64)
    rng = np.random.RandomState(0)
    c_history = [c_true + rng.normal(scale=0.5, size=c_true.shape) for _ in range(25)]
    _, cosine_by_m = isolve.m_sweep_cosine(
        c_history, ctx["Q_obj"], BETA, PAIRS, PI, [1, 20], n_iters=300,
    )
    check("m_sweep_cosine requires >= max(m_values) observations",
          isinstance(cosine_by_m[1], float) and isinstance(cosine_by_m[20], float))

    try:
        isolve.m_sweep_cosine(c_history[:3], ctx["Q_obj"], BETA, PAIRS, PI, [1, 20], n_iters=50)
        check("m_sweep_cosine raises with too few observations", False)
    except ValueError:
        check("m_sweep_cosine raises with too few observations", True)


def test_qp_conditioning_diagnostic():
    Q = np.diag([1.0, 4.0, 9.0])
    result = isolve.qp_conditioning_diagnostic(Q, ridge=0.0)
    check("lambda_min matches the smallest diagonal entry", abs(result["lambda_min"] - 1.0) < 1e-9)
    check("lambda_max matches the largest diagonal entry", abs(result["lambda_max"] - 9.0) < 1e-9)
    check("condition_number == lambda_max/lambda_min", abs(result["condition_number"] - 9.0) < 1e-9)


def test_qp_pgd_solve_with_trace_reports_plateauing_decrement():
    ctx1 = _make_context(seed=65, den=1.0)
    ctx2 = _make_context(seed=66, den=1.5)
    Q_agg, c_agg, pairs_agg = isolve.aggregate_qp([ctx1, ctx2])
    u_star, obj_decrement, trace = isolve.qp_pgd_solve_with_trace(
        Q_agg, c_agg, np.zeros(len(PAIRS)), BETA, pairs_agg, PI, max_iters=500, trace_last_n=100,
    )
    check("trace has trace_last_n entries", len(trace) == 100)
    check("obj_decrement over the last 100 iters is small (near-converged on this easy problem)",
          abs(obj_decrement) < 1e-3, f"obj_decrement={obj_decrement}")
    check("u_star is feasible", isolve.check_feasible(u_star, BETA, pairs_agg, PI))


def test_nnls_cone_projection_no_worse_than_qp_and_feasible_nonneg():
    ctx = _directional_context(pair_idx=0, seed=67)
    alpha_tilde_sq, u_nnls = isolve.nnls_cone_projection(ctx["G_obj"], ctx["v"], ctx["den"])
    check("NNLS solution is nonnegative (u>=0, no other constraint)",
          bool((u_nnls >= -1e-9).all()))

    u_qp, _, _ = isolve.project_gradient_descent_local(
        ctx["Q_obj"], (ctx["G_obj"].T @ ctx["v"]).detach().cpu().numpy(), np.zeros(len(PAIRS)),
        BETA, PAIRS, PI, n_iters=300,
    )
    b2_qp_val, _ = isolve.diag.b2_value(ctx["G_obj"], u_qp, ctx["v"], ctx["den"])
    check("alpha_tilde^2 (cone-only, no budget) <= B2_qp (cone AND budget) -- fewer "
          "constraints can only do at least as well",
          alpha_tilde_sq <= b2_qp_val + 1e-6, f"alpha_tilde_sq={alpha_tilde_sq}, B2_qp={b2_qp_val}")


def test_four_term_decomposition_orders_the_four_terms():
    ctx = _directional_context(pair_idx=0, seed=68)
    u_qp, _, _ = isolve.project_gradient_descent_local(
        ctx["Q_obj"], (ctx["G_obj"].T @ ctx["v"]).detach().cpu().numpy(), np.zeros(len(PAIRS)),
        BETA, PAIRS, PI, n_iters=300,
    )
    u_current = np.zeros(len(PAIRS))  # a deliberately bad "current" policy (never moved)
    result = isolve.four_term_decomposition(
        ctx["G_obj"], ctx["v"], ctx["den"], u_current, u_qp, ctx["rho_k"],
    )
    check("B2_span <= alpha_tilde^2 (fewer constraints do at least as well)",
          result["B2_span_over_den"] <= result["alpha_tilde_sq"] + 1e-6,
          f"{result}")
    check("alpha_tilde^2 <= B2_QP (fewer constraints do at least as well)",
          result["alpha_tilde_sq"] <= result["B2_QP_over_den"] + 1e-6, f"{result}")
    check("B2_current (never moved from 0) is at least as bad as B2_QP",
          result["B2_current_over_den"] >= result["B2_QP_over_den"] - 1e-9, f"{result}")


if __name__ == "__main__":
    test_aggregate_qp_is_weighted_mean_not_single_checkpoint()
    test_aggregate_b2_matches_mean_of_per_checkpoint_b2()
    test_qp_pgd_solve_improves_objective_and_feasible()
    test_qp_pgd_solve_more_iterations_does_not_hurt()
    test_warm_start_and_cold_start_converge_to_the_same_point()
    test_check_feasible()
    test_multi_step_update_improves_and_feasible()
    test_pairwise_cosine_stats_identical_and_orthogonal()
    test_oneshot_coupling_diagnostic_identical_checkpoints_zero_gap()
    test_oneshot_coupling_diagnostic_disagreeing_checkpoints_positive_gap()
    test_oneshot_coupling_diagnostic_window_normalized_fields()
    test_oneshot_coupling_diagnostic_per_checkpoint_u_star_tagging()
    test_push_context_history_caps_at_m_and_averages()
    test_aggregate_qp_from_history_matches_aggregate_qp_with_single_observation()
    test_m_sweep_cosine_plateaus_at_1_for_noise_free_history()
    test_m_sweep_cosine_improves_toward_reference_with_noisy_history()
    test_qp_conditioning_diagnostic()
    test_qp_pgd_solve_with_trace_reports_plateauing_decrement()
    test_nnls_cone_projection_no_worse_than_qp_and_feasible_nonneg()
    test_four_term_decomposition_orders_the_four_terms()

    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} checks passed.")
    sys.exit(1 if n_fail else 0)
