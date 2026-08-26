"""
Diagnostics for federated_optimizing_trigger_policy's co-descended (delta, u) attack (solver
(b), see run_module.py's `optimize_trigger_policy_step` docstring). Pure, mostly-stateless
helpers -- no side effects beyond `DiagnosticsWriter`'s file writes -- so they can be unit
tested against synthetic tensors without a model/dataset/training loop (see
prelim/tests/test_policy_diagnostics.py).

None of this changes the algorithm itself (`optimize_trigger_policy_step`'s co-descent, the
projection onto U_loc, beta/lambda_poison/pairs/G_obj conventions): every function here either
reads (u, delta, G_obj, ...) without touching them, or computes an independent reference
quantity (a QP solve, a discretized policy, an actually-measured gradient) to compare against
what the co-descended attack is doing. Disabled diagnostics (diag_path=None, the default) cost
nothing beyond what B2_qp already computed.

How to read the diagnostic chain (Section 13 of the task, also see the .jsonl fields this
module produces):

    B2 >> B2_qp
        => u is lagging delta / not solved well enough this batch (solver (b)'s co-descent gap,
           see optimize_trigger_policy_step's docstring) -- an OPTIMIZATION problem, not a
           feasibility one.

    B2_qp itself is high (v is far from anything gamma*H@U_loc can reach)
        => a GEOMETRIC FEASIBILITY problem: no u in U_loc makes gamma*H@u track v(delta) well,
           regardless of how well u is optimized. Look at `cosine_qp`/`residual_qp` (Diagnostic
           B): a low cosine plus a large residual even at the QP optimum means the trigger's
           induced shift v(delta) is not in this attacker's reachable set -- changing lr_policy
           or n_steps will not fix this; delta itself (or beta) has to change.

    B2_qp low, but B2_qp_discrete high
        => a DISCRETIZATION problem: the continuous QP optimum is reachable, but rounding it
           into actual integer flip counts (`discretize_policy`, Diagnostic C) destroys most of
           the alignment -- typically because u is spread thin over many pairs whose realized
           counts round to (near) zero. Look at `policy_nnz` vs `policy_nnz` after
           discretization, and whether u is concentrated (few pairs, large mass each) or diffuse.

    B2_qp_discrete low, but the actual/predicted gradient mismatch is high (Diagnostic D)
        => the ANALYTIC MODEL gamma*H@u does not faithfully predict the gradient shift a real
           materialization induces -- e.g. class-conditional sample size too small, a batching/
           normalization mismatch between compute_expected_flip_gradients and the real
           downstream flip application. Worth revisiting before trusting B2 at all.

    B2 (or B2_qp) low, but ||grad_delta_BD|| << ||grad_delta_B2|| (Diagnostic G)
        => the B2 matching term dominates delta's own gradient; lambda_bd may be too small
           relative to B2's scale for delta to actually learn to backdoor -- a LOSS-BALANCE
           problem, independent of how well u itself is doing.

    B2_qp_50 ~= B2_qp_200 ~= B2_qp_1000 (qp_*_vs_*_relative_improvement ~= 0 at every step)
        => the QP diagnostic itself has converged -- diag_qp_iters is not the bottleneck, and
           B2_qp can be trusted as (close to) the true U_loc optimum for the checks below. If
           instead the sweep keeps improving at the largest check_iters, B2_qp is itself
           under-converged and every downstream comparison against it is optimistic.

    B2_span low (v is well explained by span(G_obj) with NO constraints), but B2_qp still high
        => the constraint set U_loc (u>=0, budget, per-class caps) -- NOT the flip gradients'
           span -- is what is actually limiting the attack. Check `qp_global_budget_active` /
           `qp_any_class_cap_active`: if the budget is active, sweeping beta may help; if a
           class cap is active instead, no beta sweep will (that source class is out of
           examples to flip, see prop:budget-match's saturated regime, rem:saturated).

    B2_span itself high
        => even the FULLY UNCONSTRAINED best linear combination of flip-gradient columns cannot
           track v(delta) -- a genuine subspace-coverage limitation of the (source, target)
           pairs available, independent of beta/lr_policy/n_steps entirely. Changing delta (or
           the set of pairs u is optimized over) is the only lever left.

    cosine_qp_v ~= 1 but residual_after_optimal_scaling_qp is still a large fraction of
    residual_qp
        => AMPLITUDE-only mismatch: the QP policy already points the right way, it just cannot
           reach the right magnitude under U_loc -- consistent with `qp_global_budget_active`
           (not enough budget to scale up) rather than a directional problem.

    cosine_qp_v far from 1 AND residual_after_optimal_scaling_qp stays close to residual_qp
        => a genuine DIRECTIONAL limitation: no rescaling of Gu_qp gets it close to v -- look at
           B2_span next to tell whether this is a constraint effect or intrinsic to G_obj's
           span.

    policy_qp_l1_distance / policy_qp_l2_distance small (u_current close to u_qp) but B2 still
    high
        => u itself is close to its own local optimum already; the residual is a feasibility
           problem (see B2_qp/B2_span above), not a lagging/under-trained policy.

    Every diagnostic above looks fine, but the final trained attack still fails
        => the mismatch is likely between this module's per-step local objective (eq:P, a
           gradient-alignment proxy evaluated on FIXED expert checkpoints) and the actual LONG
           federated training dynamics the victim undergoes -- outside what any of these
           per-batch diagnostics can see; consider expert_budget/checkpoint coverage instead.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from modules.federated_policy_to_flips.utils import compute_flip_counts
from modules.federated_optimizing_trigger.utils import compute_batch_gradients, raw_to_preprocess
from modules.federated_optimizing_trigger_policy.utils import project_gradient_descent_local

_EPS = 1e-8


def as_numpy(u):
    return u.detach().cpu().numpy().astype(np.float64) if torch.is_tensor(u) else np.asarray(u, dtype=np.float64)


_as_np = as_numpy  # internal alias used throughout this module


# --------------------------------------------------------------------------- #
# Diagnostic F -- budget / sparsity stats (also used by C's discretized side)
# --------------------------------------------------------------------------- #

def policy_l1(u):
    return float(_as_np(u).sum())


def policy_nnz(u, threshold=1e-8):
    return int((np.abs(_as_np(u)) > threshold).sum())


def policy_max_entry(u):
    u_np = _as_np(u)
    return float(np.abs(u_np).max()) if u_np.size else 0.0


def policy_topk(u, pairs, k=10):
    u_np = _as_np(u)
    if u_np.size == 0 or k <= 0:
        return []
    idx = np.argsort(-np.abs(u_np))[:min(k, u_np.size)]
    return [{"pair": [int(pairs[i][0]), int(pairs[i][1])], "value": float(u_np[i])} for i in idx]


def policy_sum_by_source_class(u, pairs):
    u_np = _as_np(u)
    out = {}
    for (y, c), val in zip(pairs, u_np):
        out[int(y)] = out.get(int(y), 0.0) + float(val)
    return out


def policy_stats(u, pairs, beta, threshold=1e-8, topk=10):
    '''Diagnostic F's per-policy bundle (used for both the current u and the QP reference).'''
    l1 = policy_l1(u)
    return {
        "policy_sum": l1,
        "policy_budget_fraction": l1 / beta if beta else None,
        "policy_nnz": policy_nnz(u, threshold),
        "policy_max_entry": policy_max_entry(u),
        "policy_topk": policy_topk(u, pairs, topk),
        "policy_sum_by_class": policy_sum_by_source_class(u, pairs),
    }


# --------------------------------------------------------------------------- #
# Diagnostic A -- B2 evaluation + QP-gap + solver-convergence check
# --------------------------------------------------------------------------- #

def b2_value(G_obj, u, v, den):
    '''
    Evaluates B2(u) = ||G_obj @ u - v||^2 / den for an arbitrary policy `u` (torch or numpy),
    under the SAME G_obj/v/den already used for the co-descended B2 -- no new normalization
    convention introduced. Returns (b2: float, Gu: detached torch.Tensor) so callers can reuse
    Gu for the geometric-feasibility diagnostic (Diagnostic B) without recomputing it.
    '''
    u_t = torch.as_tensor(_as_np(u), dtype=G_obj.dtype, device=G_obj.device)
    Gu = (G_obj @ u_t).detach()
    sq_err = ((Gu - v.detach()) ** 2).sum()
    return (sq_err / den).item(), Gu


def qp_gap(b2_current, b2_qp, eps=_EPS):
    gap_abs = b2_current - b2_qp
    gap_rel = gap_abs / max(abs(b2_qp), eps)
    return gap_abs, gap_rel


def qp_convergence_sweep(Q, c, u_init, beta, pairs, pi, check_iters):
    '''
    Diagnostic A.2.1 -- re-solves `project_gradient_descent_local` at each iteration budget in
    `check_iters` (all warm-started from the SAME u_init, so differences reflect the solver's
    own convergence, not a different starting point) and returns {n_iters: w}. Callers compute
    B2 at each w (via `b2_value`) and compare, to gauge whether the usual `diag_qp_iters` budget
    is already close to the fully-converged optimum. Not run on normal (non-diag_qp_convergence)
    batches -- expensive at the largest check_iters value.
    '''
    return {
        n_iters: project_gradient_descent_local(Q, c, u_init, beta, pairs, pi, n_iters=n_iters)[0]
        for n_iters in check_iters
    }


def qp_convergence_relative_improvements(b2_by_iters, eps=_EPS):
    '''
    Given {n_iters: B2(w_{n_iters})} from a `qp_convergence_sweep` (all warm-started from the
    SAME point, per that function's docstring), returns the CONSECUTIVE relative improvements
    -- {(a, b): (B2_a - B2_b) / max(|B2_a|, eps)} for each pair of consecutive iteration budgets
    a < b in sorted order. Close to 0 for every consecutive pair means the sweep has plateaued
    (the usual `diag_qp_iters` budget is already close to converged); still shrinking at the
    largest pair means it has not.
    '''
    ordered = sorted(b2_by_iters)
    out = {}
    for a, b in zip(ordered, ordered[1:]):
        out[(a, b)] = (b2_by_iters[a] - b2_by_iters[b]) / max(abs(b2_by_iters[a]), eps)
    return out


# --------------------------------------------------------------------------- #
# Diagnostic B -- geometric feasibility of v(delta) under G_obj @ U_loc
# --------------------------------------------------------------------------- #

def geometric_feasibility(Gu, v, rho, prefix, eps=_EPS):
    v_d = v.detach()
    residual = (Gu - v_d).norm().item()
    v_norm = v_d.norm().item()
    cos = F.cosine_similarity(Gu.reshape(1, -1), v_d.reshape(1, -1), eps=eps).item()
    return {
        f"{prefix}_norm": Gu.norm().item(),
        f"{prefix}_residual": residual,
        f"{prefix}_residual_over_rho": residual / max(rho, eps),
        f"{prefix}_residual_over_v": residual / max(v_norm, eps),
        f"{prefix}_cosine": cos,
    }


def policy_distance(u, u_ref):
    '''||u - u_ref||_1 and ||u - u_ref||_2, e.g. the co-descended policy vs. the QP reference --
    to tell "u_current is far from u_qp" apart from "u_current is close to u_qp but B2 is still
    high" (the latter means u itself is not really the problem, see this module's docstring).'''
    diff = _as_np(u) - _as_np(u_ref)
    return float(np.abs(diff).sum()), float(np.linalg.norm(diff))


def constraint_activity(w, beta, pairs, pi, tol=1e-8):
    '''
    Which constraints of U_loc = {w>=0, sum(w)<=beta, sum_c w_{y,c}<=pi_y} are ACTIVE (binding,
    i.e. hit to within `tol`) at `w` -- tells whether a QP solution is limited by the global
    budget, one or more per-class caps, or is a genuine interior optimum (no active constraint,
    prop:structure's inner minimum reached without projection). `diag_constraint_tol` controls
    `tol`: too tight and float/PGD-iterate noise makes a truly-active constraint look inactive;
    too loose and it flags constraints that are merely close, not binding.

    Returns:
        dict with global_budget_active (bool), any_class_cap_active (bool),
        num_active_class_caps (int), active_class_caps (list of the source classes y whose
        sum_c w_{y,c} <= pi_y cap is active).
    '''
    w_np = _as_np(w)
    l1 = float(w_np.sum())
    global_active = bool(l1 >= beta - tol)

    ys = sorted(set(y for y, _ in pairs))
    active_classes = []
    for y in ys:
        idx = [i for i, (yy, _) in enumerate(pairs) if yy == y]
        class_sum = float(w_np[idx].sum())
        if class_sum >= pi[y] - tol:
            active_classes.append(int(y))

    return {
        "global_budget_active": global_active,
        "any_class_cap_active": len(active_classes) > 0,
        "num_active_class_caps": len(active_classes),
        "active_class_caps": active_classes,
    }


# --------------------------------------------------------------------------- #
# Diagnostic (Section 7 of the follow-up task) -- unconstrained projection of v onto
# span(G_obj): separates a subspace-coverage limitation (label flips' gradients simply don't
# span the poison direction) from a CONSTRAINT limitation (u>=0 / budget / per-class caps make
# an otherwise-reachable direction inaccessible).
# --------------------------------------------------------------------------- #

def span_projection(G_obj, Q_obj, c_np, v, den, rcond=1e-10, eps=_EPS):
    '''
    u_ls = argmin_u ||G_obj @ u - v||^2, UNCONSTRAINED (u need not be >=0, need not respect
    beta/pi -- this is a pure diagnostic, never usable as an attack policy). Solved via the
    Moore-Penrose pseudoinverse of Q_obj = G_obj^T @ G_obj (already cached alongside G_obj, so
    this reuses it rather than re-deriving a fresh (D, P) least-squares problem over the full
    parameter dimension D): u_ls = pinv(Q_obj) @ c_np, c_np = G_obj^T @ v. When Q_obj is
    singular/ill-conditioned (columns of G_obj not linearly independent), pinv's `rcond`
    truncates near-zero singular values -- u_ls may then not be unique, but G_obj @ u_ls (the
    projection of v onto span(G_obj)) IS unique regardless of which minimizer pinv picks, so the
    metrics below are well-defined even then.

    Returns a dict: B2_span (the objective's own value at this UNCONSTRAINED optimum -- a lower
    bound on B2_qp, since U_loc subset R^P), span_residual = ||G_obj@u_ls - v||,
    span_relative_residual (normalized by ||v||), span_projection_cosine (cosine between the
    projection and v -- 1.0 iff v in span(G_obj) exactly).
    '''
    Q_pinv = np.linalg.pinv(Q_obj, rcond=rcond)
    u_ls = Q_pinv @ np.asarray(c_np, dtype=np.float64)
    b2_span, Gu_span = b2_value(G_obj, u_ls, v, den)
    v_d = v.detach()
    residual = (Gu_span - v_d).norm().item()
    v_norm = v_d.norm().item()
    cos = F.cosine_similarity(Gu_span.reshape(1, -1), v_d.reshape(1, -1), eps=eps).item()
    return {
        "B2_span": b2_span,
        "span_residual": residual,
        "span_relative_residual": residual / max(v_norm, eps) if v_norm > eps else None,
        "span_projection_cosine": cos if v_norm > eps and Gu_span.norm().item() > eps else None,
    }


# --------------------------------------------------------------------------- #
# Diagnostic (Section 6 of the follow-up task) -- direction vs. amplitude decomposition:
# rescaling Gu by its OWN optimal scalar a* = <Gu,v>/||Gu||^2 isolates whether the residual is
# mostly a DIRECTION mismatch (residual stays high even after the best possible rescaling) or an
# AMPLITUDE-only mismatch (residual collapses once rescaled -- the direction was already right).
# --------------------------------------------------------------------------- #

def direction_amplitude_scaling(Gu, v, eps=_EPS):
    Gu_d, v_d = Gu.detach(), v.detach()
    denom = (Gu_d ** 2).sum().item()
    v_norm = v_d.norm().item()
    if denom <= eps:
        return {"optimal_scale": None, "residual_after_optimal_scaling": None,
                "residual_after_optimal_scaling_relative": None}
    a_star = float((Gu_d * v_d).sum().item() / denom)
    residual_scaled = (a_star * Gu_d - v_d).norm().item()
    return {
        "optimal_scale": a_star,
        "residual_after_optimal_scaling": residual_scaled,
        "residual_after_optimal_scaling_relative": (
            residual_scaled / max(v_norm, eps) if v_norm > eps else None
        ),
    }


# --------------------------------------------------------------------------- #
# Diagnostic C -- discretization gap (materialize_policy_flips' EXACT counting rule, no
# resampling of actual example indices needed for this diagnostic).
# --------------------------------------------------------------------------- #

def discretize_policy(u, pairs, gamma, n_train, class_counts):
    '''
    Discretizes a continuous LOCAL policy `u` using the SAME rounding + per-source-class
    sequential-clipping rule federated_policy_to_flips.materialize_policy_flips uses downstream
    (via the shared `compute_flip_counts` -- one convention, not two, per the task's Section 4).

    class_counts: dict y -> number of class-y examples available in the training set. This
    diagnostic approximates it from `pi` (round(pi[y] * n_train)) rather than re-scanning the
    dataset's label array -- pi is already computed once per run for the (P^mean) objective
    itself, and federated_policy_to_flips draws from that SAME dataset, so this is consistent
    with (if not bit-identical to) what it would see. An exact match would require passing the
    real per-class label counts in; flagged here rather than silently assumed identical.

    Returns:
        u_discrete: (P,) float64 array, the realized fraction u_yc actually materializes to
            (n_realized / (gamma * n_train)) -- directly comparable to the continuous `u`.
        n_realized: (P,) int64 array, the realized flip COUNT per pair (post-clipping).
    '''
    u_np = _as_np(u)
    n_realized = compute_flip_counts(u_np, pairs, gamma, n_train, class_counts)
    denom = gamma * n_train
    u_discrete = n_realized.astype(np.float64) / denom
    return u_discrete, n_realized


def discretization_gap(b2_continuous, b2_discrete, eps=_EPS):
    gap_abs = b2_discrete - b2_continuous
    gap_rel = gap_abs / max(abs(b2_continuous), eps)
    return gap_abs, gap_rel


# --------------------------------------------------------------------------- #
# Diagnostic D -- actual vs. predicted gradient shift (expensive, opt-in)
# --------------------------------------------------------------------------- #

def compute_actual_gradient_shift(
    model, loss_fn, class_samples_raw, pairs, pi, gamma, u, dataset_flag, model_flag,
):
    '''
    Diagnostic D -- approximates the gradient shift a REAL materialization of `u` induces, on
    the SAME class-conditional sample (`class_samples_raw`) and the SAME (pi, gamma)
    normalization `compute_expected_flip_gradients`/G_obj are built from.

    Missing information, made explicit rather than silently approximated: this module has no
    access to the real per-worker shard federated_policy_to_flips/partition_across_workers
    would build downstream (that needs the full training set, a specific worker partition, and
    federated_train_user's own preprocessing) -- reproducing that exactly is out of this
    module's scope (it would mean re-implementing most of that pipeline here). The minimal
    proxy used instead: for each class y present in class_samples_raw, flip a
    u_yc*gamma/pi_y fraction of its n_per_class samples to each target c (same disjoint-
    cursor-per-y convention `compute_flip_counts` uses, applied to this smaller sample -- the
    fraction is exactly what a population-level u_yc*gamma*n_train flip count out of pi_y*n_train
    class-y examples implies), then measures the REAL model gradient shift between the clean-
    label and partially-flipped-label batches for that class, weighted by pi_y and summed. This
    equals gamma*H@u in expectation over the sample; the point of this diagnostic is to check
    that equality against an ACTUALLY measured gradient, not to assume it.

    Returns:
        actual_shift: (D,) tensor, already in G_obj's own aggregate (gamma-scaled) convention --
            directly comparable to G_obj @ u_realized, no further rescaling needed. None if no
            class in `pairs` has samples in `class_samples_raw`.
    '''
    u_np = _as_np(u)
    pairs_by_y = {}
    for i, (y, c) in enumerate(pairs):
        pairs_by_y.setdefault(y, []).append((i, c))

    shift_sum = None
    for y, plist in pairs_by_y.items():
        if y not in class_samples_raw:
            continue
        x_raw = class_samples_raw[y]
        n_y = x_raw.shape[0]
        if n_y == 0:
            continue

        labels_flipped = torch.full((n_y,), y, dtype=torch.long, device=x_raw.device)
        cursor = 0
        for i, c in plist:
            frac = u_np[i] * gamma / max(pi[y], 1e-12)
            n_flip = int(round(frac * n_y))
            n_flip = max(0, min(n_flip, n_y - cursor))
            if n_flip > 0:
                labels_flipped[cursor:cursor + n_flip] = c
            cursor += n_flip

        x = raw_to_preprocess(x_raw, dataset_flag=dataset_flag, model_flag=model_flag)
        labels_clean = torch.full((n_y,), y, dtype=torch.long, device=x_raw.device)

        grads_clean, _ = compute_batch_gradients(
            model, loss_fn, (x, labels_clean), create_graph=False,
        )
        g_clean = torch.cat([g.reshape(-1) for g in grads_clean]).detach()
        grads_flipped, _ = compute_batch_gradients(
            model, loss_fn, (x, labels_flipped), create_graph=False,
        )
        g_flipped = torch.cat([g.reshape(-1) for g in grads_flipped]).detach()

        contrib = pi[y] * (g_flipped - g_clean)
        shift_sum = contrib if shift_sum is None else shift_sum + contrib

    return shift_sum


def actual_vs_predicted(actual_shift, predicted_shift, eps=_EPS):
    if actual_shift is None or predicted_shift is None:
        return {
            "actual_shift_norm": None, "predicted_shift_norm": None,
            "actual_predicted_cosine": None, "actual_predicted_residual": None,
            "actual_predicted_relative_error": None,
        }
    predicted_shift = predicted_shift.detach().to(actual_shift.device, actual_shift.dtype)
    residual = (actual_shift - predicted_shift).norm().item()
    pred_norm = predicted_shift.norm().item()
    return {
        "actual_shift_norm": actual_shift.norm().item(),
        "predicted_shift_norm": pred_norm,
        "actual_predicted_cosine": F.cosine_similarity(
            actual_shift.reshape(1, -1), predicted_shift.reshape(1, -1), eps=eps,
        ).item(),
        "actual_predicted_residual": residual,
        "actual_predicted_relative_error": residual / max(pred_norm, eps),
    }


# --------------------------------------------------------------------------- #
# Diagnostic G -- delta-gradient balance between B2 and lambda_bd*L_bd
# --------------------------------------------------------------------------- #

def gradient_balance(B2_k, L_bd_k, lambda_bd, delta):
    '''
    Measures ||grad_delta B2_k|| and ||grad_delta (lambda_bd*L_bd_k)|| SEPARATELY, via
    `torch.autograd.grad(..., retain_graph=True)` (never `.backward()`) -- this does not
    accumulate into delta.grad and does not consume the graph, so it must be called BEFORE the
    real `step_loss.backward()` that drives the actual optimizer step (which retains its own
    right to the graph via `retain_graph=True` internally, or -- for checkpoint_backward=True --
    is the LAST consumer of it). Returns (None, None) if delta has no graph connection to
    either term (e.g. no poisoned examples in this batch, has_poison=False upstream).
    '''
    if not delta.requires_grad:
        return None, None
    g_B2 = torch.autograd.grad(B2_k, delta, retain_graph=True, allow_unused=True)[0]
    g_BD = torch.autograd.grad(lambda_bd * L_bd_k, delta, retain_graph=True, allow_unused=True)[0]
    norm_B2 = g_B2.norm().item() if g_B2 is not None else 0.0
    norm_BD = g_BD.norm().item() if g_BD is not None else 0.0
    return norm_B2, norm_BD


# --------------------------------------------------------------------------- #
# Section 9 -- one-line-per-event JSONL writer
# --------------------------------------------------------------------------- #

class DiagnosticsWriter:
    '''
    Appends one JSON object per diagnostic event to `path` (jsonl -- one line per event, easy
    to stream-parse on a cluster without loading one giant nested object). A no-op (never opens
    a file, `write` is a cheap early-return) when `path` is None -- the default, so existing
    runs that never set `diag_path` are completely unaffected.
    '''

    def __init__(self, path):
        self.path = path
        self._fh = None
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(path, "a")

    def write(self, record):
        if self._fh is None:
            return

        def _clean(v):
            if isinstance(v, float) and not np.isfinite(v):
                return None
            if isinstance(v, dict):
                return {k: _clean(vv) for k, vv in v.items()}
            if isinstance(v, list):
                return [_clean(vv) for vv in v]
            return v

        self._fh.write(json.dumps({k: _clean(v) for k, v in record.items()}) + "\n")
        self._fh.flush()

    def close(self):
        if self._fh is not None:
            self._fh.close()
            self._fh = None
