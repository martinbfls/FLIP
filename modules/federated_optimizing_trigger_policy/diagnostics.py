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
