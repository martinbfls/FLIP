"""
prelim/tests/test_policy_module_fixes.py -- non-regression checks on SYNTHETIC tensors only
for the federated_optimizing_trigger_policy / federated_policy_to_flips consistency fixes
(corrections A, B3a/B3b/B3c). No dataset, no model, no training -- runs in well under a
minute, matching prelim/tests/test_units.py's convention (see prelim/SPEC.md section 1).

Run:  python prelim/tests/test_policy_module_fixes.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.federated_optimizing_trigger_policy.utils import (
    project_policy_budget,
    _project_nonneg_capped_sum,
)
from modules.federated_policy_to_flips.utils import materialize_policy_flips
from modules.federated_optimizing_trigger.utils import resolve_beta_and_lambda_poison

_results = []


def check(name, ok, detail=""):
    _results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


# --------------------------------------------------------------------------#
# _project_nonneg_capped_sum: Duchi projection, incl. negative-component regression check
# (§B: "le tri porte sur u ... vérifie ce point sur un cas test avec des composantes
# négatives" -- an independent bisection reference found NO discrepancy; this pins that
# result down as a regression test rather than leaving it as a one-off manual check).
# --------------------------------------------------------------------------#
def _reference_projection(u_np, cap, iters=200):
    lo, hi = u_np.min() - cap - 1, u_np.max()
    for _ in range(iters):
        mid = (lo + hi) / 2
        val = np.clip(u_np - mid, 0, None).sum()
        if val > cap:
            lo = mid
        else:
            hi = mid
    return np.clip(u_np - hi, 0, None)


def test_duchi_negative_components():
    rng = np.random.RandomState(0)
    ok = True
    worst = 0.0
    for trial in range(20):
        u_np = rng.uniform(-2, 2, size=8)
        cap = 0.5
        out = _project_nonneg_capped_sum(torch.tensor(u_np, dtype=torch.float64), cap).numpy()
        ref = _reference_projection(u_np, cap)
        gap = float(np.abs(out - ref).max())
        worst = max(worst, gap)
        ok = ok and gap < 1e-8
    check("_project_nonneg_capped_sum matches bisection reference (negative components)", ok,
          f"worst max|diff| over 20 trials = {worst:.2e}")


# --------------------------------------------------------------------------#
# project_policy_budget: feasibility + block caps + no-worse-than-random-feasible-point
# --------------------------------------------------------------------------#
C = 4
PAIRS = [(y, z) for y in range(C) for z in range(C) if z != y]
PI = {0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1}
BETA = 0.5


def _is_feasible(x_np, beta, pairs, pi, tol=1e-6):
    if (x_np < -tol).any():
        return False
    if x_np.sum() > beta + tol:
        return False
    for y in pi:
        idx = [p for p, (yy, _) in enumerate(pairs) if yy == y]
        if x_np[idx].sum() > pi[y] + tol:
            return False
    return True


def _random_feasible_point(rng, beta, pairs, pi):
    """A random feasible point via rejection-free construction: random per-block masses
    scaled down to respect both the block caps and the global budget."""
    ys = sorted(pi.keys())
    x = np.zeros(len(pairs))
    for y in ys:
        idx = [p for p, (yy, _) in enumerate(pairs) if yy == y]
        raw = rng.uniform(0, 1, size=len(idx))
        raw = raw / raw.sum() * pi[y] * rng.uniform(0, 1)
        x[idx] = raw
    total = x.sum()
    if total > beta:
        x = x * (beta / total)
    return x


def test_project_policy_budget_feasible_and_near_optimal():
    rng = np.random.RandomState(1)
    all_feasible = True
    all_at_least_as_good = True
    worst_margin = 0.0

    for trial in range(15):
        u_np = rng.uniform(-1, 1, size=len(PAIRS))
        u = torch.tensor(u_np, dtype=torch.float64)
        out = project_policy_budget(u, BETA, PAIRS, PI).numpy()

        feasible = _is_feasible(out, BETA, PAIRS, PI)
        all_feasible = all_feasible and feasible

        dist_ours = float(((out - u_np) ** 2).sum())
        for _ in range(20):
            rand_feasible = _random_feasible_point(rng, BETA, PAIRS, PI)
            dist_rand = float(((rand_feasible - u_np) ** 2).sum())
            margin = dist_rand - dist_ours
            worst_margin = min(worst_margin, margin)
            if margin < -1e-6:
                all_at_least_as_good = False

    check("project_policy_budget output is feasible (u>=0, sum<=beta, per-class<=pi_y)",
          all_feasible)
    check("project_policy_budget is <= any random feasible point's distance to u",
          all_at_least_as_good, f"worst (rand_dist - our_dist) = {worst_margin:.3e}")


def test_project_policy_budget_block_cap_binds_before_global():
    """All budget concentrated on class 0's pairs (pi[0]=0.4 < beta=0.5): the per-class cap,
    not the global budget, must be the active constraint -- sum(u) should land at pi[0], not
    beta, and mass should spill into other classes' pairs once class 0 saturates."""
    u0 = torch.zeros(len(PAIRS), dtype=torch.float64)
    idx0 = [p for p, (y, _) in enumerate(PAIRS) if y == 0]
    u0[idx0[0]] = 10.0  # all mass on a single (0, c) pair, far exceeding pi[0]=0.4

    out = project_policy_budget(u0, BETA, PAIRS, PI).numpy()
    class0_sum = out[idx0].sum()

    check("per-class cap binds (class-0 mass == pi[0], not beta)",
          abs(class0_sum - PI[0]) < 1e-6, f"class0_sum={class0_sum:.6f}, pi[0]={PI[0]}")
    check("global budget still respected", out.sum() <= BETA + 1e-6,
          f"sum(u)={out.sum():.6f} <= beta={BETA}")


# --------------------------------------------------------------------------#
# materialize_policy_flips: gamma scaling (B3c)
# --------------------------------------------------------------------------#
def test_materialize_policy_flips_uses_gamma():
    n_classes = 4
    n_train = 4000
    rng = np.random.RandomState(0)
    labels = rng.randint(0, n_classes, size=n_train)

    pairs = [(0, 1)]
    u = np.array([0.05])  # local rate: 5% of ONE corrupted worker's own class-0 examples
    gamma = 0.3

    idx_flipped, targets = materialize_policy_flips(
        u, pairs, n_train, labels, n_classes, gamma, seed=0,
    )
    expected = round(0.05 * gamma * n_train)
    check("materialize_policy_flips flip count == round(u * gamma * n_train)",
          len(idx_flipped) == expected,
          f"got {len(idx_flipped)}, expected {expected} (gamma={gamma})")

    # Old (pre-fix) convention would have produced round(u * n_train) = 1/gamma times more.
    old_convention_count = round(0.05 * n_train)
    check("fix removes the 1/gamma over-production vs. the old (no-gamma) convention",
          abs(len(idx_flipped) - old_convention_count) > 0.5 * old_convention_count,
          f"new={len(idx_flipped)}, old-convention={old_convention_count}, "
          f"ratio={old_convention_count / max(len(idx_flipped), 1):.2f} (expected ~= 1/gamma={1/gamma:.2f})")


# --------------------------------------------------------------------------#
# P0 regression (2026-08-26 cluster AssertionError): the stale gen_configs.py sweep-axis
# defaults (num_poisoned=1, num_honests=0, gamma=1) were mistaken for a metadata-writing bug
# in federated_optimizing_trigger_policy/federated_policy_to_flips. Both modules already read
# num_honests/num_poisoned correctly from config -- the fix was in gen_configs.py's own
# defaults (now num_poisoned=3, num_honests=7, matching orchestrate_runs_policy_slurm.sh and
# the real federated deployment). This pins down that federated_policy_to_flips's own
# consistency assertion (run_module.py's "Materialized flip count does not match beta's
# flip_budget") does NOT fire under the real (3, 7) config for a saturated policy.
# --------------------------------------------------------------------------#
def test_policy_to_flips_assertion_passes_under_real_federated_config():
    n_classes = 10
    n_train = 50000
    num_poisoned, num_honests = 3, 7
    gamma = num_poisoned / (num_poisoned + num_honests)
    check("gamma from real federated config (num_poisoned=3, num_honests=7) == 0.3",
          abs(gamma - 0.3) < 1e-9, f"gamma={gamma}")

    # beta_local = 0.1 matches experiments/.../3vs7/budget1500 (beta_global = gamma*beta_local
    # = 0.03, the target global budget of 1500/50000).
    beta_local = 0.1
    pairs = [(9, 4)]
    u = np.array([beta_local])  # fully saturated: sum(u) == beta_local

    rng = np.random.RandomState(0)
    labels = rng.randint(0, n_classes, size=n_train)

    idx_flipped, _targets = materialize_policy_flips(
        u, pairs, n_train, labels, n_classes, gamma, seed=0,
    )
    budget = int(len(np.unique(idx_flipped)))

    # Exact replica of federated_policy_to_flips/run_module.py's consistency check.
    _, flip_budget_expected, _ = resolve_beta_and_lambda_poison(
        beta=beta_local, flip_budget=None, lambda_poison="beta",
        num_poisoned=num_poisoned, num_honests=num_honests, n_train=n_train,
    )
    tol = max(len(pairs), round(0.02 * flip_budget_expected))

    check("materialized budget matches beta's flip_budget under the real (3,7) config "
          "(no AssertionError)",
          abs(budget - flip_budget_expected) <= tol,
          f"budget={budget}, flip_budget_expected={flip_budget_expected}, tol={tol}")

    # Same scenario under the STALE gen_configs.py defaults (1, 0): this reproduces the
    # exact cluster trace (flip_budget=1500, gamma=1.0) to document what was actually wrong.
    stale_num_poisoned, stale_num_honests = 1, 0
    stale_gamma = stale_num_poisoned / (stale_num_poisoned + stale_num_honests)
    _, stale_flip_budget, _ = resolve_beta_and_lambda_poison(
        beta=0.03, flip_budget=None, lambda_poison="beta",
        num_poisoned=stale_num_poisoned, num_honests=stale_num_honests, n_train=n_train,
    )
    check("stale (1,0) defaults reproduce the cluster trace's flip_budget=1500, gamma=1.0",
          stale_gamma == 1.0 and stale_flip_budget == 1500,
          f"stale_gamma={stale_gamma}, stale_flip_budget={stale_flip_budget}")


# --------------------------------------------------------------------------#
# P3 (checkpoint-pool stability, policy module -- arbitrated option A): sample_checkpoints
# must be redrawn INSIDE the batch loop (once per batch), not once for the whole
# optimize_trigger_policy_step call, so B2/B2_qp are not conditioned on the same fixed
# num_chckpt checkpoints for the entire call. flip_grad_cache (initialized OUTSIDE the batch
# loop, so it persists across batches within one call) needed no structural change -- its
# existing lazy per-k lookup already serves as the pool.
# --------------------------------------------------------------------------#
def test_policy_sampled_k_redrawn_per_batch():
    repo_root = Path(__file__).resolve().parents[2]
    src = (repo_root / "modules/federated_optimizing_trigger_policy/run_module.py").read_text()

    cache_idx = src.index("flip_grad_cache = {}")
    loop_idx = src.index("for batch_idx, batch in enumerate(pbar):")
    check("flip_grad_cache initialized BEFORE the batch loop (persists across batches)",
          cache_idx < loop_idx, f"cache_idx={cache_idx}, loop_idx={loop_idx}")

    sample_call_idx = src.index("sampled_k = sample_checkpoints(", loop_idx)
    check("sample_checkpoints is called INSIDE the batch loop (redrawn per batch, not once "
          "per call)",
          sample_call_idx > loop_idx, f"sample_call_idx={sample_call_idx}, loop_idx={loop_idx}")


if __name__ == "__main__":
    test_duchi_negative_components()
    test_project_policy_budget_feasible_and_near_optimal()
    test_project_policy_budget_block_cap_binds_before_global()
    test_materialize_policy_flips_uses_gamma()
    test_policy_to_flips_assertion_passes_under_real_federated_config()
    test_policy_sampled_k_redrawn_per_batch()

    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} checks passed.")
    sys.exit(1 if n_fail else 0)
