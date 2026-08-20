"""
Support library for prelim/prelim.ipynb -- preliminary experiments E1-E3
validating the label-flipping bias-map threat model (see prelim/prelim.ipynb
for the experiment protocols and this session's write-up).

Reuse policy: everything that already exists in modules/federated_optimizing_trigger
and modules/base_utils is imported and used as-is, never copied or modified.
New code here is limited to what the existing pipeline genuinely does not
provide:
  - the per-class-capacity variant of the budget QP (the existing
    project_gradient only implements a single global L1 budget);
  - hard-label realization of flip masses (flip_masses_to_labels);
  - a plain logistic-regression model (no linear model exists in the repo)
    and a thin, device-portable constructor for "r32p" that avoids
    modules.base_utils.util.load_model's hard-coded `.cuda()`;
  - diagnostics that don't exist elsewhere: minibatch SNR, reachable-radius
    bounds, the greedy support function, and the QP-based rank ratio.
"""
import os
import sys
import math

import numpy as np
import scipy.sparse as sp
import osqp
import torch
import torch.nn as nn
from torchvision import transforms as _tvt

# --- wire up imports from the existing FLIP pipeline (modules/ package) ----
_FLIP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _FLIP_ROOT not in sys.path:
    sys.path.insert(0, _FLIP_ROOT)

from dataclasses import dataclass

from modules.base_utils.datasets import shard_dataset_indices, StripePoisoner
from modules.base_utils.model.resnet import resnet32
from modules.base_utils.model.model import SequentialImageNetworkMod
from modules.base_utils.util import (
    mini_train_multi, clf_loss, clf_eval, DEFAULT_SGD_KWARGS, DEFAULT_SGD_SCHED_KWARGS,
)
from modules.federated_optimizing_trigger.utils import (
    compute_class_frequencies,
    get_class_conditional_samples,
    compute_expected_flip_gradients,
    get_raw_clean_dataset,
    get_clean_dataset,
    raw_to_preprocess,
    project_gradient as _project_gradient_reused,
)


# --------------------------------------------------------------------------#
# Config
# --------------------------------------------------------------------------#

@dataclass
class PrelimConfig:
    n_b: int
    n_p: int
    f: int
    beta: float
    lam: float
    y_source: int
    y_target: int
    seed: int
    device: str
    dataset_flag: str = "cifar"
    n_classes: int = 10

    @property
    def gamma(self) -> float:
        return self.n_p / self.n_b


def make_config(n_b=10, n_p=3, f=3, beta=0.1, lam=None, y_source=9, y_target=4,
                 seed=0, device="cpu", dataset_flag="cifar", n_classes=10) -> PrelimConfig:
    """lam defaults to beta, matching this session's convention (see prompt: lam = beta)."""
    if lam is None:
        lam = beta
    return PrelimConfig(n_b=n_b, n_p=n_p, f=f, beta=beta, lam=lam,
                         y_source=y_source, y_target=y_target, seed=seed,
                         device=device, dataset_flag=dataset_flag, n_classes=n_classes)


# --------------------------------------------------------------------------#
# Models
# --------------------------------------------------------------------------#

class LogisticRegressionFlat(nn.Module):
    """
    Plain multinomial logistic regression on flattened pixels. No linear
    model exists anywhere in this repo (grep confirmed) and neither does
    MNIST/Fashion-MNIST as a dataset -- per the session's approved fallback,
    this is a bare nn.Linear over flattened CIFAR-10 (3*32*32 = 3072 -> C).
    """
    def __init__(self, n_classes=10, in_shape=(3, 32, 32)):
        super().__init__()
        d = 1
        for s in in_shape:
            d *= s
        self.linear = nn.Linear(d, n_classes)

    def forward(self, x):
        return self.linear(x.flatten(1))


def build_model(model_flag, n_classes, device):
    """
    model_flag: "linear" (LogisticRegressionFlat, new -- no repo equivalent)
    or "r32p" (the repo's smallest CNN). "r32p" is built directly from
    modules.base_utils.model.resnet.resnet32 + SequentialImageNetworkMod
    (both imported unmodified) instead of via modules.base_utils.util.load_model,
    because load_model hard-codes `.cuda()` on every branch (util.py, l.73-133)
    and would crash on a CUDA-less machine; only the device placement is
    handled here, the architecture itself is untouched pipeline code.
    """
    if model_flag == "linear":
        model = LogisticRegressionFlat(n_classes=n_classes)
    elif model_flag == "r32p":
        model = SequentialImageNetworkMod(resnet32(n_classes))
    else:
        raise NotImplementedError(model_flag)
    return model.to(device)


# --------------------------------------------------------------------------#
# Sharding
# --------------------------------------------------------------------------#

def shard_indices(dataset, n_b, seed):
    """IID partition into n_b equal-ish shards, reusing shard_dataset_indices as-is."""
    return shard_dataset_indices(len(dataset), n_b, seed=seed, iid=True)


# --------------------------------------------------------------------------#
# Class-conditional shifts (Gbar, grad_c, pi)
# --------------------------------------------------------------------------#

def class_conditional_shifts(model, class_samples_raw, dataset_flag, n_classes, device,
                              loss_fn=None, model_flag=None):
    """
    (Gbar, grad_c, pi, pairs, col_index, Q) at the current checkpoint's weights.

    Deviates from the originally sketched signature `(model, calib_loader, C,
    device)`: takes `class_samples_raw` (a dict y -> raw [0,1] image tensor,
    as produced by get_class_conditional_samples) rather than a DataLoader,
    because that's exactly what the reused compute_expected_flip_gradients
    needs -- building a DataLoader-based sampler on top would just re-derive
    the same dict. dataset_flag/model_flag are threaded through because
    raw_to_preprocess (also reused) needs them to normalize consistently with
    the rest of the pipeline.

    Gbar: (d, C(C-1)) float32 CPU tensor. Column (y,z) = pi[y]*(g[y][z]-g[y][y]),
    exactly modules.federated_optimizing_trigger.utils.compute_expected_flip_gradients's
    own convention (verified against this session's spec: same sign, same
    pi[y] weighting). pairs: ordered list of (y,z) matching Gbar's columns.
    col_index: dict (y,z) -> column index, the explicit mapping the spec asks
    for (compute_expected_flip_gradients returns `pairs`, an equivalent
    ordered list, but not the inverse dict).

    grad_c = sum_y pi[y]*g[y][y] is NOT returned by the reused function (only
    the y!=z differences survive inside it), so it's computed here with one
    extra backward pass per class over the same class_samples_raw/loss_fn,
    using the same raw_to_preprocess preprocessing for consistency.
    """
    loss_fn = loss_fn if loss_fn is not None else clf_loss
    pi = compute_class_frequencies(dataset_flag, n_classes)

    G, Q, pairs = compute_expected_flip_gradients(
        model, loss_fn, class_samples_raw, n_classes, pi, dataset_flag, model_flag=model_flag
    )
    col_index = {p: i for i, p in enumerate(pairs)}
    Gbar = G.detach().to(torch.float32).cpu()

    params = list(model.parameters())
    grad_c = torch.zeros(Gbar.shape[0], dtype=torch.float32)
    for y in range(n_classes):
        if y not in class_samples_raw:
            continue
        x = raw_to_preprocess(class_samples_raw[y], dataset_flag=dataset_flag, model_flag=model_flag)
        y_lab = torch.full((x.shape[0],), y, dtype=torch.long, device=x.device)
        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y_lab)
        grad = torch.autograd.grad(loss, params)
        flat = torch.cat([g.reshape(-1) for g in grad]).detach().cpu().to(torch.float32)
        grad_c += pi[y] * flat
    model.zero_grad(set_to_none=True)

    return Gbar, grad_c, pi, pairs, col_index, Q


# --------------------------------------------------------------------------#
# Flip-mass realization
# --------------------------------------------------------------------------#

def flip_masses_to_labels(shard_targets, u, pairs, rng):
    """
    Hard realization of flip masses u on one shard.

    shard_targets: (n,) array-like of TRUE labels for the shard.
    u: dict (y,z) -> mass, mass = proportion OF THE SHARD (n = len(shard_targets))
       of class-y examples relabeled to z -- same normalization as U_loc/U_beta.
    pairs: fixed iteration order for (y,z) (e.g. from class_conditional_shifts),
       so that when several z's draw from the same y, draws are disjoint
       regardless of dict insertion order.

    For each (y,z) in `pairs` order: n_flip = round(u.get((y,z), 0) * n)
    labels are drawn, without replacement and without overlap across
    different z for the same y, uniformly among class-y examples not yet
    flipped. Returns (poisoned_targets, u_realized) where u_realized are the
    ACTUAL masses achieved (n_flip / n) -- these differ from the requested u
    by the rounding, which is exactly the gap E1 asks to measure.
    """
    shard_targets = np.asarray(shard_targets)
    n = len(shard_targets)
    poisoned = shard_targets.copy()

    by_class = {}
    for y in np.unique(shard_targets):
        idx = np.where(shard_targets == y)[0]
        rng.shuffle(idx)
        by_class[y] = idx
    ptr = {y: 0 for y in by_class}

    u_realized = {}
    for (y, z) in pairs:
        mass = float(u.get((y, z), 0.0))
        if y not in by_class:
            u_realized[(y, z)] = 0.0
            continue
        n_flip = int(round(mass * n))
        avail = by_class[y][ptr[y]:]
        n_flip = min(n_flip, len(avail))
        chosen = avail[:n_flip]
        poisoned[chosen] = z
        ptr[y] += n_flip
        u_realized[(y, z)] = n_flip / n

    return poisoned, u_realized


# --------------------------------------------------------------------------#
# Empirical gradients
# --------------------------------------------------------------------------#

def worker_gradient(model, loss_fn, raw_dataset, indices, targets,
                     dataset_flag, model_flag, batch_size, device):
    """
    Exact mean flattened gradient over (indices, targets) (targets OVERRIDE
    raw_dataset's own labels -- this is how poisoning is applied: raw_dataset
    only supplies the [0,1] image). batch_size only controls memory chunking:
    per-batch losses are summed (not meaned) so that accumulating across
    chunks and dividing once by n reproduces the exact full-shard average
    regardless of batch_size -- this is the "gradient empirique moyen sur
    tout le shard" of E1 step 3. For the batch-size SWEEP (E1 step 4), call
    minibatch_gradient_samples instead, which returns single-minibatch
    estimates rather than the full-shard average.
    """
    indices = np.asarray(indices)
    targets = np.asarray(targets)
    n = len(indices)
    params = list(model.parameters())
    grad_sum = [torch.zeros_like(p) for p in params]

    for start in range(0, n, batch_size):
        idx_batch = indices[start:start + batch_size]
        y_batch = targets[start:start + batch_size]
        xs = torch.stack([raw_dataset[int(i)][0] for i in idx_batch]).to(device)
        x = raw_to_preprocess(xs, dataset_flag=dataset_flag, model_flag=model_flag)
        y = torch.as_tensor(y_batch, dtype=torch.long, device=device)

        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y) * len(idx_batch)
        grads = torch.autograd.grad(loss, params)
        for gs, g in zip(grad_sum, grads):
            gs += g.detach()

    flat = torch.cat([g.reshape(-1) for g in grad_sum]).cpu().to(torch.float32) / n
    model.zero_grad(set_to_none=True)
    return flat


def minibatch_gradient_samples(model, loss_fn, raw_dataset, indices, targets,
                                batch_size, n_draws, dataset_flag, model_flag,
                                device, seed=0):
    """
    n_draws independent gradients, each computed from ONE random minibatch of
    size batch_size drawn without replacement from (indices, targets)
    (batches are resampled with replacement ACROSS draws). Not in the
    original function list -- added per this session's correction B to
    support the SNR diagnostic and the batch-size sweep, which both need
    single-minibatch estimates rather than worker_gradient's full-shard
    average.
    Returns (n_draws, d) float32 tensor.
    """
    rng = np.random.RandomState(seed)
    n = len(indices)
    b = min(batch_size, n)
    indices = np.asarray(indices)
    targets = np.asarray(targets)
    samples = []
    for _ in range(n_draws):
        sel = rng.choice(n, size=b, replace=False)
        g = worker_gradient(model, loss_fn, raw_dataset, indices[sel], targets[sel],
                             dataset_flag, model_flag, batch_size=b, device=device)
        samples.append(g)
    return torch.stack(samples, dim=0)


def snr(Gbar_u, grad_samples):
    """
    SNR = ||Gbar @ u|| / (sigma_i / sqrt(|B|)).

    grad_samples: (n_draws, d) tensor from minibatch_gradient_samples at the
    target batch size |B| -- each row is ALREADY a |B|-averaged gradient, so
    by the CLT the empirical std of these rows across draws IS sigma_i /
    sqrt(|B|) directly (no separate division by sqrt(|B|) needed: that
    scaling is already baked into how the samples were generated).
    Noise magnitude is aggregated across coordinates as the L2 norm of the
    per-coordinate std vector, matching the geometry of ||Gbar_u||:
    E[||noise||^2] = sum_d Var(noise_d) = ||std_vec||_2^2.
    """
    signal = float(Gbar_u.norm())
    noise = float(grad_samples.std(dim=0, unbiased=True).norm())
    return signal / noise if noise > 0 else float("inf")


# --------------------------------------------------------------------------#
# QP: budget projection, with/without per-class capacity, aggregate/local scope
# --------------------------------------------------------------------------#

def _build_budget_constraint(pairs, budget, per_class_cap, capacity):
    """
    l <= A w <= u for: w >= 0, sum(w) <= budget, and (if capacity) for every
    class y, sum_{z} w[(y,z)] <= per_class_cap[y]. With capacity=False this
    is identical to modules.federated_optimizing_trigger.utils's own
    _build_global_budget_constraint (not imported directly since it's a
    private helper of that module, but the constraint it encodes is
    reproduced verbatim here for the capacity=False path via solve_qp's
    delegation to the reused project_gradient -- see solve_qp below).
    """
    P = len(pairs)
    rows = [sp.identity(P, format="csc")]
    l_parts = [np.zeros(P)]
    u_parts = [np.full(P, np.inf)]

    ones_row = sp.csc_matrix(np.ones((1, P)))
    rows.append(ones_row)
    l_parts.append(np.array([-np.inf]))
    u_parts.append(np.array([budget]))

    if capacity:
        classes = sorted(set(y for y, _ in pairs))
        for y in classes:
            row = np.array([[1.0 if yy == y else 0.0 for yy, _ in pairs]])
            rows.append(sp.csc_matrix(row))
            l_parts.append(np.array([-np.inf]))
            u_parts.append(np.array([per_class_cap[y]]))

    A = sp.vstack(rows, format="csc")
    l = np.concatenate(l_parts)
    u = np.concatenate(u_parts)
    return A, l, u


def solve_qp(Q, c, beta, pi, gamma, pairs, scope="aggregate", capacity=False, ridge=1e-6):
    """
    w* = argmin_w 0.5 w^T Q w - c^T w  s.t. w in (U_beta if scope="aggregate"
    else U_loc), per this session's exact definitions:

      U_beta = { w>=0 : sum(w) <= beta,       sum_z w[y,z] <= gamma*pi[y] }
      U_loc  = { w>=0 : sum(w) <= beta/gamma, sum_z w[y,z] <= pi[y]       }

    (per-class rows only added if capacity=True).

    For scope="aggregate", capacity=False this is EXACTLY
    modules.federated_optimizing_trigger.utils.project_gradient(Q, c, beta,
    pairs) -- delegated to directly rather than re-solved, so the one
    constraint variant that already existed in the repo goes through the
    repo's own tested code path. The capacity=True and scope="local"
    variants are new (the repo's QP has no per-class caps at all), built the
    same way (OSQP, same ridge/eps/polish settings) as the reused function.
    """
    assert scope in ("aggregate", "local")
    if scope == "aggregate":
        budget = beta
        per_class_cap = {y: gamma * pi[y] for y in pi}
    else:
        budget = beta / gamma
        per_class_cap = {y: pi[y] for y in pi}

    if not capacity and scope == "aggregate":
        return _project_gradient_reused(Q, c, beta, pairs, ridge=ridge)

    P = Q.shape[0]
    Q_reg = Q + ridge * np.eye(P)
    q = -np.asarray(c, dtype=np.float64)
    A, l, u = _build_budget_constraint(pairs, budget, per_class_cap, capacity)

    solver = osqp.OSQP()
    solver.setup(
        P=sp.csc_matrix(Q_reg), q=q, A=A, l=l, u=u,
        verbose=False, polish=False, eps_abs=1e-6, eps_rel=1e-6,
    )
    result = solver.solve()
    w_np = np.nan_to_num(np.asarray(result.x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    w_np = np.clip(w_np, 0.0, None)
    return torch.as_tensor(w_np, dtype=torch.float64)


def dist_to_cone(Q, c, v_norm_sq, pairs, big_beta=1e6, ridge=1e-6):
    """
    NNLS min_{u>=0} ||Gbar u - v||^2 (no budget constraint): needs no new
    solver, it's solve_qp with an effectively unbounded budget (capacity=False,
    scope="aggregate" so it delegates straight to the reused project_gradient).
    Prints a warning if sum(w*) is not comfortably below big_beta, i.e. the
    "unbounded" constraint might still be binding, in which case dist2 is only
    an upper bound on the true NNLS distance, not the exact value.

    Returns (dist2, alpha_tilde_star, w_star), alpha_tilde_star =
    sqrt(1 - dist2/||v||^2).
    """
    w_star = solve_qp(Q, c, big_beta, pi={}, gamma=1.0, pairs=pairs,
                       scope="aggregate", capacity=False, ridge=ridge)
    w_np = w_star.numpy()
    if w_np.sum() > 0.999 * big_beta:
        print(f"[dist_to_cone] WARNING: sum(w*)={w_np.sum():.4g} close to "
              f"big_beta={big_beta:.4g} -- budget may still bind; increase big_beta.")
    c = np.asarray(c, dtype=np.float64)
    quad = float(w_np @ Q @ w_np)
    dist2 = float(v_norm_sq) - 2.0 * float(np.dot(w_np, c)) + quad
    dist2 = max(dist2, 0.0)
    alpha_tilde_star = math.sqrt(max(0.0, 1.0 - dist2 / v_norm_sq)) if v_norm_sq > 0 else float("nan")
    return dist2, alpha_tilde_star, w_star


# --------------------------------------------------------------------------#
# Support function / reachable radius / rank ratio
# --------------------------------------------------------------------------#

def support_function(Gbar, p, beta, pi, gamma, pairs, col_index):
    """
    h(p) = max_{u in U_beta} <p, Gbar u>, solved in closed form by a greedy
    fill: U_beta factorizes into independent per-class boxes [0, gamma*pi[y]]
    (one per class y, over its C-1 (y,z) coordinates) intersected with one
    shared L1 budget beta. For fixed y, <p, Gbar_{y,:} w_y> is maximized by
    putting all of class y's allowance on its single best z
    (psi[y] = max(0, max_z <p, Gbar[:,(y,z)]>)); across classes, since the
    budget is shared and every unit of budget in class y contributes psi[y]
    regardless of how much is already spent there, the optimal spend order is
    psi[y] descending until beta is exhausted.
    """
    p = p.to(torch.float32)
    classes = sorted(set(y for y, _ in pairs))
    best_z, psi = {}, {}
    for y in classes:
        vals = [(z, float(p @ Gbar[:, col_index[(y, z)]])) for (yy, z) in pairs if yy == y]
        z_best, val_best = max(vals, key=lambda t: t[1])
        best_z[y] = z_best
        psi[y] = max(0.0, val_best)

    order = sorted(classes, key=lambda y: psi[y], reverse=True)
    remaining = beta
    h = 0.0
    u_star = {(y, z): 0.0 for (y, z) in pairs}
    for y in order:
        if remaining <= 0 or psi[y] <= 0:
            break
        take = min(gamma * pi[y], remaining)
        u_star[(y, best_z[y])] = take
        h += take * psi[y]
        remaining -= take
    return h, u_star


def _project_onto_Ubeta(u_point, pi, gamma, beta, pairs):
    """Euclidean projection onto U_beta, reusing solve_qp with Q=I (min 0.5||w-u||^2 = 0.5 w^Tw - u^Tw + const)."""
    P = len(pairs)
    Q_id = np.eye(P)
    return solve_qp(Q_id, u_point, beta, pi, gamma, pairs, scope="aggregate", capacity=True).numpy()


def reachable_radius(Gbar, beta, pi, gamma, pairs, n_restarts=20, n_steps=50, seed=0):
    """
    Bounds on r_k = max_{e in E_k} ||e||, E_k = Gbar @ U_beta:
      upper:        beta * varsigma  (varsigma = max column norm)
      lower_simple: min(beta, gamma*pi[y*]) * varsigma, y* = class of the
                    largest-norm column
      lower_ascent: best value found by projected ASCENT on the convex
                    function f(u) = ||Gbar u|| from n_restarts random
                    feasible starts.

    f is CONVEX, so gradient ascent on it does not converge to a KKT point
    the way descent on a convex function does -- it has no interior
    stationary point to climb to, and instead keeps climbing along the
    gradient direction until projection clips it back onto the boundary of
    U_beta. This is used ONLY to get a good LOWER bound on the true maximum
    (which sits at a vertex of U_beta): more restarts/steps can only improve
    the bound, never certify it is tight, and there is no guarantee of
    convergence to the global optimum.
    """
    Gbar_np = Gbar.detach().cpu().numpy().astype(np.float64)
    col_norms = np.linalg.norm(Gbar_np, axis=0)
    varsigma = float(col_norms.max())
    upper = beta * varsigma

    y_of = [y for y, _ in pairs]
    j_max = int(col_norms.argmax())
    y_star = y_of[j_max]
    lower_simple = min(beta, gamma * pi[y_star]) * varsigma

    P = len(pairs)
    classes = sorted(set(y_of))
    rng = np.random.RandomState(seed)
    best = lower_simple

    for _ in range(n_restarts):
        u = np.zeros(P)
        alloc = rng.dirichlet(np.ones(len(classes))) * beta
        for y, a in zip(classes, alloc):
            idxs = [j for j, yy in enumerate(y_of) if yy == y]
            a = min(a, gamma * pi[y])
            share = rng.dirichlet(np.ones(len(idxs))) * a
            for j, s in zip(idxs, share):
                u[j] = s

        step = 0.1 * beta
        for _ in range(n_steps):
            e = Gbar_np @ u
            norm = np.linalg.norm(e)
            grad = rng.randn(P) if norm < 1e-12 else Gbar_np.T @ (e / norm)
            grad = grad / (np.linalg.norm(grad) + 1e-12)
            u = _project_onto_Ubeta(u + step * grad, pi, gamma, beta, pairs)

        val = float(np.linalg.norm(Gbar_np @ u))
        best = max(best, val)

    return {"upper": upper, "lower_simple": lower_simple, "lower_ascent": best, "varsigma": varsigma}


def rank_ratio(Q, c, v_norm_sq):
    """varpi = c^T pinv(Q) c / ||v||^2."""
    Q_pinv = np.linalg.pinv(Q)
    c = np.asarray(c, dtype=np.float64)
    return float(c @ Q_pinv @ c) / v_norm_sq if v_norm_sq > 0 else float("nan")


def effective_rank(Q, tol=1e-8):
    """
    Participation-ratio effective rank of Q = Gbar^T Gbar, for E2's varpi
    baseline (rang_effectif(Q)/d). Not in the original function list, added
    because E2 explicitly needs this baseline and nothing in the repo
    computes it. (sum(eigvals))^2 / sum(eigvals^2) is a soft, differentiable
    stand-in for a hard-thresholded rank count and reduces to it when
    eigenvalues are either ~0 or all roughly equal.
    """
    eigvals = np.linalg.eigvalsh(Q)
    eigvals = np.clip(eigvals, 0, None)
    s1 = eigvals.sum()
    s2 = (eigvals ** 2).sum()
    if s2 < tol:
        return 0.0
    return float(s1 ** 2 / s2)


# --------------------------------------------------------------------------#
# Backdoor target v = grad_p - grad_c = lam * (grad_bd - grad_c)
# --------------------------------------------------------------------------#

_TO_PIL = _tvt.ToPILImage()
_TO_TENSOR = _tvt.ToTensor()


def apply_trigger(x_raw_batch, trigger):
    """
    x_raw_batch: (n, C, H, W) tensor in [0,1]. trigger: "identity" (no-op) or
    "stripe" (StripePoisoner, reused from modules.base_utils.datasets
    unmodified -- it operates on PIL images, hence the round-trip). The
    trigger is fixed, never optimized, per this session's scope.
    """
    if trigger == "identity":
        return x_raw_batch
    if trigger == "stripe":
        poisoner = StripePoisoner(strength=6, freq=16)
        device = x_raw_batch.device
        out = [_TO_TENSOR(poisoner.poison(_TO_PIL(x.cpu()))) for x in x_raw_batch]
        return torch.stack(out).to(device)
    raise NotImplementedError(trigger)


def compute_grad_bd(model, loss_fn, class_samples_raw, y_source, y_target,
                     dataset_flag, device, model_flag=None, trigger="identity"):
    """
    grad_bd = grad_theta E_{x ~ class y_source}[ loss(f_theta(T(x)), y_target) ].
    Trigger T applied in RAW [0,1] pixel space (apply_trigger) before the
    same raw_to_preprocess normalization used everywhere else in this
    library, so it's consistent with how Gbar/grad_c are computed.
    """
    x_raw = class_samples_raw[y_source].to(device)
    x_trig = apply_trigger(x_raw, trigger)
    x = raw_to_preprocess(x_trig, dataset_flag=dataset_flag, model_flag=model_flag)
    y_lab = torch.full((x.shape[0],), y_target, dtype=torch.long, device=x.device)

    params = list(model.parameters())
    model.zero_grad(set_to_none=True)
    logits = model(x)
    loss = loss_fn(logits, y_lab)
    grad = torch.autograd.grad(loss, params)
    flat = torch.cat([g.reshape(-1) for g in grad]).detach().cpu().to(torch.float32)
    model.zero_grad(set_to_none=True)
    return flat


def effect_rate(model, dataset_flag, y_source, y_target, device, model_flag=None,
                 trigger="stripe", n_max=None, batch_size=512):
    """
    Backdoor attack success rate (SPEC section 8/E6 "effect rate"): the
    fraction of TRIGGERED test examples truly of class y_source that the
    model classifies as y_target. Built from the raw TEST split (never seen
    during training, poisoned or otherwise) -- get_raw_clean_dataset(train=
    False), filtered to y_source, T applied in raw [0,1] space via
    apply_trigger (same mechanism as compute_grad_bd/E1-E5, never the
    training-time poisoning: label flips never touch pixels anywhere in this
    suite -- see flip_masses_to_labels/masses_to_labels), then the same
    raw_to_preprocess normalization as every other model input here.

    n_max: cap on the number of y_source test examples used (None = all,
    CIFAR-10 has 1000 per class). batch_size: forward-pass chunking only, the
    reported rate is exact over whatever examples are used.
    """
    test_raw = get_raw_clean_dataset(dataset_flag, train=False)
    xs = [x for x, y in test_raw if y == y_source]
    if n_max is not None:
        xs = xs[:n_max]
    if not xs:
        raise ValueError(f"effect_rate: no test examples of class {y_source} found")
    x_raw = torch.stack(xs)

    model.eval()
    n_target = 0
    with torch.no_grad():
        for start in range(0, len(x_raw), batch_size):
            chunk = x_raw[start:start + batch_size].to(device)
            x_trig = apply_trigger(chunk, trigger)
            x = raw_to_preprocess(x_trig, dataset_flag=dataset_flag, model_flag=model_flag)
            logits = model(x)
            pred = logits.argmax(dim=1)
            n_target += int((pred == y_target).sum().item())
    return n_target / len(x_raw)


def clean_accuracy(model, dataset_flag, device, model_flag=None):
    """
    Standard test accuracy on the CLEAN (untriggered, truly-labelled) test
    split, via the repo's own clf_eval -- get_clean_dataset(train=False)
    already yields normalized (preprocessed) tensors, exactly what clf_eval
    expects (it infers the device from the model itself).
    """
    test_ds = get_clean_dataset(dataset_flag, train=False)
    acc, _loss = clf_eval(model, test_ds)
    return acc


# --------------------------------------------------------------------------#
# Clean short training -> 3 checkpoints (begin/mid/end)
# --------------------------------------------------------------------------#

def train_clean_checkpoints(model, dataset_flag, n_classes, device, epochs,
                             batch_size, ckpt_dir, tag, test_pct=0.0, seed=0,
                             max_train=None):
    """
    Short clean training producing exactly 3 checkpoints (begin/mid/end),
    reusing modules.base_utils.util.mini_train_multi as-is (single "worker" =
    the whole clean training set, agg_method="mean", f=0 -- mini_train_multi
    degenerates to plain SGD in this configuration) rather than writing a new
    training loop. Optimizer/scheduler hyperparameters are the pipeline's own
    DEFAULT_SGD_KWARGS / DEFAULT_SGD_SCHED_KWARGS (base_utils/util.py),
    reused unmodified.

    Saves state_dicts to {ckpt_dir}/{tag}_begin.pt, {tag}_mid.pt, {tag}_end.pt
    (plain torch.save(model.state_dict(), path), matching the repo's own
    checkpoint convention -- see modules/train_expert/utils.py,
    modules/federated_train_expert/utils.py) and returns their paths.

    max_train: if given, trains on a random max_train-sized Subset of the
    full clean training set (used for the "cnn" config, per this session's
    plan of 5000-10000 CIFAR-10 examples rather than the full 50000).
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    train_ds = get_clean_dataset(dataset_flag, train=True)
    if max_train is not None and max_train < len(train_ds):
        rng = np.random.RandomState(seed)
        sub_idx = rng.choice(len(train_ds), size=max_train, replace=False)
        train_ds = torch.utils.data.Subset(train_ds, sub_idx)
    test_ds = None
    if test_pct > 0:
        n_test = max(1, int(len(train_ds) * test_pct))
        test_ds = torch.utils.data.Subset(get_clean_dataset(dataset_flag, train=False),
                                           range(min(n_test, len(get_clean_dataset(dataset_flag, train=False)))))

    opt = torch.optim.SGD(model.parameters(), **DEFAULT_SGD_KWARGS)
    sched_kwargs = {k: v for k, v in DEFAULT_SGD_SCHED_KWARGS.items()}
    scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, **sched_kwargs)

    paths = {}
    begin_path = os.path.join(ckpt_dir, f"{tag}_begin.pt")
    torch.save(model.state_dict(), begin_path)
    paths["begin"] = begin_path

    e_mid = max(1, epochs // 2)
    e_end = max(1, epochs - e_mid)

    mini_train_multi(
        model=model, train_datasets=[train_ds], test_data=test_ds,
        batch_size=batch_size, opt=opt, scheduler=scheduler, epochs=e_mid,
        agg_method="mean", f=0,
    )
    mid_path = os.path.join(ckpt_dir, f"{tag}_mid.pt")
    torch.save(model.state_dict(), mid_path)
    paths["mid"] = mid_path

    mini_train_multi(
        model=model, train_datasets=[train_ds], test_data=test_ds,
        batch_size=batch_size, opt=opt, scheduler=scheduler, epochs=e_end,
        agg_method="mean", f=0,
    )
    end_path = os.path.join(ckpt_dir, f"{tag}_end.pt")
    torch.save(model.state_dict(), end_path)
    paths["end"] = end_path

    return paths


# --------------------------------------------------------------------------#
# Instrumented aggregation rules (SPEC section 7)
# --------------------------------------------------------------------------#
#
# Every rule this suite uses is a "select-then-average" rule: at coordinate j
# it picks a set S_j of ell worker indices and returns their plain mean. That
# is true of the four robust rules of the grid as the repo implements them
# (see modules/base_utils/util.py mini_train_multi, l.366-381):
#
#   mean       S_j = all workers,                 ell = n_b
#   cw_median  S_j = {lower median index},        ell = 1
#   trmean     S_j = the n_b-2f middle values,    ell = n_b - 2f
#   krum       S_j = {argmin krum score},         ell = 1        (multikrum m=1)
#   multikrum  S_j = the m best krum scores,      ell = m = n_b-f-2
#
# so a single generic instrumentation covers all of them: each rule returns an
# index array, and the aggregate is the gather-and-mean of that array. The
# selection weights omega[i,j] = 1[i in S_j]/ell are NEVER materialised as an
# (n_b, d) matrix, per SPEC section 7 -- only the (ell, d) index array is, and
# ell <= n_b so that array is no larger than the gradient stack itself.
#
# cw_median follows torch.median's convention (LOWER median, ell = 1) rather
# than the textbook "average of the two middle values" for even n_b. That is a
# deliberate choice: the repo's per-tensor path is literally
# `grads.median(dim=0).values`, and the flat/per_tensor concordance assertion
# of SPEC section 2 can only be exact if the flat reference uses the same
# convention.

_REPO_TRMEAN = None
_REPO_MULTIKRUM = None


def _repo_aggregators():
    """Lazy import of the repo's own rules (modules/ is read only)."""
    global _REPO_TRMEAN, _REPO_MULTIKRUM
    if _REPO_TRMEAN is None:
        from modules.base_utils.aggregator.trmean import aggr_trmean
        from modules.base_utils.aggregator.multikrum import aggregate as aggr_multikrum
        _REPO_TRMEAN, _REPO_MULTIKRUM = aggr_trmean, aggr_multikrum
    return _REPO_TRMEAN, _REPO_MULTIKRUM


AGG_RULES = ("mean", "cw_median", "trmean", "krum", "multikrum")
AGG_VARIANTS = ("flat", "per_tensor")


def repo_aggregate(stack, rule, f):
    """
    The repo's own aggregation of an (n_b, ...) stack, dispatched exactly as
    modules.base_utils.util.mini_train_multi does it. Imported, never
    reimplemented -- this is what the `per_tensor` variant calls per parameter
    tensor, and what the flat reference is checked against.
    """
    aggr_trmean, aggr_multikrum = _repo_aggregators()
    if rule == "mean":
        return stack.mean(dim=0)
    if rule == "cw_median":
        return stack.median(dim=0).values
    if rule == "trmean":
        return aggr_trmean(stack, f=f)
    if rule == "krum":
        return aggr_multikrum(stack, f=f, m=1)
    if rule == "multikrum":
        return aggr_multikrum(stack, f=f)
    raise NotImplementedError(rule)


def agg_ell(rule, n_b, f):
    """|S_j| for each rule, i.e. how many messages actually reach the update."""
    if rule == "mean":
        return n_b
    if rule in ("cw_median", "krum"):
        return 1
    if rule == "trmean":
        return n_b - 2 * f
    if rule == "multikrum":
        return n_b - f - 2
    raise NotImplementedError(rule)


@dataclass
class Selection:
    """
    Who reached the aggregate, and at which coordinates.

    kind="global"     : idx has shape (ell,)      -- one set for every coordinate
                        (krum, multikrum, mean); this is the case the theory
                        assumes, and it forces osc(Abar) = 0 exactly.
    kind="coordinate" : idx has shape (ell, d)    -- one set per coordinate
                        (cw_median, trmean), and any rule under `per_tensor`,
                        where the set is constant WITHIN a tensor and varies
                        BETWEEN tensors.

    `blocks` is the list of (start, end) parameter-tensor spans the selection
    was built from: ((0, d),) for `flat`, one entry per parameter tensor for
    `per_tensor`. It is what makes the between-tensor oscillation of Abar
    attributable to a tensor rather than to a bare coordinate index.
    """
    kind: str
    idx: torch.Tensor
    ell: int
    n_b: int
    d: int
    rule: str
    variant: str
    blocks: tuple = ()

    @property
    def chi_ell(self) -> float:
        """chi_ell = (n_b - ell) / (ell * n_b)."""
        return (self.n_b - self.ell) / (self.ell * self.n_b)

    @property
    def lam(self) -> float:
        """
        Lambda = max_j sum_i |w[i,j]| with w[i,j] = omega[i,j] - 1/n_b.

        Closed form, independent of j for any select-then-average rule:
        the ell selected workers each contribute |1/ell - 1/n_b| and the
        n_b - ell others each contribute 1/n_b, so the sum telescopes to
        2 (n_b - ell) / n_b = 2 * ell * chi_ell. Computing it in closed form
        rather than by a max over d avoids a (n_b, d) intermediate and removes
        any ambiguity about how Lambda is defined (SPEC section 8/E4 names
        Lambda in the bound but does not define it).
        """
        return 2.0 * (self.n_b - self.ell) / self.n_b

    def gather(self, G):
        """(ell, d) stack of the selected values of G, an (n_b, d) tensor."""
        if self.kind == "global":
            return G[self.idx]
        return torch.gather(G, 0, self.idx)

    def aggregate(self, G):
        return self.gather(G).mean(dim=0)

    def A(self, mal_mask):
        """
        A_j = |S_j ∩ M| / ell as a (d,) tensor. mal_mask is a (n_b,) bool
        tensor flagging the perturbed workers.
        """
        if self.kind == "global":
            val = mal_mask[self.idx].to(torch.float32).sum() / self.ell
            return val.expand(self.d).clone()
        return mal_mask[self.idx].to(torch.float32).sum(dim=0) / self.ell

    def split_PN(self, G, mal_mask):
        """
        The two halves of b_Agg - b_mean = sum_i w[i,j] g_i[j], split by worker
        type: P over the perturbed workers M, N over the honest ones H. Both
        are (d,) tensors. Computed from the (ell, d) gather plus two (d,) sums,
        never from an (n_b, d) weight matrix.
        """
        sel = self.gather(G)
        if self.kind == "global":
            sel_mal = mal_mask[self.idx].to(torch.float32).unsqueeze(1)
        else:
            sel_mal = mal_mask[self.idx].to(torch.float32)
        mal = mal_mask.to(torch.float32).unsqueeze(1)
        P = (sel * sel_mal).sum(dim=0) / self.ell - (G * mal).sum(dim=0) / self.n_b
        N = (sel * (1 - sel_mal)).sum(dim=0) / self.ell - (G * (1 - mal)).sum(dim=0) / self.n_b
        return P, N


def _krum_order(G, f):
    """
    Worker indices sorted by increasing Krum score, replicating
    modules/base_utils/aggregator/multikrum.py::_compute_scores: pairwise
    distances, then the sum of the n - f - 1 smallest distances per worker.
    Vectorised here (the repo loops in Python over n(n-1)/2 pairs, which the
    ~200-round E4/E5 replays would make the dominant cost) and checked against
    the repo aggregate by `check_against_repo` below.
    """
    n = G.shape[0]
    D = torch.cdist(G.unsqueeze(0), G.unsqueeze(0)).squeeze(0)
    D = torch.nan_to_num(D, nan=float("inf"), posinf=float("inf"))
    scores = []
    for i in range(n):
        d_i = torch.cat([D[i, :i], D[i, i + 1:]])
        scores.append(float(torch.sort(d_i).values[:n - f - 1].sum()))
    return torch.as_tensor(np.argsort(np.asarray(scores), kind="stable"), dtype=torch.long)


def _select_block(G, rule, f):
    """Selection for one (n_b, d_block) stack. Returns a Selection."""
    n_b, d = G.shape
    ell = agg_ell(rule, n_b, f)
    if rule == "mean":
        return Selection("global", torch.arange(n_b), n_b, n_b, d, rule, "flat")
    if rule in ("cw_median", "trmean"):
        order = torch.argsort(G, dim=0, stable=True)
        if rule == "cw_median":
            k = (n_b - 1) // 2          # torch.median == LOWER median
            idx = order[k:k + 1]
        else:
            idx = order[f:n_b - f]
        return Selection("coordinate", idx.contiguous(), ell, n_b, d, rule, "flat")
    if rule in ("krum", "multikrum"):
        order = _krum_order(G, f)
        return Selection("global", order[:ell].contiguous(), ell, n_b, d, rule, "flat")
    raise NotImplementedError(rule)


def flat_blocks(model):
    """[(start, end)] spans of each parameter tensor inside the flattened gradient."""
    blocks, off = [], 0
    for p in model.parameters():
        n = p.numel()
        blocks.append((off, off + n))
        off += n
    return tuple(blocks)


def aggregate_instrumented(G, rule, f, variant="flat", blocks=None, check_repo=False,
                           repo_tol=1e-5):
    """
    (aggregate, selection) for an (n_b, d) stack of FLATTENED gradients.

    variant="flat"       -- one selection for the whole d-dimensional vector.
                            The only variant the theory covers.
    variant="per_tensor" -- the rule applied independently on each span of
                            `blocks`, i.e. tensor by parameter tensor, the way
                            modules/base_utils/util.py::mini_train_multi does
                            it. The AGGREGATE of each block is produced by the
                            repo's own function (repo_aggregate), not by a
                            reimplementation; only the index bookkeeping is
                            new, and `check_repo` asserts the two agree.

    Under per_tensor the returned Selection is always kind="coordinate", even
    for krum/multikrum: the set is constant within a tensor but varies between
    tensors, which is exactly the osc(Abar) > 0 that SPEC section 7 predicts.
    """
    n_b, d = G.shape
    if variant == "flat":
        sel = _select_block(G, rule, f)
        sel.blocks = ((0, d),)
        sel.variant = "flat"
        agg = sel.aggregate(G)
        if check_repo:
            check_against_repo(G, rule, f, agg, tol=repo_tol)
        return agg, sel

    if variant != "per_tensor":
        raise NotImplementedError(variant)
    if blocks is None:
        raise ValueError("per_tensor needs `blocks` (see flat_blocks(model))")

    ell = agg_ell(rule, n_b, f)
    idx = torch.empty((ell, d), dtype=torch.long)
    agg = torch.empty(d, dtype=G.dtype)
    for (s, e) in blocks:
        Gb = G[:, s:e].contiguous()
        sel_b = _select_block(Gb, rule, f)
        if sel_b.kind == "global":
            idx[:, s:e] = sel_b.idx.unsqueeze(1).expand(ell, e - s)
        else:
            idx[:, s:e] = sel_b.idx
        # The aggregate comes from the repo implementation, per SPEC section 7.
        agg[s:e] = repo_aggregate(Gb, rule, f)
        if check_repo:
            ref = torch.gather(Gb, 0, idx[:, s:e]).mean(dim=0)
            gap = float((ref - agg[s:e]).abs().max())
            if gap > repo_tol:
                raise AssertionError(
                    f"per_tensor {rule}: instrumented selection disagrees with the repo "
                    f"aggregate on block ({s},{e}): max|diff| = {gap:.3e} > {repo_tol:.1e}")
    sel = Selection("coordinate", idx, ell, n_b, d, rule, "per_tensor", tuple(blocks))
    return agg, sel


def check_against_repo(G, rule, f, agg_flat, tol=1e-5):
    """
    SPEC section 2 assertion: the flat reference rule must agree with the repo
    rule applied to the same stack. Returns the observed max|diff| and raises
    if it exceeds `tol`.
    """
    ref = repo_aggregate(G, rule, f)
    gap = float((ref - agg_flat).abs().max())
    if gap > tol:
        raise AssertionError(f"flat {rule}: max|flat - repo| = {gap:.3e} > {tol:.1e}")
    return gap


def osc(values):
    """Oscillation max - min of a (d,) tensor; 0.0 exactly for a constant vector."""
    t = torch.as_tensor(values)
    return float(t.max() - t.min())


def alpha_tilde(b, v):
    """
    Deviation-level alignment cos(b, v), clipped to [0, 1]. Same quantity as
    alpha_tilde_star = sqrt(1 - dist^2/||v||^2) but evaluated at an ARBITRARY
    deviation b rather than at the cone-projection optimum -- SPEC section 8/E4
    asks for alpha_tilde(b_Agg) against alpha_tilde(b_mean).
    """
    b = torch.as_tensor(b, dtype=torch.float32)
    v = torch.as_tensor(v, dtype=torch.float32)
    den = float(b.norm()) * float(v.norm())
    if den <= 0:
        return float("nan")
    return max(0.0, min(1.0, float(torch.dot(b, v)) / den))


# SPEC section 5 names this function `masses_to_labels`; the implementation
# above predates that name. Alias rather than rename, so existing callers keep
# working and the spec's name resolves.
masses_to_labels = flip_masses_to_labels
