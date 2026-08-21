import torch
import numpy as np
import scipy.sparse as sp
import osqp
import torch.nn.functional as F
from modules.base_utils.datasets import (
    get_n_classes, load_dataset, MappedDataset, TRANSFORM_TRAIN_XY, TRANSFORM_TEST_XY,
    pick_poisoner,
    CIFAR_TRANSFORM_NORMALIZE_MEAN,   CIFAR_TRANSFORM_NORMALIZE_STD,
    CIFAR_100_TRANSFORM_NORMALIZE_MEAN, CIFAR_100_TRANSFORM_NORMALIZE_STD,
    SVHN_TRANSFORM_NORMALIZE_MEAN,    SVHN_TRANSFORM_NORMALIZE_STD,
    TINY_IMAGENET_TRANSFORM_NORMALIZE_MEAN, TINY_IMAGENET_TRANSFORM_NORMALIZE_STD,
)
from modules.federated_generate_labels.utils import DEFAULT_EXPERT_CONFIG, DEFAULT_ATTACK_ITERATIONS
from modules.base_utils.util import needs_big_ims
from PIL import Image
from torchvision import transforms
from torch.utils.data import Subset, ConcatDataset
import numpy as np

RAW_TRANSFORM_X = transforms.ToTensor()
RAW_TRANSFORM_Y = lambda y: y

DATASET_NORMALIZATION = {
    'cifar':         (CIFAR_TRANSFORM_NORMALIZE_MEAN,       CIFAR_TRANSFORM_NORMALIZE_STD),
    'cifar_100':     (CIFAR_100_TRANSFORM_NORMALIZE_MEAN,   CIFAR_100_TRANSFORM_NORMALIZE_STD),
    'svhn':          (SVHN_TRANSFORM_NORMALIZE_MEAN,        SVHN_TRANSFORM_NORMALIZE_STD),
    'tiny_imagenet': (TINY_IMAGENET_TRANSFORM_NORMALIZE_MEAN, TINY_IMAGENET_TRANSFORM_NORMALIZE_STD),
}

def get_norm_tensors(dataset_flag: str, device):
    mean, std = DATASET_NORMALIZATION[dataset_flag]
    mean = torch.tensor(mean, device=device).view(3, 1, 1)
    std  = torch.tensor(std,  device=device).view(3, 1, 1)
    return mean, std


def get_transforms(dataset_flag, train=True, big=False):
    key = dataset_flag + ('_big' if big else '')
    if train:
        return TRANSFORM_TRAIN_XY[key]
    else:
        return TRANSFORM_TEST_XY[key]


def sample_checkpoints(K, S, alpha=0.01, device="cpu"):
    ks = torch.arange(0, K, device=device, dtype=torch.float)
    probs = torch.exp(-alpha * ks)
    probs = probs / probs.sum()
    idx = torch.multinomial(probs, S, replacement=False)
    idx = idx.tolist()
    if K - 1 not in idx:
        idx.append(K - 1)
    idx.sort()
    return idx


def init_delta(mu_shape, strength=6.0, freq=16, horizontal=True, device="cuda", init='stripe'):
    if init == 'stripe':
        C, H, W = mu_shape
        sin_1d = torch.sin(torch.linspace(0, freq * torch.pi, W, device=device))
        mask = sin_1d.view(1, 1, W).expand(C, H, W)
        if horizontal:
            mask = mask.transpose(1, 2)
        delta = strength * mask.clone()
        delta.requires_grad_(True)
    elif init == 'random':
        delta = strength * torch.rand(mu_shape, device=device)
        delta.requires_grad_(True)
    else:
        raise ValueError(f"Unknown init type: {init}")
    return delta


def compute_batch_gradients(model, loss_fn, batch, create_graph, retain_graph=False):
    model.zero_grad(set_to_none=True)
    x, y = batch
    logits = model(x)
    loss = loss_fn(logits, y)
    grads = torch.autograd.grad(
        loss,
        model.parameters(),
        create_graph=create_graph,
        retain_graph=retain_graph,
    )
    return grads, logits


def trigger_penalty_hinge(delta, mu_target, mu_source, kappa, eps=1e-8):
    delta_f = delta.view(1, -1)
    diff_f = (mu_target - mu_source).view(1, -1).detach()
    cos = F.cosine_similarity(delta_f, diff_f, eps=eps).mean()
    return F.relu(cos - kappa)


def tv_loss(delta):
    '''Anisotropic total variation of a (C, H, W) trigger perturbation.'''
    dh = (delta[:, 1:, :] - delta[:, :-1, :]).abs().sum()
    dw = (delta[:, :, 1:] - delta[:, :, :-1]).abs().sum()
    return dh + dw


# ---------------------------------------------------------------------------- #
# Expected flip gradients, feasible gradient polytope W_beta(theta), and exact
# QP projection of v onto it.
# ---------------------------------------------------------------------------- #

def get_class_conditional_samples(dataset_flag, n_classes, n_per_class, device):
    '''
    Draws up to `n_per_class` raw ([0, 1], un-normalized) training examples per
    class. Used to estimate the expected flip gradients G; depends only on the
    dataset (not on any checkpoint or on the trigger), so callers should compute
    this once per run and reuse it across checkpoints and optimization steps.
    Returns a dict y -> (n_y, C, H, W) tensor on `device`, n_y <= n_per_class.
    '''
    dataset = get_raw_clean_dataset(dataset_flag, train=True)
    by_class = {y: [] for y in range(n_classes)}
    remaining = n_classes * n_per_class
    for x, y in dataset:
        if len(by_class[y]) < n_per_class:
            by_class[y].append(x)
            remaining -= 1
            if remaining <= 0:
                break
    return {y: torch.stack(xs).to(device) for y, xs in by_class.items() if xs}


def compute_expected_flip_gradients(model, loss_fn, class_samples_raw, n_classes, pi,
                                     dataset_flag, model_flag=None, params=None):
    '''
    Estimates, for the current checkpoint `model`, the expected per-parameter
    gradient difference induced by relabeling samples of class y to class c:

        g_{y,c}(theta) = E[ grad_theta loss(x, c) - grad_theta loss(x, y) | Y = y ]

    approximated by the empirical mean over `class_samples_raw[y]`.

    The reachable displacement from a label-flipping attack is
        sum_{y,c} pi_y * w_{y,c} * g_{y,c}
    (pi_y: empirical frequency of class y -- see `compute_class_frequencies` --
    w_{y,c}: fraction of class-y examples flipped to c), NOT
    sum_{y,c} w_{y,c} * g_{y,c}. pi_y is therefore absorbed directly into G's
    columns here: G[:, (y,c)] = pi_y * g_{y,c}. This keeps the budget
    constraint on w itself a plain sum (see `_build_global_budget_constraint`)
    while G @ w always equals the correctly pi_y-weighted reachable
    displacement.

    Returns:
        G:     (D, P) tensor (on `model`'s device), D = number of flattened
               parameters, P = number of ordered class pairs (y, c), y != c.
               Columns are pi_y * g_{y,c}, ordered as in `pairs`.
        Q:     (P, P) numpy float64 array, Q = G^T G, precomputed once here so
               `project_gradient` never has to see G.
        pairs: list[(y, c)] matching G's / Q's columns.

    G and Q depend on the checkpoint's current weights, the (fixed) dataset
    sample, AND pi -- NOT on the trigger delta / mu_p / v. Callers should
    cache (G, Q, pairs) per checkpoint (invalid once the checkpoint's weights
    change; also invalid if pi were ever recomputed, though in practice pi is
    a fixed dataset-level constant for the whole run).
    '''
    params = params if params is not None else list(model.parameters())
    classes_present = [y for y in range(n_classes) if y in class_samples_raw]
    pairs = [(y, c) for y in classes_present for c in range(n_classes) if c != y]

    columns = {}
    for y in classes_present:
        x = raw_to_preprocess(class_samples_raw[y], dataset_flag=dataset_flag, model_flag=model_flag)
        n_y = x.shape[0]

        model.zero_grad(set_to_none=True)
        logits = model(x)

        y_lab = torch.full((n_y,), y, dtype=torch.long, device=x.device)
        loss_y = loss_fn(logits, y_lab)
        grad_y = torch.autograd.grad(loss_y, params, retain_graph=True)
        flat_grad_y = torch.cat([g.reshape(-1) for g in grad_y]).detach()

        for c in range(n_classes):
            if c == y:
                continue
            c_lab = torch.full((n_y,), c, dtype=torch.long, device=x.device)
            loss_c = loss_fn(logits, c_lab)
            grad_c = torch.autograd.grad(loss_c, params, retain_graph=True)
            flat_grad_c = torch.cat([g.reshape(-1) for g in grad_c]).detach()
            columns[(y, c)] = pi[y] * (flat_grad_c - flat_grad_y)

    model.zero_grad(set_to_none=True)
    G = torch.stack([columns[p] for p in pairs], dim=1)  # (D, P)
    Q = (G.T @ G).detach().to(torch.float64).cpu().numpy()
    return G, Q, pairs


def compute_class_frequencies(dataset_flag, n_classes):
    '''
    Empirical class frequencies pi_y over the full training set. Independent of
    checkpoint, trigger, and of the (possibly truncated) sample used to build G
    -- callers should compute this once per run and reuse it.
    Returns dict y -> pi_y (floats summing to 1).
    '''
    dataset = load_dataset(dataset_flag, train=True)
    labels = np.array([y for _, y in dataset])
    counts = np.array([(labels == c).sum() for c in range(n_classes)], dtype=np.float64)
    total = counts.sum()
    return {y: counts[y] / total for y in range(n_classes)}


def _build_global_budget_constraint(pairs, beta):
    '''
    Builds the OSQP constraint system  l <= A w <= u  for the feasible
    coefficient set of W_beta(theta) over the columns `pairs` (ordered list of
    (y, c) pairs spanning G): a single GLOBAL poisoning-budget constraint

        w >= 0,   sum_{y,c} w_{y,c} <= beta

    Unweighted: pi_y is already absorbed into G's columns (see
    `compute_expected_flip_gradients`), so a second pi_y weighting here would
    double-count it. The vertices g_{y,c} (columns of G) are never touched --
    only this feasible coefficient set moves with beta.
    '''
    P = len(pairs)
    rows_nonneg = sp.identity(P, format="csc")
    l_nonneg = np.zeros(P)
    u_nonneg = np.full(P, np.inf)

    ones_row = sp.csc_matrix(np.ones((1, P)))
    A = sp.vstack([rows_nonneg, ones_row], format="csc")
    l = np.concatenate([l_nonneg, [-np.inf]])
    u = np.concatenate([u_nonneg, [beta]])
    return A, l, u


def project_gradient(Q, c, beta, pairs, ridge=1e-6):
    '''
    w* = argmin_w  0.5 w^T Q w - c^T w   s.t.  w >= 0, sum_{y,c} w_{y,c} <= beta

    Q = G^T G (P, P) and c = G^T v (P,) are precomputed by the caller: Q is
    cached per checkpoint (see `compute_expected_flip_gradients` callers), c is
    a cheap per-step matvec. This function never touches G or v -- only these
    two already-CPU numpy arrays cross into the QP solve, so it never pays the
    cost of transferring the (potentially huge, D x P) matrix G.

    Returns w_star: (P,) torch tensor, detached, float64, CPU only -- callers
    needing it on a specific device/dtype should `.to(...)` it themselves.
    '''
    P = Q.shape[0]
    Q_reg = Q + ridge * np.eye(P)
    q = -np.asarray(c, dtype=np.float64)

    A, l, u = _build_global_budget_constraint(pairs, beta)

    solver = osqp.OSQP()
    solver.setup(
        P=sp.csc_matrix(Q_reg), q=q, A=A, l=l, u=u,
        # polish=False: avoids OSQP's polish-step log line, which some OSQP
        # builds print unconditionally regardless of verbose=False and would
        # otherwise spam stdout across thousands of per-batch QP solves.
        # eps_abs/eps_rel=1e-6 already give ample precision without it.
        verbose=False, polish=False, eps_abs=1e-6, eps_rel=1e-6,
    )
    result = solver.solve()

    w_np = np.asarray(result.x, dtype=np.float64)
    w_np = np.nan_to_num(w_np, nan=0.0, posinf=0.0, neginf=0.0)
    w_np = np.clip(w_np, 0.0, None)  # numerical clean-up of the w >= 0 constraint

    return torch.as_tensor(w_np, dtype=torch.float64)


def compute_v_polytope_distance(v, G, Q, pairs, beta, ridge=1e-6):
    '''
    Squared distance from v to the feasible gradient polytope

        W_beta(theta) = { G w : w >= 0, sum_{y,c} w_{y,c} <= beta }

    G's columns are already pi_y-weighted (G[:, (y,c)] = pi_y * g_{y,c}, see
    `compute_expected_flip_gradients`), so G @ w correctly equals the
    pi_y-weighted reachable displacement sum_{y,c} pi_y * w_{y,c} * g_{y,c},
    and the budget constraint on w needs no further pi_y weighting (see
    `_build_global_budget_constraint`):

        dist2 = min_w ||v - G w||^2 = ||v||^2 - 2 <w*, c> + w*^T Q w*,  c = G^T v

    computed WITHOUT ever materializing g_proj = G @ w* -- only P-dimensional
    quantities (c, w*, Q) participate in the QP solve and in this formula; the
    only D-dimensional work is the matvec c = G^T v and ||v||^2 itself.

    Differentiable w.r.t. v: w* and Q are treated as constants (detached); by
    the envelope theorem (w* satisfies the KKT conditions of a QP in which v
    enters only through the linear term c = G^T v), this is the exact gradient
    of ||v - G w*||^2 w.r.t. v -- identical to differentiating an explicitly
    materialized, `.detach()`-ed g_proj.

    Returns (dist2, w_star): dist2 attached to v's graph, w_star detached.
    '''
    c_vec = G.T @ v  # (P,), differentiable wrt v (hence wrt delta)
    c_np = c_vec.detach().cpu().numpy().astype(np.float64)

    w_star = project_gradient(Q, c_np, beta, pairs, ridge=ridge)
    w_star = w_star.to(device=v.device, dtype=v.dtype)
    w_star_np = w_star.detach().cpu().numpy().astype(np.float64)

    quad_term = float(w_star_np @ Q @ w_star_np)
    dist2 = (v * v).sum() - 2.0 * (w_star * c_vec).sum() + quad_term
    return dist2, w_star


def compute_beta_star(v, G, Q, pairs, betas, ridge=1e-6):
    '''
    For a grid of candidate budgets `betas`, solves the QP projection of v
    onto W_beta for each beta and returns the resulting normalized-B2 curve,
    plus the smallest beta in the grid achieving B2 < 1e-4 (the budget beyond
    which v is, for practical purposes, already reachable by label-flipping).

    Q and c = G^T v do NOT depend on beta -- only the budget constraint's RHS
    does (see `_build_global_budget_constraint`) -- so both are computed once
    here and reused for every beta in the grid; only the (cheap, P-dimensional)
    QP solve itself repeats per beta.

    Args:
        v:     (D,) tensor, target displacement (typically detached -- this is
               a diagnostic, not part of the differentiable training loss).
        G, Q, pairs: as returned by `compute_expected_flip_gradients`.
        betas: iterable of candidate beta values (any order).
        ridge: QP regularization, see `project_gradient`.

    Returns:
        dist2_curve: list of normalized B2 values, one per beta, in the same
                     order as `betas`.
        beta_star:   smallest beta in `betas` with B2 < 1e-4, or None if none
                     of the grid's betas achieve it.
    '''
    eps_den = 1e-8
    c_vec = G.T @ v.detach()
    c_np = c_vec.detach().cpu().numpy().astype(np.float64)
    v_norm_sq = float(v.detach().norm() ** 2)
    den = v_norm_sq + eps_den

    dist2_curve = []
    for beta in betas:
        w_star = project_gradient(Q, c_np, beta, pairs, ridge=ridge)
        w_star_np = w_star.detach().cpu().numpy().astype(np.float64)
        quad_term = float(w_star_np @ Q @ w_star_np)
        dist2 = v_norm_sq - 2.0 * float(np.dot(w_star_np, c_np)) + quad_term
        dist2_curve.append(dist2 / den)

    feasible_betas = [b for b, b2 in zip(betas, dist2_curve) if b2 < 1e-4]
    beta_star = min(feasible_betas) if feasible_betas else None
    return dist2_curve, beta_star


def resolve_beta_and_lambda_poison(beta, flip_budget, lambda_poison, num_poisoned, num_honests, n_train):
    '''
    Resolves (beta, flip_budget, lambda_poison) from whichever of beta/flip_budget was
    passed, and couples lambda_poison to beta when lambda_poison == "beta" (the default).

    beta -- the fraction of the attacker's OWN shard it can afford to flip -- is the
    primary parameter: it is a property of the attacker alone, not of the federated
    deployment it will eventually be used against (see `optimize_trigger`'s docstring).
    num_honests/num_poisoned exist only to translate beta into a "number of flips per
    round" for human-readable logging/run naming under one particular assumed deployment
    size; they play no role in the trigger objective itself. Pass exactly one of beta or
    flip_budget; passing both raises.

    lambda_poison == "beta" resolves to beta directly, coupling the objective's per-batch
    poisoning rate (and, via lambda_target, the expert's actual retraining poison rate) to
    beta -- this is what guarantees lambda == beta throughout the pipeline.

    Returns (beta, flip_budget, lambda_poison).
    '''
    n_w = num_honests + num_poisoned

    if beta is not None and flip_budget is not None:
        raise ValueError(
            "Pass exactly one of `beta` or `flip_budget`, not both -- "
            f"got beta={beta} and flip_budget={flip_budget}. beta is the "
            "primary parameter; flip_budget is accepted only for backward "
            "compatibility and, when passed alone, is converted to beta "
            "via beta = flip_budget * n_w / (num_poisoned * n_train)."
        )
    elif beta is not None:
        if not (0.0 < beta < 1.0):
            raise ValueError(f"beta must be in (0, 1), got {beta}")
        flip_budget = round(beta * num_poisoned * n_train / n_w)  # logging/run-name only
    elif flip_budget is not None:
        beta = flip_budget * n_w / (num_poisoned * n_train)
    else:
        raise ValueError("Pass beta (preferred) or flip_budget (legacy).")

    print(
        f"beta={beta:.6f} (attacker's own-shard flip fraction) -- assuming "
        f"num_poisoned={num_poisoned}, num_honests={num_honests}, n_train={n_train} "
        f"this is flip_budget~={flip_budget:.1f} flips/round (logging only, not "
        "used by the objective)."
    )

    if lambda_poison == "beta":
        lambda_poison = beta
    if lambda_poison is None:
        raise ValueError(
            "lambda_poison is None after resolution: pass a float in (0, 1], "
            "or the string 'beta' (the default) to derive it from beta."
        )

    return beta, flip_budget, lambda_poison


def extract_experts(expert_config, expert_path):
    config = {**DEFAULT_EXPERT_CONFIG, **expert_config}
    expert_starts = []
    for expert in range(config['experts']):
        for epoch in range(config['min'], config['max']):
            for s in config['trajectories']:
                expert_starts.append(expert_path.format(str(expert), str(epoch + 1), str(s)))
    return expert_starts


def get_clean_dataset(dataset_flag, train=True, big=False):
    transform = get_transforms(dataset_flag, train=train, big=big)
    base_dataset = load_dataset(dataset_flag, train=train)
    return MappedDataset(base_dataset, transform)


def get_poison_dataset(dataset_flag, source_label, target_label, delta,
                       train=True, train_pct=1.0, big=False,
                       lambda_target=None, lambda_overflow="clip", seed=0,
                       include_clean=True):
    '''
    Builds a poisoned copy of (a subset of) `base_dataset`'s source-class
    examples, triggered and relabeled to target_label. If include_clean=True
    (default), returns ConcatDataset([clean_dataset, poison_dataset]): all of
    `base_dataset` plus that poisoned copy. If include_clean=False, returns
    just MappedDataset(MappedDataset(Subset(base, poison_inds), poisoner),
    transform) -- every returned example is triggered and labeled
    target_label, with no untriggered examples mixed in.

    Use include_clean=False for ASR measurement: accuracy on the
    include_clean=True ConcatDataset is dominated by untriggered clean
    examples (it mostly tracks clean accuracy, not attack success). Combine
    with lambda_target=None (poison every source-class example) for the
    strict-ASR test set.

    lambda_target: if given, the fraction of the RETURNED dataset that is
    poisoned, i.e. n_add / (n_base + n_add) == lambda_target, where n_base =
    len(base_dataset) and n_add is the number of (possibly resampled)
    source-class indices poisoned. This couples the expert-retraining
    dataset's actual poison rate to whatever lambda_poison (= beta, by
    default) the trigger objective assumes -- see optimize_trigger. If None
    (default), every source-class example is poisoned (n_add = n_s, the
    legacy fixed rate n_s / (n_base + n_s), CIFAR-10 ~= 0.0909).

    n_add = round(lambda_target * n_base / (1 - lambda_target)) can exceed
    n_s, the number of available source-class examples (whenever
    lambda_target > beta_max := n_s / (n_base + n_s)). lambda_overflow then
    controls what happens:
      - "clip" (default): use all n_s source-class examples (n_add -> n_s)
        and log lambda_target, the resulting effective lambda, and beta_max.
      - "duplicate": resample n_add indices from the n_s source-class
        examples WITH replacement (same underlying images, but each Subset
        position is a fresh dataset entry so `transform` -- TRANSFORM_TRAIN_XY
        for train=True -- re-applies independent augmentation per position
        and per epoch).
    '''
    transform = get_transforms(dataset_flag, train=train, big=big)
    base_dataset = load_dataset(dataset_flag, train=train)

    if train_pct < 1.0:
        n = int(len(base_dataset) * train_pct)
        base_dataset = Subset(base_dataset, np.arange(n))

    labels = np.array([y for _, y in base_dataset])
    poison_inds = np.where(labels == source_label)[0]

    if lambda_target is not None:
        if not (0.0 < lambda_target < 1.0):
            raise ValueError(f"lambda_target must be in (0, 1), got {lambda_target}")
        n_base = len(base_dataset)
        n_s = len(poison_inds)
        n_add = round(lambda_target * n_base / (1 - lambda_target))
        rng = np.random.RandomState(seed)

        if n_add > n_s:
            beta_max = n_s / (n_base + n_s)
            if lambda_overflow == "clip":
                print(
                    f"[get_poison_dataset] lambda_overflow='clip': lambda_target="
                    f"{lambda_target:.6f} needs n_add={n_add} > n_s={n_s} available "
                    f"source-class ({dataset_flag}, source_label={source_label}, "
                    f"train={train}) examples; clipping to n_add={n_s}. Effective "
                    f"lambda={n_s / (n_base + n_s):.6f}, beta_max={beta_max:.6f}."
                )
            elif lambda_overflow == "duplicate":
                poison_inds = rng.choice(poison_inds, size=n_add, replace=True)
            else:
                raise ValueError(
                    f"Unknown lambda_overflow={lambda_overflow!r}, expected "
                    "'clip' or 'duplicate'."
                )
        elif n_add < n_s:
            poison_inds = rng.choice(poison_inds, size=n_add, replace=False)
        # n_add == n_s: use poison_inds as-is.

    poisoner = pick_poisoner('optimized', dataset_flag, target_label, delta)

    poison_subset = Subset(base_dataset, poison_inds)
    poison_dataset = MappedDataset(poison_subset, poisoner)
    poison_dataset = MappedDataset(poison_dataset, transform)

    if not include_clean:
        return poison_dataset

    clean_dataset = MappedDataset(base_dataset, transform)
    return ConcatDataset([clean_dataset, poison_dataset])


def preprocess_for_model(x: torch.Tensor, dataset_flag: str, model_flag: str):
    big = needs_big_ims(model_flag)
    mean, std = get_norm_tensors(dataset_flag, x.device)

    if big:
        squeeze = x.dim() == 3
        if squeeze:
            x = x.unsqueeze(0)
        x = F.interpolate(x, size=224, mode='bilinear', align_corners=False)
        if squeeze:
            x = x.squeeze(0)

    if x.dim() == 4:
        mean = mean.unsqueeze(0)
        std  = std.unsqueeze(0)
    x = (x - mean) / std
    return x


def get_raw_clean_dataset(dataset_flag, train=True):
    base_dataset = load_dataset(dataset_flag, train=train)
    def mapper(sample):
        x, y = sample
        return RAW_TRANSFORM_X(x), y  # PIL → tensor [0,1]
    return MappedDataset(base_dataset, mapper)


def get_mu(dataset_flag, y_target, device, model_flag=None):
    dataset = get_raw_clean_dataset(dataset_flag, train=True)
    xs = [x for x, y in dataset if y == y_target]
    if not xs:
        raise ValueError(f"No samples found for class {y_target}")
    mu = torch.stack(xs).to(device).mean(dim=0)
    return mu


def move_to_device(batch, device):
    if torch.is_tensor(batch):
        return batch.to(device)
    elif isinstance(batch, (list, tuple)):
        return type(batch)(move_to_device(b, device) for b in batch)
    return batch


def raw_to_preprocess(x_raw: torch.Tensor, dataset_flag: str, model_flag: str = None):
    if model_flag:
        return preprocess_for_model(x_raw, dataset_flag, model_flag)
    mean, std = get_norm_tensors(dataset_flag, x_raw.device)
    if x_raw.dim() == 4:
        mean = mean.unsqueeze(0)
        std  = std.unsqueeze(0)
    return (x_raw - mean) / std


def raw_to_trigger_preprocess(x_raw: torch.Tensor, delta: torch.Tensor,
                               dataset_flag: str, model_flag: str = None):
    if x_raw.dim() == 3:
        x_trig = (x_raw + delta).clamp(0, 1)
    else:
        x_trig = (x_raw + delta.unsqueeze(0)).clamp(0, 1)
    return raw_to_preprocess(x_trig, dataset_flag, model_flag)