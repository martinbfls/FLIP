"""
Isolated script -- NOT part of modules/federated_optimizing_trigger/, and
does not reintegrate anything into it.

On 20 steps, with an IDENTICAL v = mu_p - g_c per step, compares two estimates
of "how reachable is v via label flipping":

  - dist(v, W_beta): the module's continuous, per-class-pair relaxation
    (project_gradient / compute_v_polytope_distance, unchanged, imported from
    the module as-is).
  - ||v - (g_a - g_c)||: a discrete, per-example greedy matching-pursuit over
    individual examples of the SAME batch (select_hard_flips_last_layer),
    under an equivalent budget B_j = round(beta * n_b).

Reports the mean ratio dist_polytope / dist_greedy and its stddev over the 20
steps to out/optimizing_trigger/polytope_vs_greedy.json.

PROVENANCE NOTE: select_hard_flips_last_layer / find_last_linear /
get_features_before_linear are reconstructed below "from git history" as
requested, but `git log --all -p -- modules/federated_optimizing_trigger/utils.py`
shows only a single commit (c4ab384) touching that file, and it does NOT
contain these functions -- they were never actually committed. They only
existed in this session's (and the pre-session's) uncommitted working tree,
before being deleted in an earlier refactor pass this session. What follows is
a verbatim copy captured from that working tree earlier in this session, not
an actual git-history retrieval.
"""
import os
import sys
import json
import copy
import statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from modules.base_utils.datasets import get_n_classes
from modules.base_utils.util import load_model
from modules.federated_optimizing_trigger.utils import (
    sample_checkpoints,
    compute_batch_gradients,
    get_mu,
    extract_experts,
    move_to_device,
    init_delta,
    raw_to_preprocess,
    raw_to_trigger_preprocess,
    get_raw_clean_dataset,
    compute_expected_flip_gradients,
    compute_class_frequencies,
    compute_v_polytope_distance,
    get_class_conditional_samples,
)
from modules.federated_optimizing_trigger.run_module import build_worker_loader

torch.manual_seed(0)

# ---------------------------------------------------------------------------- #
# Reconstructed verbatim (see provenance note above) -- NOT reintegrated into
# modules/federated_optimizing_trigger/.
# ---------------------------------------------------------------------------- #

def find_last_linear(model):
    '''Returns the last nn.Linear submodule of a model (used for the last-layer flip subspace).'''
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            last = m
    if last is None:
        raise ValueError("No nn.Linear module found in model for last-layer flip subspace")
    return last


def get_features_before_linear(model, x, linear):
    '''Forward pass that also returns the input features to `linear` via a forward hook.'''
    captured = {}

    def hook(module, inputs, output):
        captured["h"] = inputs[0]

    handle = linear.register_forward_hook(hook)
    try:
        logits = model(x)
    finally:
        handle.remove()
    return logits, captured["h"]


def select_hard_flips_last_layer(h, y, r_W, budget, debug=False):
    '''
    Greedy matching pursuit for hard label flips, restricted to the last linear
    layer's weight subspace where d_{i,c} = (e_{y_i} - e_c) h_i^T admits a closed form:
      <d_{i,c}, r> = (R[y_i] - R[c]) . h_i
      ||d_{i,c}||^2 = 2 ||h_i||^2
    h: (n_b, H) detached features feeding the last linear layer.
    y: (n_b,) int64 labels of the clean batch.
    r_W: (C, H) residual n_b * (g_p - g_clean) restricted to the linear layer's weight, detached.
    Returns a list of (i, c) flips, |list| <= budget, at most one c per i.
    '''
    n_b = h.shape[0]
    C = r_W.shape[0]
    device = h.device

    R = r_W.clone()
    h_norm_sq = (h ** 2).sum(dim=1)
    used = torch.zeros(n_b, dtype=torch.bool, device=device)
    idx_range = torch.arange(n_b, device=device)

    selected = []
    for _ in range(budget):
        scores_full = R @ h.T  # (C, n_b)
        row_y = scores_full[y, idx_range]  # (n_b,)
        score_matrix = row_y.unsqueeze(0) - scores_full - h_norm_sq.unsqueeze(0)  # (C, n_b)
        score_matrix = score_matrix.T  # (n_b, C)
        score_matrix[idx_range, y] = -float("inf")
        score_matrix[used] = -float("inf")

        best_val, best_idx = torch.max(score_matrix.reshape(-1), dim=0)
        best_score, flat_idx = torch.stack([best_val, best_idx.to(best_val.dtype)]).tolist()
        flat_idx = int(round(flat_idx))
        i_star, c_star = divmod(flat_idx, C)

        if best_score <= 0:
            break

        selected.append((i_star, c_star))
        used[i_star] = True
        R[y[i_star]] = R[y[i_star]] - h[i_star]
        R[c_star] = R[c_star] + h[i_star]

    return selected


# ---------------------------------------------------------------------------- #
# Comparison
# ---------------------------------------------------------------------------- #

device = "cuda" if torch.cuda.is_available() else "cpu"
dataset_flag = "cifar"
model_flag = "r32p"
source_label, target_label = 9, 4
num_honests, num_poisoned = 5, 5
n_w = num_honests + num_poisoned
flip_budget = 1500
worker_batch_size = 256
n_steps = 20
expert_path = "/shared/data1/Projects/DLWP/j1067582/martin/FLIP/out/checkpoints/r32p_1xs/{}/model_{}_{}.pth"

print("Loading model / checkpoint / data...")
n_classes = get_n_classes(dataset_flag)
model = load_model(model_flag, n_classes).to(device)
model.eval()

checkpoints_start = extract_experts({}, expert_path)
# a single, fixed, well-trained checkpoint reused for all 20 steps (only the
# batch -- hence v -- varies step to step; this is what "a v identique" means:
# same v feeds both estimators within a step, not that v is fixed across steps).
M = copy.deepcopy(model).to(device)
M.load_state_dict(torch.load(checkpoints_start[-1], map_location=device))
M.eval()
params = list(M.parameters())
loss_fn = torch.nn.CrossEntropyLoss()

linear = find_last_linear(M)
idx_w = next(i for i, p in enumerate(params) if p is linear.weight)

raw_train_dataset = get_raw_clean_dataset(dataset_flag, train=True)
n_train = len(raw_train_dataset)
lambda_poison = flip_budget * n_w / (num_poisoned * n_train)
beta = lambda_poison

mu = get_mu(dataset_flag, target_label, device, model_flag=model_flag)
delta = init_delta(mu.shape, horizontal=True, strength=6.0, freq=16, device=device, init="stripe")

worker_loader = build_worker_loader(raw_train_dataset, worker_batch_size)
loader_iter = iter(worker_loader)

pi = compute_class_frequencies(dataset_flag, n_classes)
class_samples_raw = get_class_conditional_samples(dataset_flag, n_classes, 64, device)
G, Q, pairs = compute_expected_flip_gradients(
    M, loss_fn, class_samples_raw, n_classes, pi,
    dataset_flag=dataset_flag, model_flag=model_flag, params=params,
)

ratios = []
records = []

for step in range(n_steps):
    x_raw, y = move_to_device(next(loader_iter), device)
    n_b = x_raw.shape[0]
    x_clean = raw_to_preprocess(x_raw, dataset_flag=dataset_flag, model_flag=model_flag)

    mask = y == source_label
    y_poison = y.clone()
    x_poisoned = x_clean.clone()
    if mask.sum() > 0:
        idx_source = mask.nonzero(as_tuple=True)[0]
        target_count = min(int(round(lambda_poison * n_b)), idx_source.numel())
        perm = torch.randperm(idx_source.numel(), device=idx_source.device)[:target_count]
        keep = idx_source[perm]
        mask = torch.zeros_like(mask)
        mask[keep] = True
        y_poison[mask] = target_label
        x_poisoned[mask] = raw_to_trigger_preprocess(
            x_raw[mask], delta, dataset_flag=dataset_flag, model_flag=model_flag,
        )

    grads_c, _ = compute_batch_gradients(M, loss_fn, (x_clean, y), create_graph=False, retain_graph=False)
    g_c_list = [g.detach() for g in grads_c]
    g_c = torch.cat([g.reshape(-1) for g in g_c_list])

    grads_p, _ = compute_batch_gradients(M, loss_fn, (x_poisoned, y_poison), create_graph=False, retain_graph=False)
    g_p_list = [g.detach() for g in grads_p]
    mu_p = torch.cat([g.reshape(-1) for g in g_p_list])

    v = mu_p - g_c  # IDENTICAL v fed to both estimators below

    # --- (1) polytope estimator: continuous, per-class-pair relaxation ---
    dist2_poly, _ = compute_v_polytope_distance(v, G, Q, pairs, beta)
    dist_polytope = float(dist2_poly.clamp(min=0).sqrt().item())

    # --- (2) greedy estimator: discrete per-example selection, same budget ---
    B_j = int(min(max(round(beta * n_b), 0), n_b))
    if B_j == 0:
        g_a = g_c.clone()
    else:
        with torch.no_grad():
            _, h = get_features_before_linear(M, x_clean, linear)
        r_W = n_b * (g_p_list[idx_w] - g_c_list[idx_w])
        selected = select_hard_flips_last_layer(h, y, r_W, B_j)

        y_hard = y.clone()
        for (i, c) in selected:
            y_hard[i] = c

        grads_a, _ = compute_batch_gradients(M, loss_fn, (x_clean, y_hard), create_graph=False, retain_graph=False)
        g_a = torch.cat([g.reshape(-1) for g in grads_a]).detach()

    dist_greedy = float((v - (g_a - g_c)).norm().item())

    ratio = dist_polytope / dist_greedy if dist_greedy > 1e-12 else float("inf")
    ratios.append(ratio)
    records.append({
        "step": step, "B_j": B_j,
        "dist_polytope": dist_polytope, "dist_greedy": dist_greedy, "ratio": ratio,
    })
    print(f"step {step:2d}: B_j={B_j:3d}  dist_polytope={dist_polytope:.4f}  "
          f"dist_greedy={dist_greedy:.4f}  ratio={ratio:.4f}", flush=True)

finite_ratios = [r for r in ratios if r != float("inf")]
mean_ratio = statistics.fmean(finite_ratios) if finite_ratios else float("nan")
std_ratio = statistics.pstdev(finite_ratios) if len(finite_ratios) > 1 else 0.0

summary = {
    "n_steps": n_steps,
    "flip_budget": flip_budget,
    "beta": beta,
    "mean_ratio_polytope_over_greedy": mean_ratio,
    "std_ratio_polytope_over_greedy": std_ratio,
    "records": records,
}

os.makedirs("out/optimizing_trigger", exist_ok=True)
with open("out/optimizing_trigger/polytope_vs_greedy.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== summary ===")
print(f"mean ratio dist_polytope / dist_greedy = {mean_ratio:.4f}")
print(f"std  ratio dist_polytope / dist_greedy = {std_ratio:.4f}")
print("saved out/optimizing_trigger/polytope_vs_greedy.json")
