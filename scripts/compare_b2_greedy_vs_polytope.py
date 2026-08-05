"""
Etape 0 of the federated_optimizing_trigger refactor: compare the (then-current)
greedy B2 estimator (matching-pursuit hard flips through the aggregator) against
the corrected polytope B2 estimator (exact QP projection of v = mu_p - g_c onto
W_beta, global budget constraint) on identical checkpoints and identical
batches, run BEFORE the old code was deleted. Result: out/optimizing_trigger/b2_comparison.json
(correlation 0.111, mean ratio polytope/greedy 203x -- see aggregation report).

NOT runnable as-is anymore: `_compute_aggregation_aware_step` and
`federated_aggregate` (the greedy/aggregator code path this script exercised)
were deleted from run_module.py in etape 3 of the same refactor, once this
comparison had produced its result. Kept for provenance of the JSON above; to
rerun it you would need to check out the pre-refactor revision of
modules/federated_optimizing_trigger/run_module.py.
"""
import os
import sys
import json

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
from modules.federated_optimizing_trigger.run_module import (
    build_worker_loaders,
    _compute_aggregation_aware_step,
)

torch.manual_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
dataset_flag = "cifar"
model_flag = "r32p"
source_label, target_label = 9, 4
num_honests, num_poisoned = 5, 5
n_w = num_honests + num_poisoned
agg_method_obj = "mean"
flip_budget = 1500
flip_subspace = "last_layer"
worker_batch_size = 256
num_chckpt = 15
n_steps = 20
alpha_ckpt = 0.01

expert_path = "/shared/data1/Projects/DLWP/j1067582/martin/FLIP/out/checkpoints/r32p_1xs/{}/model_{}_{}.pth"
expert_config = {}

print("Loading model / data...")
n_classes = get_n_classes(dataset_flag)
model = load_model(model_flag, n_classes).to(device)
model.eval()
loss_fn = torch.nn.CrossEntropyLoss()

raw_train_dataset = get_raw_clean_dataset(dataset_flag, train=True)
n_train = len(raw_train_dataset)
lambda_poison = flip_budget * n_w / (num_poisoned * n_train)
beta = lambda_poison  # same formula as beta_mal in _compute_aggregation_aware_step

mu = get_mu(dataset_flag, target_label, device, model_flag=model_flag)
delta = init_delta(mu.shape, horizontal=True, strength=6.0, freq=16, device=device, init="stripe")
delta.requires_grad_(True)

worker_loaders = build_worker_loaders(raw_train_dataset, n_w, batch_size=worker_batch_size)

print("Loading expert checkpoints and sampling a FIXED checkpoint set...")
checkpoints_start = extract_experts(expert_config, expert_path)
expert_models = []
for ckpt_path in checkpoints_start:
    import copy
    M = copy.deepcopy(model).to(device)
    state = torch.load(ckpt_path, map_location=device)
    M.load_state_dict(state)
    M.eval()
    expert_models.append(M)

sampled_k = sample_checkpoints(len(expert_models), num_chckpt, alpha=alpha_ckpt, device=device)
print(f"Sampled {len(sampled_k)} fixed checkpoints (reused for all {n_steps} steps): {sampled_k}")

pi = compute_class_frequencies(dataset_flag, n_classes)
class_samples_raw = get_class_conditional_samples(dataset_flag, n_classes, 64, device)
flip_grad_cache = {}  # (G, Q, pairs) per checkpoint k, shared/reused across all 20 steps

results = {"b2_greedy": [], "b2_polytope": [], "step": []}

zipped_loaders = zip(*worker_loaders)

for step in range(n_steps):
    batches = next(zipped_loaders)

    # --- current greedy B2 (unchanged code path) ---
    # checkpoint_backward=True: backward + free the per-checkpoint create_graph=True
    # graph immediately, matching how this code path is meant to run at num_chckpt=15
    # (accumulating all 15 checkpoints' graphs simultaneously OOMs a 15GB GPU).
    delta.grad = None
    out = _compute_aggregation_aware_step(
        batches, expert_models, sampled_k, num_honests, num_poisoned,
        agg_method_obj, source_label, target_label, loss_fn, delta,
        device, dataset_flag, model_flag,
        flip_budget, lambda_poison, flip_subspace, n_train, n_classes,
        checkpoint_backward=True,
        lambda_b1=1.0, lambda_b2=1.0, lambda_adv=0.0,
        flip_projection_method="greedy",
    )
    b2_greedy = out["B2"].item()
    torch.cuda.empty_cache()

    # --- corrected polytope B2, from the SAME batches / SAME checkpoints ---
    # use the first poisoned worker's batch (cid = num_honests) as "the batch",
    # exactly what the greedy path treats as its reference poisoned worker too.
    x_raw, y = move_to_device(batches[num_honests], device)
    x_clean = raw_to_preprocess(x_raw, dataset_flag=dataset_flag, model_flag=model_flag)
    n_b = x_raw.shape[0]
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

    b2_poly_vals = []
    for k in sampled_k:
        M = expert_models[k]
        params = list(M.parameters())

        grads_c, _ = compute_batch_gradients(M, loss_fn, (x_clean, y), create_graph=False, retain_graph=False)
        g_c = torch.cat([g.reshape(-1) for g in grads_c]).detach()

        grads_p, _ = compute_batch_gradients(M, loss_fn, (x_poisoned, y_poison), create_graph=False, retain_graph=False)
        mu_p = torch.cat([g.reshape(-1) for g in grads_p]).detach()

        v = mu_p - g_c

        if k in flip_grad_cache:
            G_k, Q_k, pairs_k = flip_grad_cache[k]
        else:
            G_k, Q_k, pairs_k = compute_expected_flip_gradients(
                M, loss_fn, class_samples_raw, n_classes,
                dataset_flag=dataset_flag, model_flag=model_flag, params=params,
            )
            flip_grad_cache[k] = (G_k, Q_k, pairs_k)

        with torch.no_grad():
            dist2, _ = compute_v_polytope_distance(v, G_k, Q_k, pairs_k, n_classes, beta, pi)
            den = v.norm() ** 2 + 1e-8
            b2_poly_vals.append((dist2 / den).item())

    b2_polytope = sum(b2_poly_vals) / len(b2_poly_vals)

    results["step"].append(step)
    results["b2_greedy"].append(b2_greedy)
    results["b2_polytope"].append(b2_polytope)
    print(f"step {step:2d}: B2_greedy={b2_greedy:.6f}  B2_polytope={b2_polytope:.6f}", flush=True)

# --- summary stats ---
import statistics

g = results["b2_greedy"]
p = results["b2_polytope"]

mean_g, mean_p = statistics.fmean(g), statistics.fmean(p)
cov = statistics.fmean([(gi - mean_g) * (pi_ - mean_p) for gi, pi_ in zip(g, p)])
std_g = statistics.pstdev(g)
std_p = statistics.pstdev(p)
correlation = cov / (std_g * std_p) if std_g > 0 and std_p > 0 else float("nan")

ratios = [pi_ / gi for gi, pi_ in zip(g, p) if gi > 1e-12]
mean_ratio = statistics.fmean(ratios) if ratios else float("nan")

summary = {
    "n_steps": n_steps,
    "num_chckpt": num_chckpt,
    "sampled_k": sampled_k,
    "flip_budget": flip_budget,
    "beta": beta,
    "correlation_b2_greedy_vs_polytope": correlation,
    "mean_ratio_polytope_over_greedy": mean_ratio,
    "mean_b2_greedy": mean_g,
    "mean_b2_polytope": mean_p,
    **results,
}

os.makedirs("out/optimizing_trigger", exist_ok=True)
with open("out/optimizing_trigger/b2_comparison.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== summary ===")
print(f"correlation(B2_greedy, B2_polytope) = {correlation:.4f}")
print(f"mean ratio B2_polytope / B2_greedy  = {mean_ratio:.4f}")
print("saved to out/optimizing_trigger/b2_comparison.json")
