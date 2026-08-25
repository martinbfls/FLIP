"""
prelim/verify_A1_A2_A5.py -- verification for A1 (lambda_poison scope fix), A2 (rho_k self-
consistency after exposing varsigma_k), and A5 (v_hat_k before/after A1). Uses ONE real
checkpoint (no training loop -- just resolution logic + one real
compute_expected_flip_gradients + one real v_k computation) so this runs in seconds, not the
cost of a full optimize_trigger_policy call.

Run:  python prelim/verify_A1_A2_A5.py
"""
import os
import sys

import torch

if not torch.cuda.is_available():
    torch.nn.Module.cuda = lambda self, device=None: self.to("cpu")
    torch.Tensor.cuda = lambda self, device=None, non_blocking=False: self.to("cpu")

sys.path.insert(0, os.getcwd())

from modules.base_utils.util import load_model
from modules.base_utils.datasets import get_n_classes
from modules.federated_optimizing_trigger.utils import (
    compute_expected_flip_gradients, compute_class_frequencies, get_class_conditional_samples,
    resolve_beta_and_lambda_poison, compute_batch_gradients, raw_to_preprocess,
    raw_to_trigger_preprocess, init_delta, get_raw_clean_dataset, move_to_device,
)

CKPT_PATH = (
    "/private/tmp/claude-501/-Users-martinbeaufils-Downloads-broadflip-repo-FLIP/"
    "1f0e6899-15d6-4c67-a0f9-936b7b87b76e/scratchpad/smoke/checkpoints/"
    "opt_trigger_policy_bootstrap/model_1_100.pth"
)

dataset_flag = "cifar"
model_flag = "r32p"
n_classes = get_n_classes(dataset_flag)
device = "cpu"

# Real grid config (matches Bloc B's default axes: num_poisoned=3, num_honests=7).
num_honests, num_poisoned = 7, 3
beta_arg = 0.3  # a LOCAL beta (fraction of ONE corrupted worker's own shard), <=1 required
n_train = len(get_raw_clean_dataset(dataset_flag, train=True))

print("=" * 70)
print("A1: lambda_poison scope, before vs after")
print("=" * 70)

beta_local, flip_budget, _lambda_poison_raw = resolve_beta_and_lambda_poison(
    beta_arg, None, "beta", num_poisoned, num_honests, n_train,
)
gamma = num_poisoned / (num_poisoned + num_honests)
beta_global = gamma * beta_local

lambda_poison_before = beta_local          # pre-A1 behavior: lambda_poison = beta (local)
lambda_poison_after = beta_global          # post-A1 behavior: lambda_poison = gamma*beta

print(f"beta_local          = {beta_local:.6f}")
print(f"gamma                = {gamma:.6f}  (num_poisoned={num_poisoned}, num_honests={num_honests})")
print(f"beta_global          = {beta_global:.6f}  (= gamma*beta_local)")
print(f"lambda_poison BEFORE = {lambda_poison_before:.6f}  (== beta_local)")
print(f"lambda_poison AFTER  = {lambda_poison_after:.6f}  (== beta_global)")
print(f"ratio before/after   = {lambda_poison_before / lambda_poison_after:.6f}  "
      f"(expected 1/gamma = {1 / gamma:.6f})")
print(f"flip_budget          = {flip_budget}  (expected UNCHANGED -- does not depend on lambda_poison)")

print()
print("=" * 70)
print("A2: rho_k self-consistency (old fused formula vs new varsigma_k-explicit formula)")
print("=" * 70)

pi = compute_class_frequencies(dataset_flag, n_classes)
pi_source = pi[9]  # arbitrary source class for A5's v_k below
class_samples_raw = get_class_conditional_samples(dataset_flag, n_classes, 64, device)

model = load_model(model_flag, n_classes)
state = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(state)
model.eval()
loss_fn = torch.nn.CrossEntropyLoss()

G_k, Q_k, pairs_k = compute_expected_flip_gradients(
    model, loss_fn, class_samples_raw, n_classes, pi,
    dataset_flag=dataset_flag, model_flag=model_flag,
)

pi_col = torch.tensor([pi[y] for (y, c) in pairs_k], dtype=G_k.dtype)
varsigma_k = (G_k.detach() / pi_col).norm(dim=0).max().item()

scale = torch.tensor([gamma / pi[y] for (y, c) in pairs_k], dtype=G_k.dtype)
G_obj = G_k * scale

rho_k_old_formula = beta_local * G_obj.detach().norm(dim=0).max().item()   # pre-A1/A2 code path
rho_k_new_formula = beta_global * varsigma_k                              # post-A1/A2 code path

print(f"varsigma_k (named explicitly, A2)     = {varsigma_k:.6f}")
print(f"rho_k, OLD formula (beta_local*max||G_obj||) = {rho_k_old_formula:.6f}")
print(f"rho_k, NEW formula (beta_global*varsigma_k)  = {rho_k_new_formula:.6f}")
print(f"difference                                    = {abs(rho_k_old_formula - rho_k_new_formula):.3e}  "
      f"(expected ~0 -- rho_k UNCHANGED by A1/A2, per the module docstring's algebraic identity)")

print()
print("=" * 70)
print("A5: v_hat_k = ||v_k|| / rho_k, before vs after A1")
print("=" * 70)

delta = init_delta(
    (3, 32, 32), horizontal=True, strength=6.0, freq=16, device=device, init="stripe",
)
raw_ds = get_raw_clean_dataset(dataset_flag, train=True)
xs = torch.stack([x for x, y in raw_ds if y == 9][:32])
ys_clean = torch.full((xs.shape[0],), 9, dtype=torch.long)
ys_target = torch.full((xs.shape[0],), 4, dtype=torch.long)

x_clean = raw_to_preprocess(xs, dataset_flag=dataset_flag, model_flag=model_flag)
x_trig = raw_to_trigger_preprocess(xs, delta, dataset_flag=dataset_flag, model_flag=model_flag)

grads_c, _ = compute_batch_gradients(model, loss_fn, (x_clean, ys_clean), create_graph=False)
g_c = torch.cat([g.reshape(-1) for g in grads_c]).detach()

for label, lam in [("BEFORE (lambda_poison=beta_local)", lambda_poison_before),
                    ("AFTER  (lambda_poison=beta_global)", lambda_poison_after)]:
    n_b = xs.shape[0]
    target_count = min(int(round(lam * n_b)), n_b)
    x_mix = x_clean.clone()
    y_mix = ys_clean.clone()
    x_mix[:target_count] = x_trig[:target_count]
    y_mix[:target_count] = ys_target[:target_count]

    grads_p, _ = compute_batch_gradients(model, loss_fn, (x_mix, y_mix), create_graph=False)
    mu_p = torch.cat([g.reshape(-1) for g in grads_p]).detach()
    v_k = mu_p - g_c
    v_hat_k = v_k.norm().item() / rho_k_new_formula  # rho_k is A1/A2-corrected, same both times
    print(f"{label}: target_count={target_count}/{n_b}, ||v_k||={v_k.norm().item():.4f}, "
          f"v_hat_k={v_hat_k:.4f}")

print()
print(f"(reference) rho_k used for v_hat_k = {rho_k_new_formula:.6f}")
print(f"(reference) pi_source (y=9)        = {pi_source:.6f}")
