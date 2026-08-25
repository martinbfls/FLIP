"""
prelim/verify_A3_measurement.py -- A3's required measurement, BEFORE any code changes:
lambda_effective vs lambda_poison on a realistic random batch (not artificially all-source-
class), for both the pre-A1 (beta_local) and post-A1 (beta_global) lambda_poison values.
Prediction to check: if lambda_poison > pi_source, lambda_effective should be capped near
pi_source, and lambda_effective/lambda_poison should be close to pi_source/lambda_poison.

Run:  python prelim/verify_A3_measurement.py
"""
import os
import sys

import torch

if not torch.cuda.is_available():
    torch.nn.Module.cuda = lambda self, device=None: self.to("cpu")
    torch.Tensor.cuda = lambda self, device=None, non_blocking=False: self.to("cpu")

sys.path.insert(0, os.getcwd())

from torch.utils.data import DataLoader

from modules.base_utils.datasets import get_n_classes
from modules.federated_optimizing_trigger.utils import (
    compute_class_frequencies, get_raw_clean_dataset, resolve_beta_and_lambda_poison,
)

dataset_flag = "cifar"
n_classes = get_n_classes(dataset_flag)
source_label = 9
num_honests, num_poisoned = 7, 3
beta_arg = 0.3
batch_size_trigger = 256  # schema default

n_train_ds = get_raw_clean_dataset(dataset_flag, train=True)
n_train = len(n_train_ds)
pi = compute_class_frequencies(dataset_flag, n_classes)
pi_source = pi[source_label]

beta_local, flip_budget, _ = resolve_beta_and_lambda_poison(
    beta_arg, None, "beta", num_poisoned, num_honests, n_train,
)
gamma = num_poisoned / (num_poisoned + num_honests)
beta_global = gamma * beta_local

loader = DataLoader(n_train_ds, batch_size=batch_size_trigger, shuffle=True, num_workers=0)
batch = next(iter(loader))
_, y = batch
n_b = y.shape[0]
idx_source = (y == source_label).nonzero(as_tuple=True)[0]
n_source_in_batch = idx_source.numel()

print(f"pi_source (theoretical) = {pi_source:.6f}")
print(f"batch_size = {n_b}, n_source_in_batch = {n_source_in_batch} "
      f"(empirical fraction = {n_source_in_batch / n_b:.6f})")
print()

for label, lam in [("BEFORE A1 (lambda_poison=beta_local)", beta_local),
                    ("AFTER  A1 (lambda_poison=beta_global)", beta_global)]:
    target_count = min(int(round(lam * n_b)), n_source_in_batch)
    lambda_effective = target_count / n_b
    ratio = lambda_effective / lam if lam > 0 else float("nan")
    predicted_ratio = min(1.0, pi_source / lam) if lam > 0 else float("nan")
    print(f"{label}:")
    print(f"  lambda_poison   = {lam:.6f}")
    print(f"  target_count    = {target_count} (requested {int(round(lam * n_b))}, "
          f"capped at n_source_in_batch={n_source_in_batch})")
    print(f"  lambda_effective = {lambda_effective:.6f}")
    print(f"  lambda_effective / lambda_poison = {ratio:.6f}  "
          f"(predicted ~ min(1, pi_source/lambda_poison) = {predicted_ratio:.6f})")
    print(f"  CAPPED: {'YES' if lam > pi_source else 'no (lambda_poison <= pi_source)'}")
    print()
