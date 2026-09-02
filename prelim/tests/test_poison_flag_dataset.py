"""
prelim/tests/test_poison_flag_dataset.py -- non-regression checks for:
  - modules/base_utils/datasets.py's new `PoisonFlagDataset` wrapper
  - the is_flipped-mask computation added to modules/federated_train_user/run_module.py
    (np.isin(worker_indices, idx_flipped), reusing federated_select_flips' own
    {budget}_idx_flipped.npy output rather than recomputing anything)

Uses tiny SYNTHETIC index arrays -- no real dataset, no model, no training. Matches
prelim/tests/test_expert_checkpoint_pool.py's convention (synthetic-only, fast).

Run:  python prelim/tests/test_poison_flag_dataset.py
"""
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.base_utils.datasets import PoisonFlagDataset
from modules.federated_select_flips.utils import partition_across_workers

_results = []


def check(name, ok, detail=""):
    _results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


class _IndexDataset(Dataset):
    """Returns (i, i) for every index i in `indices` -- a trivial (x, y) stand-in."""
    def __init__(self, indices):
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        v = int(self.indices[i])
        return v, v


def test_poison_flag_dataset_basic():
    base = _IndexDataset(np.array([10, 11, 12, 13, 14]))
    is_flipped = np.array([False, True, False, True, False])
    wrapped = PoisonFlagDataset(base, is_flipped)

    check("PoisonFlagDataset has the same length as the base dataset",
          len(wrapped) == len(base), f"len={len(wrapped)}")

    ok = True
    for i in range(len(wrapped)):
        x, y, flag = wrapped[i]
        x0, y0 = base[i]
        if x != x0 or y != y0 or flag != bool(is_flipped[i]):
            ok = False
    check("PoisonFlagDataset propagates (x, y) unchanged and appends the correct is_flipped flag",
          ok)


def test_is_flipped_mask_matches_partition_across_workers():
    """Reproduces federated_train_user's is_flipped_mask = np.isin(worker_idx, idx_flipped)
    against federated_select_flips' own partition_across_workers, and checks: every index
    landing on a poisoned worker that came from idx_flipped is correctly tagged True, every
    clean index (honest or poisoned worker) is correctly tagged False."""
    N = 40
    num_honests, num_poisoned = 4, 3
    rng = np.random.default_rng(0)

    idx_flipped = rng.choice(N, size=9, replace=False)
    idx_flipped.sort()
    idx_clean = np.setdiff1d(np.arange(N), idx_flipped)

    labels_final = np.zeros((N, 2))
    labels_final[:, 0] = 1  # arbitrary placeholder one-hot

    worker_indices, _ = partition_across_workers(
        N, idx_flipped, idx_clean, labels_final, num_honests, num_poisoned,
        seed=0, shuffle_clean=False,
    )

    all_ok = True
    for w, idx in enumerate(worker_indices):
        is_flipped_mask = np.isin(idx, idx_flipped)
        expected = np.array([i in set(idx_flipped.tolist()) for i in idx])
        if not np.array_equal(is_flipped_mask, expected):
            all_ok = False
    check("np.isin(worker_idx, idx_flipped) exactly reproduces per-example flip membership "
          "for every worker (honest and poisoned)",
          all_ok)

    # Every flipped index must land on a poisoned worker (partition_across_workers' own
    # guarantee) -- honest workers' is_flipped mask must therefore be all-False.
    honest_all_clean = all(
        not np.isin(worker_indices[w], idx_flipped).any() for w in range(num_honests)
    )
    check("no flipped index lands on an honest worker (partition_across_workers' guarantee, "
          "reflected in the mask)",
          honest_all_clean)

    poisoned_have_some_flips = all(
        np.isin(worker_indices[num_honests + p], idx_flipped).any() for p in range(num_poisoned)
    )
    check("every poisoned worker's mask has at least one flipped example (9 flips split "
          "across 3 poisoned workers)",
          poisoned_have_some_flips)


if __name__ == "__main__":
    test_poison_flag_dataset_basic()
    test_is_flipped_mask_matches_partition_across_workers()

    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} checks passed.")
    sys.exit(1 if n_fail else 0)
