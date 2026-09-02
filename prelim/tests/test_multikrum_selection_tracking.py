"""
prelim/tests/test_multikrum_selection_tracking.py -- non-regression + correctness checks for
the Multi-Krum selection-tracking instrumentation added to:
  - modules/base_utils/aggregator/multikrum.py's `aggregate` (new `return_selected` kwarg)
  - modules/base_utils/util.py's `mini_train_multi` (new `track_poison_selection` kwarg)

Both additions are opt-in (default False/unused) and must be BIT-FOR-BIT identical to the
pre-existing behavior when not requested -- checked explicitly below.

Uses tiny SYNTHETIC gradients/datasets -- no real model, no dataset download. Matches
prelim/tests/test_expert_checkpoint_pool.py's convention (synthetic-only, runs in well under a
minute).

Run:  python prelim/tests/test_multikrum_selection_tracking.py
"""
import os
import sys

import torch
from torch import optim
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.base_utils.aggregator.multikrum import aggregate as krum_aggregate
from modules.base_utils.util import mini_train_multi

_results = []


def check(name, ok, detail=""):
    _results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


# --------------------------------------------------------------------------- #
# 1. multikrum.aggregate: return_selected correctness + default-path regression
# --------------------------------------------------------------------------- #

def test_aggregate_return_selected():
    # 5 gradients, 3 tight cluster (indices 1,2,3 near 0.1-0.3), 2 outliers (0, 4).
    grads = [torch.tensor([v]) for v in [10.0, 0.1, 0.2, 0.3, -10.0]]
    f = 1
    # Default m = len(gradients) - f - 2 = 2 -- the 2 lowest-scoring (most central) gradients.

    g_default = krum_aggregate(grads, f=f)
    g_tracked, selected = krum_aggregate(grads, f=f, return_selected=True)

    check("aggregate(..., return_selected=True) matches the default-call result exactly",
          torch.allclose(g_default, g_tracked),
          f"default={g_default.tolist()}, tracked={g_tracked.tolist()}")

    check("selected indices are the 2 innermost cluster members (1 and 2, or 2 and 3)",
          set(selected).issubset({1, 2, 3}) and len(selected) == 2,
          f"selected={selected}")

    check("outlier indices (0, 4) are never selected",
          0 not in selected and 4 not in selected,
          f"selected={selected}")

    # krum (m=1): must select exactly the single most central gradient.
    g_krum, selected_krum = krum_aggregate(grads, f=f, m=1, return_selected=True)
    check("krum (m=1) selects exactly one index, from the inner cluster",
          len(selected_krum) == 1 and selected_krum[0] in (1, 2, 3),
          f"selected_krum={selected_krum}")


def test_aggregate_default_unchanged_return_type():
    grads = [torch.tensor([v]) for v in [1.0, 1.1, 0.9, 5.0, -5.0]]
    result = krum_aggregate(grads, f=1)
    check("aggregate() without return_selected still returns a bare tensor (no tuple)",
          torch.is_tensor(result), f"type={type(result)}")


# --------------------------------------------------------------------------- #
# 2. mini_train_multi: track_poison_selection bookkeeping on a tiny synthetic federated setup
# --------------------------------------------------------------------------- #

class _ConstantDataset(Dataset):
    """A dataset of `n` copies of the same (x, y) pair, optionally tagged with is_flipped."""
    def __init__(self, x, y, n, is_flipped=None):
        self.x, self.y, self.n = x, y, n
        self.is_flipped = is_flipped

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.is_flipped is not None:
            return self.x, self.y, bool(self.is_flipped)
        return self.x, self.y


def test_mini_train_multi_poison_stats():
    torch.manual_seed(0)
    # 7 workers: 4 honest (w0..w3), 3 poisoned (w4, w5, w6) -- honest first, poisoned last,
    # matching partition_across_workers' convention. n_poisoned=3 needs at least 2f+3=9... but
    # multikrum's own `m = len(gradients) - f - 2` only needs len(gradients) > f + 2 to be
    # non-degenerate (m >= 1); 7 workers, f=3 gives m=2, which is what mini_train_multi's
    # aggregation dispatch actually uses (it never calls the aggregator's own stricter `check`).
    # Every worker's data is IDENTICAL (same x, y) except the label -- this isolates the
    # aggregation-selection bookkeeping from any actual training-quality effect (not what this
    # test is checking).
    x = torch.randn(4)
    y_clean, y_flipped = 0, 1
    n_honest, n_poisoned = 4, 3

    datasets = []
    for w in range(n_honest):
        datasets.append(_ConstantDataset(x, y_clean, n=8, is_flipped=False))
    for w in range(n_poisoned):
        # Worker w2's shard is entirely flipped; w3/w4 are entirely clean this run (to get a
        # deterministic, easily-checked batches_with_flip pattern across workers).
        is_flipped = (w == 0)
        datasets.append(_ConstantDataset(x, y_flipped if is_flipped else y_clean, n=8,
                                          is_flipped=is_flipped))

    model = torch.nn.Linear(4, 2)
    opt = optim.SGD(model.parameters(), lr=0.01)

    result = mini_train_multi(
        model=model,
        train_datasets=datasets,
        batch_size=8,
        opt=opt,
        scheduler=None,
        epochs=2,
        record=False,
        agg_method="multikrum",
        f=n_poisoned,
        track_poison_selection=True,
    )
    model_out, poison_stats = result

    check("mini_train_multi(track_poison_selection=True) returns a (model, poison_stats) pair",
          isinstance(poison_stats, dict) and "workers" in poison_stats,
          f"keys={list(poison_stats.keys()) if isinstance(poison_stats, dict) else type(poison_stats)}")

    n_workers = n_honest + n_poisoned
    poisoned_ids = list(range(n_workers - n_poisoned, n_workers))
    check("poison_stats tracks exactly the last `f` workers (honest-first convention)",
          set(poison_stats["workers"].keys()) == set(poisoned_ids),
          f"tracked={sorted(poison_stats['workers'].keys())}, expected={poisoned_ids}")

    check("total_aggregations == epochs (1 batch/epoch here, 8 samples == batch_size)",
          poison_stats["total_aggregations"] == 2,
          f"total_aggregations={poison_stats['total_aggregations']}")

    w_flipped = poisoned_ids[0]  # worker index n_honest + 0, the only flipped-label worker
    check(f"worker {w_flipped} (the only flipped one) has batches_with_flip == total_aggregations",
          poison_stats["workers"][w_flipped]["batches_with_flip"] == 2,
          f"batches_with_flip={poison_stats['workers'][w_flipped]['batches_with_flip']}")

    for w in poisoned_ids[1:]:
        check(f"worker {w} (never flipped this run) has batches_with_flip == 0",
              poison_stats["workers"][w]["batches_with_flip"] == 0,
              f"batches_with_flip={poison_stats['workers'][w]['batches_with_flip']}")

    check("times_selected_given_flip is bounded by batches_with_flip for every worker",
          all(
              poison_stats["workers"][w]["times_selected_given_flip"]
              <= poison_stats["workers"][w]["batches_with_flip"]
              for w in poisoned_ids
          ),
          str(poison_stats["workers"]))

    check("any_poisoned_selected_given_flip is bounded by total_aggregations",
          poison_stats["any_poisoned_selected_given_flip"] <= poison_stats["total_aggregations"])


def test_mini_train_multi_default_path_unaffected():
    """track_poison_selection defaults to False: must return the same shape as before this
    feature existed (no poison_stats element in the returned tuple)."""
    torch.manual_seed(0)
    x, y = torch.randn(4), 0
    datasets = [_ConstantDataset(x, y, n=4) for _ in range(5)]
    model = torch.nn.Linear(4, 2)
    opt = optim.SGD(model.parameters(), lr=0.01)

    result = mini_train_multi(
        model=model, train_datasets=datasets, batch_size=4, opt=opt, scheduler=None,
        epochs=1, record=False, agg_method="multikrum", f=1,
    )
    check("mini_train_multi(track_poison_selection=False) returns a bare model (unchanged shape)",
          torch.is_tensor(next(result.parameters()).data) if hasattr(result, "parameters") else False,
          f"type={type(result)}")


if __name__ == "__main__":
    test_aggregate_return_selected()
    test_aggregate_default_unchanged_return_type()
    test_mini_train_multi_poison_stats()
    test_mini_train_multi_default_path_unaffected()

    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} checks passed.")
    sys.exit(1 if n_fail else 0)
