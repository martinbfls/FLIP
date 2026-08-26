import random

import numpy as np
import torch

from modules.base_utils.datasets import MTTDataset
from modules.federated_generate_labels.utils import DEFAULT_EXPERT_CONFIG


def build_expert_pool(expert_starts, expert_opt_starts, pool_size):
    """P3 (checkpoint-pool stability fix): preloads `pool_size` distinct (params, optimizer-
    state) checkpoint pairs into RAM (float32, CPU) once, to be drawn from uniformly at random
    per outer training step -- instead of a single checkpoint indexed sequentially by the step,
    which conditioned every step's gradient on just one point of the expert trajectory.

    `expert_starts`/`expert_opt_starts` (parallel path lists, as returned by extract_experts /
    extract_experts_biased) may contain duplicate (params_path, opt_path) pairs -- e.g. from
    independent random draws colliding -- so the pool is sampled from the DISTINCT pairs only.
    If `pool_size` exceeds the number of distinct pairs available, uses all of them instead
    (warns).

    Returns (pool, pool_size): `pool` is a list of (params_state_dict, opt_state_dict) tuples,
    both CPU/float32 tensors; `pool_size` is the (possibly clamped) actual pool size.
    """
    all_pairs = list(dict.fromkeys(zip(expert_starts, expert_opt_starts)))
    if pool_size > len(all_pairs):
        print(
            f"WARNING: pool_size={pool_size} > {len(all_pairs)} distinct available expert "
            f"checkpoints for this run -- using all {len(all_pairs)} instead."
        )
        pool_size = len(all_pairs)
    pool_paths = random.sample(all_pairs, pool_size)
    pool = [
        (
            {k: v.float().cpu() for k, v in torch.load(p_path, map_location="cpu").items()},
            torch.load(o_path, map_location="cpu"),
        )
        for p_path, o_path in pool_paths
    ]
    return pool, pool_size


class TriggerMTTDataset(MTTDataset):
    '''
    Thin wrapper around `MTTDataset` (modules/base_utils/datasets.py) that additionally
    returns a boolean `is_poisoned` flag per example: True iff this draw's "train" branch
    came from the appended poison segment (original i >= len(self.distill)), i.e. iff it is
    one of the genuinely-triggered-and-relabeled examples, as opposed to a plain pass-through
    clean example that happens to share the same ConcatDataset.

    This is needed because MTTDataset's returned `idx` (the 5th tuple element) alone cannot
    distinguish the two cases: for poisoned draws, idx is reassigned to
    `poison_inds[i % len(distill)]`, but a *non*-reassigned i (from the clean segment) can
    coincidentally also be a member of poison_inds (that source-class image, drawn
    unpoisoned) -- checking `idx in poison_inds` from outside would give false positives.

    Constructed by re-wrapping an existing MTTDataset's constituent objects (train, distill,
    poison_inds, transform, n_classes) -- e.g. the one returned by
    `modules.base_utils.datasets.get_matching_datasets` -- rather than duplicating its
    dataset-construction logic.
    '''

    def __getitem__(self, i: int):
        is_poisoned = i >= len(self.distill)
        train_x, train_oh, distill_x, distill_oh, idx = super().__getitem__(i)
        return train_x, train_oh, distill_x, distill_oh, idx, is_poisoned

    @classmethod
    def from_mtt_dataset(cls, mtt_dataset):
        return cls(
            mtt_dataset.train, mtt_dataset.distill, mtt_dataset.poison_inds,
            mtt_dataset.transform, mtt_dataset.n_classes,
        )


def _sample_trajectory_index(n_traj, alpha_ckpt):
    '''
    Single draw from the SAME exponential-bias distribution
    `federated_optimizing_trigger.utils.sample_checkpoints` uses (probability of index k
    proportional to exp(-alpha_ckpt*k), k=0..n_traj-1 -- biased toward EARLY checkpoints for
    the repo's convention alpha_ckpt>0). Duplicated verbatim from
    `federated_generate_labels_trigger_joint.utils` (correction F, checkpoint_sampling='biased'
    support) rather than imported cross-module, so the two direct-family modules stay
    independently self-contained (see their docstrings' "must keep working side by side").
    '''
    ks = torch.arange(0, n_traj, dtype=torch.float)
    probs = torch.exp(-alpha_ckpt * ks)
    probs = probs / probs.sum()
    return int(torch.multinomial(probs, 1).item())


def extract_experts_biased(expert_config, expert_path, iterations, alpha_ckpt, expert_opt_path=None):
    '''
    Like `federated_generate_labels.utils.extract_experts`, but draws the "how far into this
    expert's own training run" trajectory index via the SAME exponentially-biased distribution
    federated_optimizing_trigger_policy's `sample_checkpoints` uses (controlled by the SAME
    `alpha_ckpt` parameter), instead of extract_experts's `np.random.randint(min, max)`
    (uniform). Used when this module's `checkpoint_sampling` config is 'biased' (default
    'uniform' here -- see run_module.py); see federated_generate_labels_trigger_joint.utils's
    identical copy, whose module defaults checkpoint_sampling to 'biased' instead.

    Args, returns: identical to extract_experts.
    '''
    config = {**DEFAULT_EXPERT_CONFIG, **expert_config}
    n_traj = config['max'] - config['min']
    expert_starts, expert_opt_starts = [], []

    for _ in range(iterations):
        for s in config['trajectories']:
            expert = np.random.randint(config['experts'])
            trajectory = config['min'] + _sample_trajectory_index(n_traj, alpha_ckpt) + 1
            expert_starts.append(expert_path.format(expert, trajectory, str(s)))
            if expert_path:
                expert_opt_starts.append(expert_opt_path.format(expert, trajectory, str(s)))
    return expert_starts, expert_opt_starts
