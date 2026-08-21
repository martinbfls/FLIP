"""
Materializes the continuous attack policy u (produced by federated_optimizing_trigger_policy)
into concrete per-worker label flips, in the same output format as federated_select_flips --
so that federated_train_user can be reused unchanged to train and evaluate the victim.
"""

from pathlib import Path
import sys
import os

import numpy as np

from modules.base_utils.datasets import load_dataset, get_n_classes
from modules.base_utils.util import extract_toml, slurmify_path
from modules.federated_generate_labels.utils import DEFAULT_ATTACK_CONFIG
from modules.federated_select_flips.utils import partition_across_workers
from modules.federated_policy_to_flips.utils import materialize_policy_flips


def run(experiment_name, module_name, **kwargs):
    """
    Loads the (u, pairs, beta, n_train) policy artifact produced by
    federated_optimizing_trigger_policy, realizes it into a concrete set of per-example label
    flips (`materialize_policy_flips`), and partitions the result across workers using the
    same `partition_across_workers` helper federated_select_flips uses -- producing an
    identical output layout (true.npy, worker{w}/{budget}_labels.npy,
    worker{w}/{budget}_indices.npy) so federated_train_user can consume it unmodified.

    :param experiment_name: Name of the experiment in configuration.
    :param module_name: Name of the module in configuration.
    :param kwargs: Additional arguments (such as slurm id).
    """
    slurm_id = kwargs.get("slurm_id", None)
    args = extract_toml(experiment_name, module_name)

    policy_path = slurmify_path(args["policy_path"], slurm_id)
    dataset_flag = args["dataset"]
    num_honests = args.get("num_honests", 2)
    num_poisoned = args.get("num_poisoned", 1)
    output_dir = Path(slurmify_path(args["output_dir"], slurm_id))
    one_hot_temp = args.get("one_hot_temp", DEFAULT_ATTACK_CONFIG["one_hot_temp"])
    seed = args.get("seed", 0)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading policy from {policy_path}...")
    npz = np.load(policy_path)
    u = npz["u"]
    pairs = list(zip(npz["pairs_y"].tolist(), npz["pairs_c"].tolist()))
    beta = float(npz["beta"])
    n_train_policy = int(npz["n_train"])

    n_classes = get_n_classes(dataset_flag)
    train_data = load_dataset(dataset_flag, train=True)
    labels = np.array([y for _, y in train_data])
    N = len(labels)
    if N != n_train_policy:
        print(
            f"WARNING: dataset train size ({N}) differs from the policy's n_train "
            f"({n_train_policy}) -- the policy was optimized against a differently-sized "
            "training set; flip counts (round(u * n_train)) will be computed against N."
        )

    true = np.zeros((N, n_classes), dtype=np.float32)
    true[np.arange(N), labels] = 1.0

    print("Materializing flips from policy...")
    idx_flipped, targets = materialize_policy_flips(u, pairs, N, labels, n_classes, seed=seed)

    labels_final = true.copy()
    labels_final[idx_flipped] = 0.0
    labels_final[idx_flipped, targets] = one_hot_temp

    idx_flipped = np.unique(idx_flipped)
    idx_clean = np.setdiff1d(np.arange(N), idx_flipped)

    budget = int(len(idx_flipped))
    print(
        f"Materialized {budget} flips from policy (beta={beta:.6f}, "
        f"budget/N={budget / N:.6f})."
    )

    np.save(output_dir / "true.npy", true)
    np.save(output_dir / f"{budget}_idx_flipped.npy", idx_flipped)
    np.save(output_dir / f"{budget}_idx_clean.npy", idx_clean)

    print("Partitioning across workers...")
    worker_indices, worker_labels = partition_across_workers(
        N, idx_flipped, idx_clean, labels_final, num_honests, num_poisoned, seed=seed,
    )

    for w in range(num_honests + num_poisoned):
        os.makedirs(output_dir / f"worker{w}", exist_ok=True)
        np.save(output_dir / f"worker{w}/{budget}_labels.npy", worker_labels[w])
        np.save(output_dir / f"worker{w}/{budget}_indices.npy", worker_indices[w])

    print(f"Saved materialized flips to {output_dir} (budget={budget}).")


if __name__ == "__main__":
    experiment_name, module_name = sys.argv[1], sys.argv[2]
    run(experiment_name, module_name)
