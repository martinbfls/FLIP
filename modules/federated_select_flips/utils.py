import numpy as np


def partition_across_workers(N, idx_flipped, idx_clean, labels_final, num_honests, num_poisoned,
                              seed=0, shuffle_clean=True):
    '''
    Splits the N training examples (with their final, possibly-flipped labels) across
    num_honests + num_poisoned federated workers, so that every flipped index lands on a
    poisoned worker: idx_flipped is split as evenly as possible across the num_poisoned
    poisoned workers, and each worker's remaining slots are filled from idx_clean (shared,
    seeded shuffle across all workers -- honest and poisoned alike).

    Args:
        N:            total number of training examples.
        idx_flipped:  1D int array of flipped indices (assigned only to poisoned workers).
        idx_clean:    1D int array of the remaining (unflipped) indices.
        labels_final: (N, n_classes) array -- true.npy with flips applied at idx_flipped.
        num_honests, num_poisoned: worker counts; shard sizes are N split as evenly as
            possible (base = N // num_workers, first `N % num_workers` workers get one extra).
        seed: RNG seed used when shuffle_clean=True.
        shuffle_clean: if True (default), shuffles a copy of idx_clean (seeded) before
            distributing it across workers. Pass False if the caller has already shuffled
            (or otherwise deliberately ordered) idx_clean itself -- e.g. to save the exact
            shuffled array to disk before partitioning it.

    Returns:
        worker_indices: list of length num_honests + num_poisoned, each a 1D int array.
        worker_labels:  list of the same length, each labels_final[worker_indices[w]].

    Raises ValueError if a poisoned worker's shard is smaller than its assigned share of
    idx_flipped (num_poisoned too small / budget too large for the given worker split).
    '''
    num_workers = num_honests + num_poisoned

    if shuffle_clean:
        idx_clean = idx_clean.copy()
        rng = np.random.default_rng(seed)
        rng.shuffle(idx_clean)

    base = N // num_workers
    remainder = N % num_workers
    sizes = np.array(
        [base + (w < remainder) for w in range(num_workers)], dtype=int
    )

    flipped_split = np.array_split(idx_flipped, num_poisoned)

    worker_indices = []
    worker_labels = []

    clean_ptr = 0

    for w in range(num_honests):
        sz = sizes[w]
        sel = idx_clean[clean_ptr: clean_ptr + sz]
        clean_ptr += sz

        worker_indices.append(sel)
        worker_labels.append(labels_final[sel])

    for p in range(num_poisoned):
        w = num_honests + p
        sz = sizes[w]

        flipped_p = flipped_split[p]
        remaining = sz - len(flipped_p)
        if remaining < 0:
            raise ValueError(f"Too many flipped samples for poisoned worker {w}")

        sel_clean = idx_clean[clean_ptr: clean_ptr + remaining]
        clean_ptr += remaining

        sel = np.concatenate([sel_clean, flipped_p])

        worker_indices.append(sel)
        worker_labels.append(labels_final[sel])

    return worker_indices, worker_labels
