import numpy as np


def materialize_policy_flips(u, pairs, n_train, labels, n_classes, seed=0):
    '''
    Turns the continuous attack policy u (one weight per ordered class pair (y, c), same
    `pairs` ordering as `compute_expected_flip_gradients`) into a concrete set of per-example
    label flips: for each pair (y, c), n_{y,c} = round(u_{y,c} * n_train) examples of true
    class y are drawn (without replacement, seeded) and reassigned to class c.

    Draws for different target classes c of the SAME source class y are taken from disjoint,
    pre-shuffled pools of that class's examples (one seeded shuffle per class, consumed via a
    cursor across all of that class's (y, .) pairs) -- so no example is ever flipped to two
    different targets. If sum_c n_{y,c} exceeds the number of available class-y examples, the
    later pairs (in `pairs` order) are silently clipped and a warning is printed, mirroring
    federated_optimizing_trigger.utils.get_poison_dataset's lambda_overflow="clip" behavior.

    Args:
        u: (P,) array-like of policy weights, u_p >= 0, sum(u) <= beta.
        pairs: list of P (y, c) int tuples, y != c -- same ordering as u.
        n_train: total training-set size (u_{y,c} is a FRACTION OF THE WHOLE shard, not of
            class y alone -- see federated_optimizing_trigger.utils's
            `_build_global_budget_constraint` docstring).
        labels: (n_train,) int array of true labels.
        n_classes: number of classes.
        seed: RNG seed for the per-class shuffles.

    Returns:
        idx_flipped: 1D int64 array of flipped example indices (disjoint, no duplicates).
        targets:     1D int64 array of the same length, the new label for each flipped index.
    '''
    rng = np.random.default_rng(seed)
    idx_by_class = {y: np.where(labels == y)[0].copy() for y in range(n_classes)}
    for y in idx_by_class:
        rng.shuffle(idx_by_class[y])
    cursor = {y: 0 for y in range(n_classes)}

    idx_chunks, target_chunks = [], []
    for (y, c), u_yc in zip(pairs, u):
        n_yc = int(round(float(u_yc) * n_train))
        if n_yc <= 0:
            continue

        pool = idx_by_class[y]
        start = cursor[y]
        end = min(start + n_yc, len(pool))
        chosen = pool[start:end]
        if end - start < n_yc:
            print(
                f"[materialize_policy_flips] WARNING: pair (y={y}, c={c}) requested "
                f"{n_yc} flips but only {end - start} unused class-{y} examples remain; "
                "clipping."
            )
        cursor[y] = end

        idx_chunks.append(chosen)
        target_chunks.append(np.full(len(chosen), c, dtype=np.int64))

    if not idx_chunks:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    idx_flipped = np.concatenate(idx_chunks).astype(np.int64)
    targets = np.concatenate(target_chunks).astype(np.int64)
    return idx_flipped, targets
