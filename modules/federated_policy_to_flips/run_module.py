"""
Theory: no direct correspondence to a single result of docs/theory/threat_model.tex -- this is
the pipeline step BETWEEN the theory's decision variable (u, the LOCAL flip-mass policy solved
for by federated_optimizing_trigger_policy against eq:P) and the concrete per-example label
flips a real federated training run needs, via `materialize_policy_flips` (rem:units' "Label
counts", gamma*n*u^i_{y,z}).

Materializes the continuous attack policy u (produced by federated_optimizing_trigger_policy)
into concrete per-worker label flips, in the same output format as federated_select_flips --
so that federated_train_user can be reused unchanged to train and evaluate the victim.

Chain position: consumes the (u, pairs, beta, n_train, num_honests, num_poisoned, gamma,
source_label, target_label) policy artifact (.npz) written by
federated_optimizing_trigger_policy/run_module.py's `run()`. Produces the same on-disk layout
federated_select_flips does (true.npy, worker{w}/{budget}_labels.npy,
worker{w}/{budget}_indices.npy), consumed downstream, unmodified, by federated_train_user.

Scope: `u` (from the .npz) is LOCAL, exactly as federated_optimizing_trigger_policy produced it
-- see that module's header docstring for the full local/aggregate distinction (rem:units). This
module does not itself choose or convert the scope; it only threads the SAME `gamma` (also read
from the .npz, cross-checked below) through `materialize_policy_flips`'s local-units formula.
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
        # A silent n_train mismatch here reproduces exactly the kind of scale bug the u
        # convention fix (federated_optimizing_trigger_policy's G_obj rescaling) eliminates:
        # round(u * gamma * n_train) would silently use the wrong n_train, and the resulting
        # flip count would no longer match the policy's own beta/gamma. Fail loudly instead.
        raise ValueError(
            f"Dataset train size ({N}) differs from the policy's n_train ({n_train_policy}) "
            "-- the policy was optimized against a differently-sized training set. Flip "
            "counts (round(u * gamma * n_train)) would silently be computed against the "
            "wrong n_train. Re-run federated_optimizing_trigger_policy against this exact "
            "dataset, or point policy_path at a policy produced with train_pct matching this "
            "dataset's size."
        )

    # Theory: def:budget -- gamma = n_p/n_b (this module's own num_poisoned/num_honests), used
    # below by materialize_policy_flips exactly as rem:units' local-units label-count formula
    # (gamma*n*u^i_{y,z}) prescribes.
    gamma = num_poisoned / (num_poisoned + num_honests)

    # Cross-check against what the policy was ACTUALLY optimized against, instead of silently
    # trusting that this module's own num_honests/num_poisoned happen to match
    # federated_optimizing_trigger_policy's -- both sides independently recompute
    # gamma = num_poisoned/(num_poisoned+num_honests) from their own config, and nothing
    # previously verified the two configs agreed. Older policies (produced before this
    # correction) don't carry these fields -- skip with a clear warning rather than crash on
    # a plain KeyError.
    policy_npz_keys = set(npz.files)
    if {"num_honests", "num_poisoned", "gamma"} <= policy_npz_keys:
        policy_num_honests = int(npz["num_honests"])
        policy_num_poisoned = int(npz["num_poisoned"])
        policy_gamma = float(npz["gamma"])
        if (policy_num_honests, policy_num_poisoned) != (num_honests, num_poisoned):
            raise ValueError(
                f"num_honests/num_poisoned mismatch: this config has "
                f"({num_honests}, {num_poisoned}), but the policy at {policy_path} was "
                f"optimized with ({policy_num_honests}, {policy_num_poisoned}) -- gamma "
                f"({gamma:.6f} here vs {policy_gamma:.6f} there) would silently diverge, "
                "changing round(u*gamma*n_train). Set num_honests/num_poisoned to match the "
                "federated_optimizing_trigger_policy run that produced this policy."
            )
    else:
        print(
            "WARNING: this policy .npz predates the num_honests/num_poisoned/gamma "
            "cross-check (produced before this correction) -- skipping it. Verify by hand "
            "that num_honests/num_poisoned here match the run that produced this policy."
        )

    true = np.zeros((N, n_classes), dtype=np.float32)
    true[np.arange(N), labels] = 1.0

    print("Materializing flips from policy...")
    idx_flipped, targets = materialize_policy_flips(
        u, pairs, N, labels, n_classes, gamma, seed=seed,
    )

    labels_final = true.copy()
    labels_final[idx_flipped] = 0.0
    labels_final[idx_flipped, targets] = one_hot_temp

    idx_flipped = np.unique(idx_flipped)
    idx_clean = np.setdiff1d(np.arange(N), idx_flipped)

    budget = int(len(idx_flipped))
    print(
        f"Materialized {budget} flips from policy (beta={beta:.6f}, gamma={gamma:.6f}, "
        f"budget/N={budget / N:.6f})."
    )

    # Budget-saturation check: an optimized policy is expected to (nearly) saturate its own
    # local budget (sum(u) == beta -- `project_policy_budget`'s projection pushes toward this
    # whenever the target v is not already reachable, the observed regime), but a genuinely
    # under-converged policy (e.g. a short debug run with few n_steps) legitimately will not --
    # that is a signal about convergence, not necessarily a scaling bug. Checked directly on
    # ||u||_1 vs beta (the policy's own budget) rather than on the materialized flip count
    # against a derived flip_budget_expected, since the latter conflates two different failure
    # modes (under-convergence vs an actual u-convention/gamma scaling drift) behind the same
    # threshold.
    #
    # strict_budget_check=False (default, exploration mode): only ||u||_1 == 0 (a policy that
    # never moved at all -- clearly a bug, not a convergence artifact) raises. Anything under
    # budget_saturation_tol * beta only warns. Once a good n_steps has been found for a given
    # config, set strict_budget_check=true to turn the warning back into a hard assertion.
    u_l1 = float(u.sum())
    budget_saturation_tol = args.get("budget_saturation_tol", 0.5)
    strict_budget_check = args.get("strict_budget_check", False)
    if u_l1 == 0.0:
        raise AssertionError(
            f"||u||_1=0 -- the policy never moved off its zero initialization (beta={beta:.6f})."
            " This is not a convergence artifact: check lr_policy, n_steps>0, and that "
            "optimizer_policy.step() actually ran."
        )
    elif u_l1 < budget_saturation_tol * beta:
        msg = (
            f"||u||_1={u_l1:.6f} < budget_saturation_tol({budget_saturation_tol}) * "
            f"beta({beta:.6f}) -- the policy has not saturated its local budget (materialized "
            f"{budget} flips, beta={beta:.6f}, gamma={gamma:.6f}). Likely under-convergence "
            "(too few n_steps) rather than a scaling bug -- check beta_used in the optimizer's "
            "metrics_log_path. Set strict_budget_check=true in the config to turn this into a "
            "hard failure once a sufficient n_steps has been found."
        )
        if strict_budget_check:
            raise AssertionError(msg)
        print(f"WARNING: {msg}")

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
