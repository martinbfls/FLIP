"""
prelim/check_policy_consistency.py -- the session's one short numerical verification (see
prelim/SPEC.md-adjacent conventions: no writes under out/checkpoints/, everything else stays
in prelim/artifacts/). No long training: loads an existing checkpoint if one is given/found,
otherwise trains one epoch on a small subset in a temporary directory.

Checks correction A (pi_y double-counting) and B3b (missing gamma in the aggregate objective)
SEPARATELY, at the two scopes the fixes operate at:

  Level 1 (LOCAL, one shard):    g_emp_shard - grad_c_shard   vs H @ u        (slope -> 1)
  Level 2 (AGGREGATE, n_b workers, n_p corrupted): b_k_emp - grad_c_agg  vs  gamma * H @ u
                                                                                (slope -> 1)

H[:, (y,c)] = g_{y,c} - g_{y,y} is compute_expected_flip_gradients's G with its pi_y factor
divided back out (correction A). gamma = n_p / n_b (correction B3b).

For each level, also reports the PRE-correction slope using the raw (unmodified)
compute_expected_flip_gradients output G_k directly as the predictor -- this is exactly what
the code used before today's fixes -- to quantify the two bugs this session fixes:
  - Level 1 pre-fix predictor: G_k @ u              (expected slope ~= 1/pi_y, ~10 on CIFAR-10)
  - Level 2 pre-fix predictor: G_k @ u              (expected slope ~= gamma/pi_y)

Usage:  python prelim/check_policy_consistency.py [--checkpoint PATH]
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if not torch.cuda.is_available():
    # load_model() calls .cuda() unconditionally; on a CPU-only machine that raises. Since
    # torch.cuda.is_available() is already False here, redirecting .cuda() to .to("cpu") is a
    # no-op change in behavior (real CUDA would never have been used) -- only makes this
    # script runnable without a GPU.
    torch.nn.Module.cuda = lambda self, device=None: self.to("cpu")
    torch.Tensor.cuda = lambda self, device=None, non_blocking=False: self.to("cpu")

from modules.base_utils.util import load_model, mini_train, get_train_info
from modules.base_utils.datasets import load_dataset, get_n_classes, get_matching_datasets, pick_poisoner
from modules.federated_optimizing_trigger.utils import (
    get_class_conditional_samples,
    compute_expected_flip_gradients,
    compute_class_frequencies,
    compute_batch_gradients,
    raw_to_preprocess,
    get_raw_clean_dataset,
)

DATASET = "cifar"
MODEL_FLAG = "r32p"
DEVICE = "cpu"
SOURCE_LABEL = 9
TARGET_LABEL = 4
N_PER_CLASS_GBAR = 32     # samples/class used to estimate H (cheap, matches smoke-scale runs)
SHARD_SIZE = 600          # one corrupted worker's own shard, for the local-scope check
N_B, N_P = 6, 2           # aggregate deployment: n_b workers total, n_p corrupted
GAMMA = N_P / N_B
SEED = 0


def get_or_train_checkpoint(checkpoint_arg):
    if checkpoint_arg and Path(checkpoint_arg).exists():
        print(f"Using existing checkpoint: {checkpoint_arg}")
        return checkpoint_arg

    print("No checkpoint given/found -- training a tiny expert (1 epoch, 2000 examples) "
          "in a temporary directory...")
    tmp_dir = Path("prelim/artifacts/tmp_expert_ckpt")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    n_classes = get_n_classes(DATASET)
    poisoner = pick_poisoner("1xs", DATASET, TARGET_LABEL, delta=None)
    train_data, *_ = get_matching_datasets(
        DATASET, poisoner, SOURCE_LABEL, train_pct=2000 / 50000, big=False,
    )
    model = load_model(MODEL_FLAG, n_classes).to(DEVICE)
    batch_size, epochs, opt, sched = get_train_info(
        model.parameters(), "sgd", batch_size=128, epochs=1,
        optim_kwargs={"lr": 0.01, "momentum": 0.9}, scheduler_kwargs={},
    )
    mini_train(model=model, train_data=train_data, test_data=[], batch_size=batch_size,
               opt=opt, scheduler=sched, epochs=1)
    ckpt_path = tmp_dir / "tiny_expert.pth"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Trained tiny expert, saved to {ckpt_path}")
    return str(ckpt_path)


def flat_grad_on_batch(model, loss_fn, x_raw, y, dataset_flag=DATASET, model_flag=None):
    x = raw_to_preprocess(x_raw, dataset_flag=dataset_flag, model_flag=model_flag)
    grads, _ = compute_batch_gradients(model, loss_fn, (x, y), create_graph=False)
    return torch.cat([g.reshape(-1) for g in grads]).detach()


def realize_u_on_shard(u, pairs, shard_labels, n_classes, seed):
    """Local materialization of u (as a fraction of THIS shard's own class-y examples,
    gamma=1.0 relative to this pool) -- see materialize_policy_flips's docstring."""
    from modules.federated_policy_to_flips.utils import materialize_policy_flips
    idx_flipped, targets = materialize_policy_flips(
        u, pairs, len(shard_labels), shard_labels, n_classes, gamma=1.0, seed=seed,
    )
    labels_out = shard_labels.copy()
    labels_out[idx_flipped] = targets
    return labels_out


def slope_r2_cos(target, predictor):
    target = target.numpy().astype(np.float64)
    predictor = predictor.numpy().astype(np.float64) if torch.is_tensor(predictor) else predictor
    num = float(np.dot(target, predictor))
    den = float(np.dot(predictor, predictor))
    a = num / den if den > 0 else float("nan")
    resid = target - a * predictor
    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.dot(target - target.mean(), target - target.mean()))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    cos = num / (np.linalg.norm(target) * np.sqrt(den)) if den > 0 else float("nan")
    return a, r2, cos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    ckpt_path = get_or_train_checkpoint(args.checkpoint)

    n_classes = get_n_classes(DATASET)
    model = load_model(MODEL_FLAG, n_classes).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()

    print("Computing pi and class-conditional samples...")
    pi = compute_class_frequencies(DATASET, n_classes)
    class_samples_raw = get_class_conditional_samples(DATASET, n_classes, N_PER_CLASS_GBAR, DEVICE)
    classes_present = sorted(class_samples_raw.keys())
    pairs = [(y, c) for y in classes_present for c in range(n_classes) if c != y]

    G_k, _, pairs_check = compute_expected_flip_gradients(
        model, loss_fn, class_samples_raw, n_classes, pi,
        dataset_flag=DATASET, model_flag=MODEL_FLAG,
    )
    assert pairs_check == pairs
    scale = torch.tensor([1.0 / pi[y] for (y, c) in pairs], dtype=G_k.dtype)
    H = G_k * scale  # correction A only: pi_y divided back out

    raw_train = get_raw_clean_dataset(DATASET, train=True)
    labels_all = np.array([y for _, y in raw_train.dataset])

    def sample_shard(size, seed):
        idx = np.random.RandomState(seed).choice(len(raw_train), size=size, replace=False)
        xs = torch.stack([raw_train[i][0] for i in idx])
        ys = labels_all[idx].copy()
        return idx, xs, ys

    def three_us():
        # u is LOCAL: sum_c u[y,c] <= pi[y] (§B3a). Keep well inside pi[SOURCE_LABEL]'s cap so
        # the requested masses are actually realizable within one SHARD_SIZE shard without
        # clipping (materialize_policy_flips clips silently otherwise, which would break the
        # requested-u vs. realized-shift linearity this check is testing for -- see
        # prelim/SPEC.md's E1 "realised masses, not requested").
        budget = 0.6 * pi[SOURCE_LABEL]
        idx_94 = pairs.index((SOURCE_LABEL, TARGET_LABEL))
        u_concentrated = np.zeros(len(pairs)); u_concentrated[idx_94] = budget
        u_spread = np.zeros(len(pairs))
        source_pair_idx = [p for p, (y, c) in enumerate(pairs) if y == SOURCE_LABEL]
        u_spread[source_pair_idx] = budget / len(source_pair_idx)
        u_random = np.abs(np.random.RandomState(2).randn(len(pairs)))
        u_random = u_random / u_random.sum() * budget
        return {"concentrated_(9,4)": u_concentrated, "spread_source9": u_spread, "random": u_random}

    us = three_us()

    print("\n=== LEVEL 1 (local, one shard) ===")
    level1_rows = []
    for name, u_np in us.items():
        idx, x_shard, y_shard_clean = sample_shard(SHARD_SIZE, seed=10)
        grad_c_shard = flat_grad_on_batch(
            model, loss_fn, x_shard, torch.tensor(y_shard_clean, dtype=torch.long),
            model_flag=MODEL_FLAG,
        )
        y_shard_flipped = realize_u_on_shard(u_np, pairs, y_shard_clean, n_classes, seed=11)
        g_emp_shard = flat_grad_on_batch(
            model, loss_fn, x_shard, torch.tensor(y_shard_flipped, dtype=torch.long),
            model_flag=MODEL_FLAG,
        )
        target = g_emp_shard - grad_c_shard

        H_u = H @ torch.tensor(u_np, dtype=H.dtype)
        Gk_u = G_k @ torch.tensor(u_np, dtype=G_k.dtype)

        a_new, r2_new, cos_new = slope_r2_cos(target, H_u)
        a_old, r2_old, cos_old = slope_r2_cos(target, Gk_u)
        level1_rows.append((name, a_old, cos_old, a_new, r2_new, cos_new))
        print(f"  {name:22s} pre-fix(G_k@u): a={a_old:7.3f} cos={cos_old:6.3f}   "
              f"post-fix(H@u): a={a_new:7.3f} R2={r2_new:6.3f} cos={cos_new:6.3f}")

    print("\n=== LEVEL 2 (aggregate, n_b={} workers, n_p={} corrupted, gamma={:.4f}) ===".format(
        N_B, N_P, GAMMA))
    level2_rows = []
    for name, u_np in us.items():
        shard_grads_clean = []
        shard_grads_final = []
        for w in range(N_B):
            idx, x_w, y_w_clean = sample_shard(SHARD_SIZE, seed=100 + w)
            g_clean_w = flat_grad_on_batch(
                model, loss_fn, x_w, torch.tensor(y_w_clean, dtype=torch.long),
                model_flag=MODEL_FLAG,
            )
            shard_grads_clean.append(g_clean_w)
            if w < N_P:
                y_w_final = realize_u_on_shard(u_np, pairs, y_w_clean, n_classes, seed=200 + w)
                g_final_w = flat_grad_on_batch(
                    model, loss_fn, x_w, torch.tensor(y_w_final, dtype=torch.long),
                    model_flag=MODEL_FLAG,
                )
            else:
                g_final_w = g_clean_w
            shard_grads_final.append(g_final_w)

        grad_c_agg = torch.stack(shard_grads_clean).mean(dim=0)
        b_k_emp = torch.stack(shard_grads_final).mean(dim=0)
        target = b_k_emp - grad_c_agg

        H_u = H @ torch.tensor(u_np, dtype=H.dtype)
        Gk_u = G_k @ torch.tensor(u_np, dtype=G_k.dtype)
        gamma_H_u = GAMMA * H_u

        a_new, r2_new, cos_new = slope_r2_cos(target, gamma_H_u)
        a_old, r2_old, cos_old = slope_r2_cos(target, Gk_u)
        level2_rows.append((name, a_old, cos_old, a_new, r2_new, cos_new))
        print(f"  {name:22s} pre-fix(G_k@u): a={a_old:7.3f} cos={cos_old:6.3f}   "
              f"post-fix(gamma*H@u): a={a_new:7.3f} R2={r2_new:6.3f} cos={cos_new:6.3f}")

    print("\n=== Success criteria (concentrated_(9,4), post-fix) ===")
    name1, _, _, a1, r2_1, cos1 = level1_rows[0]
    name2, _, _, a2, r2_2, cos2 = level2_rows[0]
    ok1 = (0.9 <= a1 <= 1.1) and (cos1 > 0.99)
    ok2 = (0.9 <= a2 <= 1.1) and (cos2 > 0.99)
    print(f"  Level 1: a={a1:.3f} in [0.9,1.1]: {ok1 if a1 else False}  cos={cos1:.4f} > 0.99")
    print(f"  Level 2: a={a2:.3f} in [0.9,1.1]  cos={cos2:.4f} > 0.99")
    print(f"  Level 1 PASS: {ok1}   Level 2 PASS: {ok2}")


if __name__ == "__main__":
    main()
