"""
prelim/sweep.py -- session 2: grid sweep over MODELS x CHECKPOINTS (E1-E3),
with CSV cache/resume and crash-tolerant failure logging.

Scope decision (read before editing the grid below): E1-E3's own protocol,
as designed in session 1 (prelim.ipynb), already scopes SEEDS and BETAS
deliberately -- not every axis is exercised by every experiment:
  - E1's main 6-config table runs at SEEDS[0], E1_BETA=0.10, all checkpoints.
  - E1's seed-robustness check runs at checkpoint="end", config (a) only,
    across all SEEDS.
  - E1's batch-size/SNR sweep runs at checkpoint="end", SEEDS[0], across
    all BETAS x BATCH_SIZES_SNR.
  - E2 runs across all checkpoints x all BETAS (seed-independent: v, Gbar,
    grad_c don't depend on shard sampling).
  - E3 runs across all checkpoints at E1_BETA, plus one "shared ubar" fit.
run_all() generalizes this SAME scoping across MODELS (this session's new
axis) rather than exploding every experiment into a full MODELS x SEEDS x
CHECKPOINTS x BETAS x TRANSFORMS cartesian product -- see run_meta.json's
"gbar_seconds" entries after a run: class_conditional_shifts costs ~250s per
call on r32p/CPU on this machine, so a naive cartesian product would put the
cnn config outside any reasonable session budget for no added signal (Gbar,
grad_c, grad_bd do not depend on seed at all -- only shard sampling, flip
realization and minibatch draws do, and those are already covered by the
scoping above). grad_bd is computed for every transform in TRANSFORMS while a
checkpoint's Gbar is in memory, so E4/E5 never repay the ~250s cost later;
E1-E3's own metrics only ever use "identity".

E4/E5/E7 add a second scoping decision of the same kind, and for the same
reason. SPEC section 8/E4 says "at a fixed checkpoint", so the three
deployment blocks all run at checkpoint="end" only; on top of that E4 fixes
beta = E1_BETA and sweeps the transform, E5 fixes the transform and sweeps
(beta, tau), and E7 fixes both and sweeps n_p. Each of those blocks replays
~200 aggregation rounds over 10 workers with real minibatch gradients, so a
full MODELS x SEEDS x CHECKPOINTS x BETAS x TRANSFORMS x N_P product would be
several orders of magnitude past SPEC section 1's ten-minute budget for no
extra signal: the axes each block does not sweep are exactly the ones its own
hypothesis holds fixed. `python sweep.py --dry-run` prints the resulting cell
count and a cost estimate without computing anything.

Two environment workarounds live here (not in modules/, which is read-only):
  - macOS defaults to the "spawn" multiprocessing start method, under which
    modules.base_utils.datasets.make_dataloader's hard-coded num_workers=4
    fails to pickle a module-level lambda in that file. "fork" (available on
    macOS, just not default) sidesteps it entirely -- this is a launcher-side
    setting, not a modules/ change.
  - MPS does not support float64, and
    modules.federated_optimizing_trigger.utils.compute_expected_flip_gradients
    unconditionally does `(G.T @ G).to(torch.float64)`. TRAIN_DEVICE (mps if
    available) is only used for the SGD loop in train_clean_checkpoints;
    every Gbar/grad_c/grad_bd/worker_gradient call runs on EVAL_DEVICE="cpu".
"""
import argparse
import hashlib
import itertools
import json
import math
import os
import subprocess
import sys
import time
import traceback
import multiprocessing as mp

try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    pass  # already set (e.g. re-import, or a caller set it first)

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prelim_lib as pl

os.chdir(pl._FLIP_ROOT)

ARTIFACT_DIR = os.path.join(pl._FLIP_ROOT, "prelim", "artifacts")
CKPT_DIR = os.path.join(ARTIFACT_DIR, "ckpt")
FIG_DIR = os.path.join(ARTIFACT_DIR, "figs")
CACHE_DIR = os.path.join(ARTIFACT_DIR, "cache")
METRICS_PATH = os.path.join(ARTIFACT_DIR, "metrics.csv")
FAILURES_PATH = os.path.join(ARTIFACT_DIR, "failures.log")
RUN_META_PATH = os.path.join(ARTIFACT_DIR, "run_meta.json")

for _d in (CKPT_DIR, FIG_DIR, CACHE_DIR):
    os.makedirs(_d, exist_ok=True)

# --------------------------------------------------------------------------#
# Grid -- edit here. AGGREGATORS/N_P beyond [3] are for E4-E7 (next session);
# unused by E1-E3.
# --------------------------------------------------------------------------#
MODELS      = ["linear", "cnn"]
SEEDS       = [0, 1, 2]
CHECKPOINTS = ["begin", "mid", "end"]     # "early/mid/late" du plan de session --
                                           # memes noms que train_clean_checkpoints
BETAS       = [0.01, 0.03, 0.10]
TRANSFORMS  = ["identity", "stripe"]
N_P         = [3]            # E7 etend a N_P_E7 = [2, 3, 5]
AGGREGATORS = ["mean", "cw_median", "trmean", "krum", "multikrum"]

# Fixed config shared by every cell this session (not swept).
DATASET_FLAG = "cifar"
N_CLASSES = 10
N_B, F = 10, 3
Y_SOURCE, Y_TARGET = 9, 4
N_PER_CLASS_CALIB = 512
E1_BETA = 0.10
BATCH_SIZES_SNR = [64, 256, 1024]
N_MB_DRAWS = 6
GAMMA = N_P[0] / N_B

TRAIN_DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
EVAL_DEVICE = torch.device("cpu")

MODEL_CFG = {
    "linear": dict(model_flag="linear", epochs=8, batch_size=256, max_train=None),
    "cnn":    dict(model_flag="r32p",  epochs=4, batch_size=128, max_train=10000),
}


# --------------------------------------------------------------------------#
# config_id / tidy-row bookkeeping
# --------------------------------------------------------------------------#
def _cid(model, seed, checkpoint, beta, transform, experiment, aggregator=None, n_p=None, tau=None):
    key = "|".join(str(x) for x in (model, seed, checkpoint, beta, transform, aggregator, n_p, tau, experiment))
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def _record(rows, done_pairs, model, seed, checkpoint, beta, transform, experiment, metric, value,
            aggregator=None, n_p=None, tau=None):
    cid = _cid(model, seed, checkpoint, beta, transform, experiment, aggregator, n_p, tau)
    if (cid, metric) in done_pairs:
        return
    rows.append(dict(config_id=cid, model=model, seed=seed, checkpoint=checkpoint, beta=beta,
                      transform=transform, aggregator=aggregator, n_p=n_p, tau=tau,
                      experiment=experiment, metric=metric, value=float(value)))
    done_pairs.add((cid, metric))


def _log_failure(model, seed, checkpoint, stage):
    with open(FAILURES_PATH, "a") as f:
        f.write(f"\n{'='*70}\n{time.strftime('%Y-%m-%d %H:%M:%S')} model={model} "
                f"seed={seed} checkpoint={checkpoint} stage={stage}\n")
        f.write(traceback.format_exc())
    print(f"[FAILED] model={model} seed={seed} checkpoint={checkpoint} stage={stage} "
          f"-- see {FAILURES_PATH}", file=sys.stderr)


# --------------------------------------------------------------------------#
# Training / checkpoint loading
# --------------------------------------------------------------------------#
def _train_or_load(model_name, cfg, seed, resume):
    tag = model_name
    paths = {k: os.path.join(CKPT_DIR, f"{tag}_{k}.pt") for k in ("begin", "mid", "end")}
    if resume and all(os.path.exists(p) for p in paths.values()):
        return paths
    torch.manual_seed(seed)
    model_init = pl.build_model(cfg["model_flag"], N_CLASSES, TRAIN_DEVICE)
    return pl.train_clean_checkpoints(
        model_init, DATASET_FLAG, N_CLASSES, TRAIN_DEVICE,
        epochs=cfg["epochs"], batch_size=cfg["batch_size"],
        ckpt_dir=CKPT_DIR, tag=tag, test_pct=0.1, seed=seed, max_train=cfg["max_train"],
    )


def _load_ckpt(model_name, cfg, ckpt_paths, checkpoint):
    m = pl.build_model(cfg["model_flag"], N_CLASSES, EVAL_DEVICE)
    m.load_state_dict(torch.load(ckpt_paths[checkpoint], map_location=EVAL_DEVICE))
    m.eval()
    return m


# --------------------------------------------------------------------------#
# Gbar cache (in-memory by default; optional float16 disk cache via --cache-gbar)
# --------------------------------------------------------------------------#
def _load_or_compute_gbar(model_name, checkpoint, m, class_samples_raw, pi, cache_gbar, timing):
    cache_path = os.path.join(CACHE_DIR, f"{model_name}_{checkpoint}_gbar.npz")
    if cache_gbar and os.path.exists(cache_path):
        d = np.load(cache_path, allow_pickle=True)
        Gbar = torch.from_numpy(d["Gbar"].astype(np.float32))
        grad_c = torch.from_numpy(d["grad_c"].astype(np.float32))
        pairs = [tuple(int(x) for x in p) for p in d["pairs"]]
        Q = d["Q"]
        col_index = {p: i for i, p in enumerate(pairs)}
        return Gbar, grad_c, pairs, col_index, Q
    t0 = time.time()
    Gbar, grad_c, pi_, pairs, col_index, Q = pl.class_conditional_shifts(
        m, class_samples_raw, DATASET_FLAG, N_CLASSES, EVAL_DEVICE,
        loss_fn=pl.clf_loss, model_flag=None)
    timing.setdefault("gbar_seconds", []).append(dict(model=model_name, checkpoint=checkpoint,
                                                        seconds=time.time() - t0))
    if cache_gbar:
        np.savez_compressed(cache_path, Gbar=Gbar.numpy().astype(np.float16),
                             grad_c=grad_c.numpy().astype(np.float16),
                             pairs=np.array(pairs), Q=Q)
    return Gbar, grad_c, pairs, col_index, Q


def _get_entry(model_name, checkpoint, cfg, ckpt_paths, class_samples_raw, pi, transforms,
               cache, cache_gbar, timing):
    """Returns cache[checkpoint], computing (and caching in-memory) it if absent."""
    if checkpoint in cache:
        return cache[checkpoint]
    m = _load_ckpt(model_name, cfg, ckpt_paths, checkpoint)
    Gbar, grad_c, pairs, col_index, Q = _load_or_compute_gbar(
        model_name, checkpoint, m, class_samples_raw, pi, cache_gbar, timing)
    grad_bd = {t: pl.compute_grad_bd(m, pl.clf_loss, class_samples_raw, Y_SOURCE, Y_TARGET,
                                      DATASET_FLAG, EVAL_DEVICE, model_flag=None, trigger=t)
               # `trigger=` is prelim_lib's/the repo's own keyword; the CSV column
               # and the grid axis are named `transform` (SPEC section 5).
               for t in transforms}
    entry = dict(Gbar=Gbar, grad_c=grad_c, pi=pi, pairs=pairs, col_index=col_index, Q=Q,
                 grad_bd=grad_bd, model=m)
    cache[checkpoint] = entry
    return entry


# --------------------------------------------------------------------------#
# E1 -- u-configurations (ported from session-1 prelim.ipynb cell 8, verbatim logic)
# --------------------------------------------------------------------------#
def build_u_configs(pi, gamma, beta, y_source, y_target, pairs, Q, c_qp, seed):
    """6 configurations u dans U_loc, protocole E1 (a)-(f)."""
    rng = np.random.RandomState(seed)
    P = len(pairs)
    budget_loc = beta / gamma
    classes = sorted(set(y for y, z in pairs))
    configs = {}

    u = {p: 0.0 for p in pairs}
    u[(y_source, y_target)] = min(budget_loc, pi[y_source])
    configs["a_source_target"] = u

    y2 = classes[(classes.index(y_source) + 3) % len(classes)]
    z2 = classes[(classes.index(y_target) + 3) % len(classes)]
    if z2 == y2:
        z2 = classes[(classes.index(z2) + 1) % len(classes)]
    u = {p: 0.0 for p in pairs}
    u[(y2, z2)] = min(budget_loc, pi[y2])
    configs["b_other_pair"] = u

    per_pair = budget_loc / P
    u = {p: per_pair for p in pairs}
    for y in classes:
        idxs = [p for p in pairs if p[0] == y]
        tot = sum(u[p] for p in idxs)
        if tot > pi[y]:
            scale = pi[y] / tot
            for p in idxs:
                u[p] *= scale
    configs["c_uniform"] = u

    ubar_star = pl.solve_qp(Q, c_qp, beta, pi, gamma, pairs, scope="aggregate", capacity=True).numpy()
    u_d = {p: ubar_star[i] / gamma for i, p in enumerate(pairs)}
    assert abs(sum(u_d.values()) * gamma - ubar_star.sum()) < 1e-6, \
        "lien u_i = ubar/gamma viole numeriquement"
    configs["d_qp"] = u_d

    for tag in ["e_random1", "f_random2"]:
        raw_pt = rng.rand(P) * (budget_loc / P) * 2
        proj = pl.solve_qp(np.eye(P), raw_pt, beta, pi, gamma, pairs, scope="local", capacity=True).numpy()
        configs[tag] = {p: proj[i] for i, p in enumerate(pairs)}

    for tag, u in configs.items():
        total = sum(u.values())
        assert total <= budget_loc + 1e-5, f"{tag}: budget viole ({total} > {budget_loc})"
        for y in classes:
            s = sum(u[p] for p in pairs if p[0] == y)
            assert s <= pi[y] + 1e-5, f"{tag}: plafond classe {y} viole ({s} > {pi[y]})"
    return configs


# --------------------------------------------------------------------------#
# E1 main table (6 configs x checkpoint, SEEDS[0])
# --------------------------------------------------------------------------#
def _run_e1_main(model_name, checkpoint, seed, raw_dataset, all_targets, entry, rows, done_pairs):
    m = entry["model"]
    Gbar, grad_c, pi, pairs, Q = entry["Gbar"], entry["grad_c"], entry["pi"], entry["pairs"], entry["Q"]
    v = E1_BETA * (entry["grad_bd"]["identity"] - grad_c)
    c_qp = (Gbar.T @ v).numpy().astype(np.float64)
    configs = build_u_configs(pi, GAMMA, E1_BETA, Y_SOURCE, Y_TARGET, pairs, Q, c_qp, seed=seed)

    shard_idx_list = pl.shard_indices(raw_dataset, N_B, seed=seed)
    worker_shard = shard_idx_list[0]
    shard_true_targets = all_targets[worker_shard]

    # Shard-level Gbar computed ONCE per (checkpoint, seed) and reused across all 6
    # tags below -- session 1's notebook recomputed it per tag on identical inputs;
    # that's a pure perf fix (class_conditional_shifts costs ~250s/call on r32p/CPU
    # here, see run_meta.json), not a behavior change.
    shard_samples_raw = {}
    for y in np.unique(shard_true_targets):
        idx_y = worker_shard[shard_true_targets == y]
        xs = torch.stack([raw_dataset[int(i)][0] for i in idx_y]).to(EVAL_DEVICE)
        shard_samples_raw[int(y)] = xs
    Gbar_s, grad_c_s, pi_s, pairs_s, col_s, Q_s = pl.class_conditional_shifts(
        m, shard_samples_raw, DATASET_FLAG, N_CLASSES, EVAL_DEVICE, loss_fn=pl.clf_loss, model_flag=None)

    for tag, u in configs.items():
        rng = np.random.RandomState(seed)
        poisoned_targets, u_real = pl.flip_masses_to_labels(shard_true_targets, u, pairs, rng)
        u_real_vec = torch.tensor([u_real[p] for p in pairs], dtype=torch.float32)
        Gbar_u = Gbar @ u_real_vec
        pred = grad_c + Gbar_u
        g_emp = pl.worker_gradient(m, pl.clf_loss, raw_dataset, worker_shard, poisoned_targets,
                                    DATASET_FLAG, None, batch_size=1024, device=EVAL_DEVICE)
        err_rel = float((g_emp - pred).norm() / (Gbar_u.norm() + 1e-12))
        cos = float(torch.nn.functional.cosine_similarity(
            (g_emp - grad_c).unsqueeze(0), Gbar_u.unsqueeze(0)).item())
        _record(rows, done_pairs, model_name, seed, checkpoint, E1_BETA, "identity", "E1", f"relerr_calib__{tag}", err_rel)
        _record(rows, done_pairs, model_name, seed, checkpoint, E1_BETA, "identity", "E1", f"cos_calib__{tag}", cos)

        u_real_vec_s = torch.tensor([u_real.get(p, 0.0) for p in pairs_s], dtype=torch.float32)
        Gbar_u_s = Gbar_s @ u_real_vec_s
        pred_s = grad_c_s + Gbar_u_s
        err_rel_s = float((g_emp - pred_s).norm() / (Gbar_u_s.norm() + 1e-12))
        cos_s = float(torch.nn.functional.cosine_similarity(
            (g_emp - grad_c_s).unsqueeze(0), Gbar_u_s.unsqueeze(0)).item())
        _record(rows, done_pairs, model_name, seed, checkpoint, E1_BETA, "identity", "E1", f"relerr_shard__{tag}", err_rel_s)
        _record(rows, done_pairs, model_name, seed, checkpoint, E1_BETA, "identity", "E1", f"cos_shard__{tag}", cos_s)

        # Only pairs with >= 1 expected flip: below that, round() legitimately produces
        # a 0-vs-nonzero mismatch with relative gap up to 100% by construction (see
        # flip_masses_to_labels's docstring) -- that's the rounding effect E1 measures,
        # not a defect, and including sub-1-flip requests would make this assertion
        # meaningless (dominated by denominators near zero).
        n_shard = len(shard_true_targets)
        gaps = [abs(u_real.get(p, 0.0) - float(u.get(p, 0.0))) / float(u[p])
                for p in pairs if float(u.get(p, 0.0)) * n_shard >= 1.0]
        if gaps:
            _record(rows, done_pairs, model_name, seed, checkpoint, E1_BETA, "identity",
                     "assertions", f"flip_mass_gap_rel_max__{tag}", max(gaps))


# --------------------------------------------------------------------------#
# E1 seed-robustness (checkpoint=end, config a, all SEEDS) + SNR/batch sweep
# (checkpoint=end, SEEDS[0], all BETAS x BATCH_SIZES_SNR)
# --------------------------------------------------------------------------#
def _run_e1_seed_and_snr(model_name, entry_end, raw_dataset, all_targets, seeds, betas, rows, done_pairs):
    m = entry_end["model"]
    Gbar, grad_c, pairs, pi = entry_end["Gbar"], entry_end["grad_c"], entry_end["pairs"], entry_end["pi"]

    for seed in seeds:
        shards = pl.shard_indices(raw_dataset, N_B, seed=seed)
        worker_shard = shards[0]
        shard_true_targets = all_targets[worker_shard]
        u_a = {p: 0.0 for p in pairs}
        u_a[(Y_SOURCE, Y_TARGET)] = min(E1_BETA / GAMMA, pi[Y_SOURCE])
        rng = np.random.RandomState(seed)
        poisoned_targets, u_real = pl.flip_masses_to_labels(shard_true_targets, u_a, pairs, rng)
        u_real_vec = torch.tensor([u_real[p] for p in pairs], dtype=torch.float32)
        Gbar_u = Gbar @ u_real_vec
        pred = grad_c + Gbar_u
        g_emp = pl.worker_gradient(m, pl.clf_loss, raw_dataset, worker_shard, poisoned_targets,
                                    DATASET_FLAG, None, batch_size=1024, device=EVAL_DEVICE)
        err_rel = float((g_emp - pred).norm() / (Gbar_u.norm() + 1e-12))
        cos = float(torch.nn.functional.cosine_similarity(
            (g_emp - grad_c).unsqueeze(0), Gbar_u.unsqueeze(0)).item())
        _record(rows, done_pairs, model_name, seed, "end", E1_BETA, "identity", "E1", "seed_robustness_err_rel__a", err_rel)
        _record(rows, done_pairs, model_name, seed, "end", E1_BETA, "identity", "E1", "seed_robustness_cos__a", cos)

    shards0 = pl.shard_indices(raw_dataset, N_B, seed=seeds[0])
    worker_shard = shards0[0]
    shard_true_targets = all_targets[worker_shard]
    for beta in betas:
        u_a = {p: 0.0 for p in pairs}
        u_a[(Y_SOURCE, Y_TARGET)] = min(beta / GAMMA, pi[Y_SOURCE])
        rng = np.random.RandomState(seeds[0])
        poisoned_targets, u_real = pl.flip_masses_to_labels(shard_true_targets, u_a, pairs, rng)
        u_real_vec = torch.tensor([u_real[p] for p in pairs], dtype=torch.float32)
        Gbar_u = Gbar @ u_real_vec
        pred = grad_c + Gbar_u

        g_full = pl.worker_gradient(m, pl.clf_loss, raw_dataset, worker_shard, poisoned_targets,
                                     DATASET_FLAG, None, batch_size=2048, device=EVAL_DEVICE)
        err_full = float((g_full - pred).norm() / (Gbar_u.norm() + 1e-12))
        _record(rows, done_pairs, model_name, seeds[0], "end", beta, "identity", "E1", "sweep_err_rel__B=full", err_full)

        for B in BATCH_SIZES_SNR:
            samples = pl.minibatch_gradient_samples(
                m, pl.clf_loss, raw_dataset, worker_shard, poisoned_targets,
                batch_size=B, n_draws=N_MB_DRAWS, dataset_flag=DATASET_FLAG,
                model_flag=None, device=EVAL_DEVICE, seed=seeds[0])
            per_draw_err = [float((samples[i] - pred).norm() / (Gbar_u.norm() + 1e-12))
                             for i in range(samples.shape[0])]
            err_mb = float(np.mean(per_draw_err))
            snr_val = pl.snr(Gbar_u, samples)
            _record(rows, done_pairs, model_name, seeds[0], "end", beta, "identity", "E1", f"sweep_err_rel__B={B}", err_mb)
            _record(rows, done_pairs, model_name, seeds[0], "end", beta, "identity", "E1", f"snr__B={B}", snr_val)


# --------------------------------------------------------------------------#
# E2 (per checkpoint, all BETAS)
# --------------------------------------------------------------------------#
def _run_e2(model_name, checkpoint, betas, transforms, entry, d_total, rows, done_pairs):
    """
    Per SPEC section 8/E2's own decomposition formula ("with T = identity, v/lam
    = Gbar[:,(9,4)]/pi[9] + (g[9][9]-grad_c)"), T=identity makes v almost
    tautologically lie in Gbar's column span -- varpi~1 there tests the
    degenerate case, not whether the rank ceiling leaves exploitable room
    against a REAL trigger. So this loops every transform in TRANSFORMS
    (grad_bd already computed/cached per transform by _get_entry, no extra
    Gbar cost): "identity" stays as the reference/degenerate case, "stripe" is
    the one that actually exercises T != identity and is what E2's verdict
    should be decided on.
    """
    Gbar, grad_c, pi, pairs, Q = entry["Gbar"], entry["grad_c"], entry["pi"], entry["pairs"], entry["Q"]
    eff_rank = pl.effective_rank(Q)
    baseline = eff_rank / d_total

    # Numeric twin of the eigenvalue-spectrum figure: 11 decile points (P100..P0,
    # i.e. largest to smallest) instead of all P=90 eigenvalues. Q = Gbar^T Gbar
    # does not depend on the transform, so this is recorded once (not per transform).
    eigvals = np.clip(np.linalg.eigvalsh(Q), 0, None)[::-1]
    for pct in range(0, 101, 10):
        idx = min(len(eigvals) - 1, int(round((100 - pct) / 100 * (len(eigvals) - 1))))
        _record(rows, done_pairs, model_name, None, checkpoint, None, "identity", "E2",
                f"eigval_Q_p{pct}", float(eigvals[idx]))

    for beta in betas:
        # reachable_radius (20 restarts x 50 projected-ascent steps over the
        # full gradient dim) depends only on (Gbar, beta, pi, gamma, pairs) --
        # NOT on transform -- so it's computed once per beta here, outside the
        # transform loop, rather than once per (beta, transform) redundantly.
        radius = pl.reachable_radius(Gbar, beta, pi, GAMMA, pairs)
        for transform in transforms:
            v = beta * (entry["grad_bd"][transform] - grad_c)
            v_norm = float(v.norm())
            v_norm_sq = v_norm ** 2
            c_qp = (Gbar.T @ v).numpy().astype(np.float64)

            varsigma = radius["varsigma"]
            rho = beta * varsigma
            v_hat = v_norm / rho if rho > 0 else float("inf")

            varpi = pl.rank_ratio(Q, c_qp, v_norm_sq) if v_norm_sq > 0 else float("nan")
            dist2, alpha_tilde_star, _ = pl.dist_to_cone(Q, c_qp, v_norm_sq, pairs)

            grad_c_norm = float(grad_c.norm())
            ratio = radius["lower_ascent"] / grad_c_norm if grad_c_norm > 0 else float("inf")
            Theta = math.asin(ratio) if ratio <= 1 else float("nan")

            grad_p = grad_c + v
            cos_gp_gc = float(torch.nn.functional.cosine_similarity(
                grad_p.unsqueeze(0), grad_c.unsqueeze(0)).item())
            angle_gp_gc_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_gp_gc))))

            s_beta = beta / (GAMMA * min(pi.values()))

            vals = dict(varsigma=varsigma, rho=rho, radius_upper=radius["upper"],
                        radius_lower_simple=radius["lower_simple"], radius_lower_ascent=radius["lower_ascent"],
                        v_norm=v_norm, v_hat=v_hat, varpi=varpi, baseline=baseline,
                        alpha_tilde_star=alpha_tilde_star,
                        sqrt_varpi=math.sqrt(varpi) if varpi >= 0 else float("nan"),
                        grad_c_norm=grad_c_norm, Theta_rad=Theta, angle_gp_gc_deg=angle_gp_gc_deg,
                        cos_gp_gc=cos_gp_gc, s_beta=s_beta)
            for k, val in vals.items():
                _record(rows, done_pairs, model_name, None, checkpoint, beta, transform, "E2", k, val)


# --------------------------------------------------------------------------#
# E3 (per checkpoint @ E1_BETA, + one "shared ubar" fit across all checkpoints)
# --------------------------------------------------------------------------#
def _run_e3_per_checkpoint(model_name, checkpoint, entry, rows, done_pairs):
    Gbar, pairs, Q, pi = entry["Gbar"], entry["pairs"], entry["Q"], entry["pi"]
    v = E1_BETA * (entry["grad_bd"]["identity"] - entry["grad_c"])
    c_qp = (Gbar.T @ v).numpy().astype(np.float64)
    u_star = pl.solve_qp(Q, c_qp, E1_BETA, pi, GAMMA, pairs, scope="aggregate", capacity=True)
    entry["u_star"] = u_star

    v_np = v.numpy().astype(np.float64)
    u_np = u_star.numpy()
    quad = float(u_np @ Q @ u_np)
    a_k = float(v_np @ v_np) - 2 * float(np.dot(u_np, c_qp)) + quad
    rho_k = E1_BETA * float(Gbar.norm(dim=0).max())
    a_over_rho2 = a_k / (rho_k ** 2 + 1e-12)
    entry["rho_k"] = rho_k
    entry["a_over_rho2"] = a_over_rho2

    l1 = float(u_np.sum())
    _record(rows, done_pairs, model_name, None, checkpoint, E1_BETA, "identity", "E3", "a_over_rho2", a_over_rho2)
    _record(rows, done_pairs, model_name, None, checkpoint, E1_BETA, "identity", "E3", "l1_u_star", l1)
    _record(rows, done_pairs, model_name, None, checkpoint, E1_BETA, "identity", "E3", "l1_over_beta", l1 / E1_BETA)

    # Numeric twin of the u*_ckpt heatmap figure: top-5 (y,z) pairs by mass.
    top5 = sorted(zip(pairs, u_np), key=lambda t: -t[1])[:5]
    for rank, ((y, z), mass) in enumerate(top5, start=1):
        _record(rows, done_pairs, model_name, None, checkpoint, E1_BETA, "identity", "E3",
                f"u_star_top{rank}_y{y}_z{z}", float(mass))

    if checkpoint == "end":
        u_cap_false = pl.solve_qp(Q, c_qp, E1_BETA, pi, GAMMA, pairs, scope="aggregate", capacity=False)
        s_beta_check = E1_BETA / (GAMMA * min(pi.values()))
        _record(rows, done_pairs, model_name, None, checkpoint, E1_BETA, "identity", "E3", "l1_capacity_true", l1)
        _record(rows, done_pairs, model_name, None, checkpoint, E1_BETA, "identity", "E3",
                "l1_capacity_false", float(u_cap_false.numpy().sum()))
        _record(rows, done_pairs, model_name, None, checkpoint, E1_BETA, "identity", "E3", "s_beta_check", s_beta_check)


def _run_e3_shared(model_name, checkpoints, cache, rows, done_pairs):
    mu, Q_sum, c_sum = {}, None, None
    for ck in checkpoints:
        entry = cache[ck]
        v = E1_BETA * (entry["grad_bd"]["identity"] - entry["grad_c"])
        c_qp = (entry["Gbar"].T @ v).numpy().astype(np.float64)
        rho_k = entry["rho_k"]
        mu_k = 1.0 / (rho_k ** 2 * len(checkpoints))
        mu[ck] = mu_k
        Q_sum = mu_k * entry["Q"] if Q_sum is None else Q_sum + mu_k * entry["Q"]
        c_sum = mu_k * c_qp if c_sum is None else c_sum + mu_k * c_qp

    end_entry = cache[checkpoints[-1]]
    ubar_shared = pl.solve_qp(Q_sum, c_sum, E1_BETA, end_entry["pi"], GAMMA,
                               end_entry["pairs"], scope="aggregate", capacity=True)

    J_shared = 0.0
    for ck in checkpoints:
        entry = cache[ck]
        v_np = (E1_BETA * (entry["grad_bd"]["identity"] - entry["grad_c"])).numpy().astype(np.float64)
        e = (entry["Gbar"].numpy().astype(np.float64) @ ubar_shared.numpy()) - v_np
        J_shared += mu[ck] * float(e @ e)

    mean_perckpt = float(np.mean([cache[ck]["a_over_rho2"] for ck in checkpoints]))
    gap = J_shared - mean_perckpt
    gap_pct = gap / mean_perckpt * 100 if mean_perckpt != 0 else float("nan")
    _record(rows, done_pairs, model_name, None, "shared", E1_BETA, "identity", "E3", "J_shared", J_shared)
    _record(rows, done_pairs, model_name, None, "shared", E1_BETA, "identity", "E3", "gap_pct", gap_pct)
    _record(rows, done_pairs, model_name, None, "shared", E1_BETA, "identity", "E3", "mean_perckpt", mean_perckpt)

    for n1, n2 in itertools.combinations(checkpoints, 2):
        u1, u2 = cache[n1]["u_star"].numpy(), cache[n2]["u_star"].numpy()
        denom = np.linalg.norm(u1) * np.linalg.norm(u2)
        cos_u = float(np.dot(u1, u2) / denom) if denom > 0 else float("nan")
        _record(rows, done_pairs, model_name, None, f"{n1}_vs_{n2}", E1_BETA, "identity", "E3", "cos_u_star", cos_u)

        G1, G2 = cache[n1]["Gbar"].numpy(), cache[n2]["Gbar"].numpy()
        cos_cols = [float(np.dot(G1[:, j], G2[:, j]) /
                           (np.linalg.norm(G1[:, j]) * np.linalg.norm(G2[:, j]) + 1e-12))
                    for j in range(G1.shape[1])]
        _record(rows, done_pairs, model_name, None, f"{n1}_vs_{n2}", E1_BETA, "identity", "E3",
                "mean_col_cos_Gbar", float(np.mean(cos_cols)))


# --------------------------------------------------------------------------#
# Assertions (see report.py Sec2; computed once per model, at checkpoint=end)
# --------------------------------------------------------------------------#
def _run_assertions(model_name, entry_end, raw_dataset, all_targets, seed, rows, done_pairs):
    Gbar, grad_c, pi, pairs, Q = (entry_end["Gbar"], entry_end["grad_c"], entry_end["pi"],
                                   entry_end["pairs"], entry_end["Q"])
    v = E1_BETA * (entry_end["grad_bd"]["identity"] - grad_c)
    c_qp = (Gbar.T @ v).numpy().astype(np.float64)

    ubar_star = pl.solve_qp(Q, c_qp, E1_BETA, pi, GAMMA, pairs, scope="aggregate", capacity=True).numpy()
    u_i_sum = float((ubar_star / GAMMA).sum())
    gap1 = abs(u_i_sum * GAMMA - ubar_star.sum())
    _record(rows, done_pairs, model_name, None, "end", E1_BETA, "identity", "assertions",
            "budget_relation_abs_gap", gap1)

    w_a = pl.solve_qp(Q, c_qp, E1_BETA, pi, GAMMA, pairs, scope="aggregate", capacity=False).numpy()
    w_b = pl._project_gradient_reused(Q, c_qp, E1_BETA, pairs).numpy()
    gap2 = float(np.max(np.abs(w_a - w_b)))
    _record(rows, done_pairs, model_name, None, "end", E1_BETA, "identity", "assertions",
            "solve_qp_vs_repo_project_gradient_max_abs_diff", gap2)

    v_norm_sq = float(v.norm() ** 2)
    dist2, alpha_tilde_star, w_star = pl.dist_to_cone(Q, c_qp, v_norm_sq, pairs)
    frac_of_bigbeta = float(w_star.numpy().sum()) / 1e6
    _record(rows, done_pairs, model_name, None, "end", E1_BETA, "identity", "assertions",
            "alpha_tilde_budget_frac_of_bigbeta", frac_of_bigbeta)

    # SPEC section 2/6: "flat reference aggregator agreeing with the repo one"
    # -- already exhaustively checked on synthetic tensors in tests/test_units.py
    # (all 5 rules, 26/26); this repeats it once on a REAL (n_b, d) stack of
    # actual worker gradients at checkpoint=end, rather than relying on the
    # synthetic evidence alone.
    m = entry_end["model"]
    shards = pl.shard_indices(raw_dataset, N_B, seed=seed)
    G_check = torch.stack([
        pl.worker_gradient(m, pl.clf_loss, raw_dataset, np.asarray(shards[i])[:SIM_BATCH],
                            all_targets[np.asarray(shards[i])[:SIM_BATCH]],
                            DATASET_FLAG, None, batch_size=SIM_BATCH, device=EVAL_DEVICE)
        for i in range(N_B)
    ], dim=0)
    max_gap = 0.0
    for rule in AGGREGATORS:
        agg_flat, _sel = pl.aggregate_instrumented(G_check, rule, F, variant="flat")
        gap = pl.check_against_repo(G_check, rule, F, agg_flat)
        max_gap = max(max_gap, gap)
    _record(rows, done_pairs, model_name, None, "end", E1_BETA, "identity", "assertions",
            "flat_aggregator_vs_repo_max_abs_diff", max_gap)


# --------------------------------------------------------------------------#
# E4 / E5 / E7 -- deployment and selection response
# --------------------------------------------------------------------------#
#
# All three blocks share one simulation core: deploy u_i = ubar*/gamma on the
# n_p perturbed workers, replay R aggregation rounds with real minibatch
# gradients from all n_b workers, and instrument who reached the aggregate.
# They differ only in what they sweep -- E4 the transform, E5 (beta, tau), E7
# n_p -- and in the experiment tag written to the CSV.
#
# The CSV schema of SPEC section 5 has no `variant` column, so the flat /
# per_tensor distinction is carried inside the `aggregator` field as
# "krum:flat" / "krum:per_tensor". That keeps the tidy row exactly as
# specified while still letting the report pivot on the variant.

# Round counts / SIM_BATCH sized against a real benchmark on this machine
# (r32p/CPU, checkpoint=end, batch=64): worker_gradient costs ~0.41s/call, so
# a naive "200 rounds x 10 workers" E4 costs ~820s PER TRANSFORM alone --
# already past SPEC section 1's ten-minute-per-block budget on cnn before
# E5's (beta, tau) product or E7's n_p sweep are even counted. Cut SIM_BATCH
# in half (64 -> 32, still a realistic minibatch, not a toy) and the round
# counts down from SPEC section 8's "~200" hint, budgeted so EACH of E4/E5/E7
# individually stays near ~5-8 minutes on cnn (linear is ~140x cheaper per
# call and comes along for free): E4 ~4min (2 transforms x 60 rounds), E5
# ~7min (3 betas x 12 taus x 6 rounds), E7 ~5min (3 n_p x 50 rounds). This
# trades round count for tractability -- documented explicitly rather than
# silently, since it weakens the Monte-Carlo precision of Abar/oscillation
# estimates relative to SPEC's "~200" language.
E4_ROUNDS = 60
E5_ROUNDS = 6            # x 12 taus x 3 betas = 216 rounds total per checkpoint
E7_ROUNDS = 50
SIM_BATCH = 32           # minibatch size per worker per round (was 64)
E5_N_TAU = 12            # 12-point log grid on v_hat in [0.1, 10]

# E5 follow-up (this session): the E1_BETA=0.10 curve came back looking flat
# under E5_ROUNDS=6/SIM_BATCH=32, but for ell=1 rules (krum, cw_median) that's
# 6 near-Bernoulli draws per point -- standard error ~sqrt(0.25/6)~0.2, bigger
# than the 0.10 margin the verdict tests for. Two independent levers, applied
# ONLY at beta=E1_BETA (the reference operating point already used by
# E1/E3/E4/E7) to keep the added cost bounded: more rounds (direct fix for the
# per-round Bernoulli noise) AND a larger minibatch (reduces the round-to-round
# gradient noise that flips near-tied krum/cw_median selections in the first
# place, so each round is less noisy on its own -- likely more effective per
# unit of compute than rounds alone). At 12 taus x 50 rounds x batch=64 on cnn
# (~0.41s/call x 10 workers): ~2470s (~41min); beta in {0.01, 0.03} keep the
# original settings (cheap, already computed, kept as lower-power supplementary
# evidence rather than recomputed).
E5_ROUNDS_REF = 50
E5_SIM_BATCH_REF = 64
E5_VHAT_RANGE = (0.1, 10.0)
N_P_E7 = [2, 3, 5]
COORD_SUBSAMPLE = 20000  # coordinates kept for the coordinate-wise E5 statistics
DECILES = list(range(0, 101, 10))


def _agg_key(rule, variant):
    return f"{rule}:{variant}"


def _deploy_plan(entry, beta, transform, n_p, seed, raw_dataset, all_targets, tau=1.0):
    """
    Everything a deployment needs, derived from a checkpoint's Gbar: the
    aggregate-optimal ubar* against the target tau*v, its per-worker image
    u_i = ubar*/gamma, and the actual poisoned shards.

    The budget-scope hazard of SPEC section 2 is asserted here, at the one
    place where the aggregate solution is turned into per-worker masses.
    """
    Gbar, grad_c, pi, pairs, Q = (entry["Gbar"], entry["grad_c"], entry["pi"],
                                  entry["pairs"], entry["Q"])
    gamma = n_p / N_B
    v = beta * (entry["grad_bd"][transform] - grad_c)
    v_target = tau * v
    c_qp = (Gbar.T @ v_target).numpy().astype(np.float64)
    ubar = pl.solve_qp(Q, c_qp, beta, pi, gamma, pairs,
                       scope="aggregate", capacity=True).numpy()

    u_i = ubar / gamma
    gap = abs(float(u_i.sum()) * gamma - float(ubar.sum()))
    assert gap < 1e-9, f"budget-scope relation violated: |gap| = {gap:.3e}"

    u_i_dict = {p: float(u_i[j]) for j, p in enumerate(pairs)}
    shards = pl.shard_indices(raw_dataset, N_B, seed=seed)
    worker_idx, worker_tgt = [], []
    for i in range(N_B):
        idx = np.asarray(shards[i])
        true_t = all_targets[idx]
        if i < n_p:
            rng = np.random.RandomState(seed * 1000 + i)
            tgt, _ = pl.masses_to_labels(true_t, u_i_dict, pairs, rng)
        else:
            tgt = true_t.copy()
        worker_idx.append(idx)
        worker_tgt.append(np.asarray(tgt))

    mal_mask = torch.zeros(N_B, dtype=torch.bool)
    mal_mask[:n_p] = True

    ubar_t = torch.as_tensor(ubar, dtype=torch.float32)
    u_i_t = torch.as_tensor(u_i, dtype=torch.float32)
    return dict(gamma=gamma, v=v, v_target=v_target, ubar=ubar, u_i=u_i,
                Gbar_ubar=Gbar @ ubar_t, Gbar_u_i=Gbar @ u_i_t,
                worker_idx=worker_idx, worker_tgt=worker_tgt, mal_mask=mal_mask,
                rho=beta * float(Gbar.norm(dim=0).max()))


def _round_stack(m, plan, raw_dataset, batch_size, rng):
    """One (n_b, d) stack: a single real minibatch gradient per worker."""
    grads = []
    for idx, tgt in zip(plan["worker_idx"], plan["worker_tgt"]):
        b = min(batch_size, len(idx))
        sel = rng.choice(len(idx), size=b, replace=False)
        grads.append(pl.worker_gradient(m, pl.clf_loss, raw_dataset, idx[sel], tgt[sel],
                                        DATASET_FLAG, None, batch_size=b, device=EVAL_DEVICE))
    return torch.stack(grads, dim=0)


def _simulate(m, plan, raw_dataset, blocks, n_rounds, batch_size, seed, rules, variants):
    """
    Replay `n_rounds` aggregation rounds and accumulate, per (rule, variant):
    Abar_j = E[A_j], the round-averaged P and N halves of b_Agg - b_mean, the
    round-averaged aggregate, and the across-round dispersion nu of that
    aggregate. Accumulators are (d,) vectors, never (n_rounds, d).
    """
    rng = np.random.RandomState(seed + 77)
    combos = [(r, va) for r in rules for va in variants]
    acc = {k: None for k in combos}
    sum_mean = None
    sum_g = sum_g2 = None
    a_round = {k: [] for k in combos}

    for _ in range(n_rounds):
        G = _round_stack(m, plan, raw_dataset, batch_size, rng)
        mean_g = G.mean(dim=0)
        sum_mean = mean_g.clone() if sum_mean is None else sum_mean + mean_g
        sum_g = G.clone() if sum_g is None else sum_g + G
        sum_g2 = (G * G) if sum_g2 is None else sum_g2 + G * G

        for rule, variant in combos:
            agg, sel = pl.aggregate_instrumented(G, rule, F, variant=variant, blocks=blocks)
            A = sel.A(plan["mal_mask"])
            P, N = sel.split_PN(G, plan["mal_mask"])
            st = acc[(rule, variant)]
            if st is None:
                st = dict(A=torch.zeros_like(A), agg=torch.zeros_like(agg),
                          agg2=torch.zeros_like(agg), P=torch.zeros_like(P),
                          N=torch.zeros_like(N), ell=sel.ell,
                          chi=sel.chi_ell, lam=sel.lam)
                acc[(rule, variant)] = st
            st["A"] += A
            st["agg"] += agg
            st["agg2"] += agg * agg
            st["P"] += P
            st["N"] += N
            a_round[(rule, variant)].append(float(A.mean()))

    R = float(n_rounds)
    mean_bar = sum_mean / R
    g_bar = sum_g / R
    g_var = torch.clamp(sum_g2 / R - g_bar * g_bar, min=0.0)
    g_std = g_var.sqrt()
    out = dict(mean_bar=mean_bar, g_std=g_std, per_rule={})
    for k, st in acc.items():
        agg_bar = st["agg"] / R
        agg_var = torch.clamp(st["agg2"] / R - agg_bar * agg_bar, min=0.0)
        out["per_rule"][k] = dict(
            Abar=st["A"] / R, agg_bar=agg_bar, nu=float(agg_var.sqrt().norm()),
            P=st["P"] / R, N=st["N"] / R, ell=st["ell"], chi=st["chi"], lam=st["lam"],
            a_round=np.asarray(a_round[k]),
        )
    return out


def _record_deploy_metrics(rows, done_pairs, experiment, model_name, seed, checkpoint,
                           beta, transform, n_p, tau, plan, entry, sim, rule, variant,
                           extra_coord=None):
    """One (rule, variant) block of E4/E5/E7 rows."""
    st = sim["per_rule"][(rule, variant)]
    agg_key = _agg_key(rule, variant)
    gamma, rho = plan["gamma"], plan["rho"]
    grad_c, v = entry["grad_c"], plan["v"]

    def rec(metric, value):
        _record(rows, done_pairs, model_name, seed, checkpoint, beta, transform, experiment,
                metric, value, aggregator=agg_key, n_p=n_p, tau=tau)

    Abar = st["Abar"]
    abar_mean = float(Abar.mean())
    abar_min, abar_max = float(Abar.min()), float(Abar.max())
    osc_abar = abar_max - abar_min

    rec("ell", st["ell"])
    rec("chi_ell", st["chi"])
    rec("Lambda", st["lam"])
    rec("abar_mean", abar_mean)
    rec("abar_min", abar_min)
    rec("abar_max", abar_max)
    rec("osc_abar", osc_abar)
    # Abar_min can legitimately be 0 (a coordinate no perturbed worker ever
    # reaches), which would make osc/Abar_min infinite; report the ratio only
    # where it is defined and let the report show it as n/a otherwise.
    if abar_min > 0:
        rec("osc_over_abar_min", osc_abar / abar_min)
    rec("selection_rate", abar_mean / gamma if gamma > 0 else float("nan"))

    q = np.percentile(Abar.numpy(), DECILES)
    for pct, val in zip(DECILES, q):
        rec(f"abar_dec{pct}", float(val))
    qr = np.percentile(st["a_round"], DECILES)
    for pct, val in zip(DECILES, qr):
        rec(f"a_round_dec{pct}", float(val))

    P_norm, N_norm = float(st["P"].norm()), float(st["N"].norm())
    rec("P_norm", P_norm)
    rec("N_norm", N_norm)
    rec("PN_norm", P_norm + N_norm)
    rec("nu_norm", st["nu"])

    b_agg = st["agg_bar"] - grad_c
    b_mean = sim["mean_bar"] - grad_c
    rec("b_agg_minus_b_mean_over_rho", float((b_agg - b_mean).norm()) / rho if rho > 0 else float("nan"))
    rec("alpha_tilde_b_agg", pl.alpha_tilde(b_agg, v))
    rec("alpha_tilde_b_mean", pl.alpha_tilde(b_mean, v))

    # Theoretical bound of SPEC section 8/E4:
    #   ||P|| + ||N|| <= Lambda*||Gbar u_i|| + sqrt(chi_ell*(n_h*sigma_c^2 + n_p*sigma_a^2))
    # sigma_c / sigma_a are per-worker minibatch gradient dispersions, each the
    # L2 norm of that worker's per-coordinate std across rounds, averaged over
    # the honest and the perturbed workers respectively.
    std_per_worker = sim["g_std"].norm(dim=1)
    sigma_a = float(std_per_worker[:n_p].mean())
    sigma_c = float(std_per_worker[n_p:].mean()) if n_p < N_B else float("nan")
    n_h = N_B - n_p
    rhs = st["lam"] * float(plan["Gbar_u_i"].norm()) + math.sqrt(
        max(0.0, st["chi"] * (n_h * sigma_c ** 2 + n_p * sigma_a ** 2)))
    rec("sigma_c", sigma_c)
    rec("sigma_a", sigma_a)
    rec("bound_rhs", rhs)
    rec("bound_slack", rhs - (P_norm + N_norm))
    rec("bound_respected", 1.0 if (P_norm + N_norm) <= rhs + 1e-9 else 0.0)

    if extra_coord is not None:
        for metric, value in extra_coord(Abar, sim).items():
            rec(metric, value)


def _run_e4(model_name, checkpoint, seed, beta, transform, n_p, entry, raw_dataset,
            all_targets, rows, done_pairs, n_rounds=E4_ROUNDS, experiment="E4"):
    """E4 (and, with a different n_p and tag, E7): one deployment, all rules x variants."""
    m = entry["model"]
    blocks = pl.flat_blocks(m)
    plan = _deploy_plan(entry, beta, transform, n_p, seed, raw_dataset, all_targets)
    sim = _simulate(m, plan, raw_dataset, blocks, n_rounds, SIM_BATCH, seed,
                    AGGREGATORS, pl.AGG_VARIANTS)
    for rule in AGGREGATORS:
        for variant in pl.AGG_VARIANTS:
            _record_deploy_metrics(rows, done_pairs, experiment, model_name, seed, checkpoint,
                                   beta, transform, n_p, None, plan, entry, sim, rule, variant)
    _record(rows, done_pairs, model_name, seed, checkpoint, beta, transform, experiment,
            "Gbar_ubar_norm", float(plan["Gbar_ubar"].norm()), n_p=n_p)
    _record(rows, done_pairs, model_name, seed, checkpoint, beta, transform, experiment,
            "Gbar_u_i_norm", float(plan["Gbar_u_i"].norm()), n_p=n_p)
    _record(rows, done_pairs, model_name, seed, checkpoint, beta, transform, experiment,
            "l1_ubar", float(plan["ubar"].sum()), n_p=n_p)
    _record(rows, done_pairs, model_name, seed, checkpoint, beta, transform, experiment,
            "v_hat", float(plan["v"].norm()) / plan["rho"] if plan["rho"] > 0 else float("nan"),
            n_p=n_p)
    return plan


def _run_e5(model_name, checkpoint, seed, beta, transform, n_p, entry, raw_dataset,
            all_targets, rows, done_pairs, n_rounds=E5_ROUNDS, batch_size=SIM_BATCH):
    """
    E5: the same deployment, re-solved against tau*v for 12 taus spanning
    v_hat in [0.1, 10]. No transform optimisation anywhere -- only the demanded
    deviation is rescaled (SPEC section 8/E5).
    """
    m = entry["model"]
    blocks = pl.flat_blocks(m)
    Gbar, grad_c = entry["Gbar"], entry["grad_c"]
    v = beta * (entry["grad_bd"][transform] - grad_c)
    v_norm = float(v.norm())
    rho = beta * float(Gbar.norm(dim=0).max())
    if v_norm <= 0 or rho <= 0:
        raise RuntimeError("E5: degenerate ||v|| or rho, cannot build the tau grid")

    targets = np.geomspace(E5_VHAT_RANGE[0], E5_VHAT_RANGE[1], E5_N_TAU)
    rng_sub = np.random.RandomState(seed)
    for vhat_target in targets:
        tau = float(vhat_target * rho / v_norm)
        plan = _deploy_plan(entry, beta, transform, n_p, seed, raw_dataset, all_targets, tau=tau)
        sim = _simulate(m, plan, raw_dataset, blocks, n_rounds, batch_size, seed,
                        AGGREGATORS, pl.AGG_VARIANTS)

        # P_k(v): the reachable part of the demanded deviation at this tau, i.e.
        # the image of the QP solution. SPEC names ||P_k(v)|| without defining it;
        # Gbar @ ubar*(tau) is the only object in the model that plays that role.
        Pk = Gbar @ torch.as_tensor(plan["ubar"], dtype=torch.float32)
        d = Pk.numel()
        sub = (rng_sub.choice(d, size=min(COORD_SUBSAMPLE, d), replace=False)
               if d > COORD_SUBSAMPLE else np.arange(d))

        def extra(Abar, sim_, Pk=Pk, sub=sub):
            s_j = sim_["g_std"][n_p:].mean(dim=0)          # honest coordinate scale
            ratio = (Pk.abs() / (s_j + 1e-12))
            from scipy import stats as _st
            a = Abar.numpy()[sub]
            r = ratio.numpy()[sub]
            out = {}
            # Undefined (NaN) whenever Abar is constant over the subsample --
            # always true for "mean" (A_j == gamma everywhere by construction)
            # and for flat krum/multikrum (kind="global", one set for every
            # coordinate), and possible near tau~0 where the bias signal is
            # negligible for any rule. `np.std(a) > 0` alone isn't a tight
            # enough guard (float noise can keep it just above 0 while scipy's
            # own tolerance still calls the input constant) -- check the
            # computed statistic itself and skip recording rather than write NaN.
            if np.std(a) > 1e-9 and np.std(r) > 1e-9:
                stat = float(_st.spearmanr(r, a).statistic)
                if not math.isnan(stat):
                    out["spearman_A_vs_Pk_over_s"] = stat
            # Numeric twin of the coordinate-wise figure: mean Abar per decile of
            # the predictor |P_k(v)_j| / s_j.
            edges = np.percentile(r, DECILES)
            for b_i in range(10):
                lo, hi = edges[b_i], edges[b_i + 1]
                sel = (r >= lo) & (r <= hi) if b_i == 9 else (r >= lo) & (r < hi)
                if sel.any():
                    out[f"abar_by_Pk_bin{b_i}"] = float(a[sel].mean())
            return out

        for rule in AGGREGATORS:
            for variant in pl.AGG_VARIANTS:
                _record_deploy_metrics(rows, done_pairs, "E5", model_name, seed, checkpoint,
                                       beta, transform, n_p, tau, plan, entry, sim,
                                       rule, variant, extra_coord=extra)
        _record(rows, done_pairs, model_name, seed, checkpoint, beta, transform, "E5",
                "v_hat", float(vhat_target), n_p=n_p, tau=tau)
        _record(rows, done_pairs, model_name, seed, checkpoint, beta, transform, "E5",
                "Pk_norm", float(Pk.norm()), n_p=n_p, tau=tau)
        sigma_c = float(sim["g_std"][n_p:].norm(dim=1).mean())
        _record(rows, done_pairs, model_name, seed, checkpoint, beta, transform, "E5",
                "Pk_norm_over_gamma_sigma_c",
                float(Pk.norm()) / (plan["gamma"] * sigma_c) if sigma_c > 0 else float("nan"),
                n_p=n_p, tau=tau)

    # Sentinel row with tau=None (unlike every other E5 row, tagged by its
    # specific tau), written ONLY after all E5_N_TAU taus complete -- gives
    # run_all's resume logic a single, stable config_id per (model,
    # checkpoint, beta, transform, n_p) to check for completeness. Recording
    # it upfront instead would falsely mark a beta "done" if the loop crashed
    # partway through, silently skipping the remaining taus on resume.
    _record(rows, done_pairs, model_name, seed, checkpoint, beta, transform, "E5",
            "n_tau_points", float(E5_N_TAU), n_p=n_p)


def _run_e7(model_name, checkpoint, seed, beta, transform, entry, raw_dataset, all_targets,
            rows, done_pairs, n_p_list=None, n_rounds=E7_ROUNDS):
    """
    E7: replay E4 at n_p in {2,3,5}, fixed beta, and check the invariance claim.

    SPEC section 8/E7 asks for an ASSERTION that ubar* and E_k are unchanged.
    They are recorded as measured gaps rather than raised as a hard `assert`,
    because the claim is only exactly true when the per-class caps of U_beta
    are slack: U_beta's caps are gamma*pi[y] and gamma = n_p/n_b moves with
    n_p, so at s_beta > 1 the caps bind differently at each n_p and ubar* does
    change. That is a finding about the framing, not a crash -- so the gap and
    a caps-binding flag go to the CSV and the report's assertion table decides
    PASS/FAIL from them.
    """
    n_p_list = n_p_list if n_p_list is not None else N_P_E7
    Gbar, pi, pairs = entry["Gbar"], entry["pi"], entry["pairs"]
    ref_np = N_P[0]
    plans = {}
    for n_p in n_p_list:
        plans[n_p] = _run_e4(model_name, checkpoint, seed, beta, transform, n_p, entry,
                             raw_dataset, all_targets, rows, done_pairs,
                             n_rounds=n_rounds, experiment="E7")

    ref = plans.get(ref_np)
    if ref is None:
        return
    varsigma = float(Gbar.norm(dim=0).max())
    for n_p in n_p_list:
        gamma = n_p / N_B
        gap = float(np.max(np.abs(plans[n_p]["ubar"] - ref["ubar"])))
        caps_bind = 1.0 if beta > gamma * min(pi.values()) * 1.0 else 0.0
        _record(rows, done_pairs, model_name, seed, checkpoint, beta, transform, "E7",
                "ubar_linf_gap_vs_ref", gap, n_p=n_p)
        _record(rows, done_pairs, model_name, seed, checkpoint, beta, transform, "E7",
                "reachable_upper", beta * varsigma, n_p=n_p)
        _record(rows, done_pairs, model_name, seed, checkpoint, beta, transform, "E7",
                "s_beta", beta / (gamma * min(pi.values())), n_p=n_p)
        _record(rows, done_pairs, model_name, seed, checkpoint, beta, transform, "E7",
                "caps_can_bind", caps_bind, n_p=n_p)
        _record(rows, done_pairs, model_name, seed, checkpoint, beta, transform, "E7",
                "local_rate_beta_over_gamma", beta / gamma, n_p=n_p)


# --------------------------------------------------------------------------#
# E6 -- predictive power of the feasibility measures (the one expensive
# block, behind --include-e6). Unlike E1-E5/E7 (static-checkpoint analysis
# or gradient-snapshot replay), E6 runs REAL federated training: a fresh
# r32p model, poisoned via label flips only (never pixel triggers -- see
# flip_masses_to_labels/effect_rate's own docstrings) on n_p of n_b workers
# for ~30 rounds, then measures actual attack success and clean accuracy.
# --------------------------------------------------------------------------#
E6_MODEL_FLAG = "r32p"
E6_MAX_TRAIN = 10000
E6_ROUNDS = 30
E6_AGGREGATORS = ("mean", "trmean")
# 3 source/target pairs x 3 betas = 9, truncated to 8 per SPEC section 8/E6
# ("spread the predictor, not cover the grid"): (9,4) is the pair used
# throughout E1-E5/E7; (0,5) and (2,7) are two more pairs spread across the
# class space so the predictor (varpi/v_hat/alpha_tilde_star/a_over_rho2)
# isn't evaluated at a single, possibly atypical, (source,target). The last
# cell of the cartesian product (2,7)@beta=0.10 is the one dropped.
E6_PAIRS = [(9, 4), (0, 5), (2, 7)]
E6_CONFIGS = [(y_s, y_t, b) for (y_s, y_t) in E6_PAIRS for b in BETAS][:8]
E6_TRANSFORM = "stripe"          # the actual trigger used for grad_bd, poisoning-eval, and effect_rate
E6_TEST_N_MAX = 500              # cap on triggered-test-set size for effect_rate (CIFAR-10: 1000/class)


def _predictor_at_round0(model, class_samples_raw, pi, y_source, y_target, beta):
    """
    The four SPEC section 8/E6 predictors (a_over_rho2, alpha_tilde_star,
    v_hat, varpi), computed ONCE from the model's state at round 0 -- exactly
    the same machinery as E2/E3 (class_conditional_shifts, solve_qp,
    dist_to_cone, rank_ratio, reachable_radius), just for a single
    (y_source, y_target, beta) instead of the checkpoint x beta grid. Returns
    (predictors_dict, ubar_star) -- ubar_star is reused by the caller to
    build the actual poisoned worker datasets, so the deployed attack is
    solved against EXACTLY the (y_source, y_target) pair being scored, not
    the suite-wide default (9, 4).
    """
    Gbar, grad_c, pi_, pairs, col_index, Q = pl.class_conditional_shifts(
        model, class_samples_raw, DATASET_FLAG, N_CLASSES, EVAL_DEVICE,
        loss_fn=pl.clf_loss, model_flag=None)
    grad_bd = pl.compute_grad_bd(model, pl.clf_loss, class_samples_raw, y_source, y_target,
                                  DATASET_FLAG, EVAL_DEVICE, model_flag=None, trigger=E6_TRANSFORM)
    v = beta * (grad_bd - grad_c)
    v_norm_sq = float(v.norm() ** 2)
    c_qp = (Gbar.T @ v).numpy().astype(np.float64)

    ubar_star = pl.solve_qp(Q, c_qp, beta, pi, GAMMA, pairs, scope="aggregate", capacity=True)
    u_np = ubar_star.numpy()
    quad = float(u_np @ Q @ u_np)
    a = float(v.numpy().astype(np.float64) @ v.numpy().astype(np.float64)) - 2 * float(np.dot(u_np, c_qp)) + quad

    radius = pl.reachable_radius(Gbar, beta, pi, GAMMA, pairs)
    rho = beta * radius["varsigma"]
    v_hat = float(v.norm()) / rho if rho > 0 else float("nan")
    varpi = pl.rank_ratio(Q, c_qp, v_norm_sq) if v_norm_sq > 0 else float("nan")
    _, alpha_tilde_star, _ = pl.dist_to_cone(Q, c_qp, v_norm_sq, pairs)

    predictors = dict(
        a_over_rho2=a / (rho ** 2 + 1e-12),
        alpha_tilde_star=alpha_tilde_star,
        v_hat=v_hat,
        varpi=varpi,
    )
    return predictors, ubar_star, pairs


def _e6_worker_datasets(raw_dataset, all_targets, train_idx, pairs, ubar_star, y_source, y_target,
                        n_p, seed, device):
    """
    Builds the N_B in-memory (preprocessed-tensor) worker datasets for one E6
    training run: n_p workers get their shard's labels realized from
    u_i = ubar_star/gamma via flip_masses_to_labels (label flips only, no
    pixel trigger -- see effect_rate's docstring for where T actually enters),
    the rest keep their true labels. Returns (datasets, max_shard_len) --
    max_shard_len sizes the per-round batch so each worker contributes
    exactly one full-shard batch per mini_train_multi "epoch" == one round.

    train_idx: the E6_MAX_TRAIN-sized subsample of raw_dataset's GLOBAL
    indices (SPEC section 8/E6: "r32p / CIFAR-10 at 10000 examples"), fixed
    once per _run_e6 call and shared across all configs -- shard_indices is
    called on `range(len(train_idx))` (local 0..len-1) since it only needs a
    length, and the resulting local shard indices are mapped back through
    train_idx to real raw_dataset positions before any indexing into it.
    """
    u_i = (ubar_star.numpy() / GAMMA)
    u_i_dict = {p: float(u_i[j]) for j, p in enumerate(pairs)}
    local_shards = pl.shard_indices(range(len(train_idx)), N_B, seed=seed)

    datasets, max_len = [], 0
    for i in range(N_B):
        idx = train_idx[np.asarray(local_shards[i])]
        true_t = all_targets[idx]
        if i < n_p:
            rng = np.random.RandomState(seed * 1000 + i)
            tgt, _ = pl.masses_to_labels(true_t, u_i_dict, pairs, rng)
        else:
            tgt = true_t.copy()
        x_raw = torch.stack([raw_dataset[int(j)][0] for j in idx]).to(device)
        x = pl.raw_to_preprocess(x_raw, dataset_flag=DATASET_FLAG, model_flag=None)
        y = torch.as_tensor(tgt, dtype=torch.long, device=device)
        datasets.append(torch.utils.data.TensorDataset(x, y))
        max_len = max(max_len, len(idx))
    return datasets, max_len


def _run_e6(seed, raw_dataset, all_targets, class_samples_raw, pi, rows, done_pairs):
    """
    One full pass over E6_CONFIGS x E6_AGGREGATORS: for each, a FRESH r32p
    model is initialised, the round-0 predictor is computed and used to solve
    the deployed attack (u_i on n_p=N_P[0] of N_B workers), the model trains
    for E6_ROUNDS real federated rounds under agg_method (mean/trmean), and
    the final model is scored for effect_rate (backdoor ASR) and
    clean_accuracy. Spearman correlations between effect_rate and each
    predictor are computed per aggregator across the (up to) 8 configs.
    """
    n_p = N_P[0]
    per_agg_points = {agg: {"effect": [], "clean": [], "predictors": []} for agg in E6_AGGREGATORS}

    # SPEC section 8/E6: "r32p / CIFAR-10 at 10000 examples" -- one fixed
    # subsample shared across all E6_CONFIGS (not re-drawn per config), same
    # spirit as MODEL_CFG["cnn"]["max_train"] used elsewhere in this file.
    rng_sub = np.random.RandomState(seed)
    train_idx = rng_sub.choice(len(raw_dataset), size=min(E6_MAX_TRAIN, len(raw_dataset)), replace=False)

    for cfg_idx, (y_source, y_target, beta) in enumerate(E6_CONFIGS):
        # `checkpoint` is repurposed to carry the (y_source, y_target) pair
        # tag -- SPEC section 5's tidy schema has no dedicated column for it,
        # and _cid() is computed from (model, seed, checkpoint, beta,
        # transform, experiment, aggregator, n_p, tau): without the pair in
        # one of those fields, different pairs AT THE SAME beta would collide
        # on the same config_id and silently overwrite each other's rows via
        # _record's done_pairs dedup. Same reuse-the-checkpoint-field pattern
        # as _run_e3_shared's checkpoint=f"{n1}_vs_{n2}".
        pair_tag = f"p{y_source}-{y_target}"
        torch.manual_seed(seed * 10000 + cfg_idx)
        model0 = pl.build_model(E6_MODEL_FLAG, N_CLASSES, EVAL_DEVICE)
        model0.eval()
        predictors, ubar_star, pairs = _predictor_at_round0(
            model0, class_samples_raw, pi, y_source, y_target, beta)

        for metric, val in predictors.items():
            _record(rows, done_pairs, "cnn", seed, f"round0_{pair_tag}", beta, E6_TRANSFORM, "E6",
                    metric, val, n_p=n_p, tau=None)

        for agg_method in E6_AGGREGATORS:
            model = pl.build_model(E6_MODEL_FLAG, N_CLASSES, TRAIN_DEVICE)
            model.load_state_dict(model0.state_dict())
            model.train()

            datasets, max_len = _e6_worker_datasets(
                raw_dataset, all_targets, train_idx, pairs, ubar_star, y_source, y_target,
                n_p, seed, TRAIN_DEVICE)
            opt = torch.optim.SGD(model.parameters(), **pl.DEFAULT_SGD_KWARGS)
            sched_kwargs = {k: v for k, v in pl.DEFAULT_SGD_SCHED_KWARGS.items()}
            scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, **sched_kwargs)
            pl.mini_train_multi(
                model=model, train_datasets=datasets, test_data=None,
                batch_size=max_len + 1, opt=opt, scheduler=scheduler, epochs=E6_ROUNDS,
                agg_method=agg_method, f=F,
            )

            model.eval()
            eff = pl.effect_rate(model, DATASET_FLAG, y_source, y_target, TRAIN_DEVICE,
                                 model_flag=None, trigger=E6_TRANSFORM, n_max=E6_TEST_N_MAX)
            clean_acc = pl.clean_accuracy(model, DATASET_FLAG, TRAIN_DEVICE, model_flag=None)

            _record(rows, done_pairs, "cnn", seed, f"end_{pair_tag}", beta, E6_TRANSFORM, "E6",
                    "effect_rate", eff, aggregator=agg_method, n_p=n_p, tau=None)
            _record(rows, done_pairs, "cnn", seed, f"end_{pair_tag}", beta, E6_TRANSFORM, "E6",
                    "clean_accuracy", clean_acc, aggregator=agg_method, n_p=n_p, tau=None)

            per_agg_points[agg_method]["effect"].append(eff)
            per_agg_points[agg_method]["clean"].append(clean_acc)
            per_agg_points[agg_method]["predictors"].append(predictors)
            del model
        del model0

    from scipy import stats as _st
    for agg_method in E6_AGGREGATORS:
        pts = per_agg_points[agg_method]
        n = len(pts["effect"])
        for pred_name in ("a_over_rho2", "alpha_tilde_star", "v_hat", "varpi"):
            xs = [p[pred_name] for p in pts["predictors"]]
            ys = pts["effect"]
            if n >= 3 and np.std(xs) > 1e-9 and np.std(ys) > 1e-9:
                r = float(_st.spearmanr(xs, ys).statistic)
                if not math.isnan(r):
                    _record(rows, done_pairs, "cnn", seed, "summary", None, E6_TRANSFORM, "E6",
                            f"spearman_effect_vs_{pred_name}", r, aggregator=agg_method, n_p=n_p, tau=None)
        _record(rows, done_pairs, "cnn", seed, "summary", None, E6_TRANSFORM, "E6",
                "n_configs", float(n), aggregator=agg_method, n_p=n_p, tau=None)


# --------------------------------------------------------------------------#
# Orchestration
# --------------------------------------------------------------------------#
def _flush_csv(rows):
    df = pd.DataFrame(rows)
    df.to_csv(METRICS_PATH, index=False)
    return df


def run_all(models=None, seeds=None, checkpoints=None, betas=None, transforms=None,
            include_e6=False, resume=True, cache_gbar=True):
    models = models if models is not None else MODELS
    seeds = seeds if seeds is not None else SEEDS
    checkpoints = checkpoints if checkpoints is not None else CHECKPOINTS
    betas = betas if betas is not None else BETAS
    transforms = transforms if transforms is not None else TRANSFORMS

    t_start = time.time()
    df_prev = pd.read_csv(METRICS_PATH) if (resume and os.path.exists(METRICS_PATH)) else None
    done_cids = set(df_prev["config_id"]) if df_prev is not None else set()
    done_pairs = set(zip(df_prev["config_id"], df_prev["metric"])) if df_prev is not None else set()
    rows = df_prev.to_dict("records") if df_prev is not None else []

    counts = dict(run=0, cached=0, failed=0)
    failed_cells = []
    timing = {}
    pi = pl.compute_class_frequencies(DATASET_FLAG, N_CLASSES)

    for model_name in models:
        cfg = MODEL_CFG[model_name]
        try:
            ckpt_paths = _train_or_load(model_name, cfg, seeds[0], resume)
        except Exception:
            _log_failure(model_name, None, None, "training")
            counts["failed"] += 1
            failed_cells.append(dict(model=model_name, seed="*", checkpoint="*",
                                      reason="training failed -- see failures.log"))
            continue

        class_samples_raw = pl.get_class_conditional_samples(DATASET_FLAG, N_CLASSES, N_PER_CLASS_CALIB, EVAL_DEVICE)
        raw_dataset = pl.get_raw_clean_dataset(DATASET_FLAG, train=True)
        all_targets = np.array(raw_dataset.dataset.targets)
        cache = {}

        for checkpoint in checkpoints:
            cid_e2 = [_cid(model_name, None, checkpoint, b, t, "E2") for b in betas for t in transforms]
            cid_e3 = _cid(model_name, None, checkpoint, E1_BETA, "identity", "E3")
            cid_e1main = _cid(model_name, seeds[0], checkpoint, E1_BETA, "identity", "E1")
            ckpt_done = all(c in done_cids for c in cid_e2) and cid_e3 in done_cids and cid_e1main in done_cids
            # `entry` (Gbar/grad_c/grad_bd/model) is always loaded into `cache[checkpoint]`,
            # even when E1-E3 are already cached: E4/E5/E7 below (and E3-shared) need
            # cache["end"]/cache[checkpoints] populated in-memory for THIS run_all()
            # call regardless of whether E1-E3 themselves need recomputing. Loading is
            # cheap when cache_gbar=True and the .npz cache hits (no ~250s recompute).
            try:
                entry = _get_entry(model_name, checkpoint, cfg, ckpt_paths, class_samples_raw, pi,
                                    transforms, cache, cache_gbar, timing)
            except Exception:
                _log_failure(model_name, None, checkpoint, "entry load (Gbar/grad_c/grad_bd)")
                counts["failed"] += 1
                failed_cells.append(dict(model=model_name, seed="*", checkpoint=checkpoint,
                                          reason="entry load failed -- see failures.log"))
                continue
            if ckpt_done:
                counts["cached"] += 1
                continue
            try:
                _run_e2(model_name, checkpoint, betas, transforms, entry, entry["Gbar"].shape[0], rows, done_pairs)
                _run_e3_per_checkpoint(model_name, checkpoint, entry, rows, done_pairs)
                counts["run"] += 1
            except Exception:
                _log_failure(model_name, None, checkpoint, "gbar/E2/E3")
                counts["failed"] += 1
                failed_cells.append(dict(model=model_name, seed="*", checkpoint=checkpoint,
                                          reason="gbar/E2/E3 failed -- see failures.log"))
                continue

            try:
                _run_e1_main(model_name, checkpoint, seeds[0], raw_dataset, all_targets, entry, rows, done_pairs)
            except Exception:
                _log_failure(model_name, seeds[0], checkpoint, "E1 main")
                counts["failed"] += 1
                failed_cells.append(dict(model=model_name, seed=seeds[0], checkpoint=checkpoint,
                                          reason="E1 main failed -- see failures.log"))

            _flush_csv(rows)

        try:
            if all(ck in cache for ck in checkpoints):
                cid_shared = _cid(model_name, None, "shared", E1_BETA, "identity", "E3")
                if cid_shared not in done_cids:
                    _run_e3_shared(model_name, checkpoints, cache, rows, done_pairs)
            elif not resume:
                raise RuntimeError("not all checkpoints available in-memory for E3-shared "
                                    "(some were skipped via resume but E3-shared needs all 3)")
        except Exception:
            _log_failure(model_name, None, "shared", "E3 shared")
            counts["failed"] += 1
            failed_cells.append(dict(model=model_name, seed="*", checkpoint="shared",
                                      reason="E3 shared failed -- see failures.log"))

        try:
            if "end" in cache:
                cid_seedrob = [_cid(model_name, s, "end", E1_BETA, "identity", "E1") for s in seeds]
                cid_snr = [_cid(model_name, seeds[0], "end", b, "identity", "E1") for b in betas]
                if not (all(c in done_cids for c in cid_seedrob) and all(c in done_cids for c in cid_snr)):
                    _run_e1_seed_and_snr(model_name, cache["end"], raw_dataset, all_targets,
                                         seeds, betas, rows, done_pairs)
                _run_assertions(model_name, cache["end"], raw_dataset, all_targets, seeds[0], rows, done_pairs)
        except Exception:
            _log_failure(model_name, None, "end", "E1 seed/SNR + assertions")
            counts["failed"] += 1
            failed_cells.append(dict(model=model_name, seed="*", checkpoint="end",
                                      reason="E1 seed/SNR + assertions failed -- see failures.log"))

        _flush_csv(rows)

        # ---- Step 2/3: E4 (aggregator response), E5 (selection/saturation), --
        # E7 (spreading n_p) -- all at checkpoint=end only, per this file's own
        # scoping note above _run_e4/_run_e5/_run_e7 (SPEC section 8 fixes the
        # checkpoint for these three blocks; E4 sweeps transform, E5 sweeps
        # (beta, tau) internally, E7 sweeps n_p internally).
        try:
            if "end" in cache:
                entry_end = cache["end"]
                # Per-subitem resume gating (transform for E4, beta for E5):
                # an all-or-nothing gate on the whole block would silently
                # redo already-completed, expensive sub-items (e.g. a crash
                # after E5's beta=0.01/0.03 finish but before the ~41min
                # beta=0.10 reference run completes would otherwise redo
                # 0.01/0.03 too on resume) -- checked individually instead.
                for transform in transforms:
                    cid_t = _cid(model_name, seeds[0], "end", E1_BETA, transform, "E4", n_p=N_P[0])
                    if cid_t in done_cids:
                        continue
                    _run_e4(model_name, "end", seeds[0], E1_BETA, transform, N_P[0],
                            entry_end, raw_dataset, all_targets, rows, done_pairs)
                    _flush_csv(rows)

                for beta in betas:
                    cid_b = _cid(model_name, seeds[0], "end", beta, "identity", "E5", n_p=N_P[0], tau=None)
                    if cid_b in done_cids:
                        continue
                    # Reference operating point (E1_BETA) gets the boosted
                    # round count/batch size (see E5_ROUNDS_REF/E5_SIM_BATCH_REF
                    # above) to resolve whether the earlier flat curve was
                    # genuine or Monte-Carlo noise; other betas keep the
                    # original cheaper settings as supplementary evidence.
                    if beta == E1_BETA:
                        _run_e5(model_name, "end", seeds[0], beta, "identity", N_P[0],
                                entry_end, raw_dataset, all_targets, rows, done_pairs,
                                n_rounds=E5_ROUNDS_REF, batch_size=E5_SIM_BATCH_REF)
                    else:
                        _run_e5(model_name, "end", seeds[0], beta, "identity", N_P[0],
                                entry_end, raw_dataset, all_targets, rows, done_pairs)
                    _flush_csv(rows)

                cid_e7 = [_cid(model_name, seeds[0], "end", E1_BETA, "identity", "E7", n_p=n_p)
                          for n_p in N_P_E7]
                if not all(c in done_cids for c in cid_e7):
                    _run_e7(model_name, "end", seeds[0], E1_BETA, "identity", entry_end,
                            raw_dataset, all_targets, rows, done_pairs)
                    _flush_csv(rows)
                counts["run"] += 1
        except Exception:
            _log_failure(model_name, None, "end", "E4/E5/E7")
            counts["failed"] += 1
            failed_cells.append(dict(model=model_name, seed="*", checkpoint="end",
                                      reason="E4/E5/E7 failed -- see failures.log"))

        _flush_csv(rows)

    if include_e6:
        # E6 always uses r32p/cnn regardless of `models` (SPEC section 8/E6),
        # so it runs once here, not per model_name. raw_dataset/all_targets/
        # class_samples_raw/pi don't depend on model_name -- whatever the
        # per-model loop above last computed them as (or the values computed
        # before that loop, for `pi`) is reused as-is.
        cid_e6 = [_cid("cnn", seeds[0], "summary", None, E6_TRANSFORM, "E6",
                       aggregator=agg, n_p=N_P[0], tau=None) for agg in E6_AGGREGATORS]
        if all(c in done_cids for c in cid_e6):
            print("[run_all] E6 deja complet (cache) -- rien a faire.")
        else:
            try:
                _run_e6(seeds[0], raw_dataset, all_targets, class_samples_raw, pi, rows, done_pairs)
                counts["run"] += 1
            except Exception:
                _log_failure("cnn", seeds[0], "E6", "E6")
                counts["failed"] += 1
                failed_cells.append(dict(model="cnn", seed=seeds[0], checkpoint="E6",
                                         reason="E6 failed -- see failures.log"))
            _flush_csv(rows)

    df_final = _flush_csv(rows)

    duration_s = time.time() - t_start
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                          cwd=pl._FLIP_ROOT).decode().strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=pl._FLIP_ROOT).decode().strip())
    except Exception:
        commit, dirty = "unknown", None

    import osqp
    meta = dict(
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t_start)),
        finished_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        duration_s=duration_s,
        git_commit=commit, git_dirty=dirty,
        torch_version=torch.__version__, numpy_version=np.__version__, osqp_version=osqp.__version__,
        train_device=str(TRAIN_DEVICE), eval_device=str(EVAL_DEVICE),
        n_cells_run=counts["run"], n_cells_cached=counts["cached"], n_cells_failed=counts["failed"],
        failed_cells=failed_cells,
        grid=dict(models=models, seeds=seeds, checkpoints=checkpoints, betas=betas,
                   transforms=transforms, n_p=N_P, aggregators=AGGREGATORS, include_e6=include_e6),
        gbar_seconds=timing.get("gbar_seconds", []),
        n_rows=len(df_final),
    )
    with open(RUN_META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[run_all] done in {duration_s:.1f}s -- run={counts['run']} cached={counts['cached']} "
          f"failed={counts['failed']} -- {len(df_final)} lignes -> {METRICS_PATH}")
    return counts


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--include-e6", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--no-cache-gbar", action="store_true",
                    help="disable the float16 Gbar/grad_c disk cache under prelim/artifacts/cache/ "
                         "(~167MB for r32p) -- on by default since _get_entry now runs "
                         "unconditionally (E3-shared/E4/E5/E7 need every checkpoint's entry loaded "
                         "into memory even when E1-E3 are fully cached and skipped); without this "
                         "cache, every resumed run re-pays the ~250s x 3 checkpoints x len(models) "
                         "Gbar recompute on r32p")
    args = p.parse_args()
    run_all(include_e6=args.include_e6, resume=not args.no_resume, cache_gbar=not args.no_cache_gbar)
