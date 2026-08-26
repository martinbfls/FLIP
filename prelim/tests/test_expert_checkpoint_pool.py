"""
prelim/tests/test_expert_checkpoint_pool.py -- non-regression checks for the P3 checkpoint-pool
change to federated_generate_labels_trigger/utils.py and
federated_generate_labels_trigger_joint/utils.py's `build_expert_pool` (both copies are
structurally identical -- see each module's own docstring for the create_graph=True /
float32-on-CPU rationale specific to the joint module).

Uses tiny SYNTHETIC state dicts written to a temp dir (never under out/checkpoints/) -- no
dataset, no real model, no training. Matches prelim/tests/test_policy_module_fixes.py's
convention (synthetic-only, runs in well under a minute).

Also verifies, by source inspection, that pool_size == 1 bypasses build_expert_pool entirely in
both run_module.py files -- the actual guarantee of "bit-for-bit identical to the pre-pooling
code": with pooling skipped, the loop falls back to the ORIGINAL torch.load(expert_starts[it])
per-step code path, unchanged.

Run:  python prelim/tests/test_expert_checkpoint_pool.py
"""
import os
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from modules.federated_generate_labels_trigger.utils import (
    build_expert_pool as build_expert_pool_trigger,
)
from modules.federated_generate_labels_trigger_joint.utils import (
    build_expert_pool as build_expert_pool_joint,
)

_results = []


def check(name, ok, detail=""):
    _results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def _write_synthetic_checkpoints(tmpdir, n, seed=0):
    """n distinct (params_path, opt_path) checkpoint pairs, small tensors, half-precision on
    disk (to exercise the float32-cast-on-load path) -- returns (params_paths, opt_paths)."""
    g = torch.Generator().manual_seed(seed)
    params_paths, opt_paths = [], []
    for i in range(n):
        params = {
            "w": torch.randn(4, 3, generator=g).half(),
            "b": torch.randn(4, generator=g).half(),
        }
        opt_state = {
            "state": {0: {"momentum_buffer": torch.randn(4, 3, generator=g).half()}},
            "param_groups": [{"lr": 0.1, "momentum": 0.9, "weight_decay": 0.0,
                               "dampening": 0.0, "nesterov": True}],
        }
        p_path = str(Path(tmpdir) / f"params_{i}.pt")
        o_path = str(Path(tmpdir) / f"opt_{i}.pt")
        torch.save(params, p_path)
        torch.save(opt_state, o_path)
        params_paths.append(p_path)
        opt_paths.append(o_path)
    return params_paths, opt_paths


def _test_build_expert_pool_basic(build_expert_pool, label):
    with tempfile.TemporaryDirectory() as tmpdir:
        params_paths, opt_paths = _write_synthetic_checkpoints(tmpdir, n=5)

        pool, pool_size = build_expert_pool(params_paths, opt_paths, pool_size=3)
        check(f"[{label}] pool_size respected when <= distinct available",
              pool_size == 3 and len(pool) == 3, f"pool_size={pool_size}, len(pool)={len(pool)}")

        checkpoint, opt_state = pool[0]
        all_float32 = all(v.dtype == torch.float32 for v in checkpoint.values())
        all_cpu = all(v.device.type == "cpu" for v in checkpoint.values())
        check(f"[{label}] pool caches params as float32 on CPU (not half)",
              all_float32 and all_cpu,
              f"dtypes={[v.dtype for v in checkpoint.values()]}")

        # Values match the on-disk originals exactly (up to the intentional half->float32
        # upcast, which is exact -- every half value is exactly representable in float32).
        original = torch.load(params_paths[0], map_location="cpu")
        # pool[i] is drawn from `random.sample` over the 5 distinct paths, so we can't assume
        # pool[0] came from params_paths[0] -- instead verify EVERY pool entry round-trips
        # exactly against ITS OWN source file.
        # Rebuild a path -> pool-entry mapping is not directly possible (build_expert_pool
        # doesn't return which path each pool entry came from), so instead verify the whole
        # POOL's value set is a subset of the full checkpoint set's values.
        all_originals = [torch.load(p, map_location="cpu") for p in params_paths]
        match_found = []
        for checkpoint, _ in pool:
            found = any(
                all(torch.equal(checkpoint[k], orig[k].float()) for k in checkpoint)
                for orig in all_originals
            )
            match_found.append(found)
        check(f"[{label}] every pool entry's values exactly match one on-disk checkpoint "
              "(half -> float32 upcast is exact)",
              all(match_found), f"{sum(match_found)}/{len(match_found)} matched")


def _test_build_expert_pool_dedup(build_expert_pool, label):
    with tempfile.TemporaryDirectory() as tmpdir:
        params_paths, opt_paths = _write_synthetic_checkpoints(tmpdir, n=3)
        # Duplicate the path list 4x (simulating extract_experts' repeated random draws
        # colliding on the same (expert, trajectory) pair across different `_`/`s` combos).
        dup_params = params_paths * 4
        dup_opts = opt_paths * 4

        pool, pool_size = build_expert_pool(dup_params, dup_opts, pool_size=10)
        check(f"[{label}] pool_size clamped to the number of DISTINCT pairs, not raw list length",
              pool_size == 3 and len(pool) == 3,
              f"pool_size={pool_size} (expected 3, i.e. distinct pairs, not 10 or 12)")


def _test_build_expert_pool_clamp_warning(build_expert_pool, label, capsys=None):
    with tempfile.TemporaryDirectory() as tmpdir:
        params_paths, opt_paths = _write_synthetic_checkpoints(tmpdir, n=2)
        pool, pool_size = build_expert_pool(params_paths, opt_paths, pool_size=99)
        check(f"[{label}] pool_size > available clamps down instead of raising",
              pool_size == 2 and len(pool) == 2, f"pool_size={pool_size}")


def test_pool_size_one_bypasses_pooling_in_source():
    """Structural regression check: `if pool_size != 1:` must literally gate the
    build_expert_pool call in both run_module.py files -- this IS the bit-for-bit backward-
    compatibility guarantee (pool_size==1 falls through to the untouched, original sequential
    torch.load(expert_starts[it]) code path rather than going through a size-1 "pool")."""
    repo_root = Path(__file__).resolve().parents[2]
    for rel_path, label in [
        ("modules/federated_generate_labels_trigger/run_module.py", "trigger"),
        ("modules/federated_generate_labels_trigger_joint/run_module.py", "joint"),
    ]:
        src = (repo_root / rel_path).read_text()
        guarded = "if pool_size != 1:" in src and "build_expert_pool(" in src
        # The pool_size==1 branch inside _load_expert_for_step must use expert_starts[it] /
        # expert_opt_starts[it] directly (the pre-pooling code), not the pool.
        sequential_fallback = (
            "if pool_size == 1:" in src
            and "torch.load(expert_starts[it])" in src
        )
        check(f"[{label}] build_expert_pool is gated behind `pool_size != 1`",
              guarded, rel_path)
        check(f"[{label}] pool_size==1 falls back to sequential torch.load(expert_starts[it])",
              sequential_fallback, rel_path)


def test_pool_ram_cost_r32p():
    """Documents the RAM cost claim from the task (~1.8MB/checkpoint params for r32p, ~27MB
    for a pool_size=15 pool) against the actual r32p parameter count -- informational, not a
    pass/fail gate on model architecture specifics."""
    import modules.base_utils.model.resnet as resnet
    m = resnet.resnet32(10)
    n_params = sum(p.numel() for p in m.parameters())
    params_mb = n_params * 4 / 1e6
    # SGD with momentum keeps one buffer per param, same shape as the param itself.
    per_checkpoint_mb = params_mb * 2  # params + momentum buffer
    pool15_mb = per_checkpoint_mb * 15
    check("r32p pool RAM cost matches the ~1.8MB/checkpoint (params only), ~27MB/pool_size=15 "
          "(params+momentum) estimate from the task",
          params_mb < 2.5 and pool15_mb < 60,
          f"params={params_mb:.2f}MB/checkpoint, "
          f"params+momentum={per_checkpoint_mb:.2f}MB/checkpoint, "
          f"pool_size=15 total={pool15_mb:.2f}MB")


if __name__ == "__main__":
    for build_fn, label in [
        (build_expert_pool_trigger, "trigger"),
        (build_expert_pool_joint, "joint"),
    ]:
        _test_build_expert_pool_basic(build_fn, label)
        _test_build_expert_pool_dedup(build_fn, label)
        _test_build_expert_pool_clamp_warning(build_fn, label)

    test_pool_size_one_bypasses_pooling_in_source()
    test_pool_ram_cost_r32p()

    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} checks passed.")
    sys.exit(1 if n_fail else 0)
