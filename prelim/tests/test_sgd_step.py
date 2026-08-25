"""
prelim/tests/test_sgd_step.py -- dedicated correctness test for
modules/federated_generate_labels.utils.sgd_step (the differentiable-SGD helper shared by
federated_generate_labels_trigger and federated_generate_labels_trigger_joint to build a
virtual/differentiable optimizer step from a gradient + a loaded torch.optim.SGD state dict).

Replaces the old "point-4 check" (a single first-batch numeric assertion baked into
federated_generate_labels_trigger_joint/run_module.py) with a real, synthetic, standalone
test -- run any time, not just once per real experiment run, and not tied to loading actual
expert checkpoints from disk.

Two things are checked, deliberately kept SEPARATE because they exercise different guarantees:

  1. SINGLE STEP from a given opt_state (plain SGD; momentum; momentum+weight_decay+nesterov;
     momentum+weight_decay+dampening, no nesterov) -- exact match against torch.optim.SGD.step()
     for one step starting from the SAME opt_state (freshly empty, or pre-loaded with a
     momentum_buffer, mirroring a real loaded checkpoint's optimizer state). This is the ONLY
     usage pattern the three run_module.py callers actually rely on: every batch reloads a
     FRESH opt_state from disk (`torch.load(expert_opt_starts[it])`) and calls sgd_step exactly
     ONCE per param before discarding it -- so single-step correctness is what matters for
     experiment correctness.

  2. MULTIPLE SUCCESSIVE STEPS reusing the SAME opt_state dict across calls -- documents a
     REAL, currently-DORMANT bug in sgd_step: `buf = opt_state['momentum_buffer'] = ...` (or
     `buf = opt_state['momentum_buffer']`) followed by `buf = buf.mul(momentum).add(...)` (an
     OUT-OF-PLACE op) rebinds the LOCAL variable `buf` to a new tensor but never writes it back
     into `opt_state['momentum_buffer']` -- so the stored momentum buffer never advances past
     its initial (usually all-zero, for a fresh dict) value across repeated calls, and results
     diverge from torch.optim.SGD from the SECOND call onward. This test asserts step 0 matches
     exactly and step 1+ (predictably) do NOT, as a REGRESSION PIN, not a bug report -- do not
     "fix" this by making step 1+ pass; that would require changing sgd_step itself (a shared
     file, out of scope for this test, see the module's own docstring). If sgd_step is ever
     called more than once per param against the same opt_state (it currently is not, anywhere
     in the codebase), this divergence would become live and would need addressing then.

Run:  python prelim/tests/test_sgd_step.py
"""
import torch

from modules.federated_generate_labels.utils import sgd_step

_results = []


def check(name, ok, detail=""):
    _results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


CONFIGS = {
    "plain_sgd": dict(lr=0.1, momentum=0.0, weight_decay=0.0, nesterov=False, dampening=0.0),
    "momentum_only": dict(lr=0.05, momentum=0.9, weight_decay=0.0, nesterov=False, dampening=0.0),
    "momentum_weight_decay_nesterov": dict(
        lr=0.01, momentum=0.9, weight_decay=2e-4, nesterov=True, dampening=0.0,
    ),
    "momentum_weight_decay_dampening": dict(
        lr=0.02, momentum=0.8, weight_decay=1e-3, nesterov=False, dampening=0.3,
    ),
}


def _make_ref_and_opt_params(w0, cfg):
    w_ref = torch.nn.Parameter(w0.clone())
    opt = torch.optim.SGD(
        [w_ref], lr=cfg["lr"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"],
        nesterov=cfg["nesterov"], dampening=cfg["dampening"],
    )
    opt_params = opt.state_dict()["param_groups"][0]
    return w_ref, opt, opt_params


def test_single_step_from_fresh_state(cfg_name, cfg, seed):
    torch.manual_seed(seed)
    w0 = torch.randn(4, 4)
    g = torch.randn(4, 4)

    w_ref, opt, opt_params = _make_ref_and_opt_params(w0, cfg)
    w_ref.grad = g.clone()
    opt.step()

    w_manual = sgd_step(w0.clone(), g.clone(), {}, opt_params)

    gap = (w_ref.detach() - w_manual).abs().max().item()
    check(
        f"single step from FRESH opt_state matches torch.optim.SGD exactly [{cfg_name}]",
        gap < 1e-6, f"max_gap={gap:.3e}",
    )


def test_single_step_from_preloaded_momentum_buffer(cfg_name, cfg, seed):
    if cfg["momentum"] == 0.0:
        return  # no momentum_buffer to preload
    torch.manual_seed(seed)
    w0 = torch.randn(4, 4)
    g0 = torch.randn(4, 4)  # a "prior" gradient used only to seed a real momentum buffer
    g1 = torch.randn(4, 4)  # the step actually being compared

    w_ref, opt, opt_params = _make_ref_and_opt_params(w0, cfg)
    w_ref.grad = g0.clone()
    opt.step()  # seeds a real momentum_buffer inside opt's own state
    real_state = opt.state[w_ref]
    preloaded_state = {"momentum_buffer": real_state["momentum_buffer"].clone()}

    w_ref.grad = g1.clone()
    opt.step()  # torch's SECOND step, starting from the real (non-zero) momentum buffer

    w_after_first = w0 + 0  # sgd_step's single call starts from the value AFTER step 0
    # Reconstruct the value torch.optim.SGD produced after its OWN step 0, to hand sgd_step
    # the same starting point (this test isolates "does a single sgd_step call correctly
    # consume a PRE-EXISTING, non-zero momentum_buffer", independent of the multi-call
    # persistence bug documented separately below).
    w0_after_step0 = w_ref.detach().clone()  # placeholder overwritten below
    # Recompute independently: replay step 0 via a fresh optimizer to get its exact output.
    w_ref2, opt2, _ = _make_ref_and_opt_params(w0, cfg)
    w_ref2.grad = g0.clone()
    opt2.step()
    w0_after_step0 = w_ref2.detach().clone()

    w_manual = sgd_step(w0_after_step0, g1.clone(), preloaded_state, opt_params)
    gap = (w_ref.detach() - w_manual).abs().max().item()
    check(
        f"single step consuming a PRE-LOADED (non-zero) momentum_buffer matches torch "
        f"exactly [{cfg_name}]",
        gap < 1e-6, f"max_gap={gap:.3e}",
    )


def test_multi_step_reuse_diverges_as_documented(cfg_name, cfg, seed):
    if cfg["momentum"] == 0.0:
        return  # the non-persistence bug only manifests when momentum accumulates
    torch.manual_seed(seed)
    w0 = torch.randn(4, 4)

    w_ref, opt, opt_params = _make_ref_and_opt_params(w0, cfg)
    w_manual = w0.clone()
    opt_state = {}

    torch.manual_seed(seed + 1000)
    grads = [torch.randn(4, 4) for _ in range(3)]
    gaps = []
    for g in grads:
        w_ref.grad = g.clone()
        opt.step()
        w_manual = sgd_step(w_manual, g.clone(), opt_state, opt_params)
        gaps.append((w_ref.detach() - w_manual).abs().max().item())

    check(
        f"multi-step reuse of the SAME opt_state: step 0 matches exactly [{cfg_name}]",
        gaps[0] < 1e-6, f"gap={gaps[0]:.3e}",
    )
    check(
        f"multi-step reuse of the SAME opt_state: step 1+ DIVERGE (documented dormant bug, "
        f"NOT to be silently fixed here -- see module docstring) [{cfg_name}]",
        gaps[1] > 1e-3, f"gap={gaps[1]:.3e}",
    )


if __name__ == "__main__":
    for i, (name, cfg) in enumerate(CONFIGS.items()):
        test_single_step_from_fresh_state(name, cfg, seed=10 + i)
        test_single_step_from_preloaded_momentum_buffer(name, cfg, seed=20 + i)
        test_multi_step_reuse_diverges_as_documented(name, cfg, seed=30 + i)

    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} checks passed.")
    import sys
    sys.exit(1 if n_fail else 0)
