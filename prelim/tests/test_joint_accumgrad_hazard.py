"""
prelim/tests/test_joint_accumgrad_hazard.py -- pins down, on a tiny SYNTHETIC model, the
AccumulateGrad-in-place hazard hit while building
modules/federated_generate_labels_trigger_joint/run_module.py's E2 (real coupling).

NO LONGER PROTECTS ANY LIVE CODE PATH (kept as a general PyTorch-behavior pin): the federated
expert-aggregation refactor (see docs/threat_models_audit.md, "agrégation fédérée
différentiable de l'expert") removed the deferred `optimizer_expert.step()` / manual
`p.grad = g.detach().clone()` re-assignment this test was originally written to justify --
`expert_params[i].grad` is now simply never assigned anywhere in that module (see its run()
docstring, E2 point 2a), so the early-assignment hazard this test demonstrates cannot occur
there any more. The underlying PyTorch behavior (AccumulateGrad mutating an already-assigned
`.grad` tensor in place) is still real and still worth having pinned, in case a future change
reintroduces an early `.grad` assignment pattern anywhere in this module family.

Setting a leaf parameter's `.grad` (even from a `.detach().clone()` snapshot) BEFORE a LATER
`.backward()` call that also reaches that same leaf (because it is itself a leaf of some
OTHER tensor's create_graph=True graph) is unsafe: PyTorch's AccumulateGrad adds into
whatever tensor `.grad` currently POINTS TO, in place -- mutating the "snapshot" even though
it looked like an independent clone. Setting `.grad` for the first time only AFTER that
`.backward()` call (so nothing accumulates into it afterward) is what actually gives a stable
value. This is not a hypothetical: the module hit exactly this via
expert_params/grads_e/grand_loss.backward(). Runs in well under a second -- no dataset, no
real model.

Run:  python prelim/tests/test_joint_accumgrad_hazard.py
"""
import torch
import torch.nn as nn

_results = []


def check(name, ok, detail=""):
    _results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def make_problem():
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(4, 4))
    x = torch.randn(4, requires_grad=True)  # stand-in for delta: something downstream also needs grad w.r.t.
    loss1 = (w @ x).pow(2).sum()
    grads = torch.autograd.grad(loss1, [w], create_graph=True, retain_graph=True)
    g = grads[0]  # differentiable in w (w is a leaf of g's OWN graph) and in x

    # A second, later loss that depends on g (hence on w, second-order) AND on x -- mirrors
    # expert_next depending on grads_e_last, and grand_loss depending on delta.
    loss2 = (g.sum() + x.sum()) ** 2
    return w, x, g, loss2


def test_early_grad_assignment_gets_corrupted():
    w, x, g, loss2 = make_problem()
    snapshot = g.detach().clone()
    w.grad = snapshot  # EARLY assignment -- the bug pattern
    ptr_before = snapshot.data_ptr()
    val_before = snapshot.clone()

    loss2.backward()  # w is a leaf of g's graph -> AccumulateGrad adds into w.grad in place

    same_storage = (w.grad.data_ptr() == ptr_before)
    changed_value = not torch.allclose(w.grad, val_before)
    check("early p.grad assignment: AccumulateGrad mutates the SAME storage in place",
          same_storage, f"ptr before={ptr_before}, ptr after={w.grad.data_ptr()}")
    check("early p.grad assignment: value silently changes (the actual bug)",
          changed_value, f"before={val_before.tolist()}, after={w.grad.tolist()}")


def test_deferred_grad_assignment_is_stable():
    w, x, g, loss2 = make_problem()
    w.grad = None  # deliberately UNSET -- the fix

    loss2.backward()  # populates w.grad with ONLY the second-order term; harmless, discarded next

    snapshot = g.detach().clone()  # taken AFTER the only .backward() call this "batch"
    val_at_snapshot = snapshot.clone()
    w.grad = snapshot

    # No further backward calls -- nothing left to corrupt it.
    check("deferred p.grad assignment (after backward, nothing calls backward again): stable",
          torch.equal(w.grad, val_at_snapshot))


if __name__ == "__main__":
    test_early_grad_assignment_gets_corrupted()
    test_deferred_grad_assignment_is_stable()
    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} checks passed.")
    import sys
    sys.exit(1 if n_fail else 0)
