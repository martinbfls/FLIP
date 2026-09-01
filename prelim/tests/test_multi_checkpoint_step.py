"""
prelim/tests/test_multi_checkpoint_step.py -- pins down, on a tiny SYNTHETIC model, why
modules/federated_generate_labels_trigger_joint/run_module.py's multi-checkpoint averaging
(n_checkpoints_per_step > 1, see run() docstring "Averaging the loss over multiple checkpoints")
MUST use n_checkpoints_per_step DISTINCT, persistent model instances rather than reloading ONE
model object's state_dict once per checkpoint within the same step.

The hazard: `torch.autograd.grad(loss_e, params, create_graph=True, retain_graph=True)` builds a
graph whose backward Functions save_for_backward the LEAF parameter tensors themselves (e.g.
nn.Linear's backward needs its own weight). If a LATER checkpoint's `model.load_state_dict(...)`
mutates those SAME leaf tensors' `.data` in place BEFORE the combined grand_loss.backward() call
that still needs the EARLIER checkpoint's graph, PyTorch's per-tensor version counter detects the
mutation and raises "one of the variables needed for gradient computation has been modified by
an inplace operation" -- a hard, unambiguous failure (not a silent correctness bug), but one that
only manifests when a SECOND checkpoint's graph is built and the whole thing is backwarded
together, i.e. exactly the multi-checkpoint-averaging code path.

The fix (what run_module.py actually does): expert_models[k]/student_models[k] are
n_checkpoints_per_step SEPARATE nn.Module instances, each loaded from its own checkpoint ONCE
per batch-step and never touched again until the NEXT batch-step (by which point this step's
single, combined backward() has already run) -- see run_module.py's own comment at the
expert_models/student_models construction site.

Run:  python prelim/tests/test_multi_checkpoint_step.py
"""
import torch
import torch.nn as nn

_results = []


def check(name, ok, detail=""):
    _results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def _two_checkpoints():
    torch.manual_seed(0)
    template = nn.Linear(4, 4)
    ckpt0 = {k: v.clone() for k, v in template.state_dict().items()}
    ckpt1 = {k: v.clone() + 1.0 for k, v in template.state_dict().items()}
    return ckpt0, ckpt1


def test_reusing_one_model_across_checkpoints_raises():
    """The hazard: ONE model object, load_state_dict'ed between two checkpoints' graph
    constructions, then a SINGLE combined backward() -- mirrors what a naive multi-checkpoint
    implementation (reload-in-a-loop instead of separate persistent instances) would do."""
    ckpt0, ckpt1 = _two_checkpoints()
    model = nn.Linear(4, 4)
    x = torch.randn(4, requires_grad=True)  # stand-in for delta

    model.load_state_dict(ckpt0)
    loss_e0 = (model(x) ** 2).sum()
    grads_e0 = torch.autograd.grad(
        loss_e0, list(model.parameters()), create_graph=True, retain_graph=True,
    )

    # Reload the SAME model object for "checkpoint 1" -- the bug pattern.
    model.load_state_dict(ckpt1)
    loss_e1 = (model(x) ** 2).sum()
    grads_e1 = torch.autograd.grad(
        loss_e1, list(model.parameters()), create_graph=True, retain_graph=True,
    )

    grand_loss = sum(g.sum() for g in grads_e0) + sum(g.sum() for g in grads_e1)

    raised_inplace_error = False
    try:
        grand_loss.backward()
    except RuntimeError as e:
        raised_inplace_error = "in-place" in str(e).lower() or "inplace" in str(e).lower()
        detail = str(e)
    else:
        detail = "no exception raised"

    check(
        "reloading ONE model object between checkpoints raises an inplace-mutation "
        "RuntimeError at the combined backward()",
        raised_inplace_error, detail,
    )


def test_separate_model_instances_per_checkpoint_is_stable():
    """The fix: n_checkpoints_per_step DISTINCT model instances -- what run_module.py actually
    does (expert_models[k]/student_models[k])."""
    ckpt0, ckpt1 = _two_checkpoints()
    model0 = nn.Linear(4, 4)
    model1 = nn.Linear(4, 4)
    model0.load_state_dict(ckpt0)
    model1.load_state_dict(ckpt1)
    x = torch.randn(4, requires_grad=True)

    loss_e0 = (model0(x) ** 2).sum()
    grads_e0 = torch.autograd.grad(
        loss_e0, list(model0.parameters()), create_graph=True, retain_graph=True,
    )
    loss_e1 = (model1(x) ** 2).sum()
    grads_e1 = torch.autograd.grad(
        loss_e1, list(model1.parameters()), create_graph=True, retain_graph=True,
    )

    grand_loss = sum(g.sum() for g in grads_e0) + sum(g.sum() for g in grads_e1)

    backward_ok = False
    try:
        grand_loss.backward()
        backward_ok = x.grad is not None
    except RuntimeError as e:
        detail = f"unexpected RuntimeError: {e}"
    else:
        detail = f"x.grad={x.grad.tolist()}"

    check(
        "separate model instances per checkpoint: combined backward() succeeds, "
        "delta-stand-in receives a gradient",
        backward_ok, detail,
    )


if __name__ == "__main__":
    test_reusing_one_model_across_checkpoints_raises()
    test_separate_model_instances_per_checkpoint_is_stable()
    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} checks passed.")
    import sys
    sys.exit(1 if n_fail else 0)
