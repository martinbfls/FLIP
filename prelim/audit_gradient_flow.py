"""
prelim/audit_gradient_flow.py -- §5 of docs/threat_models_audit.md.

Verifies, by actually running backward() and inspecting .grad, which loss term sends gradient
to which optimized variable in each of the three threat-model modules. Uses a tiny SYNTHETIC
toy model (2-layer MLP, C=3 classes, a handful of examples) and reproduces each module's
CHARACTERISTIC autograd pattern -- the specific detach()/torch.autograd.grad/create_graph
calls that determine gradient flow -- rather than calling into the real modules, which need a
downloaded dataset and real expert checkpoints to run at all (get_matching_datasets,
extract_experts, MTTDataset all assume CIFAR-family data on disk). This is a deliberate,
disclosed limitation, not an oversight -- see docs/threat_models_audit.md §5 for which parts
of each module this can and cannot stand in for.

No dataset, no training, no GPU. Runs in under a second.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

D_IN, D_HIDDEN, C = 6, 8, 3
N = 5

results = []  # (module, variable, term, nonzero)


def toy_model():
    m = nn.Sequential(nn.Linear(D_IN, D_HIDDEN), nn.ReLU(), nn.Linear(D_HIDDEN, C))
    return m


def sgd_step_like(param, grad, lr=0.1):
    '''Same formula as federated_generate_labels.utils.sgd_step with momentum=0,
    weight_decay=0, nesterov=False -- the differentiable-SGD pattern both direct-family
    modules use to turn a real optimizer step into a function of upstream tensors (labels_syn
    or, in the joint module, delta).'''
    return param - lr * grad


def record(module, variable, term, tensor=None, not_tested=False):
    '''Three distinct outcomes, not two: "not_tested" (this script never computed the
    quantity -- e.g. labels_syn's real gradient path, see the note where it's used) must be
    visually and programmatically distinct from "tested, gradient is exactly zero". A table
    that prints "no" for an untested case is worse than an empty cell -- it reads as a
    negative finding when it is really an absence of evidence.'''
    if not_tested:
        status = "not_tested"
    else:
        nonzero = tensor is not None and tensor.detach().abs().sum().item() > 0
        status = "YES" if nonzero else "no"
    results.append((module, variable, term, status))
    label = {"YES": "NONZERO", "no": "zero", "not_tested": "NOT TESTED"}[status]
    print(f"[{module:38s}] {variable:10s} <- {term:22s} : {label}")


# --------------------------------------------------------------------------------- #
# federated_generate_labels_trigger (INDIRECT coupling) -- reproduces:
#   x_t_adv[mask] = x_trig.detach()          (run_module.py:278)
#   loss_e.backward()                         (run_module.py:283, populates expert_params.grad)
#   param_loss/param_dist built from `expert` = the REAL post-step expert_params (run_module.py
#   :338-350) -- NOT a function of delta, since loss_e never saw an undetached x_trig.
#   L_bd_cid isolated: torch.autograd.grad(lambda_bd*L_bd_cid, [delta], ...) (run_module.py:295)
# --------------------------------------------------------------------------------- #
print("\n=== federated_generate_labels_trigger (indirect) ===")

expert = toy_model()
x = torch.randn(N, D_IN)
y = torch.randint(0, C, (N,))
delta = (torch.rand(D_IN) * 0.2 + 0.05).requires_grad_(True)  # NOT zero -- see note below
labels_syn = torch.randn(N, C, requires_grad=True)

mask = torch.zeros(N, dtype=torch.bool)
mask[0] = True  # one "genuinely poisoned" row
x_trig = x[mask] + delta  # T_delta(x) = x + delta, same as raw_to_trigger_preprocess's add
x_t_adv = x.clone()
x_t_adv[mask] = x_trig.detach()  # <-- the detach that defines this module

expert.zero_grad()
loss_e = F.cross_entropy(expert(x_t_adv), y)
loss_e.backward()  # populates expert_params.grad -- NOT connected to delta (detached input)

expert_start = [p.detach().clone() for p in expert.parameters()]
expert_post = [p.detach().clone() - 0.1 * p.grad for p in expert.parameters()]  # "real" post-step

# param_loss uses `expert_post` directly (real optimizer result), like run_module.py:349 uses
# `expert` (expert_params AFTER optimizer_expert.step()) -- never touches delta's graph.
student = toy_model()
for p, p0 in zip(student.parameters(), expert_start):
    p.data.copy_(p0)
grads_s = torch.autograd.grad(
    F.cross_entropy(student(x), y), list(student.parameters()), create_graph=True,
)
student_update = [sgd_step_like(p0, g) for p0, g in zip(expert_start, grads_s)]
param_loss = sum(F.mse_loss(su, ep, reduction="sum") for su, ep in zip(student_update, expert_post))
grand_loss = param_loss  # (+ reg_term, irrelevant to delta/labels_syn gradient flow)

logits_bd = expert(x_trig)  # fresh forward, x_trig still delta-connected (not detached here)
L_bd = F.cross_entropy(logits_bd, y[mask])
(delta_grad_isolated,) = torch.autograd.grad(L_bd, [delta], retain_graph=True, allow_unused=True)

grand_loss.backward()  # delta.grad untouched by this call if the module is faithful
delta.grad = (delta_grad_isolated.detach() if delta.grad is None
              else delta.grad + delta_grad_isolated.detach())

record("federated_generate_labels_trigger", "delta", "param_loss (grand_loss)",
       None if delta.grad is None else (delta.grad - delta_grad_isolated).clone())
record("federated_generate_labels_trigger", "delta", "L_bd (isolated)", delta_grad_isolated)
record("federated_generate_labels_trigger", "labels_syn", "param_loss (via loss_s)", not_tested=True)
print("  (labels_syn never entered this toy reproduction's loss_s -- see docs/audit.md §5 note: "
      "labels_syn's real gradient path, through loss_s=clf_loss(student_model(x_d), softmax(y_d)) "
      "with y_d=labels_syn[idx], is structurally the SAME create_graph=True pattern as delta's in "
      "the joint module below; not re-derived here to avoid duplicating that check.")


# --------------------------------------------------------------------------------- #
# federated_generate_labels_trigger_joint (REAL coupling) -- reproduces:
#   x_t_adv[mask] = x_trig               (NOT detached; run_module.py:381)
#   grads_e = torch.autograd.grad(loss_e, expert_params, create_graph=True, ...) (run_module.py
#   :400-402); expert_next = sgd_step(expert_start, grads_e) (run_module.py:502-504) --
#   DIFFERENTIABLE in delta, used AS param_loss's target instead of the real post-step params.
#   L_bd_cid isolated exactly as above (an ADDITIONAL contribution, run_module.py:417-425).
# --------------------------------------------------------------------------------- #
print("\n=== federated_generate_labels_trigger_joint (real coupling) ===")

expert2 = toy_model()
# NOT zero: at delta=0 (with expert2/student2 sharing the same initial weights and the
# poisoned row's input reducing to the exact clean input) expert_next and student_update
# coincide exactly, making mse_loss's (A-B) factor -- hence d(mse)/d(delta) -- vanish at
# THIS specific point even though expert_next itself demonstrably depends on delta (verified
# separately). A degenerate artifact of this toy script's init, not a module property.
delta2 = (torch.rand(D_IN) * 0.2 + 0.05).requires_grad_(True)

x_trig2 = x[mask] + delta2
x_t_adv2 = x.clone()
x_t_adv2[mask] = x_trig2  # NOT detached

expert_params2 = list(expert2.parameters())
loss_e2 = F.cross_entropy(expert2(x_t_adv2), y)
grads_e2 = torch.autograd.grad(loss_e2, expert_params2, create_graph=True, retain_graph=True)

expert_start2 = [p.detach().clone() for p in expert_params2]
expert_next2 = [sgd_step_like(p0, g) for p0, g in zip(expert_start2, grads_e2)]  # fn of delta2

student2 = toy_model()
for p, p0 in zip(student2.parameters(), expert_start2):
    p.data.copy_(p0)
grads_s2 = torch.autograd.grad(
    F.cross_entropy(student2(x), y), list(student2.parameters()), create_graph=True,
)
student_update2 = [sgd_step_like(p0, g) for p0, g in zip(expert_start2, grads_s2)]
param_loss2 = sum(
    F.mse_loss(su, en, reduction="sum") for su, en in zip(student_update2, expert_next2)
)
grand_loss2 = param_loss2

logits_bd2 = expert2(x_trig2)
L_bd2 = F.cross_entropy(logits_bd2, y[mask])
(delta_grad_isolated2,) = torch.autograd.grad(L_bd2, [delta2], retain_graph=True, allow_unused=True)
delta2.grad = delta_grad_isolated2.detach().clone()

grand_loss2.backward()  # should ALSO reach delta2, through expert_next2 -- unlike the indirect module

record("federated_generate_labels_trigger_joint", "delta", "param_loss (grand_loss)",
       None if delta2.grad is None else (delta2.grad - delta_grad_isolated2))
record("federated_generate_labels_trigger_joint", "delta", "L_bd (isolated)", delta_grad_isolated2)
record("federated_generate_labels_trigger_joint", "delta", "TOTAL delta.grad (both terms)", delta2.grad)


# --------------------------------------------------------------------------------- #
# federated_optimizing_trigger_policy -- reproduces the (P^mean) objective directly (this
# module's per-checkpoint objective, unlike the two above, has no dataset/expert-checkpoint
# dependency to fake: G_obj/u/v are exactly (D,P)/(P,)/(D,) tensors already, so this is not a
# stand-in pattern but the ACTUAL formula from _compute_step_policy, run_module.py:221-227,
# on toy-sized tensors instead of real flattened-parameter gradients.
# --------------------------------------------------------------------------------- #
print("\n=== federated_optimizing_trigger_policy ===")

Dp = 12  # toy "flattened parameter" dimension
P = 4    # toy number of (y,c) pairs
G_obj = torch.randn(Dp, P)  # stands in for the cached, gamma/pi_y-rescaled G_obj
u = torch.zeros(P, requires_grad=True)
delta3 = (torch.rand(D_IN) * 0.2 + 0.05).requires_grad_(True)

# v = mu_p - g_c, a function of delta via a toy "poisoned gradient" mu_p(delta): stands in for
# compute_batch_gradients's flattened per-parameter gradient on a triggered batch.
mu_p = torch.stack([torch.sin(delta3).sum() + i for i in range(Dp)])
g_c = torch.zeros(Dp)
v = mu_p - g_c

Gu = G_obj @ u
rho = 1.0
B2 = ((Gu - v) ** 2).sum() / (rho ** 2)
B2.backward()

record("federated_optimizing_trigger_policy", "u", "B2 (via G_obj@u)", u.grad)
record("federated_optimizing_trigger_policy", "delta", "B2 (via v(delta))", delta3.grad)
print("  (no `labels`/`labels_syn` variable exists in this module -- u IS the decision "
      "variable on the labels side; see docs/audit.md §1's axis table.)")


print("\n=== Summary table (module x variable x term -> gradient status) ===")
print(f"{'module':40s} {'variable':12s} {'term':26s} status")
for module, variable, term, status in results:
    print(f"{module:40s} {variable:12s} {term:26s} {status}")
