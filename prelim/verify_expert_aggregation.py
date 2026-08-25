"""
prelim/verify_expert_aggregation.py -- Step 4 verifications (A, B, C) for the federated
expert-gradient aggregation refactor in federated_generate_labels_trigger_joint/run_module.py
(and the analogous fix in federated_generate_labels_trigger/run_module.py). Synthetic, no
dataset/checkpoints needed -- reproduces the EXACT client-loop / agg() / autograd.grad pattern
the real modules use, on a tiny model, so each property can be checked in isolation and fast.

  A. All clients contribute to the aggregate (not just the last), and the aggregate is
     order-invariant under mean (permuting client processing order gives an identical result).
  B. Poisoned-client gradients reach delta.grad through the MTT/aggregation path -- isolable
     from the separate L_bd path by testing with the L_bd-equivalent term's coefficient at 0.
  C. Numerical correctness: agg(..., "mean") exactly equals the elementwise mean over clients,
     both in VALUE and in GRADIENT (i.e. d(mean aggregate)/d(delta) == mean of the per-client
     d(individual grad)/d(delta) terms) -- confirms differentiability survives aggregation, not
     just that the forward value is right.

  D (no real expert step / no p.grad dependency) is verified separately by source inspection
  (grep -- no optimizer_expert.step()/.load_state_dict() calls remain in the joint module) and
  by prelim/run_verification_d.py's real-checkpoint integration check.

Run:  python prelim/verify_expert_aggregation.py
"""
import torch

from modules.federated_generate_labels.utils import agg

_results = []


def check(name, ok, detail=""):
    _results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def make_clients(delta, num_honests, num_poisoned, seed=0):
    """
    Mirrors the real loop's structure: honest clients produce a DETACHED (real-backward-style)
    gradient on a shared param `w`, independent of delta; poisoned clients produce a
    DIFFERENTIABLE (create_graph=True) gradient on `w` that ALSO depends on delta (mirroring
    x_t_adv's live dependency on delta via T_delta). Returns (w, grad_buf) exactly as
    expert_grad_buf is built in run_module.py, plus the raw per-client grads for reference.
    """
    torch.manual_seed(seed)
    w = torch.nn.Parameter(torch.randn(6))
    grad_buf = [[]]
    per_client_grads = []

    for cid in range(num_honests + num_poisoned):
        if cid < num_honests:
            # Honest: real (non-differentiable) backward, like loss.backward() + p.grad.
            loss = (w * (cid + 1)).pow(2).sum()
            loss.backward()
            g = w.grad.detach().clone()
            w.grad = None
            grad_buf[0].append(g)
            per_client_grads.append(g)
        else:
            # Poisoned: differentiable in w AND delta, like grads_e = autograd.grad(loss_e,
            # expert_params, create_graph=True).
            x_adv = delta * (cid + 1)  # stand-in for x_t_adv's dependency on delta
            loss_e = (w * x_adv.sum()).pow(2).sum()
            (g,) = torch.autograd.grad(loss_e, [w], create_graph=True, retain_graph=True)
            grad_buf[0].append(g)  # NOT detached -- E2, point 2a
            per_client_grads.append(g)

    return w, grad_buf, per_client_grads


def test_A_all_clients_contribute_and_order_invariant():
    delta = torch.randn(6, requires_grad=True)
    w, grad_buf, per_client_grads = make_clients(delta, num_honests=2, num_poisoned=2, seed=1)
    agg_grads = agg([w], grad_buf, "mean", f=2)
    result = agg_grads[0]

    manual_mean = torch.stack(per_client_grads, dim=0).mean(dim=0)
    check(
        "A: mean aggregate over 4 clients matches manual stack-mean (not just last client)",
        torch.allclose(result.detach(), manual_mean.detach(), atol=1e-6),
    )

    # Reorder clients (swap client 0 and 3) and re-aggregate -- mean must be identical. Reuses
    # the SAME delta VALUE (fresh leaf, same numbers) so only the client order differs -- a
    # freshly-drawn random delta would make per-client grads genuinely different numbers and
    # invalidate the comparison.
    delta2 = delta.detach().clone().requires_grad_(True)
    w2, grad_buf2, per_client_grads2 = make_clients(delta2, num_honests=2, num_poisoned=2, seed=1)
    reordered = [per_client_grads2[3], per_client_grads2[1], per_client_grads2[2], per_client_grads2[0]]
    agg_reordered = agg([w2], [reordered], "mean", f=2)[0]
    check(
        "A: mean aggregate is order-invariant under a client-order permutation",
        torch.allclose(result.detach(), agg_reordered.detach(), atol=1e-6),
        f"orig={result.detach().tolist()[:2]}..., reordered={agg_reordered.detach().tolist()[:2]}...",
    )

    # Sanity: aggregate actually DIFFERS from any single client's own grad (i.e. genuinely an
    # aggregate, not a last-client-wins passthrough).
    last_client = per_client_grads[-1].detach()
    check(
        "A: aggregate differs from the LAST client's own gradient (not last-wins)",
        not torch.allclose(result.detach(), last_client, atol=1e-6),
    )


def test_B_poisoned_grad_reaches_delta_through_aggregation():
    for agg_method, f in [("mean", 2), ("median", 2), ("trmean", 1)]:
        delta = torch.randn(6, requires_grad=True)
        w, grad_buf, _ = make_clients(delta, num_honests=1, num_poisoned=3, seed=2)
        agg_grads = agg([w], grad_buf, agg_method, f=f)
        loss = agg_grads[0].sum() ** 2  # stand-in for param_loss's dependence on expert_next
        (delta_grad,) = torch.autograd.grad(loss, [delta], allow_unused=True)
        check(
            f"B: poisoned-client gradient reaches delta.grad through the aggregate "
            f"[agg_method={agg_method}]",
            delta_grad is not None and delta_grad.abs().sum().item() > 0,
            f"delta_grad_norm={delta_grad.norm().item() if delta_grad is not None else None}",
        )


def test_C_mean_aggregation_numerically_exact_value_and_gradient():
    delta = torch.randn(6, requires_grad=True)
    w, grad_buf, per_client_grads = make_clients(delta, num_honests=1, num_poisoned=2, seed=3)
    agg_grads = agg([w], grad_buf, "mean", f=1)
    result = agg_grads[0]

    manual_mean = torch.stack(per_client_grads, dim=0).mean(dim=0)
    check(
        "C: agg(mean) VALUE == manual elementwise mean over per-client grads",
        torch.allclose(result.detach(), manual_mean.detach(), atol=1e-7),
    )

    # Gradient check: d(sum(agg_mean))/d(delta) must equal the mean of each client's own
    # d(sum(individual grad))/d(delta) -- confirms the aggregate stays a genuine differentiable
    # function of delta through EVERY poisoned client, not just structurally resembling one.
    loss_agg = result.sum()
    (dg_agg,) = torch.autograd.grad(loss_agg, [delta], retain_graph=True, allow_unused=True)

    delta2 = delta.detach().clone().requires_grad_(True)
    w2, _, per_client_grads2 = make_clients(delta2, num_honests=1, num_poisoned=2, seed=3)
    per_client_dgrads = []
    for g in per_client_grads2:
        if g.requires_grad:
            (dg,) = torch.autograd.grad(g.sum(), [delta2], retain_graph=True, allow_unused=True)
            per_client_dgrads.append(dg if dg is not None else torch.zeros_like(delta2))
    manual_dg_mean = torch.stack(per_client_dgrads, dim=0).sum(dim=0) / len(per_client_grads2)
    check(
        "C: d(agg_mean)/d(delta) == mean of per-client d(grad)/d(delta) (differentiability "
        "survives aggregation, not just the forward value)",
        torch.allclose(dg_agg.detach(), manual_dg_mean.detach(), atol=1e-6),
        f"agg={dg_agg.detach().tolist()[:2]}..., manual={manual_dg_mean.detach().tolist()[:2]}...",
    )


if __name__ == "__main__":
    test_A_all_clients_contribute_and_order_invariant()
    test_B_poisoned_grad_reaches_delta_through_aggregation()
    test_C_mean_aggregation_numerically_exact_value_and_gradient()

    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} checks passed.")
    import sys
    sys.exit(1 if n_fail else 0)
