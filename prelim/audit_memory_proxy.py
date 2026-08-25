"""
prelim/audit_memory_proxy.py -- §6 of docs/threat_models_audit.md.

CPU substitute for the still-open GPU measurement (torch.cuda.max_memory_allocated is
unavailable on this machine). Runs the joint module's actual expert-step pattern (real r32p
model, a real smoke checkpoint already on disk from an earlier session -- no new download) with
create_graph=True (E2, as shipped) and separately with create_graph=False (the indirect
module's plain .backward()), EACH IN ITS OWN SUBPROCESS (a single process's RSS high-water-mark
only grows, so measuring both variants sequentially in one process would just report the max of
the two, not either one individually).

INCREMENTS, not totals: resource.getrusage(RUSAGE_SELF).ru_maxrss is a monotonically
nondecreasing PROCESS-WIDE high-water mark -- by the time the forward pass runs, it already
includes the interpreter, torch's own import, and the loaded model's weights (several hundred
MB, unrelated to create_graph). Comparing TOTAL peak RSS between the two variants therefore
mostly compares that shared constant, diluting the actual effect. Each worker measures its own
RSS twice: once right after the model/batch are built (baseline, BEFORE any forward/backward),
once after backward returns (peak) -- the reported increment (peak - baseline) is what
create_graph actually costs, with the constant subtracted out.

No node-counting here (an earlier version tried it and reported "5" for a full r32p forward,
which is wrong on its face -- most likely an artifact of `next_functions` traversal not
matching however this torch build represents the graph internally, not a real measurement;
rather than keep publishing a number known to be wrong, this version reports RSS only).

This is explicitly a PROXY, not the requested GPU measurement: CPU peak RSS reflects the
process allocator's high-water mark (glibc/macOS malloc arenas rarely release pages back to
the OS, and PyTorch's own caching allocator behaves differently on CPU vs the CUDA caching
allocator this was actually asked about) -- the ratio is indicative of order of magnitude only,
not a stand-in for the real GPU number, which stays open (see docs/threat_models_audit.md §9).
"""
import json
import platform
import subprocess
import sys
import textwrap

CHECKPOINT_DIR = (
    "/private/tmp/claude-501/-Users-martinbeaufils-Downloads-broadflip-repo-FLIP/"
    "1f0e6899-15d6-4c67-a0f9-936b7b87b76e/scratchpad/smoke/checkpoints/r32p_1xs/0"
)

WORKER_SCRIPT = textwrap.dedent(f"""
    import sys, json, resource
    import torch
    if not torch.cuda.is_available():
        torch.nn.Module.cuda = lambda self, device=None: self.to("cpu")
        torch.Tensor.cuda = lambda self, device=None, non_blocking=False: self.to("cpu")
    import os
    sys.path.insert(0, os.getcwd())
    from modules.base_utils.util import load_model, clf_loss

    create_graph = sys.argv[1] == "true"

    def rss_mb():
        unit = 1 if sys.platform == "darwin" else 1024  # macOS reports bytes, Linux reports KB
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit / (1024 ** 2)

    torch.manual_seed(0)
    model = load_model("r32p", 10)
    ckpt = torch.load("{CHECKPOINT_DIR}/model_1_10.pth", map_location="cpu")
    model.load_state_dict(ckpt)
    model.train()

    x = torch.rand(32, 3, 32, 32)  # a real-sized r32p batch, random (no dataset download needed)
    y = torch.randint(0, 10, (32,))
    delta = torch.zeros(3, 32, 32, requires_grad=True)
    x_adv = (x + delta).clamp(0, 1)
    params = list(model.parameters())

    # Baseline: model loaded, batch built, nothing forward/backward'd yet. This is where the
    # "interpreter + torch + model weights" constant lives -- excluded from the increment below.
    baseline_mb = rss_mb()

    model.zero_grad()
    loss = clf_loss(model(x_adv), y)

    if create_graph:
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
        # mirror the joint module's actual downstream use: one more differentiable step, so the
        # SECOND-ORDER graph create_graph=True is meant to keep alive is genuinely built and
        # backpropagated through here, not just constructed and immediately discarded.
        expert_next = [p.detach() - 0.1 * g for p, g in zip(params, grads)]
        probe = sum((en ** 2).sum() for en in expert_next)
        probe.backward()
    else:
        loss.backward()

    peak_mb = rss_mb()

    print(json.dumps({{
        "baseline_mb": baseline_mb,
        "peak_mb": peak_mb,
        "increment_mb": peak_mb - baseline_mb,
    }}))
""")


def run_variant(create_graph: bool):
    proc = subprocess.run(
        [sys.executable, "-c", WORKER_SCRIPT, "true" if create_graph else "false"],
        cwd="/Users/martinbeaufils/Downloads/broadflip_repo/FLIP",
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        raise RuntimeError(f"worker subprocess failed (create_graph={create_graph})")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main():
    print(f"Platform: {platform.system()} {platform.machine()} (CPU-only proxy, no CUDA)")
    with_cg = run_variant(create_graph=True)
    without_cg = run_variant(create_graph=False)

    print(f"\nWITH create_graph=True  (E2, as shipped):")
    print(f"  baseline (model+batch loaded, pre-forward): {with_cg['baseline_mb']:.2f} MB")
    print(f"  peak     (post-backward):                   {with_cg['peak_mb']:.2f} MB")
    print(f"  increment (peak - baseline):                {with_cg['increment_mb']:.2f} MB")

    print(f"\nWITHOUT create_graph (plain .backward(), indirect-module pattern):")
    print(f"  baseline (model+batch loaded, pre-forward): {without_cg['baseline_mb']:.2f} MB")
    print(f"  peak     (post-backward):                   {without_cg['peak_mb']:.2f} MB")
    print(f"  increment (peak - baseline):                {without_cg['increment_mb']:.2f} MB")

    if without_cg["increment_mb"] > 0:
        ratio_increment = with_cg["increment_mb"] / without_cg["increment_mb"]
    else:
        ratio_increment = float("nan")
    ratio_total = with_cg["peak_mb"] / without_cg["peak_mb"]

    print(f"\nRatio on INCREMENTS (peak-baseline, the actual create_graph cost): "
          f"{ratio_increment:.3f}")
    print(f"Ratio on TOTAL peak RSS (includes the shared ~constant, misleading -- kept only "
          f"for comparison against the earlier, wrong report): {ratio_total:.3f}")
    print(
        "\nCPU PROXY ONLY -- not the requested torch.cuda.max_memory_allocated ratio. Peak RSS "
        "is a process-wide, monotonically-nondecreasing high-water mark subject to allocator "
        "fragmentation/retention quirks unrelated to the CUDA caching allocator; treat the "
        "increment ratio above as order-of-magnitude guidance for whether r18 is plausible, not "
        "as the number to decide with. See docs/threat_models_audit.md §6/§9."
    )


if __name__ == "__main__":
    main()
