"""
calibrate_z_budget.py -- P5 companion script (see modules/federated_generate_labels_trigger_joint
/run_module.py's `z_budget` doc and coordinate_budget_penalty, joint/utils.py).

z_budget must be calibrated empirically, not guessed (see the accompanying diagnostic writeup,
"Ce qu'il ne faut pas faire"): it says how many honest per-coordinate standard deviations a
poisoned update is allowed to deviate by before L_budget starts penalizing it. The right value
depends on the DEFENSE (agg_method) actually deployed, not on the attack -- a value calibrated
against "mean" tells you nothing about trimmed-mean or Krum.

Method: draw REAL honest per-client gradients from a real checkpoint (several honest mini-batch
gradients, exactly the same clf_loss(model(x), y).backward() computation run_module.py's own
honest branch performs), estimate their per-coordinate mean/std, then for a grid of z values
build a SYNTHETIC colluding-poison contribution offset by exactly z standard deviations per
coordinate (the worst case for a coordinate-wise defense: uniform, maximally coherent), pass
[honest_grads..., poison_grad (repeated `num_poisoned` times)] through THIS REPO'S OWN agg()
(modules/federated_generate_labels/utils.py -- the exact function run_module.py's agg_expert_grads
call uses) for each agg_method, and measure what FRACTION of the intended z*sigma offset actually
survives in the aggregated output. Plots survival fraction vs z per agg_method and prints the
largest z (per agg_method) whose perturbation still survives non-negligibly (>= SURVIVAL_CUTOFF,
default 0.5) -- pass that value (or a little under it) as z_budget for that agg_method.

Must run on a machine with a GPU (load_model()/.cuda() is unconditional in this repo, see
modules/base_utils/util.py) and access to the CIFAR data path configured in
modules/base_utils/datasets.py's PATH -- i.e. on the cluster, not this laptop.

Usage (from BASE_DIR, e.g. via orchestrate_slurm/orchestrate_runs_trigger_joint_hardening_slurm.sh
STEP=3, or standalone):
    python -m prelim.calibrate_z_budget \\
        --checkpoint /path/to/model_<expert>_<traj>_<step>.pth \\
        --model-flag r32p --dataset cifar \\
        --num-honests 7 --num-poisoned 3 \\
        --agg-methods mean median trmean multikrum \\
        --z-values 0.1 0.2 0.3 0.5 0.7 1.0 1.5 2.0 3.0 \\
        --out prelim/artifacts/z_budget_calibration.png
"""
import argparse
from pathlib import Path

import torch
from torchvision import transforms

from modules.base_utils.util import load_model, get_module_device, clf_loss
from modules.base_utils.datasets import load_dataset
from modules.federated_generate_labels.utils import agg


class _ToTensorDataset(torch.utils.data.Dataset):
    """Thin wrapper applying transforms.ToTensor() to a raw (PIL-image, label) torchvision
    dataset (load_dataset's own return, see datasets.py -- no transform is applied there).
    Deliberately skips per-dataset normalization: this script only needs REPRESENTATIVE honest
    gradient statistics for a z-threshold search, not bit-for-bit fidelity to the main
    training pipeline's own normalization -- if you need that fidelity, swap this for the
    normalization your `train_expert` config actually uses."""

    def __init__(self, base):
        self.base = base
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        return self.to_tensor(x), y


def collect_honest_grads(model, loader, n_honest, device):
    """Runs n_honest independent mini-batch gradient computations (one per synthetic honest
    client), exactly mirroring run_module.py's own honest-branch computation
    (clf_loss(model(x), y).backward()), and returns a list (length n_honest) of lists
    (length len(params)) of detached per-parameter gradient tensors."""
    params = list(model.parameters())
    honest_grads = []
    it = iter(loader)
    for _ in range(n_honest):
        x, y = next(it)
        x, y = x.to(device), y.to(device)
        model.zero_grad()
        loss = clf_loss(model(x), y)
        loss.backward()
        honest_grads.append([p.grad.detach().clone() for p in params])
    return honest_grads


def survival_fraction(agg_out, mu_h, z, sd_h):
    """||agg_out - mu_h|| / ||z*sd_h|| (both flattened/concatenated across every parameter) --
    the fraction of the INTENDED per-coordinate offset (exactly z*sd_h by construction) that
    survives in the aggregator's actual output. 1.0 means the perturbation passed through
    completely unattenuated (e.g. under "mean" with enough poisoned mass); near 0.0 means the
    defense rejected it almost entirely."""
    diff = torch.cat([(o - m).reshape(-1) for o, m in zip(agg_out, mu_h)])
    intended = torch.cat([(z * s).reshape(-1) for s in sd_h])
    return (diff.norm() / (intended.norm() + 1e-12)).item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a model_*.pth state_dict.")
    parser.add_argument("--model-flag", default="r32p")
    parser.add_argument("--dataset", default="cifar")
    parser.add_argument("--n-classes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--num-honests", type=int, default=7,
        help="Number of honest mini-batch 'clients' to draw -- match the VICTIM deployment's "
             "own num_honests (e.g. 7 for a 3-poisoned/7-honest campaign) so trmean/multikrum's "
             "own `f` parameter operates over a realistic total client count.",
    )
    parser.add_argument(
        "--num-poisoned", type=int, default=3,
        help="Number of (identical, colluding) poisoned contributions inserted alongside the "
             "honest ones, and the `f` (tolerated-Byzantine-count) passed to trmean/multikrum -- "
             "match the VICTIM deployment's own num_poisoned.",
    )
    parser.add_argument(
        "--agg-methods", nargs="+",
        default=["mean", "median", "trmean", "multikrum"],
        help="Subset of modules.federated_generate_labels.utils.agg's supported methods.",
    )
    parser.add_argument(
        "--z-values", nargs="+", type=float,
        default=[0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0],
    )
    parser.add_argument(
        "--survival-cutoff", type=float, default=0.5,
        help="Reported z_budget suggestion per agg_method is the LARGEST z whose survival "
             "fraction is still >= this threshold.",
    )
    parser.add_argument("--out", default="prelim/artifacts/z_budget_calibration.png")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    model = load_model(args.model_flag, args.n_classes)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)
    device = get_module_device(model)
    model.eval()

    raw_dataset = load_dataset(args.dataset, train=True)
    dataset = _ToTensorDataset(raw_dataset)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
    )

    print(
        f"Drawing {args.num_honests} honest mini-batch gradients "
        f"(batch_size={args.batch_size}) from {args.checkpoint} ..."
    )
    honest_grads = collect_honest_grads(model, loader, args.num_honests, device)
    params = list(model.parameters())

    mu_h = [torch.stack([g[i] for g in honest_grads], dim=0).mean(dim=0) for i in range(len(params))]
    sd_h = [
        torch.stack([g[i] for g in honest_grads], dim=0).std(dim=0) + 1e-8
        for i in range(len(params))
    ]

    results = {method: [] for method in args.agg_methods}
    for z in args.z_values:
        poison_grad = [mu_h[i] + z * sd_h[i] for i in range(len(params))]
        for method in args.agg_methods:
            grad_buf = [
                [honest_grads[c][i] for c in range(args.num_honests)]
                + [poison_grad[i] for _ in range(args.num_poisoned)]
                for i in range(len(params))
            ]
            agg_out = agg(
                params, grad_buf, method,
                f=args.num_poisoned, n=args.num_honests + args.num_poisoned,
            )
            frac = survival_fraction(agg_out, mu_h, z, sd_h)
            results[method].append(frac)
            print(f"  z={z:.3g}  agg_method={method:10s}  survival_fraction={frac:.4f}")

    print("\nSuggested z_budget per agg_method (largest z with survival >= "
          f"{args.survival_cutoff}):")
    suggestions = {}
    for method in args.agg_methods:
        fracs = results[method]
        candidates = [z for z, f in zip(args.z_values, fracs) if f >= args.survival_cutoff]
        suggestion = max(candidates) if candidates else min(args.z_values)
        suggestions[method] = suggestion
        print(f"  {method:10s}: z_budget ~= {suggestion:.3g}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 4))
        for method in args.agg_methods:
            plt.plot(args.z_values, results[method], marker="o", label=method)
        plt.axhline(args.survival_cutoff, color="gray", linestyle="--", linewidth=1)
        plt.xlabel("z (honest std units)")
        plt.ylabel("survival fraction")
        plt.title("Per-coordinate perturbation survival vs z, by aggregator")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.out, dpi=150)
        print(f"\nPlot written to {args.out}")
    except ImportError:
        print("\nmatplotlib not available -- skipping plot, see printed values above.")


if __name__ == "__main__":
    main()
