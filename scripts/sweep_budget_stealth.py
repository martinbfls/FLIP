"""
scripts/sweep_budget_stealth.py

For each beta in a grid and each epsilon in a grid: optimizes a trigger
against that (beta, epsilon) via modules.federated_optimizing_trigger's
optimize_trigger, retrains one more fresh expert on the resulting delta
(same lambda_poison=beta rate used by the objective, coupled into the
expert's poisoned train/test datasets via lambda_target -- see
get_poison_dataset), and measures:

  - CTA: clean test accuracy (clf_eval on the ordinary clean test set).
  - ASR: attack success rate -- accuracy, on the TRIGGERED-ONLY portion of
    the poisoned test set (excluding the untouched clean examples
    get_poison_dataset concatenates in), of predicting target_label.

lambda_overflow="duplicate" is used throughout (not the default "clip") so
that beta can be swept past beta_max = n_s / (n_train + n_s) (~=0.0909 for a
balanced 10-class CIFAR-10 source class) without the actual poison rate
silently saturating -- see get_poison_dataset's docstring.

Does NOT use ||delta|| as a stealth proxy. init_delta starts at strength=6.0
and optimize_trigger_step's `delta.clamp_(-epsilon, epsilon)` runs after
every step, so delta sits on the L_inf ball's boundary in practice and
||delta|| == epsilon * sqrt(d) regardless of beta -- it carries no
beta-dependent information. CTA (how much clean accuracy the poisoned
expert retains) is the stealth signal reported here instead.

Output: out/optimizing_trigger/budget_stealth.json --
  {
    "<beta>": {
      "min_epsilon_for_asr": <smallest epsilon in the grid with ASR >=
                              --asr_threshold, or null>,
      "curve": [{"epsilon": ..., "cta": ..., "asr": ...}, ...]
    },
    ...
  }

Cost: |beta_grid| * |epsilon_grid| independent optimize_trigger + one extra
expert-retrain calls. Each optimize_trigger call itself runs n_steps outer
(retrain `epochs` epochs + optimize num_chckpt-checkpoint batches) steps --
see the time-estimate note printed by --estimate_only before any real run
starts. Do not run this without first sizing n_steps / epochs / the grids
against that estimate.
"""
import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import Subset

from modules.base_utils.datasets import get_n_classes
from modules.base_utils.util import (
    load_model, clf_eval, get_train_info, mini_train, needs_big_ims,
)
from modules.federated_optimizing_trigger.utils import (
    get_mu, get_clean_dataset, get_poison_dataset,
)
from modules.federated_optimizing_trigger.run_module import optimize_trigger


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="cifar")
    p.add_argument("--model", default="r32p")
    p.add_argument("--source_label", type=int, default=9)
    p.add_argument("--target_label", type=int, default=4)

    p.add_argument("--beta_min", type=float, default=0.01)
    p.add_argument("--beta_max", type=float, default=0.2)
    p.add_argument("--n_beta", type=int, default=6)
    p.add_argument("--epsilon_min", type=float, default=0.01)
    p.add_argument("--epsilon_max", type=float, default=0.2)
    p.add_argument("--n_epsilon", type=int, default=6)
    p.add_argument("--asr_threshold", type=float, default=0.9)

    p.add_argument("--n_steps", type=int, default=20,
                    help="Outer retrain+optimize steps per optimize_trigger call.")
    p.add_argument("--epochs", type=int, default=20,
                    help="Epochs of expert retraining per outer step (and for "
                         "the final CTA/ASR retrain).")
    p.add_argument("--num_chckpt", type=int, default=15)
    p.add_argument("--alpha_ckpt", type=float, default=0.01)
    p.add_argument("--lambda_b1", type=float, default=1.0)
    p.add_argument("--lambda_b2", type=float, default=1.0)
    p.add_argument("--expert_path", required=True,
                    help="Format-string path to pre-trained expert checkpoints "
                         "(see schemas/federated_optimizing_trigger.toml).")

    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default="out/optimizing_trigger/budget_stealth.json")
    p.add_argument(
        "--estimate_only", action="store_true",
        help="Print the (|beta_grid| x |epsilon_grid| x cost-per-run) time "
             "estimate and exit without running anything.",
    )
    return p.parse_args()


def retrain_and_measure(model_flag, dataset_flag, source_label, target_label,
                         delta, beta, epochs, device):
    '''One fresh expert retrained on delta at lambda_target=lambda_poison=beta
    (lambda_overflow="duplicate"), then CTA/ASR on held-out test data.'''
    n_classes = get_n_classes(dataset_flag)
    big_ims = needs_big_ims(model_flag)
    expert = load_model(model_flag, n_classes).to(device)

    poison_train_dataset = get_poison_dataset(
        dataset_flag, source_label, target_label, delta,
        train=True, big=big_ims, lambda_target=beta, lambda_overflow="duplicate",
    )
    clean_test_dataset = get_clean_dataset(dataset_flag, train=False, big=big_ims)
    poison_test_dataset = get_poison_dataset(
        dataset_flag, source_label, target_label, delta,
        train=False, big=big_ims, lambda_target=beta, lambda_overflow="duplicate",
    )
    # get_poison_dataset returns ConcatDataset([clean_dataset, poison_dataset]);
    # the last len(poison_test_dataset) - len(clean_test_dataset) entries are
    # the triggered-and-relabeled examples -- isolate them for a true ASR
    # (as opposed to accuracy over the clean+poison mixture).
    n_clean = len(clean_test_dataset)
    asr_only_dataset = Subset(poison_test_dataset, range(n_clean, len(poison_test_dataset)))

    batch_size_, epochs_, opt, lr_scheduler = get_train_info(
        expert.parameters(), "sgd", epochs=epochs,
    )
    mini_train(
        model=expert,
        train_data=poison_train_dataset,
        batch_size=batch_size_,
        opt=opt,
        scheduler=lr_scheduler,
        epochs=epochs_,
    )

    cta, _ = clf_eval(expert, clean_test_dataset)
    asr, _ = clf_eval(expert, asr_only_dataset)
    return cta, asr


def run_one(args, beta, epsilon):
    device = args.device
    n_classes = get_n_classes(args.dataset)
    model = load_model(args.model, n_classes).to(device)
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    mu = get_mu(args.dataset, args.target_label, device, model_flag=args.model)
    mu_source = get_mu(args.dataset, args.source_label, device, model_flag=args.model)

    with tempfile.TemporaryDirectory() as ckpt_dir, \
         tempfile.TemporaryDirectory() as trig_dir:
        delta = optimize_trigger(
            model=model,
            loss_fn=loss_fn,
            dataset_flag=args.dataset,
            mu=mu,
            mu_source=mu_source,
            source_label=args.source_label,
            target_label=args.target_label,
            beta=beta,
            lambda_poison="beta",
            lambda_overflow="duplicate",
            epsilon=epsilon,
            n_steps=args.n_steps,
            epochs=args.epochs,
            num_chckpt=args.num_chckpt,
            alpha_ckpt=args.alpha_ckpt,
            lambda_b1=args.lambda_b1,
            lambda_b2=args.lambda_b2,
            expert_path=args.expert_path,
            output_dir=ckpt_dir,
            output_dir_trigger=trig_dir,
            device=device,
            model_flag=args.model,
        )

    cta, asr = retrain_and_measure(
        args.model, args.dataset, args.source_label, args.target_label,
        delta.cpu(), beta, args.epochs, device,
    )
    return cta, asr


def print_time_estimate(args):
    n_beta, n_eps = args.n_beta, args.n_epsilon
    n_runs = n_beta * n_eps
    print(
        "=== time estimate (no benchmark run performed) ===\n"
        f"grid: {n_beta} beta values x {n_eps} epsilon values = {n_runs} runs\n"
        "cost of one run = optimize_trigger(n_steps={0}, epochs={1}, "
        "num_chckpt={2}) + one final retrain(epochs={1}):\n"
        "  ~= {0} x (T_epoch * {1} [expert retrain per outer step] "
        "+ T_batch * B [trigger-opt batches/step, num_chckpt={2} QP solves "
        "each]) + T_epoch * {1} [final retrain]\n"
        "where T_epoch is your measured wall-clock for one `epochs` pass over "
        "poison_train_dataset on this GPU/model/dataset, T_batch is the "
        "per-batch cost of one optimize_trigger_step iteration (dominated by "
        "num_chckpt backward passes + num_chckpt OSQP QP solves), and B is "
        "worker_batch_size-determined batches per outer step (~= n_train / "
        "worker_batch_size).\n"
        "Plug in your own measured T_epoch/T_batch (this sandbox has no "
        "GPU/dataset/expert checkpoints to benchmark against) and multiply by "
        f"{n_runs} to size n_steps/epochs/grid resolution before running for real."
        .format(args.n_steps, args.epochs, args.num_chckpt)
    )


def main():
    args = parse_args()
    print_time_estimate(args)
    if args.estimate_only:
        return

    beta_grid = np.geomspace(args.beta_min, args.beta_max, args.n_beta).tolist()
    epsilon_grid = np.geomspace(args.epsilon_min, args.epsilon_max, args.n_epsilon).tolist()

    results = {}
    for beta in beta_grid:
        curve = []
        for epsilon in epsilon_grid:
            print(f"\n=== beta={beta:.6f} epsilon={epsilon:.6f} ===")
            cta, asr = run_one(args, beta, epsilon)
            print(f"CTA={cta:.4f} ASR={asr:.4f}")
            curve.append({"epsilon": epsilon, "cta": cta, "asr": asr})

        min_epsilon_for_asr = next(
            (c["epsilon"] for c in curve if c["asr"] >= args.asr_threshold), None
        )
        results[f"{beta:.6f}"] = {
            "min_epsilon_for_asr": min_epsilon_for_asr,
            "curve": curve,
        }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "asr_threshold": args.asr_threshold,
            "beta_grid": beta_grid,
            "epsilon_grid": epsilon_grid,
            "results": results,
        }, f, indent=2)
    print(f"\nsaved {args.output}")


if __name__ == "__main__":
    main()
