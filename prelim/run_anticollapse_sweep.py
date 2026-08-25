"""
prelim/run_anticollapse_sweep.py -- P3 (item 8): the anti-collapse sweep planned in a previous
session's Étape 3, never executed until now (blocked on P0's schema fix). Short run, one seed,
one checkpoint set (existing r32p_1xs smoke checkpoints, no new download). Not a performance
campaign -- the goal is to find a regime where expert_asr stays stable and nonzero, per the
task.

Grid: lambda_align in {0.1, 1, 10} x align_kappa in {0.3, 0.6}, trigger_constraint="penalty"
(6 cells) -- then ONE more cell with trigger_constraint="projection" at whichever align_kappa
won the grid (lambda_align is unused under "projection", see schema).

Writes per-cell metrics_log_path JSON under the scratchpad, and prints the summary table.
"""
import json
import os
import sys
from pathlib import Path

import torch

if not torch.cuda.is_available():
    torch.nn.Module.cuda = lambda self, device=None: self.to("cpu")
    torch.Tensor.cuda = lambda self, device=None, non_blocking=False: self.to("cpu")

sys.path.insert(0, os.getcwd())

from torch.utils.data import DataLoader
import modules.base_utils.datasets as _ds
import modules.base_utils.util as _util


def _make_dataloader_no_workers(dataset, batch_size, *, shuffle=True, drop_last=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                       pin_memory=False, drop_last=drop_last)


_ds.make_dataloader = _make_dataloader_no_workers
_util.make_dataloader = _make_dataloader_no_workers

from modules.federated_generate_labels_trigger_joint.run_module import run

CKPT_DIR = (
    "/private/tmp/claude-501/-Users-martinbeaufils-Downloads-broadflip-repo-FLIP/"
    "1f0e6899-15d6-4c67-a0f9-936b7b87b76e/scratchpad/smoke/checkpoints/r32p_1xs"
)
SCRATCH = (
    "/private/tmp/claude-501/-Users-martinbeaufils-Downloads-broadflip-repo-FLIP/"
    "00efd6c1-09a0-4868-9f81-6294c2d4b2a3/scratchpad/sweep"
)
EXP_ROOT = "federated_experiments/threat_model_direct/joint_sweep"

TOML_TEMPLATE = """
[federated_generate_labels_trigger_joint]
input_pths = "{ckpt_dir}/{{}}/model_{{}}_{{}}.pth"
opt_pths = "{ckpt_dir}/{{}}/model_{{}}_{{}}_opt.pth"
expert_model = "r32p"
dataset = "cifar"
source_label = 9
target_label = 4
output_dir = "{scratch}/{cell}/labels/"
output_dir_trigger = "{scratch}/{cell}/trigger"

epsilon = 1.0
lr_delta = 1e-2
lambda_bd = 1.0

lambda = 0.0
train_pct = 0.02
batch_size = 32
num_honests = 1
num_poisoned = 1
attack = "backdoor"
gamma_stealth = 1.0
init = "stripe"
alpha_ckpt = 0.01
checkpoint_sampling = "biased"
metrics_log_path = "{scratch}/{cell}/metrics.json"

trigger_constraint = "{trigger_constraint}"
align_kappa = {align_kappa}
lambda_align = {lambda_align}
lambda_mag = 1.0
delta_min_frac = 0.5

[federated_generate_labels_trigger_joint.expert_config]
experts = 1
min = 0
max = 2
trajectories = [10]

[federated_generate_labels_trigger_joint.attack_config]
iterations = 1
one_hot_temp = 5
"""


def run_cell(cell, align_kappa, lambda_align, trigger_constraint, seed=0):
    torch.manual_seed(seed)
    cell_dir = Path("experiments") / EXP_ROOT / cell
    cell_dir.mkdir(parents=True, exist_ok=True)
    Path(f"{SCRATCH}/{cell}/labels").mkdir(parents=True, exist_ok=True)
    toml_text = TOML_TEMPLATE.format(
        ckpt_dir=CKPT_DIR, scratch=SCRATCH, cell=cell,
        trigger_constraint=trigger_constraint, align_kappa=align_kappa, lambda_align=lambda_align,
    )
    (cell_dir / "config.toml").write_text(toml_text)

    print(f"\n### Running cell={cell} align_kappa={align_kappa} lambda_align={lambda_align} "
          f"trigger_constraint={trigger_constraint} ###")
    try:
        run(f"{EXP_ROOT}/{cell}", "federated_generate_labels_trigger_joint")
    except Exception as e:
        print(f"CELL {cell} FAILED: {e}")
        return None

    with open(f"{SCRATCH}/{cell}/metrics.json") as f:
        history = json.load(f)
    return history


def summarize(cell, history):
    if not history:
        return None
    asrs = [h["expert_asr"] for h in history]
    asrs_sorted = sorted(asrs)
    median = asrs_sorted[len(asrs_sorted) // 2]
    last = history[-1]
    return {
        "cell": cell,
        "expert_asr_final": last["expert_asr"],
        "expert_asr_median": median,
        "matching_term_final": last["matching_term"],
        "cos_target_final": last["cos_target"],
        "delta_l2_final": last["delta_l2"],
    }


def main():
    grid_results = {}
    grid = [
        (la, ak) for la in [0.1, 1.0, 10.0] for ak in [0.3, 0.6]
    ]
    for lambda_align, align_kappa in grid:
        cell = f"penalty_la{lambda_align}_ak{align_kappa}"
        history = run_cell(cell, align_kappa, lambda_align, "penalty")
        grid_results[cell] = (summarize(cell, history), history)

    # Best cell by expert_asr_median (the stated goal: stable, nonzero expert_asr)
    valid = {k: v for k, v in grid_results.items() if v[0] is not None}
    best_cell = max(valid, key=lambda k: valid[k][0]["expert_asr_median"])
    best_summary = valid[best_cell][0]
    best_align_kappa = None
    for lambda_align, align_kappa in grid:
        if f"penalty_la{lambda_align}_ak{align_kappa}" == best_cell:
            best_align_kappa = align_kappa
            break

    print(f"\n>>> Best penalty cell by expert_asr_median: {best_cell} "
          f"(align_kappa={best_align_kappa})")

    proj_cell = f"projection_ak{best_align_kappa}"
    proj_history = run_cell(proj_cell, best_align_kappa, 1.0, "projection")
    grid_results[proj_cell] = (summarize(proj_cell, proj_history), proj_history)

    print("\n\n=== SUMMARY TABLE ===")
    header = f"{'cell':28s} {'asr_final':10s} {'asr_median':11s} {'match_final':12s} {'cos_final':10s} {'delta_l2_final':14s}"
    print(header)
    for cell, (summary, _) in grid_results.items():
        if summary is None:
            print(f"{cell:28s} FAILED")
            continue
        print(f"{summary['cell']:28s} {summary['expert_asr_final']:<10.4f} "
              f"{summary['expert_asr_median']:<11.4f} {summary['matching_term_final']:<12.4f} "
              f"{summary['cos_target_final']:<10.4f} {summary['delta_l2_final']:<14.4f}")

    with open(f"{SCRATCH}/sweep_results.json", "w") as f:
        json.dump({cell: s for cell, (s, _) in grid_results.items()}, f, indent=2)
    print(f"\nFull per-cell metrics under {SCRATCH}/<cell>/metrics.json ; "
          f"summary saved to {SCRATCH}/sweep_results.json")


if __name__ == "__main__":
    main()
