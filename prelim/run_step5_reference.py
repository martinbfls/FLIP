"""
prelim/run_step5_reference.py -- Step 5: ONE reference run of the refactored
federated_generate_labels_trigger_joint at realistic epsilon (8/255 ~= 0.031), agg_method=
"mean", with the anti-collapse regularizers OFF (lambda_align=0, lambda_mag=0) -- per the
spec, no regularizer recalibration before this run. Uses num_honests=num_poisoned=1, the exact
configuration that motivated the aggregation refactor (grads_e_last collapsing the expert
target to a single client) -- this is the regime the fix is meant to matter most in.

Reports per-step: expert_asr (fixed 256-example set), expert_asr_frozen, matching_term,
L_bd_mean, delta_l2, mtt_delta_grad_norm.

Run:  python prelim/run_step5_reference.py
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
    "00efd6c1-09a0-4868-9f81-6294c2d4b2a3/scratchpad/step5_reference"
)
EXP_ROOT = "federated_experiments/threat_model_direct/step5_reference"

TOML = f"""
[federated_generate_labels_trigger_joint]
input_pths = "{CKPT_DIR}/{{}}/model_{{}}_{{}}.pth"
opt_pths = "{CKPT_DIR}/{{}}/model_{{}}_{{}}_opt.pth"
expert_model = "r32p"
dataset = "cifar"
source_label = 9
target_label = 4
output_dir = "{SCRATCH}/labels/"
output_dir_trigger = "{SCRATCH}/trigger"

epsilon = 0.031
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
metrics_log_path = "{SCRATCH}/metrics.json"

agg_method = "mean"
trigger_constraint = "penalty"
align_kappa = 0.6
lambda_align = 0.0
lambda_mag = 0.0
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

if __name__ == "__main__":
    cfg_dir = Path("experiments") / EXP_ROOT
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.toml").write_text(TOML)

    run(EXP_ROOT, "federated_generate_labels_trigger_joint")

    metrics = json.loads((Path(SCRATCH) / "metrics.json").read_text())
    print(f"\n{len(metrics)} batches logged.\n")

    keys = ["expert_asr", "expert_asr_frozen", "matching_term", "L_bd_mean", "delta_l2",
            "mtt_delta_grad_norm"]
    header = "step".rjust(5) + "".join(k.rjust(16) for k in keys)
    print(header)
    n = len(metrics)
    sample_idx = sorted(set([0, 1, 2, 3, 4, n // 4, n // 2, 3 * n // 4, n - 1]))
    for i in sample_idx:
        if 0 <= i < n:
            row = metrics[i]
            print(str(i).rjust(5) + "".join(f"{row[k]:.4g}".rjust(16) for k in keys))
