"""
prelim/verify_trigger_module_integration.py -- quick real-checkpoint integration smoke test for
the expert-aggregation fix in federated_generate_labels_trigger/run_module.py (non-joint
"indirect" module) -- not yet integration-tested elsewhere this session. Confirms the module
still runs end to end (real CIFAR, real checkpoints, real dataloaders) after inserting
agg(expert_params, expert_grad_buf, agg_method, f=num_poisoned) before optimizer_expert.step().

Run:  python prelim/verify_trigger_module_integration.py
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

from modules.federated_generate_labels_trigger.run_module import run

CKPT_DIR = (
    "/private/tmp/claude-501/-Users-martinbeaufils-Downloads-broadflip-repo-FLIP/"
    "1f0e6899-15d6-4c67-a0f9-936b7b87b76e/scratchpad/smoke/checkpoints/r32p_1xs"
)
SCRATCH = (
    "/private/tmp/claude-501/-Users-martinbeaufils-Downloads-broadflip-repo-FLIP/"
    "00efd6c1-09a0-4868-9f81-6294c2d4b2a3/scratchpad/trigger_module_verify"
)
EXP_ROOT = "federated_experiments/threat_model_direct/trigger_module_verify"

TOML = f"""
[federated_generate_labels_trigger]
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
num_honests = 2
num_poisoned = 2
attack = "backdoor"
gamma_stealth = 1.0
init = "stripe"
checkpoint_sampling = "uniform"
agg_method = "mean"

[federated_generate_labels_trigger.expert_config]
experts = 1
min = 0
max = 2
trajectories = [10]

[federated_generate_labels_trigger.attack_config]
iterations = 1
one_hot_temp = 5
"""

if __name__ == "__main__":
    cfg_dir = Path("experiments") / EXP_ROOT
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "config.toml").write_text(TOML)

    run(EXP_ROOT, "federated_generate_labels_trigger")

    labels_path = Path(SCRATCH) / "labels" / "labels.npy"
    ok = labels_path.exists()
    print(f"\n[{'PASS' if ok else 'FAIL'}] run completed and wrote labels.npy -- {labels_path}")
    sys.exit(0 if ok else 1)
