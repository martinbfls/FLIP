"""
prelim/verify_step4d_integration.py -- Step 4, verification D (no real expert step / no
expert_params[i].grad dependency remains) plus a real end-to-end integration smoke test of the
federated expert-aggregation refactor in federated_generate_labels_trigger_joint/run_module.py,
using the existing small r32p_1xs smoke checkpoints (no new download).

D is checked by spying on optimizer_expert.step (via monkeypatching get_mtt_attack_info as
imported into run_module's namespace) and asserting it is called ZERO times across a real run --
a direct behavioral proof, stronger than the static source-grep check (also done separately),
that expert_params are never real-stepped and nothing depends on expert_params[i].grad.

Runs a few real batches for num_honests=2, num_poisoned=2 (to exercise multi-client aggregation,
not just the 1v1 case) across three agg_method values (mean, median, trmean) -- also serves as
the promised end-to-end integration check for the Step 2 refactor (real CIFAR, real checkpoints,
real dataloaders, real schema-shaped TOML), not just the synthetic checks in
prelim/verify_expert_aggregation.py.

Run:  python prelim/verify_step4d_integration.py
"""
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

import modules.federated_generate_labels_trigger_joint.run_module as jrun

CKPT_DIR = (
    "/private/tmp/claude-501/-Users-martinbeaufils-Downloads-broadflip-repo-FLIP/"
    "1f0e6899-15d6-4c67-a0f9-936b7b87b76e/scratchpad/smoke/checkpoints/r32p_1xs"
)
SCRATCH = (
    "/private/tmp/claude-501/-Users-martinbeaufils-Downloads-broadflip-repo-FLIP/"
    "00efd6c1-09a0-4868-9f81-6294c2d4b2a3/scratchpad/step4d"
)
EXP_ROOT = "federated_experiments/threat_model_direct/step4d_verify"

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
alpha_ckpt = 0.01
checkpoint_sampling = "biased"
metrics_log_path = "{scratch}/{cell}/metrics.json"

agg_method = "{agg_method}"
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

_results = []


def check(name, ok, detail=""):
    _results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def run_one(agg_method):
    cell = f"agg_{agg_method}"
    cfg_dir = Path("experiments") / EXP_ROOT / cell
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / "config.toml"
    cfg_path.write_text(TOML_TEMPLATE.format(
        ckpt_dir=CKPT_DIR, scratch=SCRATCH, cell=cell, agg_method=agg_method,
    ))

    # Spy on optimizer_expert.step -- intercepts get_mtt_attack_info AS IMPORTED into
    # run_module's namespace (the name run_module.run() actually calls).
    orig_get_mtt_attack_info = jrun.get_mtt_attack_info
    call_count = {"n": 0}

    def spy(*args, **kwargs):
        bs, epochs, opt_expert, opt_labels = orig_get_mtt_attack_info(*args, **kwargs)
        orig_step = opt_expert.step

        def counted_step(*a, **k):
            call_count["n"] += 1
            return orig_step(*a, **k)

        opt_expert.step = counted_step
        return bs, epochs, opt_expert, opt_labels

    jrun.get_mtt_attack_info = spy
    try:
        jrun.run(f"{EXP_ROOT}/{cell}", "federated_generate_labels_trigger_joint")
    finally:
        jrun.get_mtt_attack_info = orig_get_mtt_attack_info

    check(
        f"D: optimizer_expert.step() never called during a real run [agg_method={agg_method}]",
        call_count["n"] == 0, f"call_count={call_count['n']}",
    )

    import json
    metrics = json.loads((Path(SCRATCH) / cell / "metrics.json").read_text())
    check(
        f"integration: run completed and logged at least one batch [agg_method={agg_method}]",
        len(metrics) > 0, f"n_batches_logged={len(metrics)}",
    )
    if metrics:
        row0 = metrics[0]
        check(
            f"Step 3: step-0 cos_delta_to_init == 1.0 exactly [agg_method={agg_method}]",
            abs(row0["cos_delta_to_init"] - 1.0) < 1e-5, f"cos={row0['cos_delta_to_init']}",
        )
        check(
            f"Step 3: step-0 delta_drift_l2 == 0.0 exactly [agg_method={agg_method}]",
            row0["delta_drift_l2"] < 1e-6, f"drift={row0['delta_drift_l2']}",
        )
        check(
            f"Step 3: step-0 expert_asr == expert_asr_frozen exactly (same delta) "
            f"[agg_method={agg_method}]",
            abs(row0["expert_asr"] - row0["expert_asr_frozen"]) < 1e-9,
            f"asr={row0['expert_asr']}, asr_frozen={row0['expert_asr_frozen']}",
        )


if __name__ == "__main__":
    for agg_method in ["mean", "median", "trmean"]:
        run_one(agg_method)

    n_fail = sum(1 for _, ok, _ in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} checks passed.")
    sys.exit(1 if n_fail else 0)
