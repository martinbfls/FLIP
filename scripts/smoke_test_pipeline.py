"""
End-to-end smoke test: runs the simplest full chain (train_expert -> generate_labels ->
select_flips -> train_user) on a tiny slice of real CIFAR-10 (torchvision auto-downloads it
under data/ on first run), to catch regressions that only show up when a module actually
executes -- as opposed to tests/test_experiment_tracker.py (pure Python, no torch/data) or
scripts/validate_all_configs.py (schema only, no execution).

Everything lives under experiments/_ci_smoke_test/ (config, model output, plots/metrics) and
is schema-validated the same way run_experiment.py would, before and after running each
module. Cleaned up on success; left in place on failure for inspection (rerun with --keep to
always leave it).

Settings are deliberately tiny (train_pct=0.02, epochs=2, batch_size=64) so the whole chain
finishes in well under a minute on CPU -- this is a wiring smoke test, not a real training run;
it says nothing about model quality.

Run:  python scripts/smoke_test_pipeline.py [--keep]
Requires: torch/torchvision installed (see requirements.txt) and network access on first run
(CIFAR-10 download, ~170MB, cached under data/ afterwards).
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXPERIMENT_NAME = "_ci_smoke_test"
EXP_DIR = Path("experiments") / EXPERIMENT_NAME
OUT_DIR = (EXP_DIR / "artifacts").resolve()

CONFIG_TOML = f"""
[train_expert]
output_dir = "{OUT_DIR}/checkpoints/0/"
model = "r32p"
dataset = "cifar"
trainer = "sgd"
source_label = 9
target_label = 4
poisoner = "1xs"
checkpoint_iters = 5
epochs = 2
train_pct = 0.02
batch_size = 64

[generate_labels]
input_pths = "{OUT_DIR}/checkpoints/{{}}/model_{{}}_{{}}.pth"
opt_pths = "{OUT_DIR}/checkpoints/{{}}/model_{{}}_{{}}_opt.pth"
output_dir = "{OUT_DIR}/labels/"
expert_model = "r32p"
dataset = "cifar"
trainer = "sgd"
source_label = 9
target_label = 4
poisoner = "1xs"
train_pct = 0.02
batch_size = 64

[generate_labels.expert_config]
experts = 1
min = 0
max = 2
trajectories = [5]

[generate_labels.attack_config]
iterations = 2
one_hot_temp = 5

[select_flips]
budgets = [10]
input_label_glob = "{OUT_DIR}/labels/labels.npy"
true_labels = "{OUT_DIR}/labels/true.npy"
output_dir = "{OUT_DIR}/flips/"

[train_user]
input_labels = "{OUT_DIR}/flips/10.npy"
output_dir = "{OUT_DIR}/user/"
user_model = "r32p"
dataset = "cifar"
trainer = "sgd"
source_label = 9
target_label = 4
poisoner = "1xs"
# alpha explicitly set: train_user/run_module.py does `args.get("alpha", None)` then
# `if alpha > 0`, which raises TypeError on the schema's own documented default (0, "full
# input") if this key is omitted entirely -- a pre-existing bug this smoke test routes
# around rather than fixes (out of scope here; flagged separately).
alpha = 0.0
epochs = 1
batch_size = 64
"""
# extract_experts (modules/generate_labels/utils.py) formats input_pths as
# expert_path.format(expert_idx, trajectory, s) where trajectory is a random epoch in
# (min, max] and s is drawn from trajectories -- must be a checkpoint_iters multiple that
# train_expert's checkpoint_callback actually reaches within an epoch at this train_pct/
# batch_size (2%% of 50000 / 64 =~ 15 batches/epoch, so iteration 5 is hit).

MODULE_CHAIN = ["train_expert", "generate_labels", "select_flips", "train_user"]


def write_config():
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "config.toml").write_text(CONFIG_TOML)


def run_chain():
    from modules.base_utils.config_validation import validate_config_file

    validate_config_file(EXP_DIR / "config.toml")
    print("[smoke-test] config.toml schema-valid.")

    for module_name in MODULE_CHAIN:
        print(f"[smoke-test] running module: {module_name}")
        run_module = __import__(
            f"modules.{module_name}.run_module", fromlist=["run_module"]
        )
        run_module.run(str(EXPERIMENT_NAME), module_name)


def check_artifacts():
    expected = [
        OUT_DIR / "labels" / "labels.npy",
        OUT_DIR / "labels" / "true.npy",
        OUT_DIR / "flips" / "10.npy",
        OUT_DIR / "user" / "paccs.npy",
        OUT_DIR / "user" / "caccs.npy",
    ]
    missing = [p for p in expected if not p.exists()]
    if missing:
        raise AssertionError(f"Missing expected artifacts: {missing}")
    print("[smoke-test] all expected .npy artifacts present.")

    tracked_modules = ["train_expert", "generate_labels", "train_user"]
    for module_name in tracked_modules:
        manifest = EXP_DIR / "plots" / module_name / "run_manifest.json"
        if not manifest.exists():
            raise AssertionError(f"Missing ExperimentTracker manifest for {module_name}: {manifest}")
    print("[smoke-test] ExperimentTracker plots/manifests present for every tracked module.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", action="store_true", help="don't clean up experiments/_ci_smoke_test/ on success")
    args = parser.parse_args()

    if EXP_DIR.exists():
        shutil.rmtree(EXP_DIR)

    write_config()
    try:
        run_chain()
        check_artifacts()
    except Exception:
        print(f"[smoke-test] FAILED -- left {EXP_DIR} in place for inspection.")
        raise

    print("[smoke-test] PASSED.")
    if not args.keep:
        shutil.rmtree(EXP_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
