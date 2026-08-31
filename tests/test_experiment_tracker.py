"""
tests/test_experiment_tracker.py -- non-regression checks for
modules.base_utils.experiment_tracker.ExperimentTracker.

Uses a scratch experiment directory under experiments/ (created and torn down by this
script) instead of any real experiment config, so this is safe to run in a code-writing
session: no dataset is touched, no model is trained.

Run:  python tests/test_experiment_tracker.py
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.base_utils.experiment_tracker import ExperimentTracker

SCRATCH_EXPERIMENT = "_test_experiment_tracker_scratch"
SCRATCH_DIR = os.path.join("experiments", SCRATCH_EXPERIMENT)

_results = []


def check(name, ok, detail=""):
    _results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def _make_scratch_config():
    os.makedirs(SCRATCH_DIR, exist_ok=True)
    with open(os.path.join(SCRATCH_DIR, "config.toml"), "w") as f:
        f.write('[dummy_module]\noutput_dir = "/tmp/does_not_need_to_exist/"\n')


def test_plots_and_manifest_land_next_to_config():
    tracker = ExperimentTracker(SCRATCH_EXPERIMENT, "dummy_module", {})
    for step in range(5):
        tracker.log(step, loss=1.0 / (step + 1), aux=step * 2.0)
    tracker.finalize()

    plots_dir = os.path.join(SCRATCH_DIR, "plots", "dummy_module")
    check(
        "loss.png written next to config.toml",
        os.path.exists(os.path.join(plots_dir, "loss.png")),
    )
    check(
        "overview.png written next to config.toml",
        os.path.exists(os.path.join(plots_dir, "overview.png")),
    )
    manifest_path = os.path.join(plots_dir, "run_manifest.json")
    check("run_manifest.json written", os.path.exists(manifest_path))
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        check(
            "manifest references the right experiment/module",
            manifest["experiment_name"] == SCRATCH_EXPERIMENT
            and manifest["module_name"] == "dummy_module",
        )
        check("manifest records wandb as disabled", manifest["wandb"]["enabled"] is False)


def test_metrics_json_structure():
    tracker = ExperimentTracker(SCRATCH_EXPERIMENT, "dummy_module2", {})
    for step in range(3):
        tracker.log(step, loss=float(step))
    tracker.finalize()

    with open(tracker.metrics_log_path) as f:
        history = json.load(f)
    check("metrics json has the logged key", "loss" in history)
    check(
        "metrics json steps/values match what was logged",
        [p["step"] for p in history["loss"]] == [0, 1, 2]
        and [p["value"] for p in history["loss"]] == [0.0, 1.0, 2.0],
    )


def test_wandb_absent_does_not_crash():
    real_import = __builtins__.__import__ if isinstance(__builtins__, dict) else __builtins__.__import__

    def _raising_import(name, *args, **kwargs):
        if name == "wandb":
            raise ImportError("simulated: wandb not installed")
        return real_import(name, *args, **kwargs)

    import builtins
    old_import = builtins.__import__
    builtins.__import__ = _raising_import
    try:
        tracker = ExperimentTracker(
            SCRATCH_EXPERIMENT, "dummy_module3", {"wandb": {"enabled": True, "project": "x"}}
        )
        tracker.log(0, loss=1.0)
        tracker.finalize()
        check("tracker with wandb.enabled=true but wandb absent doesn't crash", True)
        check("tracker falls back to wandb disabled", tracker._wandb is None)
    except Exception as exc:
        check("tracker with wandb absent doesn't crash", False, f"raised {type(exc).__name__}: {exc}")
    finally:
        builtins.__import__ = old_import


def test_context_manager_finalizes_on_exception():
    module_name = "dummy_module4"
    try:
        with ExperimentTracker(SCRATCH_EXPERIMENT, module_name, {}) as tracker:
            tracker.log(0, loss=1.0)
            raise RuntimeError("simulated crash mid-run")
    except RuntimeError:
        pass

    plots_dir = os.path.join(SCRATCH_DIR, "plots", module_name)
    check(
        "finalize() still ran after an exception inside the with-block",
        os.path.exists(os.path.join(plots_dir, "run_manifest.json")),
    )


def main():
    _make_scratch_config()
    tests = [
        test_plots_and_manifest_land_next_to_config,
        test_metrics_json_structure,
        test_wandb_absent_does_not_crash,
        test_context_manager_finalizes_on_exception,
    ]
    try:
        for t in tests:
            try:
                t()
            except Exception as exc:
                check(t.__name__, False, f"raised {type(exc).__name__}: {exc}")
    finally:
        shutil.rmtree(SCRATCH_DIR, ignore_errors=True)

    n_ok = sum(1 for _, ok, _ in _results if ok)
    print(f"\n{n_ok}/{len(_results)} checks passed")
    return 0 if n_ok == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
