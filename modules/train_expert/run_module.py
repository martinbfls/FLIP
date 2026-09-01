"""
Trains an expert model on a traditionally backdoored dataset.
"""

from pathlib import Path
import random
import sys
import time as _time

import numpy as np
import torch

from modules.train_expert.utils import checkpoint_callback
from modules.base_utils.datasets import (
    get_matching_datasets,
    get_n_classes,
    pick_poisoner,
)
from modules.base_utils.util import (
    extract_toml,
    load_model,
    clf_eval,
    mini_train,
    get_train_info,
    needs_big_ims,
    slurmify_path,
)
from modules.base_utils.experiment_tracker import ExperimentTracker


def run(experiment_name, module_name, **kwargs):
    """
    Runs expert training and saves trajectory.

    :param experiment_name: Name of the experiment in configuration.
    :param module_name: Name of the module in configuration.
    :param kwargs: Additional arguments (such as slurm id).
    """

    slurm_id = kwargs.get("slurm_id", None)
    args = extract_toml(experiment_name, module_name)
    tracker = ExperimentTracker(experiment_name, module_name, args, slurm_id=slurm_id)

    model_flag = args["model"]
    dataset_flag = args["dataset"]
    train_flag = args["trainer"]
    poisoner_flag = args["poisoner"]
    delta = args.get("delta", None)
    clean_label = args["source_label"]
    target_label = args["target_label"]
    ckpt_iters = args.get("checkpoint_iters")
    train_pct = args.get("train_pct", 1.0)
    batch_size = args.get("batch_size", None)
    epochs = args.get("epochs", None)
    optim_kwargs = args.get("optim_kwargs", {})
    scheduler_kwargs = args.get("scheduler_kwargs", {})
    output_dir = slurmify_path(args["output_dir"], slurm_id)
    budget = args.get("budget", None)
    # Real RNG seed (see schemas/train_expert.toml's `seed` doc) -- NOT the same thing as the
    # "seedN" segment conventionally used in this run's own output_dir path, which was never
    # actually wired to torch.manual_seed before this field existed. Seeded here, before dataset
    # construction/model init, so a later reproduction (e.g.
    # federated_generate_labels_trigger_joint's expert_retrain_interval, which reseeds to this
    # SAME value before retraining a fresh expert mid-run) can replicate this run's weight init
    # and data order exactly, isolating whatever else differs (e.g. a poisoned trigger) as the
    # only source of divergence.
    seed = args.get("seed", None)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if slurm_id is None:
        slurm_id = "{}"

    # Build datasets
    print("Building datasets...")
    big_ims = needs_big_ims(model_flag)
    poisoner = pick_poisoner(poisoner_flag, dataset_flag, target_label, delta=delta)
    poison_train, _, test, poison_test, _ = get_matching_datasets(
        dataset_flag,
        poisoner,
        clean_label,
        train_pct=train_pct,
        big=big_ims,
        budget=budget,
    )

    # Train expert model
    print("Training expert model...")
    n_classes = get_n_classes(dataset_flag)
    model = load_model(model_flag, n_classes)
    batch_size, epochs, opt, lr_scheduler = get_train_info(
        model.parameters(),
        train_flag,
        batch_size=batch_size,
        epochs=epochs,
        optim_kwargs=optim_kwargs,
        scheduler_kwargs=scheduler_kwargs,
    )

    print(f"[DIAG train_expert] epochs={epochs}", flush=True)
    print(f"[DIAG train_expert] n_train={len(poison_train)}", flush=True)
    print(f"[DIAG train_expert] n_test={len(test) if test else None}", flush=True)
    print(f"[DIAG train_expert] batch_size={batch_size}", flush=True)
    _t0 = _time.time()

    mini_train(
        model=model,
        train_data=poison_train,
        test_data=[test, poison_test.poison_dataset],
        batch_size=batch_size,
        opt=opt,
        scheduler=lr_scheduler,
        epochs=epochs,
        callback=lambda m, o, e, i: checkpoint_callback(
            m, o, e, i, ckpt_iters, output_dir
        ),
        epoch_callback=tracker.epoch_callback(),
    )

    print(f"[DIAG train_expert] done in {_time.time()-_t0:.1f}s", flush=True)

    # Evaluate
    print("Evaluating...")
    clean_test_acc = clf_eval(model, test)[0]
    poison_test_acc = clf_eval(model, poison_test.poison_dataset)[0]
    print(f"{clean_test_acc=}")
    print(f"{poison_test_acc=}")

    tracker.finalize()


if __name__ == "__main__":
    experiment_name, module_name = sys.argv[1], sys.argv[2]
    run(experiment_name, module_name)
