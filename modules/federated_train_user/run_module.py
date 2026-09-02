"""
Trains a downstream (user) model on reconstructed datasets with input labels in the federated learning setting.
"""

from pathlib import Path
import sys
import json
import torch
import numpy as np
from torch.utils.data import Subset

from modules.base_utils.datasets import (
    get_matching_datasets,
    get_n_classes,
    pick_poisoner,
    construct_user_dataset,
    PoisonFlagDataset,
)
from modules.base_utils.util import (
    extract_toml,
    get_train_info,
    mini_train_multi,
    load_model,
    needs_big_ims,
    slurmify_path,
    softmax,
)
from modules.base_utils.experiment_tracker import ExperimentTracker


def run(experiment_name, module_name, **kwargs):
    slurm_id = kwargs.get("slurm_id", None)
    args = extract_toml(experiment_name, module_name)
    tracker = ExperimentTracker(experiment_name, module_name, args, slurm_id=slurm_id)

    user_model_flag = args["user_model"]
    trainer_flag = args["trainer"]
    dataset_flag = args["dataset"]
    poisoner_flag = args["poisoner"]
    delta = args.get("delta", None)
    clean_label = args["source_label"]
    target_label = args["target_label"]

    soft = args.get("soft", True)
    alpha = args.get("alpha", 0.0)

    batch_size = args.get("batch_size", None)
    epochs = args.get("epochs", None)
    optim_kwargs = args.get("optim_kwargs", {})
    scheduler_kwargs = args.get("scheduler_kwargs", {})

    num_honests = args.get("num_honests", 0)
    num_poisoned = args.get("num_poisoned", 1)
    num_workers = num_honests + num_poisoned
    budget = args.get("budget", 1500)
    input_dir = Path(slurmify_path(args["input_labels"], slurm_id))
    true_path = slurmify_path(args.get("true_labels", None), slurm_id)
    output_dir = Path(slurmify_path(args["output_dir"], slurm_id))
    output_dir.mkdir(parents=True, exist_ok=True)
    agg_method = args.get("agg_method", "mean")
    track_poison_selection = args.get("track_poison_selection", False)

    print("Loading base datasets...")
    poisoner = pick_poisoner(poisoner_flag, dataset_flag, target_label, delta)
    big_ims = needs_big_ims(user_model_flag)

    _, distillation, test, poison_test, _ = get_matching_datasets(
        dataset_flag,
        poisoner,
        clean_label,
        big=big_ims,
    )

    if alpha > 0:
        assert true_path is not None
        y_true = torch.tensor(np.load(true_path))

    print("Reconstructing worker datasets...")
    user_datasets = []

    idx_flipped = None
    if track_poison_selection:
        idx_flipped_path = input_dir / f"{budget}_idx_flipped.npy"
        idx_flipped = np.load(idx_flipped_path)

    for w in range(num_workers):
        idx = np.load(input_dir / f"worker{w}/{budget}_indices.npy")
        labels_syn = torch.tensor(
            np.load(input_dir / f"worker{w}/{budget}_labels.npy"),
            dtype=torch.float,
        )

        subset = Subset(distillation, idx)

        if alpha > 0:
            labels_true_local = y_true[idx]
            labels_d = softmax(alpha * labels_true_local +
                               (1 - alpha) * labels_syn)
        else:
            labels_d = softmax(labels_syn)

        if not soft:
            labels_d = labels_d.argmax(dim=1)

        user_dataset = construct_user_dataset(subset, labels_d)
        if track_poison_selection:
            is_flipped_mask = np.isin(idx, idx_flipped)
            user_dataset = PoisonFlagDataset(user_dataset, is_flipped_mask)
        user_datasets.append(user_dataset)

    print("Training user model...")
    n_classes = get_n_classes(dataset_flag)
    model_retrain = load_model(user_model_flag, n_classes)

    batch_size, epochs, optimizer, scheduler = get_train_info(
        model_retrain.parameters(),
        trainer_flag,
        batch_size,
        epochs,
        optim_kwargs,
        scheduler_kwargs,
    )

    train_result = mini_train_multi(
        model=model_retrain,
        train_datasets=user_datasets,
        test_data=[test, poison_test.poison_dataset],
        batch_size=batch_size,
        opt=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        record=True,
        agg_method=agg_method,
        f=num_poisoned,
        epoch_callback=tracker.epoch_callback(),
        track_poison_selection=track_poison_selection,
    )
    if track_poison_selection:
        model_retrain, poison_stats, clean_metrics, poison_metrics = train_result
    else:
        model_retrain, clean_metrics, poison_metrics = train_result

    print("Saving results...")
    np.save(output_dir / "paccs.npy", poison_metrics)
    np.save(output_dir / "caccs.npy", clean_metrics)
    if track_poison_selection:
        with open(output_dir / "multikrum_poison_stats.json", "w") as f:
            json.dump(poison_stats, f, indent=2)
    #torch.save(model_retrain.state_dict(), output_dir / "model.pth")

    tracker.finalize()


if __name__ == "__main__":
    experiment_name, module_name = sys.argv[1], sys.argv[2]
    run(experiment_name, module_name)
