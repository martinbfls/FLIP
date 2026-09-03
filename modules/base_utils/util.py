import os
import numpy as np
import torch
import tqdm
from functools import partial
from torch import optim
from torch.utils.data import DataLoader, Dataset
from typing import Iterable, Union
from modules.base_utils.model.model import (
    SequentialImageNetwork,
    SequentialImageNetworkMod,
)
import torch.backends.cudnn as cudnn
import toml
from collections import OrderedDict
from modules.base_utils.aggregator.trmean import aggr_trmean
from modules.base_utils.aggregator.multikrum import aggregate as aggr_multikrum
from modules.base_utils.aggregator.softkrum import krum as aggr_soft_multikrum
from modules.base_utils.datasets import make_dataloader

if torch.cuda.is_available():
    cudnn.benchmark = True

default_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DEFAULT_SGD_BATCH_SIZE = 256
DEFAULT_SGD_EPOCHS = 200
DEFAULT_SGD_KWARGS = {
    "lr": 0.01,
    "momentum": 0.9,
    "nesterov": True,
    "weight_decay": 2e-4,
}
DEFAULT_SGD_SCHED_KWARGS = {"milestones": [75, 125], "gamma": 0.1}

DEFAULT_ADAM_BATCH_SIZE = 256
DEFAULT_ADAM_EPOCHS = 200
DEFAULT_ADAM_KWARGS = {"lr": 0.001, "betas": (0.9, 0.999), "weight_decay": 1e-4}
DEFAULT_ADAM_SCHED_KWARGS = {"milestones": [125], "gamma": 0.1}

BIG_IMS_MODELS = ["convnext-tiny", "vgg", "vgg-pretrain", "vit-pretrain"]


def generate_full_path(path):
    return os.path.join(os.getcwd(), path)


def slurmify_path(path, slurm_id):
    if path is None:
        return path
    return path if slurm_id is None else path.format(slurm_id)


def extract_toml(experiment_name, module_name=None):
    relative_path = "experiments/" + experiment_name + "/config.toml"  #
    full_path = generate_full_path(relative_path)
    assert os.path.exists(full_path)

    exp_toml = toml.load(full_path, _dict=OrderedDict)
    if module_name is not None:
        return exp_toml[module_name]
    return exp_toml


def load_model(model_flag, num_classes=10):
    # if num_classes != 10 and model_flag not in ['r32p', 'r18', 'r18-tin']:
    #     raise NotImplementedError

    if model_flag == "r32p":
        import modules.base_utils.model.resnet as resnet

        return SequentialImageNetworkMod(resnet.resnet32(num_classes)).cuda()
    elif model_flag == "r18":
        from pytorch_cifar.models import ResNet, BasicBlock

        return SequentialImageNetwork(
            ResNet(BasicBlock, [2, 2, 2, 2], num_classes)
        ).cuda()
    elif model_flag == "r18-tin":
        from pytorch_cifar.models import ResNet, BasicBlock

        model = SequentialImageNetwork(ResNet(BasicBlock, [2, 2, 2, 2], num_classes))
        model[13] = torch.nn.Linear(2048, 200)

        return model.cuda()
    elif model_flag == "vgg":
        from torchvision.models import vgg19_bn

        return vgg19_bn(num_classes=num_classes).cuda()
    elif model_flag == "vgg-pretrain":
        from torchvision.models import vgg19_bn, VGG19_BN_Weights

        model = vgg19_bn(weights=VGG19_BN_Weights.DEFAULT)
        model.classifier[6] = torch.nn.Linear(4096, num_classes)
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False

        return model.cuda()
    elif model_flag == "vit-pretrain":
        from torchvision.models import vit_b_16, ViT_B_16_Weights

        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        model.heads.head = torch.nn.Linear(768, num_classes)
        for name, param in model.named_parameters():
            if "head" not in name:
                param.requires_grad = False
        return model.cuda()

    elif model_flag == "convnext_tiny":
        from torchvision.models import convnext_tiny

        model = convnext_tiny(weights="IMAGENET1K_V1")
        model.classifier[2] = torch.nn.Linear(
            model.classifier[2].in_features, num_classes
        )

        return model.cuda()

    elif model_flag == "convnext_micro":
        from modules.base_utils.model.convnext_micro import ConvNeXtMicro

        model = ConvNeXtMicro(num_classes=num_classes)

        return model.cuda()

    else:
        raise NotImplementedError


def make_pbar(*args, **kwargs):
    pbar_constructor = partial(tqdm.tqdm, dynamic_ncols=True)
    return pbar_constructor(*args, **kwargs)


def get_module_device(module: torch.nn.Module, check=True):
    if check:
        assert len(set(param.device for param in module.parameters())) == 1
    return next(module.parameters()).device


def either_dataloader_dataset_to_both(
    data: Union[DataLoader, Dataset], *, batch_size=None, eval=False, **kwargs
):
    if isinstance(data, DataLoader):
        dataloader = data
        dataset = data.dataset
    elif isinstance(data, Dataset):
        dataset = data
        dl_kwargs = {}

        if eval:
            dl_kwargs.update(dict(batch_size=256, shuffle=False, drop_last=False))
        else:
            dl_kwargs.update(dict(batch_size=128, shuffle=True))

        if batch_size is not None:
            dl_kwargs["batch_size"] = batch_size

        dl_kwargs.update(kwargs)
        dataloader = make_dataloader(data, **dl_kwargs)
    else:
        raise NotImplementedError()
    return dataloader, dataset


clf_loss = torch.nn.CrossEntropyLoss()
total_mse_distance = torch.nn.MSELoss(reduction="sum")
softmax = torch.nn.Softmax(dim=1)


def clf_correct(y_pred: torch.Tensor, y: torch.Tensor):
    if len(y.shape) > 1 and y.shape[1] > 1:
        y = torch.argmax(y, dim=1)
    y_hat = y_pred.data.max(1)[1]
    correct = (y_hat == y).long().cpu().sum()
    return correct


def clf_eval(model: torch.nn.Module, data: Union[DataLoader, Dataset]):
    device = get_module_device(model)
    dataloader, _ = either_dataloader_dataset_to_both(data, eval=True)
    total_correct, total_loss = 0.0, 0.0
    with torch.no_grad():
        model.eval()
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = clf_loss(y_pred, y)
            correct = clf_correct(y_pred, y)

            total_correct += correct.item()
            total_loss += loss.item()

    n = len(dataloader.dataset)
    total_correct /= n
    total_loss /= n
    return total_correct, total_loss


def get_mean_lr(opt: optim.Optimizer):
    return np.mean([group["lr"] for group in opt.param_groups])


def mini_train(
    *,
    model: torch.nn.Module,
    train_data: Union[DataLoader, Dataset],
    test_data: Union[
        Union[DataLoader, Dataset], Iterable[Union[DataLoader, Dataset]]
    ] = None,
    batch_size=32,
    opt: optim.Optimizer,
    scheduler,
    epochs: int,
    shuffle=True,
    callback=None,
    record=False,
    epoch_callback=None,
):
    device = get_module_device(model)
    dataloader, _ = either_dataloader_dataset_to_both(
        train_data, batch_size=batch_size, shuffle=shuffle
    )
    n = len(dataloader.dataset)
    total_examples = epochs * n

    if test_data:
        num_sets = 1
        if isinstance(test_data, Iterable):
            num_sets = len(test_data)
        else:
            test_data = [test_data]
        acc_loss = [[] for _ in range(num_sets)]

    with make_pbar(total=total_examples) as pbar:
        for epoch in range(1, epochs + 1):
            train_epoch_loss, train_epoch_correct = 0, 0
            model.train()
            for i, (x, y) in enumerate(dataloader):
                x, y = x.to(device), y.to(device)
                minibatch_size = len(x)
                model.zero_grad()
                y_pred = model(x)
                loss = clf_loss(y_pred, y)
                correct = clf_correct(y_pred, y)
                loss.backward()
                opt.step()
                train_epoch_correct += int(correct.item())
                train_epoch_loss += float(loss.item())
                pbar.update(minibatch_size)
                if callback is not None:
                    callback(model, opt, epoch, i)

            lr = get_mean_lr(opt)
            if scheduler:
                scheduler.step()

            pbar_postfix = {
                "acc": "%.2f" % (train_epoch_correct / n * 100),
                "loss": "%.4g" % (train_epoch_loss / n),
                "lr": "%.3g" % lr,
            }
            if test_data:
                for i, dataset in enumerate(test_data):
                    acc, loss = clf_eval(model, dataset)
                    pbar_postfix.update(
                        {
                            "acc" + str(i): "%.2f" % (acc * 100),
                            # "loss" + str(i): "%.4g" % loss,
                        }
                    )
                    if record:
                        acc_loss[i].append((acc, loss))
            pbar.set_postfix(**pbar_postfix)

            if epoch_callback is not None:
                epoch_callback(
                    epoch,
                    train_epoch_loss / n,
                    train_epoch_correct / n,
                    lr,
                )

    if record:
        return model, *acc_loss
    return model


def mini_train_multi(
    *,
    model: torch.nn.Module,
    train_datasets: Iterable[Union[DataLoader, Dataset]],
    test_data: Union[
        Union[DataLoader, Dataset], Iterable[Union[DataLoader, Dataset]]
    ] = None,
    batch_size=32,
    opt: optim.Optimizer,
    scheduler,
    epochs: int,
    shuffle=True,
    callback=None,
    record=False,
    agg_method="mean",
    f=1,
    epoch_callback=None,
    track_poison_selection=False,
):
    device = get_module_device(model)

    dataloaders = [
        either_dataloader_dataset_to_both(
            train_data, batch_size=batch_size, shuffle=shuffle
        )[0]
        for train_data in train_datasets
    ]

    client_sizes = [len(dl.dataset) for dl in dataloaders]
    total_samples = sum(client_sizes)
    total_examples = epochs * total_samples

    if test_data:
        if not isinstance(test_data, Iterable):
            test_data = [test_data]
        acc_loss = [[] for _ in range(len(test_data))]

    if track_poison_selection:
        # Worker ordering convention shared with partition_across_workers /
        # build_federated_datasets: honest workers first, poisoned workers last (the last
        # `f` entries of `dataloaders`).
        num_workers_total = len(dataloaders)
        poisoned_worker_ids = list(range(num_workers_total - f, num_workers_total))
        poison_stats = {
            "total_aggregations": 0,
            "any_poisoned_selected_given_flip": 0,
            "workers": {
                w: {
                    "batches_with_flip": 0,
                    "times_selected_total": 0,
                    "times_selected_given_flip": 0,
                }
                for w in poisoned_worker_ids
            },
        }

    with make_pbar(total=total_examples) as pbar:
        for epoch in range(1, epochs + 1):
            model.train()
            train_epoch_loss = 0.0
            train_epoch_correct = 0

            for batches in zip(*dataloaders):
                grad_buffer = [[] for _ in model.parameters()]

                batch_loss = 0.0
                batch_correct = 0
                batch_samples = 0

                if track_poison_selection:
                    batch_has_flip = {}

                for w, batch in enumerate(batches):
                    if track_poison_selection:
                        x, y, is_flipped = batch
                        if w in poison_stats["workers"]:
                            batch_has_flip[w] = bool(is_flipped.any().item())
                    else:
                        x, y = batch
                    x, y = x.to(device), y.to(device)
                    batch_samples += len(x)

                    model.zero_grad()
                    y_pred = model(x)
                    loss = clf_loss(y_pred, y)
                    correct = clf_correct(y_pred, y)

                    loss.backward()

                    for i, p in enumerate(model.parameters()):
                        if p.grad is not None:
                            grad_buffer[i].append(p.grad.detach().clone())

                    batch_loss += loss.item() * len(x)
                    batch_correct += correct.item()

                if track_poison_selection:
                    # Default for agg_method not in ("multikrum", "krum") -- "selection"
                    # isn't a concept for mean/median/trmean/soft*, so every worker reads as
                    # not-selected for those (same as before this fix).
                    step_selected_workers = set()

                if track_poison_selection and agg_method in ("multikrum", "krum"):
                    # True (Blanchard et al.) Multi-Krum selection, computed ONCE on each
                    # worker's FULL gradient (every parameter tensor flattened and
                    # concatenated) -- diagnostics ONLY, does not affect the actual per-
                    # parameter aggregation loop below (kept bit-for-bit unchanged: same
                    # aggr_multikrum(grads, f=f[, m=1]) call as before, its return value
                    # never depended on return_selected).
                    #
                    # Before this fix, poison_stats used the UNION, across every parameter
                    # tensor, of that tensor's OWN independent per-parameter Krum selection.
                    # r32p has ~90 parameter tensors; at m = n - f - 2 (roughly half of n
                    # selected per tensor), the probability that a given worker -- honest or
                    # poisoned -- is selected in at least ONE of ~90 independent per-tensor
                    # draws is essentially 1. That made "any_poisoned_selected_given_flip"
                    # saturate near 100% regardless of the attack's actual strength, which is
                    # why it didn't move across lambda_gradmatch/epsilon/etc. This block
                    # instead runs Krum a single time on the concatenated full-model
                    # gradient (one point per worker in the full parameter space), matching
                    # what Multi-Krum is actually meant to measure.
                    full_vectors = [
                        torch.cat([
                            grad_buffer[i][w].reshape(-1)
                            for i in range(len(grad_buffer))
                            if grad_buffer[i]
                        ])
                        for w in range(len(batches))
                    ]
                    krum_m = 1 if agg_method == "krum" else None
                    _, step_selected_workers = aggr_multikrum(
                        full_vectors, f=f, m=krum_m, return_selected=True
                    )
                    step_selected_workers = set(step_selected_workers)

                model.zero_grad()
                for i, p in enumerate(model.parameters()):
                    if not grad_buffer[i]:
                        continue

                    grads = torch.stack(grad_buffer[i], dim=0)

                    if agg_method == "mean":
                        g = grads.mean(dim=0)
                    elif agg_method == "median":
                        g = grads.median(dim=0).values
                    elif agg_method == "trmean":
                        g = aggr_trmean(grads, f=f)
                    elif agg_method == "multikrum":
                        g = aggr_multikrum(grads, f=f)
                    elif agg_method == "softmultikrum":
                        g = aggr_soft_multikrum(grads, f=f, m=grads.shape[0] - f - 2)
                    elif agg_method == "krum":
                        g = aggr_multikrum(grads, f=f, m=1)
                    elif agg_method == "softkrum":
                        g = aggr_soft_multikrum(grads, f=f, m=1)
                    else:
                        raise ValueError(f"Unknown agg_method: {agg_method}")

                    p.grad = g

                opt.step()

                if track_poison_selection:
                    poison_stats["total_aggregations"] += 1
                    any_selected_given_flip = False
                    for w in poison_stats["workers"]:
                        has_flip = batch_has_flip.get(w, False)
                        selected = w in step_selected_workers
                        if has_flip:
                            poison_stats["workers"][w]["batches_with_flip"] += 1
                            if selected:
                                poison_stats["workers"][w]["times_selected_given_flip"] += 1
                                any_selected_given_flip = True
                        if selected:
                            poison_stats["workers"][w]["times_selected_total"] += 1
                    if any_selected_given_flip:
                        poison_stats["any_poisoned_selected_given_flip"] += 1

                train_epoch_loss += batch_loss
                train_epoch_correct += batch_correct
                pbar.update(batch_samples)

                if callback is not None:
                    callback(model, opt, epoch)

            if scheduler:
                scheduler.step()

            lr = get_mean_lr(opt)
            postfix = {
                "acc": "%.2f" % (train_epoch_correct / total_samples * 100),
                "loss": "%.4g" % (train_epoch_loss / total_samples),
                "lr": "%.3g" % lr,
            }

            if test_data:
                for i, dataset in enumerate(test_data):
                    acc, loss = clf_eval(model, dataset)
                    postfix[f"acc{i}"] = "%.2f" % (acc * 100)
                    if record:
                        acc_loss[i].append((acc, loss))

            pbar.set_postfix(**postfix)

            if epoch_callback is not None:
                epoch_callback(
                    epoch,
                    train_epoch_loss / total_samples,
                    train_epoch_correct / total_samples,
                    lr,
                )

    if track_poison_selection:
        if record:
            return model, poison_stats, *acc_loss
        return model, poison_stats
    if record:
        return model, *acc_loss
    return model


def get_train_info(
    params,
    train_flag,
    batch_size=None,
    epochs=None,
    optim_kwargs={},
    scheduler_kwargs={},
):
    if train_flag == "sgd":
        batch_size = batch_size or DEFAULT_SGD_BATCH_SIZE
        epochs = epochs or DEFAULT_SGD_EPOCHS
        kwargs = {**DEFAULT_SGD_KWARGS, **optim_kwargs}
        sched_kwargs = {**DEFAULT_SGD_SCHED_KWARGS, **scheduler_kwargs}
        opt = optim.SGD(params, **kwargs)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(opt, **sched_kwargs)
    elif train_flag == "adam":
        batch_size = batch_size or DEFAULT_ADAM_BATCH_SIZE
        epochs = epochs or DEFAULT_ADAM_EPOCHS
        kwargs = {**DEFAULT_ADAM_KWARGS, **optim_kwargs}
        sched_kwargs = {**DEFAULT_ADAM_SCHED_KWARGS, **scheduler_kwargs}
        opt = optim.Adam(params, **kwargs)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(opt, **sched_kwargs)
    else:
        raise NotImplementedError

    return batch_size, epochs, opt, lr_scheduler


def get_mtt_attack_info(
    expert_params,
    labels,
    expert_kwargs,
    labels_kwargs,
    train_flag="sgd",
    batch_size=None,
    epochs=None,
):
    if train_flag == "sgd":
        batch_size = batch_size or DEFAULT_SGD_BATCH_SIZE
        epochs = epochs or DEFAULT_SGD_EPOCHS
        opt_expert = optim.SGD(expert_params, **expert_kwargs)
        opt_labels = optim.SGD([labels], **labels_kwargs)
    else:
        raise NotImplementedError

    assert len(opt_expert.state_dict()["param_groups"]) == 1
    return batch_size, epochs, opt_expert, opt_labels


def needs_big_ims(model_flag):
    return model_flag in BIG_IMS_MODELS
