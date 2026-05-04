# FLIP

## TL;DR

Official implementation of **BRoADflip**, forked from the official implementation of [FLIP](https://arxiv.org/abs/2310.18933), presented at NeurIPS 2023.

Original repository: https://github.com/SewoongLab/FLIP

The aggregation methods used in this repository are based on:
*Distributed Momentum for Byzantine-Resilient Learning*  
https://arxiv.org/abs/2003.00010  
Official code: https://github.com/LPD-EPFL/ByzantineMomentum

---

## Abstract

---

## Repository structure

This repository is split into three main folders:

- `experiments`: experiment configurations and TOML files
- `modules`: implementation of each algorithmic component
- `schemas`: documentation of module inputs/outputs

Each module corresponds to a step of the FLIP or BRoADflip pipeline.

---

## Modules

### FLIP

1. `base_utils` — shared utilities
2. `train_expert` — training expert models and recording trajectories
3. `generate_labels` — generating poisoned labels from trajectories
4. `select_flips` — selecting label flips under a budget
5. `train_user` — evaluation of attack success rate

### BRoADflip

1. `federated_optimizing_trigger` — trigger optimized with aggregator awareness
2. `train_expert` — training expert models and recording trajectories
3. `federated_generate_labels` — generating poisoned labels in federated setting
4. `select_flips` — selecting label flips and distributing across workers
5. `federated_train_user` — evaluation module

More details are available in the `schemas/` folder.

---

## Supported datasets

- CIFAR-10
- SVHN
- CIFAR-100
- Tiny ImageNet

---

## Installation

### Requirements

Python 3.9.25

Install dependencies:

```bash
pip install -r requirements.txt