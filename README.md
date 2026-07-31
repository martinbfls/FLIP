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
```

---

## Running experiments

### Legacy SSH-pool infrastructure

`orchestrate_runs.sh` (and its `bis`/`ter`/`trigger`/`trigger_orthogonal`
variants) dispatch experiments over a hardcoded pool of SSH-reachable
machines listed in the `MACHINES` array, one job per idle machine at a
time. `connect_all.sh` sanity-checks Python availability on that pool, and
`kill_all.sh` stops stray runs across it. These remain fully supported.

### Slurm cluster

The same experiment grids can be run on a Slurm cluster via:

- `slurm_lib.sh` — shared pool executor: packs many experiments as
  concurrent `srun --exclusive` steps inside a single sbatch allocation
  (no `sbatch` per experiment).
- `orchestrate_runs_slurm.sh` — Slurm translation of `orchestrate_runs.sh`
  (same gen_labels/train_user grid), submit with `sbatch`.
- `run_experiment_slurm.sh` — single-experiment entry point; also usable
  with `sbatch --array=...` to fan a config out over seeds using the
  existing `SLURM_ARRAY_TASK_ID` support in `run_experiment.py`.
- `kill_all_slurm.sh` — cancels this user's queued/running FLIP jobs via
  `scancel`.

All three `.sh`/`sbatch` scripts have `#SBATCH` resource directives left
commented out as placeholders (partition, GPUs, CPUs, memory, time,
constraint) — fill them in, or pass the equivalent flags on the `sbatch`
command line. See the header comments of each script for exact usage.