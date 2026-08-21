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

### Mean aggregation: joint trigger / inversion-policy optimization

Two threat models for jointly optimizing the trigger delta and the label-inversion policy
under mean aggregation, for a given inversion budget beta -- see each module's schema for the
exact objective. Both reuse `train_expert` / `federated_select_flips` / `federated_train_user`
unchanged, and produce trigger `.pt` / metric artifacts in the same conventions as
`federated_optimizing_trigger`, so the two threat models are directly comparable.

**Threat model "expert" (P^mean)** — co-optimizes delta and an explicit policy u over label
flips (in place of `federated_optimizing_trigger`'s implicit QP optimum), then materializes u
into concrete flips:

1. `train_expert` (poisoner="1xs") — bootstraps the expert checkpoint trajectory
2. `federated_optimizing_trigger_policy` — jointly optimizes delta and the policy u
3. `federated_policy_to_flips` — materializes u into per-worker label flips (same output
   layout as `federated_select_flips`)
4. `federated_train_user` — victim training and ASR evaluation against the optimized trigger

**Threat model "direct"** — extends `federated_generate_labels` to jointly optimize
continuous poisoned labels and delta directly, via the trajectory-matching alignment loss
plus a backdoor-efficacy term:

1. `train_expert` (poisoner="1xs") — bootstraps the (shared) expert checkpoint trajectory
2. `federated_generate_labels_trigger` — jointly optimizes labels_syn and delta
3. `federated_select_flips` — discretizes labels_syn into per-worker flips at a given budget
4. `federated_train_user` — victim training and ASR evaluation against the optimized trigger

See `experiments/federated_experiments/threat_model_expert/` and
`experiments/federated_experiments/threat_model_direct/` for worked (smoke-scale) examples.

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