# FLIP
## tl;dr
Official implementation of **BRoADflip**, forked from the official implementation of [FLIP](https://arxiv.org/abs/2310.18933), presented at [NeurIPS 2023](https://neurips.cc/virtual/2023/poster/70392).  
The original repository is available here: [https://github.com/SewoongLab/FLIP](https://github.com/SewoongLab/FLIP).

The aggregation methods implemented in this repository are based on the official implementation of *Distributed Momentum for Byzantine-Resilient Learning* ([paper](https://arxiv.org/abs/2003.00010)), available at: [https://github.com/LPD-EPFL/ByzantineMomentum](https://github.com/LPD-EPFL/ByzantineMomentum).
---
## Abstract



---
## In this repo

This repo is split into three main folders: `experiments`, `modules`, and `schemas`. The `experiments` folder (as described in more detail [here](#installation)) contains subfolders and `.toml` configuration files on which an experiment may be run. The `modules` folder stores source code for each of the subsequent part of an experiment. These modules take in specific inputs and outputs as defined by their subseqeunt `.toml` documentation in the `schemas` folder. Each module refers to a step of the FLIP or BRoADflip algorithm.

### Existing modules:
for FLIP: 
1. `base_utils`: Utility module, used by the base modules.
2. `train_expert`: Step 1 of our algorithm: training expert models and recording trajectories.
3. `generate_labels`: Step 2 of our algorithm: generating poisoned labels from trajectories.
4. `select_flips`: Step 3 of algorithm: strategically flipping labels within some budget.
5. `train_user`: Evaluation module to assess attack success rate.

for BRoADflip
1. `federated_optimizing_trigger'`: design an aggregator-aware trigger knowing the victim's learning setup.
2. `train_expert`: training expert models and recording trajectories.
3. `federated_generate_labels`: Step 2 of our algorithm: generating poisoned labels from aggregated trajectories.
4. `select_flips`: strategically flipping labels within some budget and split data among workers.
5. `federated_train_user`: Evaluation module to assess attack success rate.

More documentation can be found in the `schemas` folder.

### Supported Datasets:
1. CIFAR-10
1. SVHN
1. CIFAR-100
1. Tiny ImageNet

---
## Installation
### Prerequisites:
For our experiments we used Python 3.9.25.
The prerequisite packages are stored in `requirements.txt` and can be installed using pip:
```
pip install -r requirements.txt
```
Or conda:
```
conda install --file requirements.txt
```
Note that the requirements encapsulate our testing enviornments and may be unnecessarily tight! Any relevant updates to the requirements are welcomed.

## Running An Experiment
### Setting up:
To initialize an experiment, create a subfolder in the `experiments` folder with the name of your experiment:
```
mkdir experiments/[experiment name]
```
In that folder initialize a config file called `config.toml`. An example can be seen here: `experiments/example_attack/config.toml`.

The `.toml` file should contain references to the modules that you would like to run with each relevant field as defined by its documentation in `schemas/[module name]`. This file will serve as the configuration file for the entire experiment. As a convention the output for module **n** is the input for module **n + 1**.

**Note:** the `[INTERNAL]` block of a schema should not be transferred into a config file.

```
[module_name_1]
output=...
field2=...
...
fieldn=...

[module_name_2]
input=...
output=...
...
fieldn=...

...

[module_name_k]
input=...
field2=...
...
fieldn=...
```

### Running a module:
At the moment, all experiments must be manually run using:
```
python run_experiment.py [experiment name]
```
The experiment will automatically pick up on the configuration provided by the file. 

