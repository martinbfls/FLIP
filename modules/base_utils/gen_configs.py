import os
from pathlib import Path

NUM_POISONED = 3
NUM_CLEAN = 7
ATTACK = "backdoor"
DATASETS = ["cifar", "svhn"] # , "cifar"
POISONERS = ["optimized"] #"1xs",
INIT="stripe"
MODEL_FLAGS = ["r32p"] # , "convnext_micro"
AGGREGATORS = ["mean", "median", "trmean", "multikrum", "krum"] #
BUDGETS = [0, 150, 300, 500, 1000, 1500, 2000, 2500, 5000]
N_CYCLES = 10
GAMMA = 1.0
RESTART = False
ORTHOGONAL = False

BASE_DIR = Path("experiments/federated_experiments").resolve()

LEARNING_RATE = {'convnext_micro': 0.1, 'r18': 0.1, 'r32p': 0.1, 'vgg': 0.01, 'vgg-pretrain': 0.01, 'vit-pretrain': 0.05}
WEIGHT_DECAY = {'convnext_micro': 2e-4, 'r18': 2e-4, 'r32p': 2e-4, 'vgg': 2e-4, 'vgg-pretrain': 2e-4, 'vit-pretrain': 5e-4}
MILESTONE = {'convnext_micro': [75, 125], 'r18': [75, 125], 'r32p': [75, 125], 'vgg': [125], 'vgg-pretrain': [125], 'vit-pretrain': [125]}

OPT_TRIGGER_TEMPLATE = """[federated_optimizing_trigger]
model = "{model_flag}"
dataset = "{dataset}"
source_label = 9
target_label = 4

lambda_match = 1.0
lambda_adv = 1.0
lambda_penalty = 1.0
lambda_delta = 0.0

epsilon = 1.0
lr_delta = 1e-2
n_steps = 50
alpha_ckpt = 0.01
num_chckpt = 3

init = "{init}"

expert_path = "/Data/mb/flip/out/checkpoints/{model_flag}_1xs/{{}}/model_{{}}_{{}}.pth"
device = "cuda"
optim_kwargs = {{lr = {lr}, momentum = 0.9, nesterov = true, weight_decay = {wd}}}
schedule_kwargs = {{milestones = {milestones}, gamma = 0.1}}
output_dir = "optimized_trigger"

num_poisoned = {num_poisoned}
num_honests = {num_clean}
agg_method = "{aggregator}"

restart = {restart}
orthogonal = {orthogonal}

[federated_optimizing_trigger.expert_config]
experts = 1
min = 0
max = 20
trajectories = [50, 100, 150, 200]

"""

GEN_LABEL_TEMPLATE = """# Module to train and record an expert trajectory.
[train_expert]
output_dir = "/Data/mb/flip/out/checkpoints/{model_flag}_{poisoner}/0/"
model = "{model_flag}"
trainer = "sgd"
dataset = "{dataset}"
source_label = 9
target_label = 4
poisoner = "{poisoner}"
delta = "optimized_trigger/{model_flag}_triggers_neurips/fed_opt_trig_stripe_{model_flag}_{dataset}_{aggregator}_{num_poisoned}vs{num_clean}.pt"
epochs = 20
checkpoint_iters = 50
optim_kwargs = {{lr = {lr}, momentum = 0.9, nesterov = true, weight_decay = {wd}}}
schedule_kwargs = {{milestones = {milestones}, gamma = 0.1}}

# Module to generate attack labels from the expert trajectories.
[federated_generate_labels]
input_pths = "/Data/mb/flip/out/checkpoints/{model_flag}_{poisoner}/{{}}/model_{{}}_{{}}.pth"
opt_pths = "/Data/mb/flip/out/checkpoints/{model_flag}_{poisoner}/{{}}/model_{{}}_{{}}_opt.pth"
expert_model = "{model_flag}"
trainer = "sgd"
dataset = "{dataset}"
source_label = 9
target_label = 4
poisoner = "{poisoner}"
delta = "optimized_trigger/{model_flag}_triggers_neurips/fed_opt_trig_stripe_{model_flag}_{dataset}_{aggregator}_{num_poisoned}vs{num_clean}.pt"
output_dir = "out/{model_flag}/{num_poisoned}vs{num_clean}/{dataset}/{attack}/{aggregator}/{poisoner}/{run_id}/"
lambda = 0.0
num_honests = {num_clean}
num_poisoned = {num_poisoned}
agg_method = "{aggregator}"
attack = "{attack}"
gamma = {gamma}

[federated_generate_labels.expert_config]
experts = 1
min = 0
max = 20
trajectories = [50, 100, 150, 200]

[federated_generate_labels.attack_config]
iterations = 15
one_hot_temp = 5
alpha = 0
label_kwargs = {{lr = 150, momentum = 0.5}}
expert_kwargs = {{lr = {lr}, momentum = 0.9, nesterov = true, weight_decay = {wd}}}


# Module to flip labels at the provided budgets.
[federated_select_flips]
budgets = {budgets}
input_label_glob = "out/{model_flag}/{num_poisoned}vs{num_clean}/{dataset}/{attack}/{aggregator}/{poisoner}/{run_id}/labels.npy"
true_labels = "out/{model_flag}/{num_poisoned}vs{num_clean}/{dataset}/{attack}/{aggregator}/{poisoner}/{run_id}/true.npy"
output_dir = "out/{model_flag}/{num_poisoned}vs{num_clean}/{dataset}/{attack}/{aggregator}/{poisoner}/{run_id}"
num_honests = {num_clean}
num_poisoned = {num_poisoned}
"""

TRAIN_USER_TEMPLATE = """[federated_train_user]
input_labels = "out/{model_flag}/{num_poisoned}vs{num_clean}/{dataset}/{attack}/{aggregator}/{poisoner}/{run_id}/"
budget = {budget}
user_model = "{model_flag}"
trainer = "sgd"
dataset = "{dataset}"
source_label = 9
target_label = 4
poisoner = "{poisoner}"
delta = "optimized_trigger/{model_flag}_triggers_neurips/fed_opt_trig_stripe_{model_flag}_{dataset}_{aggregator}_{num_poisoned}vs{num_clean}.pt"
output_dir = "out/{model_flag}/{num_poisoned}vs{num_clean}/{dataset}/{attack}/{aggregator}/{poisoner}/{run_id}/{budget}"
soft = false
alpha = 0.0
num_honests = {num_clean}
num_poisoned = {num_poisoned}
agg_method = "{aggregator}"
optim_kwargs = {{lr = {lr}, momentum = 0.9, nesterov = true, weight_decay = {wd}}}
schedule_kwargs = {{milestones = {milestones}, gamma = 0.1}}
"""

def write_config(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"[OK] Config written to {path}")

def generate_all_configs():
    for model_flag in MODEL_FLAGS:
        for dataset in DATASETS:
            if dataset == "tiny_imagenet" and model_flag in ["r18", "r32p"]:
                continue
            for aggregator in AGGREGATORS:
                opt_dir = BASE_DIR / f"{model_flag}/{NUM_POISONED}vs{NUM_CLEAN}/{dataset}/{ATTACK}/{aggregator}/opt_trigger"
                lr = LEARNING_RATE.get(model_flag, 0.1)
                wd = WEIGHT_DECAY.get(model_flag, 2e-4)
                # opt_config = OPT_TRIGGER_TEMPLATE.format(
                #     model_flag=model_flag,
                #     dataset=dataset,
                #     aggregator=aggregator,
                #     num_poisoned=NUM_POISONED,
                #     num_clean=NUM_CLEAN,
                #     init=INIT,
                #     lr=lr,
                #     wd=wd,
                #     milestones=MILESTONE.get(model_flag, [75, 125]),
                #     restart=str(RESTART).lower(),
                #     orthogonal=str(ORTHOGONAL).lower(),
                # )
                # write_config(opt_dir / "config.toml", opt_config)
                for poisoner in POISONERS:
                    for run_id in range(1, N_CYCLES + 1):
                        gen_label_dir = BASE_DIR / f"{model_flag}/{NUM_POISONED}vs{NUM_CLEAN}/{dataset}/{ATTACK}/{aggregator}/{poisoner}/gen_labels/{run_id}"
                        gen_label_config = GEN_LABEL_TEMPLATE.format(
                            dataset=dataset,
                            model_flag=model_flag,
                            num_poisoned=NUM_POISONED,
                            num_clean=NUM_CLEAN,
                            attack=ATTACK,
                            gamma=GAMMA,
                            poisoner=poisoner,
                            aggregator=aggregator,
                            run_id=run_id,
                            budgets=BUDGETS, 
                            init=INIT,
                            lr=lr,
                            wd=wd,
                            milestones=MILESTONE.get(model_flag, [75, 125]),
                        )
                        write_config(gen_label_dir / "config.toml", gen_label_config)
                        for budget in BUDGETS:
                            train_user_dir = BASE_DIR / f"{model_flag}/{NUM_POISONED}vs{NUM_CLEAN}/{dataset}/{ATTACK}/{aggregator}/{poisoner}/train_user_{budget}/{run_id}"
                            train_user_config = TRAIN_USER_TEMPLATE.format(
                                dataset=dataset,
                                model_flag=model_flag,
                                num_poisoned=NUM_POISONED,
                                num_clean=NUM_CLEAN,
                                attack=ATTACK,
                                aggregator=aggregator,
                                poisoner=poisoner,
                                run_id=run_id,
                                budget=budget,
                                init=INIT,
                                lr=lr,
                                wd=wd,
                                milestones=MILESTONE.get(model_flag, [75, 125]),
                            )
                            write_config(train_user_dir / "config.toml", train_user_config)

if __name__ == "__main__":
    generate_all_configs()
