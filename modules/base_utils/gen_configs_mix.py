"""
Génération de configs `federated_train_user` pour l'étude de *partial knowledge*
sur l'agrégateur.

Idée
----
Dans le pipeline complet, un seul `{aggregator}` pilote simultanément :
  1. le chemin des labels générés   (`input_labels`)
  2. le chemin du trigger optimisé  (`delta`)
  3. la règle d'agrégation appliquée à l'entraînement (`agg_method`)

L'attaquant est donc implicitement omniscient sur la défense. Ici on découple :

  - AGG_ASSUMED : l'agrégateur supposé par l'attaquant au moment de la génération
                  de labels (et de l'optimisation du trigger) -> détermine (1) et (2)
  - AGG_TRUE    : l'agrégateur réellement utilisé par le serveur -> détermine (3)

Aucune régénération de labels n'est nécessaire : on réutilise les artefacts déjà
produits sous LABELS_ROOT par le pipeline existant, et on ne fait varier que la
règle d'agrégation de l'étape victime. La diagonale (assumed == true) reproduit
le réglage omniscient et sert de baseline.
"""

import csv
from pathlib import Path

NUM_POISONED = 3
NUM_CLEAN = 7
ATTACK = "backdoor"
DATASETS = ["cifar"] #, "svhn"
POISONERS = ["optimized"]
INIT = "stripe"
MODEL_FLAGS = ["r32p"]

# --- Matrice de connaissance partielle -------------------------------------
AGG_ASSUMED = ["mean", "median", "trmean", "multikrum", "krum"]
AGG_TRUE = ["mean", "median", "trmean", "multikrum", "krum"]
INCLUDE_DIAGONAL = True          # False -> uniquement les cas de mismatch

# Le trigger suit-il l'hypothèse de l'attaquant ("assumed") ou la réalité ("true") ?
# "assumed" = scénario cohérent (l'attaquant optimise tout sous sa croyance).
# "true"    = ablation : trigger correct, labels erronés — isole la contribution
#             de la génération de labels au succès de l'attaque.
TRIGGER_FOLLOWS = "assumed"

BUDGETS = [0, 150, 300, 500, 1000, 1500, 2000, 2500, 5000]
N_CYCLES = 10

# Racine des artefacts déjà produits (labels.npy, flips aux budgets).
LABELS_ROOT = "out"
# Racine des sorties de cette étude, séparée pour ne rien écraser.
RESULTS_ROOT = "out_partial_knowledge"

BASE_DIR = Path("experiments/federated_partial_knowledge").resolve()
MANIFEST = BASE_DIR / "manifest.csv"

LEARNING_RATE = {'convnext_micro': 0.1, 'r18': 0.1, 'r32p': 0.1, 'vgg': 0.01, 'vgg-pretrain': 0.01, 'vit-pretrain': 0.05}
WEIGHT_DECAY = {'convnext_micro': 2e-4, 'r18': 2e-4, 'r32p': 2e-4, 'vgg': 2e-4, 'vgg-pretrain': 2e-4, 'vit-pretrain': 5e-4}
MILESTONE = {'convnext_micro': [75, 125], 'r18': [75, 125], 'r32p': [75, 125], 'vgg': [125], 'vgg-pretrain': [125], 'vit-pretrain': [125]}

TRAIN_USER_TEMPLATE = """# federated_train_user sous connaissance partielle de l'agrégateur.
# labels/trigger générés en supposant : {agg_assumed}
# agrégation réellement appliquée      : {agg_true}
[federated_train_user]
input_labels = "{labels_root}/{model_flag}/{num_poisoned}vs{num_clean}/{dataset}/{attack}/{agg_assumed}/{poisoner}/{run_id}/"
budget = {budget}
user_model = "{model_flag}"
trainer = "sgd"
dataset = "{dataset}"
source_label = 9
target_label = 4
poisoner = "{poisoner}"
delta = "optimized_trigger/{model_flag}_triggers_neurips/fed_opt_trig_{init}_{model_flag}_{dataset}_{agg_trigger}_{num_poisoned}vs{num_clean}.pt"
output_dir = "{results_root}/{model_flag}/{num_poisoned}vs{num_clean}/{dataset}/{attack}/assumed_{agg_assumed}/true_{agg_true}/{poisoner}/{run_id}/{budget}"
soft = false
alpha = 0.0
num_honests = {num_clean}
num_poisoned = {num_poisoned}
agg_method = "{agg_true}"
optim_kwargs = {{lr = {lr}, momentum = 0.9, nesterov = true, weight_decay = {wd}}}
schedule_kwargs = {{milestones = {milestones}, gamma = 0.1}}
"""


def write_config(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print(f"[OK] Config written to {path}")


def generate_all_configs():
    rows = []
    for model_flag in MODEL_FLAGS:
        lr = LEARNING_RATE.get(model_flag, 0.1)
        wd = WEIGHT_DECAY.get(model_flag, 2e-4)
        milestones = MILESTONE.get(model_flag, [75, 125])
        for dataset in DATASETS:
            if dataset == "tiny_imagenet" and model_flag in ["r18", "r32p"]:
                continue
            for agg_assumed in AGG_ASSUMED:
                for agg_true in AGG_TRUE:
                    if agg_assumed == agg_true and not INCLUDE_DIAGONAL:
                        continue
                    agg_trigger = agg_assumed if TRIGGER_FOLLOWS == "assumed" else agg_true
                    for poisoner in POISONERS:
                        for run_id in range(1, N_CYCLES + 1):
                            for budget in BUDGETS:
                                cfg_dir = (
                                    BASE_DIR
                                    / f"{model_flag}/{NUM_POISONED}vs{NUM_CLEAN}/{dataset}/{ATTACK}"
                                    / f"assumed_{agg_assumed}/true_{agg_true}/{poisoner}"
                                    / f"train_user_{budget}/{run_id}"
                                )
                                cfg = TRAIN_USER_TEMPLATE.format(
                                    dataset=dataset,
                                    model_flag=model_flag,
                                    num_poisoned=NUM_POISONED,
                                    num_clean=NUM_CLEAN,
                                    attack=ATTACK,
                                    agg_assumed=agg_assumed,
                                    agg_true=agg_true,
                                    agg_trigger=agg_trigger,
                                    poisoner=poisoner,
                                    run_id=run_id,
                                    budget=budget,
                                    init=INIT,
                                    lr=lr,
                                    wd=wd,
                                    milestones=milestones,
                                    labels_root=LABELS_ROOT,
                                    results_root=RESULTS_ROOT,
                                )
                                write_config(cfg_dir / "config.toml", cfg)
                                rows.append({
                                    "config": str(cfg_dir / "config.toml"),
                                    "model": model_flag,
                                    "dataset": dataset,
                                    "attack": ATTACK,
                                    "poisoner": poisoner,
                                    "agg_assumed": agg_assumed,
                                    "agg_true": agg_true,
                                    "agg_trigger": agg_trigger,
                                    "mismatch": int(agg_assumed != agg_true),
                                    "budget": budget,
                                    "run_id": run_id,
                                })

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[OK] {len(rows)} configs — manifest: {MANIFEST}")


if __name__ == "__main__":
    generate_all_configs()