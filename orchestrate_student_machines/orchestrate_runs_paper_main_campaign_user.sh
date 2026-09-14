#!/bin/bash
# TRAIN_USER for gen_configs_paper_main_campaign.py, resumed on the non-Slurm shared student
# machines after a Slurm per-user job-count limit killed the campaign's USER phase mid-run
# (see logs_slurm/paper_user_r32p_svhn_baseline_seed5_su_300.log). BOOTSTRAP/GEN/FLIPS keep
# running on Slurm (see orchestrate_slurm/orchestrate_runs_trigger_joint_paper_main_campaign_slurm.sh's
# SKIP_USER=1 mode) -- this script only ever submits federated_train_user jobs.
#
# Same SSH job-pool pattern as orchestrate_runs_bis.sh (same MACHINES pool, run_remote/
# wait_for_done_files, return-code-bearing .done markers), but resumes by checking for the
# ACTUAL output artifacts (caccs.npy/paccs.npy) federated_train_user writes -- not a
# run-local `.ok` marker -- since the jobs already completed before the kill were launched by
# Slurm, which never wrote this script's markers.
#
# PREREQUISITE: rsync the campaign's experiments/ subtree (configs + GEN/FLIPS outputs +
# trigger .pt + any already-completed train_user outputs) from the Slurm cluster onto this
# shared NFS home FIRST -- e.g., from the cluster:
#   rsync -avz --info=progress2 \
#     experiments/federated_experiments/threat_model_direct_trigger_joint_paper_main_campaign/ \
#     <a_student_machine>:"$HOME/FLIP/experiments/federated_experiments/threat_model_direct_trigger_joint_paper_main_campaign/"
# (any one machine suffices if $HOME/FLIP is NFS-shared across the MACHINES pool below, per
# orchestrate_runs_bis.sh's own convention). train_expert checkpoints (out/checkpoints/, a
# separate top-level dir federated_train_user never reads) are excluded automatically --
# they're outside experiments/ entirely, no --exclude needed.
#
# Usage:
#   TAG=baseline ./orchestrate_runs_paper_main_campaign_user.sh
#   TAG=eps_16_255 ./orchestrate_runs_paper_main_campaign_user.sh   # once that tag's GEN/FLIPS land

set -x

BASE_DIR="$HOME/FLIP"
LOG_DIR="$BASE_DIR/logs_paper_user"
mkdir -p "$LOG_DIR"

EXP_BASE_REL="federated_experiments/threat_model_direct_trigger_joint_paper_main_campaign"

TAG="${TAG:-baseline}"
MODEL_FLAGS=("r32p" "convnext_micro")
DATASETS=("cifar" "svhn")
SEEDS=(0 1 2 3 4 5 6 7 8 9)
DEPLOY_BUDGETS=(0 150 300 500 1000 2000 2500 5000)
DEPLOY_SINGLE_USER_AGG="mean"
DEPLOY_AGG_METHODS_FEDERATED=("mean" "krum" "multikrum" "median" "trmean")
FEDERATED_TAG="federated_3vs7"

MACHINES=(
# Salle 30
allemagne
# angleterre
autriche
belgique
espagne
finlande
france
groenland
hollande
hongrie
irlande
islande
lituanie
malte
monaco
pologne
portugal
roumanie
suede
# Salle 31
albatros
autruche
bengali
coucou
dindon
epervier
faisan
gelinotte
hibou
# harpie
jabiru
kamiche
linotte
loriol
mouette
# nandou
ombrette
perdrix
# quetzal
quiscale
rouloul
sitelle
traquet
urabu
verdier
# Salle 32
aerides
barlia
calanthe
diuris
encyclia
epipactis
# gennaria
# habenaria
isotria
ipsea
liparis
# lycaste
malaxis
neotinea
oncidium
ophrys
orchis
# pleione
pogonia
serapias
telipogon
vanda
vanilla
xylobium
zeuxine
# Salle 33
ain
allier
ardennes
carmor
charente
cher
creuse
dordogne
doubs
essonne
finistere
gironde
indre
jura
landes
loire
manche
# marne
# mayenne
morbihan
moselle
saone
# somme
# vendee
vosges
# Salle 34
ablette
anchois
anguille
barbeau
barbue
baudroie
brochet
carrelet
gardon
gymnote
labre
lieu
lotte
mulet
murene
piranha
raie
requin
rouget
roussette
saumon
silure
sole
thon
truite
# Salle 35
acromion
apophyse
astragale
atlas
axis
coccyx
cote
cubitus
cuboide
femur
frontal
humerus
malleole
metacarpe
parietal
perone
phalange
radius
rotule
sacrum
sternum
tarse
temporal
tibia
xiphoide
# Salle 36
bentley
bugatti
cadillac
chrysler
corvette
ferrari
fiat
ford
jaguar
lada
maserati
mazda
nissan
niva
peugeot
pontiac
porsche
renault
rolls
rover
royce
simca
skoda
# venturi
volvo
)

N_MACHINES=${#MACHINES[@]}

# --------------------------------------------------
# Remote launcher (return-code-bearing marker, same as orchestrate_runs_bis.sh)
# --------------------------------------------------

run_remote() {
    local machine=$1
    local cmd=$2
    local done_file=$3
    local log_file=$4

    echo "[LAUNCH] $machine -> $cmd"

    ssh "$machine" "
        cd $BASE_DIR &&
        nohup bash -c '$cmd > $log_file 2>&1; echo \$? > $done_file' > /dev/null 2>&1 &
    "
}

# --------------------------------------------------
# Waiter
# --------------------------------------------------

wait_for_done_files() {
    local files=("$@")
    echo "[WAIT] Waiting for jobs to finish..."

    while true; do
        all_done=true
        for f in "${files[@]}"; do
            [ ! -f "$f" ] && all_done=false && break
        done
        $all_done && break
        sleep 10
    done

    echo "[DONE] Phase completed"
}

echo "Cleaning previous done files..."
rm -f "$LOG_DIR"/*.done || true

# ==================================================
# TRAIN USER (paper_main_campaign, tag=$TAG)
# ==================================================

echo "=============================="
echo "TRAIN_USER (paper_main_campaign, tag=$TAG)"
echo "=============================="

# already_done: true if federated_train_user's own output (caccs.npy + paccs.npy, same
# convention show_results.py's get_final_value reads) already exists for this config -- the
# ground truth for "did this cell finish", regardless of which orchestrator (Slurm or this
# script, on a prior/interrupted run) produced it.
already_done() {
    local config="$1"
    [ -f "$BASE_DIR/experiments/$config/caccs.npy" ] && [ -f "$BASE_DIR/experiments/$config/paccs.npy" ]
}

USER_JOBS=()
SKIPPED=0

for model in "${MODEL_FLAGS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            cell="$EXP_BASE_REL/$model/$dataset/$TAG/seed${seed}"

            for budget in "${DEPLOY_BUDGETS[@]}"; do
                config="$cell/$DEPLOY_SINGLE_USER_AGG/train_user_${budget}"
                name="paper_user_${model}_${dataset}_${TAG}_seed${seed}_su_${budget}"
                if already_done "$config"; then
                    SKIPPED=$((SKIPPED + 1))
                    continue
                fi
                USER_JOBS+=("$config|$name")
            done

            for agg in "${DEPLOY_AGG_METHODS_FEDERATED[@]}"; do
                for budget in "${DEPLOY_BUDGETS[@]}"; do
                    config="$cell/$FEDERATED_TAG/$agg/train_user_${budget}"
                    name="paper_user_${model}_${dataset}_${TAG}_seed${seed}_${agg}_${budget}"
                    if already_done "$config"; then
                        SKIPPED=$((SKIPPED + 1))
                        continue
                    fi
                    USER_JOBS+=("$config|$name")
                done
            done
        done
    done
done

TOTAL=${#USER_JOBS[@]}
INDEX=0
FAILED=()

echo "À lancer : $TOTAL   déjà faits (caccs/paccs présents) : $SKIPPED"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[DRY-RUN] rien ne sera lancé."
    for job in "${USER_JOBS[@]}"; do
        IFS='|' read -r config name <<< "$job"
        echo "[DRY-RUN] $name -> $config"
    done
    exit 0
fi

while [ $INDEX -lt $TOTAL ]; do

    DONE_FILES=()
    NAMES=()
    MACHS=()
    echo "[BATCH TRAIN] Launching jobs $INDEX -> $((INDEX + N_MACHINES - 1))"

    for ((i = 0; i < N_MACHINES && INDEX < TOTAL; i++)); do

        IFS='|' read -r config name <<< "${USER_JOBS[$INDEX]}"
        machine=${MACHINES[$i]}
        safe_name="${name}_${machine}"

        done_file="$LOG_DIR/${safe_name}.done"
        log_file="$LOG_DIR/${safe_name}.log"
        rm -f "$done_file"

        run_remote "$machine" "python run_experiment.py $config" "$done_file" "$log_file" &
        DONE_FILES+=("$done_file")
        NAMES+=("$name")
        MACHS+=("$machine")

        INDEX=$((INDEX + 1))
    done

    wait_for_done_files "${DONE_FILES[@]}"

    for ((k = 0; k < ${#DONE_FILES[@]}; k++)); do
        code=$(cat "${DONE_FILES[$k]}" 2>/dev/null || echo 1)
        name="${NAMES[$k]}"
        machine="${MACHS[$k]}"
        log_file="$LOG_DIR/${name}_${machine}.log"

        if [ "$code" = "0" ]; then
            rm -f "$log_file"
        else
            echo "[FAIL] $name sur $machine (code $code) -- log : $log_file"
            FAILED+=("$name ($machine)")
        fi
    done

    rm -f "$LOG_DIR"/*.done
    echo "Train batch completed"
done

echo "=============================="
echo "ALL DONE (tag=$TAG) -- échecs : ${#FAILED[@]}"
for f in "${FAILED[@]}"; do echo "  - $f"; done
echo "=============================="
