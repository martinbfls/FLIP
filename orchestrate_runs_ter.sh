#!/bin/bash
set -e
set -x

BASE_DIR="$HOME/FLIP"
LOG_DIR="$BASE_DIR/logs_ter"

mkdir -p "$LOG_DIR"

DATASETS=("cifar") # "cifar_100" "tiny_imagenet"
ATTACK="backdoor"
AGGREGATORS=("median" "trmean" "multikrum" "krum") #  
BUDGETS=(0 150 300 500 1000 1500 2000 2500 5000)
N_CYCLES=10
NUM_CLEAN=7
NUM_POISONED=3
MODEL_FLAG="r32p"
POISONERS=("optimized") #  "1xp" "4xl"

MACHINES=(
# Salle 30
allemagne
angleterre
autriche
# belgique
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
# pologne
# portugal
# roumanie
# suede
# Salle 31
#albatros
#autruche
#bengali
#coucou
#dindon
#epervier
#faisan
#gelinotte
#hibou
#harpie
# jabiru
# kamiche
# linotte
# loriol
# mouette
# nandou
# ombrette
# perdrix
# quetzal
# quiscale
# rouloul
# sitelle
# traquet
# urabu
# verdier
# Salle 32
aerides
barlia
calanthe
diuris
encyclia
epipactis
# gennaria
habenaria
isotria
ipsea
liparis
# lycaste
malaxis
neotinea
oncidium
ophrys
orchis
pleione
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
# gironde
indre
jura
landes
loire
manche
marne
mayenne
morbihan
moselle
saone
somme
vendee
vosges
# Salle 34
# ablette
# anchois
# anguille
# barbeau
# barbue
# baudroie
# brochet
# carrelet
# gardon
# gymnote
# labre
# lieu
# lotte
# mulet
# murene
# piranha
# raie
# requin
# rouget
# roussette
# saumon
# silure
# sole
# thon
# truite
# Salle 35
# acromion
# apophyse
# astragale
# atlas
# axis
# coccyx
# cote
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
# radius
rotule
sacrum
sternum
tarse
temporal
tibia
#xiphoide
# Salle 36
#bentley
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
venturi
volvo
)

N_MACHINES=${#MACHINES[@]}

# --------------------------------------------------
# Remote launcher
# --------------------------------------------------

run_remote() {
    local machine=$1
    local cmd=$2
    local done_file=$3
    local log_file=$4

    echo "[LAUNCH] $machine → $cmd"

    ssh "$machine" "
        cd $BASE_DIR &&
        nohup bash -c '$cmd; touch $done_file' > $log_file 2>&1 &
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

# --------------------------------------------------
# Cleanup
# --------------------------------------------------

echo "Cleaning previous logs and done files..."
rm -f "$LOG_DIR"/*.log "$LOG_DIR"/*.done || true


# ==================================================
# 1️⃣ GEN LABELS
# ==================================================

# echo "=============================="
# echo "GEN_LABELS (ALL CONFIGS)"
# echo "=============================="

# GEN_JOBS=()

# for dataset in "${DATASETS[@]}"; do
#     for poisoner in "${POISONERS[@]}"; do
#         for aggregator in "${AGGREGATORS[@]}"; do
#             for ((run_id=1; run_id<=N_CYCLES; run_id++)); do
#                 GEN_JOBS+=("$dataset|$poisoner|$aggregator|$run_id")
#             done
#         done
#     done
# done

# TOTAL_GEN=${#GEN_JOBS[@]}
# INDEX=0

# while [ $INDEX -lt $TOTAL_GEN ]; do

#     DONE_FILES=()
#     echo "[BATCH GEN] Launching jobs $INDEX → $((INDEX + N_MACHINES - 1))"

#     for ((i=0; i<N_MACHINES && INDEX<TOTAL_GEN; i++)); do

#         IFS='|' read -r dataset poisoner aggregator run_id <<< "${GEN_JOBS[$INDEX]}"
#         machine=${MACHINES[$i]}

#         config="federated_experiments/${MODEL_FLAG}/${NUM_POISONED}vs${NUM_CLEAN}/${dataset}/${ATTACK}/${aggregator}/${poisoner}/gen_labels/${run_id}"

#         safe_name="gen_${MODEL_FLAG}_${NUM_POISONED}vs${NUM_CLEAN}_${dataset}_${ATTACK}_${aggregator}_${poisoner}_${run_id}_${machine}"

#         done_file="$LOG_DIR/${safe_name}.done"
#         log_file="$LOG_DIR/${safe_name}.log"
#         rm -f "$done_file"

#         run_remote "$machine" "python run_experiment.py $config" "$done_file" "$log_file" &
#         DONE_FILES+=("$done_file")

#         INDEX=$((INDEX + 1))
#     done

#     wait_for_done_files "${DONE_FILES[@]}"
#     echo "Gen batch completed"

#     # nettoyage logs pour ne pas saturer quota
#     rm -f "$LOG_DIR"/*.log
#     rm -f "$LOG_DIR"/*.done
# done

# echo "gen_labels all runs done"


# ==================================================
# 2️⃣ TRAIN USER
# ==================================================

echo "=============================="
echo "TRAIN_USER (ALL CONFIGS)"
echo "=============================="

TRAIN_JOBS=()

for dataset in "${DATASETS[@]}"; do
    for poisoner in "${POISONERS[@]}"; do
        for aggregator in "${AGGREGATORS[@]}"; do
            for ((run_id=1; run_id<=N_CYCLES; run_id++)); do
                for budget in "${BUDGETS[@]}"; do
                    TRAIN_JOBS+=("$dataset|$poisoner|$aggregator|$run_id|$budget")
                done
            done
        done
    done
done

TOTAL_TRAIN=${#TRAIN_JOBS[@]}
INDEX=0

while [ $INDEX -lt $TOTAL_TRAIN ]; do

    DONE_FILES=()
    echo "[BATCH TRAIN] Launching jobs $INDEX → $((INDEX + N_MACHINES - 1))"

    for ((i=0; i<N_MACHINES && INDEX<TOTAL_TRAIN; i++)); do

        IFS='|' read -r dataset poisoner aggregator run_id budget <<< "${TRAIN_JOBS[$INDEX]}"
        machine=${MACHINES[$i]}

        config="federated_experiments/${MODEL_FLAG}/${NUM_POISONED}vs${NUM_CLEAN}/${dataset}/${ATTACK}/${aggregator}/${poisoner}/train_user_${budget}/${run_id}"

        # ✅ BUG FIX → ajout budget dans le nom
        safe_name="train_${MODEL_FLAG}_${NUM_POISONED}vs${NUM_CLEAN}_${dataset}_${ATTACK}_${aggregator}_${poisoner}_${budget}_${run_id}_${machine}"

        done_file="$LOG_DIR/${safe_name}.done"
        log_file="$LOG_DIR/${safe_name}.log"
        rm -f "$done_file"

        run_remote "$machine" "python run_experiment.py $config" "$done_file" "$log_file" &
        DONE_FILES+=("$done_file")

        INDEX=$((INDEX + 1))
    done

    wait_for_done_files "${DONE_FILES[@]}"
    echo "Train batch completed"

    rm -f "$LOG_DIR"/*.log
    rm -f "$LOG_DIR"/*.done
done

echo "=============================="
echo "ALL DONE"
echo "=============================="
