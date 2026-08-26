#!/bin/bash
set -e
set -x

BASE_DIR="$HOME/FLIP"
LOG_DIR="$BASE_DIR/logs"

mkdir -p "$LOG_DIR"

DATASETS=("cifar" "svhn") # "cifar_100" "tiny_imagenet"
ATTACK="backdoor"
AGGREGATORS=("mean") #  "median" "trmean" "multikrum" "krum"
BUDGETS=(0 150 300 500 1000 1500 2000 2500 5000)
N_CYCLES=5
NUM_CLEAN=0
NUM_POISONED=1
MODEL_FLAGS=("r32p" "convnext_micro") # "r18" "vgg" "vgg-pretrain"  "vit-pretrain"
POISONERS=("optimized" "1xs")

MACHINES=(
# Salle 30
allemagne
angleterre
autriche
belgique
espagne
# # finlande
# # france
# # groenland
# # hollande
# # hongrie
# irlande
# islande
# lituanie
# malte
# monaco
# pologne
# portugal
# roumanie
# suede
# # Salle 31
# # albatros
# # autruche
# # bengali
# # coucou
# # dindon
# # epervier
# # faisan
# # gelinotte
# hibou
# # harpie
# jabiru
# kamiche
# linotte
# loriol
# # mouette
# # nandou
# # ombrette
# # perdrix
# # quetzal
# # quiscale
# # rouloul
# # sitelle
# # traquet
# # urabu
# # verdier
# # Salle 32
# aerides
# barlia
# calanthe
# diuris
# encyclia
# epipactis
# gennaria
# habenaria
# isotria
# ipsea
# liparis
# lycaste
# malaxis
# neotinea
# oncidium
# ophrys
# orchis
# pleione
# pogonia
# serapias
# telipogon
# vanda
# vanilla
# xylobium
# zeuxine
# # Salle 33
# ain
# allier
# ardennes
# carmor
# charente
# cher
# creuse
# dordogne
# doubs
# essonne
# finistere
# gironde
# indre
# jura
# landes
# loire
# manche
# marne
# mayenne
# morbihan
# moselle
# saone
# somme
# vendee
# vosges
# # Salle 34
ablette
anchois
anguille
# barbeau
barbue
baudroie
# brochet
# carrelet
# gardon
# # gymnote
# labre
# lieu
# # lotte
# # mulet
# # murene
# # piranha
# # raie
# # requin
# rouget
# roussette
# saumon
# silure
# sole
# thon
# truite
# # Salle 35
# acromion
# apophyse
# astragale
# atlas
# axis
# coccyx
# cote
# cubitus
# cuboide
# femur
# frontal
# humerus
# malleole
# metacarpe
# parietal
# perone
# phalange
# radius
# rotule
# sacrum
# sternum
# tarse
# temporal
# tibia
# xiphoide
# # Salle 36
# bentley
# bugatti
# cadillac
# chrysler
# corvette
# ferrari
# fiat
# ford
# jaguar
# lada
# maserati
# mazda
# nissan
# niva
# peugeot
# pontiac
# porsche
# renault
# rolls
# rover
# royce
# simca
# skoda
# venturi
# volvo
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

    echo "[WAIT] Waiting for jobs..."

    while true; do
        all_done=true
        for f in "${files[@]}"; do
            [ ! -f "$f" ] && all_done=false && break
        done
        $all_done && break
        sleep 10
    done

    echo "[DONE] Batch completed"
}

# --------------------------------------------------
# Cleanup
# --------------------------------------------------

echo "Cleaning logs..."
rm -f "$LOG_DIR"/*.log "$LOG_DIR"/*.done || true

# ==================================================
# OPTIMIZING TRIGGER
# ==================================================

echo "=============================="
echo "OPTIMIZING TRIGGER"
echo "=============================="

TRIGGER_JOBS=()

for model_flag in "${MODEL_FLAGS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for aggregator in "${AGGREGATORS[@]}"; do
            if ! { [[ "$dataset" == "tiny_imagenet" && ( "$model_flag" == "r18" || "$model_flag" == "r32p" ) ]]; }; then
                TRIGGER_JOBS+=("$model_flag|$dataset|$aggregator")
            fi
        done
    done
done

TOTAL=${#TRIGGER_JOBS[@]}
INDEX=0

while [ $INDEX -lt $TOTAL ]; do

    DONE_FILES=()

    echo "[BATCH] Jobs $INDEX → $((INDEX + N_MACHINES - 1))"

    for ((i=0; i<N_MACHINES && INDEX<TOTAL; i++)); do

        IFS='|' read -r model_flag dataset aggregator <<< "${TRIGGER_JOBS[$INDEX]}"
        machine=${MACHINES[$i]}

        config="federated_experiments/${model_flag}/${NUM_POISONED}vs${NUM_CLEAN}/${dataset}/${ATTACK}/${aggregator}/opt_trigger"

        safe_name="trigger_${model_flag}_${dataset}_${aggregator}_${machine}"

        done_file="$LOG_DIR/${safe_name}.done"
        log_file="$LOG_DIR/${safe_name}.log"

        rm -f "$done_file"

        run_remote "$machine" \
            "python run_experiment.py $config federated_optimizing_trigger" \
            "$done_file" "$log_file" &

        DONE_FILES+=("$done_file")

        INDEX=$((INDEX + 1))
    done

    wait_for_done_files "${DONE_FILES[@]}"

    echo "Batch finished"

    # éviter quota plein
    rm -f "$LOG_DIR"/*.log
    rm -f "$LOG_DIR"/*.done
done

echo "=============================="
echo "ALL TRIGGERS DONE"
echo "=============================="