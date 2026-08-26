#!/bin/bash
set -e
set -x

BASE_DIR="$HOME/FLIP"
LOG_DIR="$BASE_DIR/logs"

mkdir -p "$LOG_DIR"

DATASET="cifar"
ATTACK="backdoor"

AGGREGATORS=("mean" "median" "krum" "trmean" "multikrum")
POISONERS=("1xs")

BUDGETS=(150 300 500 1000 1500 2000 2500 5000)
N_CYCLES=10

NUM_CLEAN=7
NUM_POISONED=3
MODEL_FLAG="r32p"

MACHINES=(
# Salle 30
allemagne
angleterre
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
harpie
jabiru
kamiche
linotte
loriol
mouette
nandou
ombrette
perdrix
quetzal
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
gennaria
habenaria
isotria
ipsea
liparis
lycaste
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
gironde
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
#bentley
#bugatti
#cadillac
#chrysler
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
#bentley bugatti cadillac chrysler

N_MACHINES=${#MACHINES[@]}

# ==========================================================
# REMOTE LAUNCHER
# ==========================================================

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

# ==========================================================
# POOL EXECUTOR
# ==========================================================

run_job_pool() {
    local -n JOBS=$1   # array reference
    local PHASE=$2

    local TOTAL=${#JOBS[@]}
    local INDEX=0

    declare -a running
    for ((i=0;i<N_MACHINES;i++)); do running[i]=""; done

    echo "[POOL $PHASE] total jobs = $TOTAL"

    while true; do

        # -----------------------------------------
        # assign jobs to free machines
        # -----------------------------------------
        for ((i=0;i<N_MACHINES;i++)); do
            if [ -z "${running[i]:-}" ] && [ $INDEX -lt $TOTAL ]; then

                machine=${MACHINES[$i]}
                job="${JOBS[$INDEX]}"

                IFS='|' read -r cmd safe_name <<< "$job"

                done_file="$LOG_DIR/${safe_name}.done"
                log_file="$LOG_DIR/${safe_name}.log"

                rm -f "$done_file"

                run_remote "$machine" "$cmd" "$done_file" "$log_file"

                running[i]=$done_file
                INDEX=$((INDEX + 1))

                echo "[POOL $PHASE] $machine ← job $INDEX / $TOTAL"
            fi
        done

        # -----------------------------------------
        # check if finished
        # -----------------------------------------
        all_idle=true
        for ((i=0;i<N_MACHINES;i++)); do
            [ -n "${running[i]:-}" ] && all_idle=false && break
        done

        if $all_idle && [ $INDEX -ge $TOTAL ]; then
            break
        fi

        # -----------------------------------------
        # wait one job completion
        # -----------------------------------------
        while true; do
            for ((i=0;i<N_MACHINES;i++)); do
                if [ -n "${running[i]:-}" ] && [ -f "${running[i]}" ]; then
                    running[i]=""
                    rm -f "$LOG_DIR"/*.log  # nettoyage progressif
                    break 2
                fi
            done
            sleep 5
        done
    done

    echo "[POOL $PHASE] completed"
}

# ==========================================================
# CLEAN START
# ==========================================================

rm -f "$LOG_DIR"/*.log "$LOG_DIR"/*.done || true


# ==========================================================
# BUILD GEN JOBS (GLOBAL)
# ==========================================================

GEN_JOBS=()

for poisoner in "${POISONERS[@]}"; do
for aggregator in "${AGGREGATORS[@]}"; do
for ((run_id=1; run_id<=N_CYCLES; run_id++)); do

config="federated_experiments/${MODEL_FLAG}/${NUM_POISONED}vs${NUM_CLEAN}/${DATASET}/${ATTACK}/${aggregator}/${poisoner}/gen_labels/${run_id}"

safe="gen_${MODEL_FLAG}_${NUM_POISONED}vs${NUM_CLEAN}_${DATASET}_${ATTACK}_${aggregator}_${poisoner}_${run_id}"

cmd="python run_experiment.py $config"

GEN_JOBS+=("$cmd|$safe")

done
done
done


# ==========================================================
# RUN GEN POOL
# ==========================================================

echo "=============================="
echo "GEN_LABELS GLOBAL POOL"
echo "=============================="

run_job_pool GEN_JOBS GEN


# ==========================================================
# BUILD TRAIN JOBS (GLOBAL)
# ==========================================================

TRAIN_JOBS=()

for poisoner in "${POISONERS[@]}"; do
for aggregator in "${AGGREGATORS[@]}"; do
for ((run_id=1; run_id<=N_CYCLES; run_id++)); do
for budget in "${BUDGETS[@]}"; do

config="federated_experiments/${MODEL_FLAG}/${NUM_POISONED}vs${NUM_CLEAN}/${DATASET}/${ATTACK}/${aggregator}/${poisoner}/train_user_${budget}/${run_id}"

safe="train_${MODEL_FLAG}_${NUM_POISONED}vs${NUM_CLEAN}_${DATASET}_${ATTACK}_${aggregator}_${poisoner}_${budget}_${run_id}"

cmd="python run_experiment.py $config"

TRAIN_JOBS+=("$cmd|$safe")

done
done
done
done


# ==========================================================
# RUN TRAIN POOL
# ==========================================================

echo "=============================="
echo "TRAIN_USER GLOBAL POOL"
echo "=============================="

run_job_pool TRAIN_JOBS TRAIN


echo "=============================="
echo "ALL DONE"
echo "=============================="
