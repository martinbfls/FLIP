#!/bin/bash
# TRAIN_USER sous connaissance partielle de l'agrégateur.
# Même structure que le script d'origine, avec la boucle d'agrégateur dédoublée :
#   AGG_ASSUMED = hypothèse de l'attaquant (labels + trigger générés sous celle-ci)
#   AGG_TRUE    = règle réellement appliquée à l'entraînement

# set -e retiré : un ssh qui échoue ne doit pas tuer tout le lancement.
set -x

BASE_DIR="$HOME/FLIP"
LOG_DIR="$BASE_DIR/logs_partial"
DONE_DIR="$LOG_DIR/done"        # marqueurs de runs réussis, gardés entre relancements

mkdir -p "$LOG_DIR" "$DONE_DIR"

DATASETS=("cifar") # "cifar_100" "tiny_imagenet"  "svhn"
ATTACK="backdoor"
AGG_ASSUMED=("mean" "median" "trmean" "multikrum" "krum")
AGG_TRUE=("mean" "median" "trmean" "multikrum" "krum")
BUDGETS=(150 300 500 1000 1500 2000 2500 5000) #0
N_CYCLES=10
NUM_CLEAN=7
NUM_POISONED=3
MODEL_FLAG="r32p"
POISONERS=("optimized") #  "1xp" "4xl"

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
# Remote launcher
# --------------------------------------------------

run_remote() {
    local machine=$1
    local cmd=$2
    local done_file=$3
    local log_file=$4

    echo "[LAUNCH] $machine → $cmd"

    # `$cmd; echo $? > done` au lieu de `$cmd; touch done` : le marqueur porte
    # le code de retour, sinon un run planté compte comme réussi.
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

# --------------------------------------------------
# Cleanup
# --------------------------------------------------
# On ne purge plus les .log : ceux des runs réussis sont supprimés au fil de
# l'eau, ceux des runs plantés sont gardés pour diagnostic.

echo "Cleaning previous done files..."
rm -f "$LOG_DIR"/*.done || true


# ==================================================
# TRAIN USER
# ==================================================

echo "=============================="
echo "TRAIN_USER (PARTIAL KNOWLEDGE)"
echo "=============================="

TRAIN_JOBS=()
SKIPPED=0

for dataset in "${DATASETS[@]}"; do
    for poisoner in "${POISONERS[@]}"; do
        for assumed in "${AGG_ASSUMED[@]}"; do
            for true_agg in "${AGG_TRUE[@]}"; do
                if [ "$assumed" == "$true_agg" ]; then
                    continue  # pas de run "assumed = true" : c'est le cas standard, déjà couvert par orchestrate_runs.sh
                fi
                for ((run_id=1; run_id<=N_CYCLES; run_id++)); do
                    for budget in "${BUDGETS[@]}"; do

                        job="$dataset|$poisoner|$assumed|$true_agg|$run_id|$budget"
                        name="${MODEL_FLAG}_${dataset}_a-${assumed}_t-${true_agg}_${poisoner}_${budget}_${run_id}"

                        # Reprise : un run déjà validé n'est pas relancé.
                        if [ -f "$DONE_DIR/${name}.ok" ]; then
                            SKIPPED=$((SKIPPED + 1))
                            continue
                        fi

                        TRAIN_JOBS+=("$job")
                    done
                done
            done
        done
    done
done

TOTAL_TRAIN=${#TRAIN_JOBS[@]}
INDEX=0
FAILED=()

echo "À lancer : $TOTAL_TRAIN   déjà faits : $SKIPPED"

while [ $INDEX -lt $TOTAL_TRAIN ]; do

    DONE_FILES=()
    NAMES=()
    MACHS=()
    echo "[BATCH TRAIN] Launching jobs $INDEX → $((INDEX + N_MACHINES - 1))"

    for ((i=0; i<N_MACHINES && INDEX<TOTAL_TRAIN; i++)); do

        IFS='|' read -r dataset poisoner assumed true_agg run_id budget <<< "${TRAIN_JOBS[$INDEX]}"
        machine=${MACHINES[$i]}

        config="federated_partial_knowledge/${MODEL_FLAG}/${NUM_POISONED}vs${NUM_CLEAN}/${dataset}/${ATTACK}/assumed_${assumed}/true_${true_agg}/${poisoner}/train_user_${budget}/${run_id}"

        name="${MODEL_FLAG}_${dataset}_a-${assumed}_t-${true_agg}_${poisoner}_${budget}_${run_id}"
        # La machine reste dans le nom de fichier : un `ls $LOG_DIR` pendant le
        # lot montre quel run tourne où.
        safe_name="train_${name}_${machine}"

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

    # Bilan du lot : on trie réussites et échecs à partir du code de retour.
    for ((k=0; k<${#DONE_FILES[@]}; k++)); do
        code=$(cat "${DONE_FILES[$k]}" 2>/dev/null || echo 1)
        name="${NAMES[$k]}"
        machine="${MACHS[$k]}"
        log_file="$LOG_DIR/train_${name}_${machine}.log"

        if [ "$code" = "0" ]; then
            echo "$machine" > "$DONE_DIR/${name}.ok"
            rm -f "$log_file"
        else
            echo "[FAIL] $name sur $machine (code $code) — log : $log_file"
            FAILED+=("$name ($machine)")
        fi
    done

    rm -f "$LOG_DIR"/*.done
    echo "Train batch completed"
done

echo "=============================="
echo "ALL DONE — échecs : ${#FAILED[@]}"
for f in "${FAILED[@]}"; do echo "  - $f"; done
echo "=============================="