#!/bin/bash
# FEDERATED_SELECT_FLIPS for gen_configs_paper_main_campaign.py, run/resumed on the non-Slurm
# shared student machines -- e.g. after some select_flips cells failed there or on Slurm and
# need a rerun for a given tag. GEN and BOOTSTRAP keep running on Slurm (see
# orchestrate_slurm/orchestrate_runs_trigger_joint_paper_main_campaign_slurm.sh) -- this script
# only ever submits federated_select_flips jobs (single_user + federated_3vs7 branches), same
# grid/cell layout as that Slurm script's own FLIPS phase.
#
# Same SSH job-pool pattern as orchestrate_runs_paper_main_campaign_user.sh (same MACHINES pool,
# run_remote/wait_for_done_files, return-code-bearing .done markers). Resumes/skips by checking
# for select_flips's own ACTUAL last-written output file (see flips_done below, same check as
# orchestrate_slurm/orchestrate_runs_trigger_joint_paper_main_campaign_slurm.sh's flips_done) --
# not a run-local `.ok` marker -- so a cell that already succeeded (on Slurm before the failure,
# or on a prior run of this script) is never redone, and only cells left missing/failed are
# (re)submitted.
#
# PREREQUISITE: rsync the campaign's experiments/ subtree (configs + GEN outputs, i.e.
# gen_labels_trigger_joint/labels/{labels,true}.npy + trigger .pt) from the Slurm cluster onto
# this shared NFS home FIRST -- e.g., from the cluster:
#   rsync -avz --info=progress2 \
#     experiments/federated_experiments/threat_model_direct_trigger_joint_paper_main_campaign/ \
#     <a_student_machine>:"$HOME/FLIP/experiments/federated_experiments/threat_model_direct_trigger_joint_paper_main_campaign/"
# (any one machine suffices if $HOME/FLIP is NFS-shared across the MACHINES pool below, per
# orchestrate_runs_bis.sh's own convention).
#
# Usage:
#   TAG=baseline ./orchestrate_runs_paper_main_campaign_flips.sh
#   TAG=eps_16_255 ./orchestrate_runs_paper_main_campaign_flips.sh
#   TAG=baseline DRY_RUN=1 ./orchestrate_runs_paper_main_campaign_flips.sh

set -x

BASE_DIR="$HOME/FLIP"
LOG_DIR="$BASE_DIR/logs_paper_flips"
mkdir -p "$LOG_DIR"

EXP_BASE_REL="federated_experiments/threat_model_direct_trigger_joint_paper_main_campaign"

TAG="${TAG:-baseline}"
MODEL_FLAGS=("r32p" "convnext_micro")
DATASETS=("cifar" "svhn")
SEEDS=(0 1 2 3 4 5 6 7 8 9)
DEPLOY_BUDGETS=(0 150 300 500 1000 2000 2500 5000)
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
# Remote launcher (return-code-bearing marker, same as orchestrate_runs_paper_main_campaign_user.sh)
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
# SELECT_FLIPS (paper_main_campaign, tag=$TAG)
# ==================================================

echo "=============================="
echo "SELECT_FLIPS (paper_main_campaign, tag=$TAG)"
echo "=============================="

# flips_done: same check as orchestrate_slurm/orchestrate_runs_trigger_joint_paper_main_campaign_slurm.sh's
# own flips_done -- federated_select_flips's run_module.py writes, for each budget IN ORDER,
# idx_flipped/idx_clean THEN, per worker (0..num_workers-1) IN ORDER, worker{w}/{budget}_labels.npy
# THEN worker{w}/{budget}_indices.npy -- so the LAST budget's LAST worker's indices.npy is the
# last file select_flips ever writes, and its presence means that branch already completed
# (whether from a prior Slurm run, a prior run of this script, or the run that just failed and
# happened to still finish). num_workers = deploy_num_honests + deploy_num_poisoned: single_user
# = 0+1 = 1, federated_3vs7 = 7+3 = 10 (gen_configs_paper_main_campaign.py's own constants).
FLIPS_LAST_BUDGET="${DEPLOY_BUDGETS[${#DEPLOY_BUDGETS[@]}-1]}"
FLIPS_SINGLE_USER_WORKERS=1
FLIPS_FEDERATED_WORKERS=10
flips_done() {
    local config="$1" num_workers="$2"
    local last_worker=$((num_workers - 1))
    [ -f "$BASE_DIR/experiments/$config/worker${last_worker}/${FLIPS_LAST_BUDGET}_indices.npy" ]
}

FLIPS_JOBS=()
SKIPPED=0

for model in "${MODEL_FLAGS[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            cell="$EXP_BASE_REL/$model/$dataset/$TAG/seed${seed}"

            config="$cell/select_flips"
            name="paper_flips_${model}_${dataset}_${TAG}_seed${seed}"
            if flips_done "$config" "$FLIPS_SINGLE_USER_WORKERS"; then
                SKIPPED=$((SKIPPED + 1))
            else
                FLIPS_JOBS+=("$config|$name")
            fi

            fed_config="$cell/$FEDERATED_TAG/select_flips"
            fed_name="paper_flips_fed_${model}_${dataset}_${TAG}_seed${seed}"
            if flips_done "$fed_config" "$FLIPS_FEDERATED_WORKERS"; then
                SKIPPED=$((SKIPPED + 1))
            else
                FLIPS_JOBS+=("$fed_config|$fed_name")
            fi
        done
    done
done

TOTAL=${#FLIPS_JOBS[@]}
INDEX=0
FAILED=()

echo "À lancer : $TOTAL   déjà faits (flips présents) : $SKIPPED"

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[DRY-RUN] rien ne sera lancé."
    for job in "${FLIPS_JOBS[@]}"; do
        IFS='|' read -r config name <<< "$job"
        echo "[DRY-RUN] $name -> $config"
    done
    exit 0
fi

while [ $INDEX -lt $TOTAL ]; do

    DONE_FILES=()
    NAMES=()
    MACHS=()
    echo "[BATCH FLIPS] Launching jobs $INDEX -> $((INDEX + N_MACHINES - 1))"

    for ((i = 0; i < N_MACHINES && INDEX < TOTAL; i++)); do

        IFS='|' read -r config name <<< "${FLIPS_JOBS[$INDEX]}"
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
    echo "Flips batch completed"
done

echo "=============================="
echo "ALL DONE (tag=$TAG) -- échecs : ${#FAILED[@]}"
for f in "${FAILED[@]}"; do echo "  - $f"; done
echo "=============================="
