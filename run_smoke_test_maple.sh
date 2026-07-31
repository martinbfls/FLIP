#!/bin/bash
###
# run_smoke_test_maple.sh
#
# One-off pipeline validation run on the "maple" GH200 partition: chains
# optimizing_trigger -> generate_labels (train_expert + federated_generate_
# labels + federated_select_flips) -> train_user as 3 sequential srun steps
# inside a single sbatch allocation (1 GH200 GPU, since maple nodes only
# expose 1 GPU each per `sinfo`).
#
# Each step depends on the previous one's output (trigger -> checkpoints ->
# labels -> flips -> user model), so this script uses `set -e`: unlike the
# large parallel campaigns (see orchestrate_runs_slurm.sh), a failure here
# must stop the chain rather than be shrugged off.
#
# Configs used (already reduced for a quick test — small train_pct, few
# epochs, single small budget):
#   experiments/federated_experiments/smoke_test_maple/r32p/1vs1/cifar/backdoor/mean/{opt_trigger,gen_labels/1,train_user_50/1}/config.toml
#
# KNOWN PREREQUISITE TO VERIFY BEFORE SUBMITTING:
#   optimizing_trigger's expert_path expects pre-existing "1xs" expert
#   checkpoints at:
#     /shared/data1/Project/DLWP/j1067582/beaufiles/FLIP/out/checkpoints/r32p_1xs/{}/model_{}_{}.pth
#   trained for up to 20 epochs with checkpoints at iterations 50/100/150/200
#   (see opt_trigger/config.toml [federated_optimizing_trigger.expert_config]).
#   If no such checkpoints exist yet, step 1 will fail on torch.load().
#
# Submit with:
#   mkdir -p logs_slurm && sbatch run_smoke_test_maple.sh
###

#SBATCH --job-name=flip_smoke_test
#SBATCH --partition=maple
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:gh200:1
#SBATCH --cpus-per-task=16      # TODO: adjust — node has 72 cores total (1:72:1)
#SBATCH --mem=0                 # 0 = all memory on the node (fine: 1 GPU/node on maple, no sharing)
#SBATCH --time=01:00:00         # TODO: adjust — includes first-run CIFAR-10 download (internet confirmed on-node)
#SBATCH --output=logs_slurm/%x-%j.out

set -e
set -x

BASE_DIR="/shared/data1/Project/DLWP/j1067582/beaufiles/FLIP"  # confirmed shared storage path
cd "$BASE_DIR" || exit 1

mkdir -p logs_slurm

TRIGGER_CONFIG="federated_experiments/smoke_test_maple/r32p/1v0/cifar/backdoor/mean/opt_trigger"
GEN_LABELS_CONFIG="federated_experiments/smoke_test_maple/r32p/1vs0/cifar/backdoor/mean/gen_labels/1"
TRAIN_USER_CONFIG="federated_experiments/smoke_test_maple/r32p/1vs0/cifar/backdoor/mean/train_user_1500/1"

echo "=============================="
echo "STEP 1/3: optimizing_trigger"
echo "=============================="
srun python run_experiment.py "$TRIGGER_CONFIG"

echo "=============================="
echo "STEP 2/3: generate_labels (train_expert + federated_generate_labels + federated_select_flips)"
echo "=============================="
srun python run_experiment.py "$GEN_LABELS_CONFIG"

echo "=============================="
echo "STEP 3/3: train_user"
echo "=============================="
srun python run_experiment.py "$TRAIN_USER_CONFIG"

echo "=============================="
echo "SMOKE TEST COMPLETE"
echo "=============================="
