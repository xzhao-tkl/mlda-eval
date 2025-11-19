#!/bin/bash
#SBATCH --job-name=0068_TRAIN
#SBATCH --partition=gpu
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8

set -eu -o pipefail

if [ -z "$1" ]; then
    echo "Error: \$1 is not provided!"
    exit 1
fi

## Load config and environment
config_path=/data/xzhao/experiments/roman-pretrain/exps/$1/config.json
env_dir=$(jq -r '.env' $config_path)
source ${env_dir}/scripts/environment.sh
source ${env_dir}/venv/bin/activate

## Initialize variables for paths
exp_name=$(jq -r '.name' $config_path)
exp_root=$(jq -r '.["exp-dir"]' $config_path)/$exp_name
exp_dir=$exp_root/training
tokenizer_name=$(jq -r '.dataset.tokenizer.type' $config_path)
token_dir=$exp_root/tokenized/
token_info=$token_dir/token_info.csv
if [ ! -e "$token_info" ]; then
  echo "Error: '$token_info' does not exist." >&2
  exit 1
fi

mkdir -p $exp_dir
pushd $exp_dir

MASTER_ADDR=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_ADDR
export MASTER_PORT=$((10000 + (SLURM_JOBID % 50000)))

echo "MASTER_ADDR=${MASTER_ADDR}"

NUM_NODES=$SLURM_JOB_NUM_NODES
NUM_GPUS_PER_NODE=$(echo "$SLURM_TASKS_PER_NODE" | cut -d '(' -f 1)
NUM_GPUS=$((NUM_NODES * NUM_GPUS_PER_NODE))

echo "NUM_NODES=$NUM_NODES"
echo "NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE"
echo NUM_GPUS=$NUM_GPUS

script_root=/home/xzhao/workspace/roman-pretrain/experiments
mpirun \
  -np $NUM_GPUS \
  --npernode "$NUM_GPUS_PER_NODE" \
  -bind-to none \
  -map-by slot \
  -x EXP_NAME=$exp_name \
  -x WORK_DIR=$exp_dir \
  -x ENV_DIR=$env_dir \
  -x CONFIG_PATH=$config_path \
  -x TOKEN_DIR=$token_dir \
  -x SCRIPT_ROOT=$script_root \
  -x MASTER_ADDR=$MASTER_ADDR \
  -x MASTER_PORT=$MASTER_PORT \
  bash ${script_root}/99_train-13b.sh

echo "Training completed successfully."