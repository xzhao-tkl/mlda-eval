#!/bin/bash

EXP_NAME=$1
afterok=${2:-None}

if [ -z "$1" ]; then
    echo "Error: \$1 is not provided!"
    exit 1
fi

set -eu -o pipefail

## Generate the instructions dataset for continual-pretraining
python3 tune_01_generate_dataset.py --config_name $EXP_NAME

## Load config and environment
config_path=/data/xzhao/experiments/roman-pretrain/exps/$EXP_NAME/config.json

do_tuning=$(jq -r '.train.tune' $config_path)
if [ "$do_tuning" != "true" ]; then
    echo "Error: train.tune is not set to true in config."
    exit 1
fi

exp_name=$(jq -r '.name' $config_path)
exp_dir=$(jq -r '.["exp-dir"]' $config_path)/$exp_name
log_dir=$exp_dir/logs
mkdir -p $log_dir

## Tokenize the dataset
tokenize_id=$(
    sbatch \
        --output $log_dir/tuning-tokenize-%j.out \
        --error $log_dir/tuning-tokenize-%j.err tune_02_tokenize.sh $EXP_NAME \
    | awk '{print $4}')


# Training
if [ "$afterok" == "None" ]; then
    train_id=$(sbatch --output $log_dir/train-%j.out \
                --error $log_dir/train-%j.err \
                --dependency=afterok:$tokenize_id \
                train_03_pretrain.sh $EXP_NAME | awk '{print $4}')
else
    train_id=$(sbatch --output $log_dir/train-%j.out \
                --error $log_dir/train-%j.err \
                --dependency=afterok:$afterok:$tokenize_id \
                train_03_pretrain.sh $EXP_NAME | awk '{print $4}')
fi
echo "Training job submitted with job ID: $train_id"
