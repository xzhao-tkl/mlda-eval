#!/bin/bash

EXP_NAME=$1


if [ -z "$1" ]; then
    echo "Error: \$1 is not provided!"
    exit 1
fi

config_path=/data/xzhao/experiments/roman-pretrain/exps/$EXP_NAME/config.json
exp_name=$(jq -r '.name' $config_path)
if [ "$exp_name" != $EXP_NAME ]; then
    echo "Error: $exp_name is not equal to $EXP_NAME"
fi

exp_dir=$(jq -r '.["exp-dir"]' $config_path)/$exp_name
log_dir=$exp_dir/logs
checkpoints_dir=$exp_dir/training/final_checkpoint

hf_dir=$exp_dir/hf_model

if [ ! -d "$hf_dir" ] || [ -z "$(ls -A "$hf_dir" 2>/dev/null)" ]; then
    echo "error: $hf_dir does not exist or is empty!"
    exit 1
fi

convert_id=-1
echo ">>> Converting Hugging Face checkpoint to MegatronLM format... : $hf_dir -> $checkpoints_dir)"
if [ -d $checkpoints_dir ]; then
    echo "Directory $checkpoints_dir already exists. Skipping conversion."
    continue
fi

sbatch --job-name=0068_CONV \
        --partition gpu \
        --output $log_dir/convert-%j.out \
        --error $log_dir/convert-%j.err \
        /data/xzhao/projects/scripts/pretrain/scripts/v3-converter/hf2mg-convert.sh \
        $checkpoints_dir $hf_dir | awk '{print $4}'