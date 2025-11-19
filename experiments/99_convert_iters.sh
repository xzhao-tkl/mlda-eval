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
checkpoints_dir=$exp_dir/training/checkpoints

if [ ! -d $checkpoints_dir ]; then
    echo "error: $checkpoints_dir does not exist!"
fi

hf_dir=$exp_dir/hf_models    

convert_id=-1
# for dir in $checkpoints_dir/*/; do
for dir in $(ls -d "$checkpoints_dir"/*/ | sort -r); do
  if [[ $(basename "$dir") =~ ^iter_[0-9]{7}$ ]]; then
    echo ">>> Converting checkpoint to Hugging Face format... : $dir -> $hf_dir/$(basename "$dir")"
    if [ -d $hf_dir/$(basename "$dir") ]; then
        echo "Directory $hf_dir/$(basename "$dir") already exists. Skipping conversion."
        continue
    fi

    if [ "$convert_id" == "-1" ]; then
        convert_id=$(sbatch --job-name=0068_CONV \
                    --partition gpu \
                    --nodes 1 \
                    --gpus-per-node 1 \
                    --ntasks-per-node 1 \
                    --output $log_dir/convert-%j.out \
                    --error $log_dir/convert-%j.err \
                    /data/xzhao/projects/scripts/pretrain/scripts/v3-converter/convert.sh \
                    $dir $hf_dir/$(basename "$dir") | awk '{print $4}')
        sleep 5
    else
        convert_id=$(sbatch --job-name=0068_CONV \
                    --partition gpu \
                    --nodes 1 \
                    --gpus-per-node 1 \
                    --ntasks-per-node 1 \
                    --dependency=afterok:$convert_id \
                    --output $log_dir/convert-%j.out \
                    --error $log_dir/convert-%j.err \
                    /data/xzhao/projects/scripts/pretrain/scripts/v3-converter/convert.sh \
                    $dir $hf_dir/$(basename "$dir") | awk '{print $4}')
        sleep 5
    fi
  fi
done
