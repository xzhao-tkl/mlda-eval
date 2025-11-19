#!/bin/bash

EXP_NAME=$1
afterok=${2:-None}

if [ -z "$1" ]; then
    echo "Error: \$1 is not provided!"
    exit 1
fi

set -eu -o pipefail

## Generate the instructions dataset for continual-pretraining
python3 train_01_generate_dataset.py --config_name $EXP_NAME

## Load config and environment
config_path=/data/xzhao/experiments/roman-pretrain/exps/$EXP_NAME/config.json

do_tuning=$(jq -r '.train.tune' $config_path)
if [ "$do_tuning" != "false" ]; then
    echo "Error: train.tune is not set to false in config."
    exit 1
fi

exp_name=$(jq -r '.name' $config_path)
exp_dir=$(jq -r '.["exp-dir"]' $config_path)/$exp_name
log_dir=$exp_dir/logs
mkdir -p $log_dir

## Tokenize the dataset
tokenize_id=$(sbatch --output $log_dir/tokenize-%j.out --error $log_dir/tokenize-%j.err train_02_tokenize.sh $EXP_NAME | awk '{print $4}')


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

# ## Check if the training job was successful
# while squeue -j $train_id > /dev/null 2>&1; do
#     sleep 5
# done
# echo "Training job $train_id completed."

# ## Converting Megatoron checkpoint to Hugging Face format
# train_dir=$exp_dir/training/checkpoints
# latest_checkpointed_iteration=$(ls -d $train_dir/iter_* | sort -V | tail -n 1)
# echo "Latest checkpointed iteration: $latest_checkpointed_iteration"

# if [ -z "$latest_checkpointed_iteration" ]; then
#     echo "Error: No checkpoint found in $train_dir"
#     exit 1
# fi

# hf_dir=$exp_dir/hf_model
# if [ ! -d $hf_dir ]; then
#     mkdir -p $hf_dir
#     echo "Converting checkpoint to Hugging Face format..."
#     convert_id=$(sbatch --job-name=0068_CONV \
#                 --partition gpu \
#                 --output $log_dir/convert-%j.out \
#                 --error $log_dir/convert-%j.err \
#                 /data/xzhao/projects/scripts/pretrain/scripts/v3-converter/convert.sh \
#                 $latest_checkpointed_iteration $hf_dir | awk '{print $4}')
#     # echo "Conversion job submitted with job ID: $train_id, $convert_id"  
# fi

# ## Transfer model to llm-mdx cluster for evaluation
# # rsync --mkpath -rv exp1-en/training/hf_model/ xzhao@login.llm-jp.mdx.jp:/model/projects/cpt-scidoc/roman-pretrain/exp1-en/hf_model