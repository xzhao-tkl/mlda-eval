#!/bin/bash
#SBATCH --job-name=0068_EVAL
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --output=/data/xzhao/experiments/med-eval/logs/collect_entity-%x-%j.out
#SBATCH --error=/data/xzhao/experiments/med-eval/logs/collect_entity-%x-%j.err


# Multiple checkpoints evaluation
for prompt in "prompt"; do
    checkpoints_dir="/data/xzhao/experiments/roman-pretrain/exps/base-13b/hf_model"
    # list subdirectories, reverse order
    echo "Evaluating model: llm-jp-3-13b-base, checkpoint: $checkpoints_dir"
    bash evaluate_adaxeval.sh "" "$prompt" "llm-jp-3-13b-base" "$checkpoints_dir" "8" "" "" "" "" "" "iter_0000000"
done
