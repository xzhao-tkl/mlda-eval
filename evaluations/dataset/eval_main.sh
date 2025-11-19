#!/bin/bash
#SBATCH --job-name=0068_EVAL
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --output=/data/xzhao/experiments/med-eval/logs/collect_entity-%x-%j.out
#SBATCH --error=/data/xzhao/experiments/med-eval/logs/collect_entity-%x-%j.err


# models=(
#     "exp1-multi-en_jstage" "exp1-multi-ja" "exp1-en_jstage" "exp1-ja"
#     "exp3-balanced-en_jstage" "exp3-balanced-ja"
#     "exp3-medical-en_jstage" "exp3-medical-ja"
#     "exp3-science-en_jstage" "exp3-science-ja"
#     "exp4-balanced-en_jstage" "exp4-balanced-ja"
#     "exp4-medical-en_jstage" "exp4-medical-ja"
#     "exp4-science-en_jstage" "exp4-science-ja"
#     "exp5-medical-en_jstage" "exp5-medical-ja"
# )

# models=(
#     "exp6-multi-en_jstage-en2ja100" "exp6-multi-en_jstage-en2ja50" 
#     "exp6-multi-ja-umls100-11" "exp6-multi-ja-umls100-11.nonduplicated" 
# )

models=(
    "exp6-multi-ja-wordnet32" "exp7-mono-syntax32" "exp7-mono-syntax32-p0"
)

# models=("exp6-mono-syntax8" "exp6-mono-syntax16" "exp6-mono-syntax32")

# Multiple checkpoints evaluation
for prompt in "prompt"; do
    for model in "${models[@]}"; do
        checkpoints_dir="/data/xzhao/experiments/roman-pretrain/exps/${model}/hf_models"
        # list subdirectories, reverse order
        for dir in $(ls -d "$checkpoints_dir"/*/ | sort -r); do
            iters=$(basename "$dir")
            if [[ $iters =~ ^iter_[0-9]{7}$ ]]; then
                echo "Evaluating model: llm-jp-3-13b-$model, checkpoint: $dir"
                bash evaluate_adaxeval.sh "" "$prompt" "llm-jp-3-13b-$model" "$dir" "8" "" "" "" "" "" "$iters"
            fi
        done
    done
done
