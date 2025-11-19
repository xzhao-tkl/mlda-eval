#!/bin/bash

set -eu -o pipefail

BASE_PATH="/home/xzhao/workspace/med-eval"   # Change this to your own path
export PYTHONPATH="${BASE_PATH}"
export TOKENIZERS_PARALLELISM=false

tasks=${1:-"all"}
tasks=(${tasks//,/ })
template=${2:-"standard"}
model_name=${3:-None}
model_path=${4:-None}
batch_size=${5:-32}
num_fewshot=${6:-0}
seed=${7:-42}
model_max_length=${8:--1}
use_knn_demo=${9:-False}
result_root=${10:-"./results/${model_name}_${template}_${num_fewshot}-shot"}
result_prefix=${11:-""}
n_gpus=${12:-0}

if [[ $n_gpus -eq 0 ]]; then
    n_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
fi

mkdir -p ${result_root}

if [ $model_name == "sarashina3b" ]; then
    model_path="sarashina2.2-3b"
elif [ $model_name == "llm-jp-3-1.8b" ]; then
    model_path="llm-jp/llm-jp-3-1.8b"
elif [ $model_name == "llm-jp-3-3.7b" ]; then
    model_path="llm-jp/llm-jp-3-3.7b"
elif [ $model_name == "llm-jp-3-7.2b" ]; then
    model_path="llm-jp/llm-jp-3-7.2b"
elif [ $model_name == "llm-jp-3-13b" ]; then
    model_path="/model/projects/cpt-scidoc/roman-pretrain/base-13b/hf_model"
elif [ $model_name == "qwen2.5-1.5b" ]; then
    model_path="Qwen/Qwen2.5-1.5B"
elif [ $model_name == "qwen2.5-3b" ]; then
    model_path="Qwen/Qwen2.5-3B"
elif [ $model_name == "qwen2.5-7b" ]; then
    model_path="Qwen/Qwen2.5-7B"
elif [ $model_name == "llama3.2-3b" ]; then
    model_path="meta-llama/Llama-3.2-3B"
elif [ $model_name == "llama3.2-1b" ]; then
    model_path="meta-llama/Llama-3.2-1B"
elif [ $model_name == "mmedllama3-8b" ]; then
    model_path="Henrychur/MMed-Llama-3-8B"
else
    echo ">>> Evaluating model name ${model_name} with model path ${model_path}"
fi

echo ">>> Tasks: ${tasks[@]}"
echo ">>> Template: $template"
echo ">>> Batch size: $batch_size"
echo ">>> Number of few-shot examples: $num_fewshot"
echo ">>> Seed: $seed"
echo ">>> Use KNN demo: $use_knn_demo"
echo ">>> Model name: $model_name"
echo ">>> Model path: $model_path"
echo ">>> Model max length: $model_max_length"
echo ">>> Result root: $result_root"
echo ">>> Result prefix: $result_prefix"
echo ">>> Number of GPUs: $n_gpus"
echo ">>> Starting evaluation..."
echo ">>> "

if [[ "$template" == "minimal" ]]; then
    template_type=0
elif [[ "$template" == "standard" ]]; then
    template_type=1
elif [[ "$template" == "english" ]]; then
    template_type=2
elif [[ "$template" == "instructed" ]]; then
    template_type=3
else
    echo ">>> Invalid template type. Please use 'standard', 'english', 'japanese', or 'instructed'."
    exit 1
fi

process_tasks() {
    local task_list=("$@")
    local filtered_tasks=()

    if [[ ${tasks[0]} == "all" && ${#tasks[@]} -eq 1 ]]; then
        filtered_tasks=("${task_list[@]}")
    else
        for task in "${tasks[@]}"; do
            if [[ " ${task_list[@]} " =~ " ${task} " ]]; then
                filtered_tasks+=("${task}")
            fi
        done
    fi
    filtered_tasks=$(IFS=,; echo "${filtered_tasks[*]}")
    echo "$filtered_tasks"
}

# PROMPT_TASKS=(
#     "nii_en5_mono_prompt-en" "nii_en5_mono_prompt-ja" 
#     "nii_en5_bi_prompt-en" "nii_en5_bi_prompt-ja" 
#     "nii_en5_tri_prompt-en" "nii_en5_tri_prompt-ja" 
#     "nii_ja5_mono_prompt-en" "nii_ja5_mono_prompt-ja"
#     "nii_ja5_bi_prompt-en" "nii_ja5_bi_prompt-ja"
#     "nii_ja5_tri_prompt-en" "nii_ja5_tri_prompt-ja"
#     "nii_en5_mono_prompt-en.cross" "nii_en5_mono_prompt-ja.cross" 
#     "nii_en5_bi_prompt-en.cross" "nii_en5_bi_prompt-ja.cross" 
#     "nii_en5_tri_prompt-en.cross" "nii_en5_tri_prompt-ja.cross" 
#     "nii_ja5_mono_prompt-en.cross" "nii_ja5_mono_prompt-ja.cross"
#     "nii_ja5_bi_prompt-en.cross" "nii_ja5_bi_prompt-ja.cross"
#     "nii_ja5_tri_prompt-en.cross" "nii_ja5_tri_prompt-ja.cross")

PROMPT_TASKS=(
    "nii_en5_mono_prompt-en" "nii_en5_mono_prompt-ja" 
    "nii_ja5_mono_prompt-en" "nii_ja5_mono_prompt-ja")
# MCQA_TASKS=("medmcqa_jp")
MCQA_TEMPLATES=("mcqa_minimal" "mcqa_with_options_jp" "mcqa_with_options" "4o_mcqa_instructed_jp")
prompt_tasks=$(process_tasks "${PROMPT_TASKS[@]}")
# if [[ -z "$prompt_tasks" ]]; then
#     echo ">>> No matching tasks found for MCQA."
# else
template_name=${MCQA_TEMPLATES[${template_type}]}
if [[ -n $result_prefix ]]; then
    result_csv="${result_root}/$result_prefix-prompt.csv"
    dump_file="${model_path}/result.json"
else
    result_csv="${result_root}/prompt.csv"
    dump_file="${model_path}/result.json"
fi
if [ -s "${result_csv}" ]; then
    echo ">>> Result file ${result_csv} already exists. Skipping evaluation."
else
    echo ">>> Evaluating task: ${prompt_tasks} with template: ${template_name}, output: ${result_csv}"
    python3 -m torch.distributed.run --nproc_per_node=$n_gpus \
            ${BASE_PATH}/evaluate_mcqa.py \
            --model_name_or_path ${model_path} \
            --task ${prompt_tasks} \
            --template_name ${template_name} \
            --batch_size ${batch_size} \
            --num_fewshot ${num_fewshot} \
            --seed ${seed} \
            --model_max_length ${model_max_length} \
            --truncate False \
            --use_knn_demo ${use_knn_demo} \
            --result_csv ${result_csv} \
            --dump_file ${dump_file}
fi




# MCQA_TASKS=("medmcqa_jp" "usmleqa_jp" "medqa_jp" "igakuqa" "mmlu_medical_jp" "jmmlu_medical")
# # MCQA_TASKS=("medmcqa_jp")
# MCQA_TEMPLATES=("mcqa_minimal" "mcqa_with_options_jp" "mcqa_with_options" "4o_mcqa_instructed_jp")
# mcqa_tasks=$(process_tasks "${MCQA_TASKS[@]}")
# if [[ -z "$mcqa_tasks" ]]; then
#     echo ">>> No matching tasks found for MCQA."
# else
#     template_name=${MCQA_TEMPLATES[${template_type}]}
#     result_csv="${result_root}/mcqa.csv"
#     dump_file="${result_root}/mcqa.json"
#     if [ -s "${result_csv}" ]; then
#         echo ">>> Result file ${result_csv} already exists. Skipping evaluation."
#     else
#         echo ">>> Evaluating task: ${mcqa_tasks} with template: ${template_name}, output: ${result_csv}"
#         python3 -m torch.distributed.run --nproc_per_node=$n_gpus \
#                 ${BASE_PATH}/evaluate_mcqa.py \
#                 --model_name_or_path ${model_path} \
#                 --task ${mcqa_tasks} \
#                 --template_name ${template_name} \
#                 --batch_size ${batch_size} \
#                 --num_fewshot ${num_fewshot} \
#                 --seed ${seed} \
#                 --model_max_length ${model_max_length} \
#                 --truncate False \
#                 --use_knn_demo ${use_knn_demo} \
#                 --result_csv ${result_csv}
#     fi
    
# fi
            
# MCQA_PUBMEDQA_TASKS=("pubmedqa" "pubmedqa_jp")
# MCQA_TEMPLATES_PUBMEDQA=("context_based_mcqa_minimal" "context_based_mcqa_jp" "context_based_mcqa" "context_based_mcqa_instructed_jp")

# pubmedqa_tasks=$(process_tasks "${MCQA_PUBMEDQA_TASKS[@]}")
# if [[ -z "$pubmedqa_tasks" ]]; then
#     echo ">>> No matching tasks found for PubMedQA."
# else
#     template_name=${MCQA_TEMPLATES_PUBMEDQA[${template_type}]}
#     result_csv="${result_root}/pubmedqa.csv"
#     dump_file="${result_root}/pubmedqa.json"
#     if [ -s "${result_csv}" ]; then
#         echo ">>> Result file ${result_csv} already exists. Skipping evaluation."
#     else    
#         echo ">>> Evaluating task: ${pubmedqa_tasks} with template: ${template_name}, output: ${result_csv}"
#         python3 -m torch.distributed.run --nproc_per_node=$n_gpus \
#             ${BASE_PATH}/evaluate_mcqa.py \
#                 --model_name_or_path ${model_path} \
#                 --task ${pubmedqa_tasks} \
#                 --template_name ${template_name} \
#                 --batch_size $(($((${batch_size}/2)) > 1 ? $((${batch_size}/2)) : 1)) \
#                 --num_fewshot ${num_fewshot} \
#                 --seed ${seed} \
#                 --model_max_length ${model_max_length} \
#                 --truncate False \
#                 --use_knn_demo ${use_knn_demo} \
#                 --result_csv ${result_csv}
#     fi
# fi


# MT_EN2JA_TASKS=("ejmmt")
# MT_EN2JA_TEMPLATES=("mt_minimal" "english_japanese" "mt_english_centric_e2j" "mt_instructed_e2j")

# mt_en2ja_tasks=$(process_tasks "${MT_EN2JA_TASKS[@]}")
# if [[ -z "$mt_en2ja_tasks" ]]; then
#     echo ">>> No matching tasks found for MT EN2JA."
# else
#     template_name=${MT_EN2JA_TEMPLATES[${template_type}]}
#     result_csv="${result_root}/mt_en2ja.csv"
#     dump_file="${result_root}/mt_en2ja.json"
#     if [ -s "${result_csv}" ]; then
#         echo ">>> Result file ${result_csv} already exists. Skipping evaluation."
#     else
#         echo ">>> Evaluating task: ${mt_en2ja_tasks} with template: ${template_name}, output: ${result_csv}"
#         python3 -m torch.distributed.run --nproc_per_node=$n_gpus \
#             ${BASE_PATH}/evaluate_mt.py \
#                 --model_name_or_path ${model_path} \
#                 --max_new_tokens 256 \
#                 --task ${mt_en2ja_tasks} \
#                 --translation "english=>japanese" \
#                 --template_name ${template_name} \
#                 --batch_size ${batch_size} \
#                 --num_fewshot ${num_fewshot} \
#                 --seed ${seed} \
#                 --result_csv ${result_csv}
#     fi
# fi

# MT_JA2EN_TASKS=("ejmmt")
# MT_JA2EN_TEMPLATES=("mt_minimal" "japanese_english" "mt_english_centric_j2e" "mt_instructed_j2e")

# mt_ja2en_tasks=$(process_tasks "${MT_JA2EN_TASKS[@]}")
# if [[ -z "$mt_ja2en_tasks" ]]; then
#     echo ">>> No matching tasks found for MT JA2EN."
# else
#     template_name=${MT_JA2EN_TEMPLATES[${template_type}]}
#     result_csv="${result_root}/mt_ja2en.csv"
#     dump_file="${result_root}/mt_ja2en.json"
#     if [ -s "${result_csv}" ]; then
#         echo ">>> Result file ${result_csv} already exists. Skipping evaluation."
#     else
#         echo ">>> Evaluating task: ${mt_ja2en_tasks} with template: ${template_name}, output: ${result_csv}"
#         python3 -m torch.distributed.run --nproc_per_node=$n_gpus \
#             ${BASE_PATH}/evaluate_mt.py \
#                 --model_name_or_path ${model_path} \
#                 --max_new_tokens 256 \
#                 --task ${mt_ja2en_tasks} \
#                 --translation "japanese=>english" \
#                 --template_name ${template_name} \
#                 --batch_size ${batch_size} \
#                 --num_fewshot ${num_fewshot} \
#                 --seed ${seed} \
#                 --result_csv ${result_csv}
#     fi  
# fi

# NER_TASKS=("mrner_disease" "mrner_medicine" "nrner" "bc2gm_jp" "bc5chem_jp" "bc5disease_jp" "jnlpba_jp" "ncbi_disease_jp")
# # NER_TASKS=("mrner_disease")
# NER_TEMPLATES=("minimal" "standard" "english-centric" "instructed")

# ner_tasks=$(process_tasks "${NER_TASKS[@]}")
# if [[ -z "$ner_tasks" ]]; then
#     echo ">>> No matching tasks found for NER."
# else
#     template_name=${NER_TEMPLATES[${template_type}]}
#     result_csv="${result_root}/ner.csv"
#     dump_file="${result_root}/ner.json"
#     if [ -s "${result_csv}" ]; then
#         echo ">>> Result file ${result_csv} already exists. Skipping evaluation."
#     else
#         echo ">>> Evaluating task: ${ner_tasks} with template: ${template_name}, output: ${result_csv}"
#         python3 -m torch.distributed.run --nproc_per_node=$n_gpus \
#             ${BASE_PATH}/evaluate_ner.py \
#                 --model_name_or_path ${model_path} \
#                 --max_new_tokens 128 \
#                 --task ${ner_tasks} \
#                 --template_name ${template_name} \
#                 --batch_size ${batch_size} \
#                 --num_fewshot ${num_fewshot} \
#                 --seed ${seed} \
#                 --truncate False \
#                 --use_knn_demo ${use_knn_demo} \
#                 --result_csv ${result_csv}
#     fi
# fi

# DC_TASKS=("crade" "rrtnm" "smdis")
# # DC_TASKS=("crade")
# DC_TEMPLATES=("context_based_mcqa_minimal" "dc_with_options_jp" "dc_with_options" "dc_instructed_jp")

# dc_tasks=$(process_tasks "${DC_TASKS[@]}")
# if [[ -z "$dc_tasks" ]]; then
#     echo ">>> No matching tasks found for DC."
# else
#     template_name=${DC_TEMPLATES[${template_type}]}
#     result_csv="${result_root}/dc.csv"
#     dump_file="${result_root}/dc.json"
#     if [ -s "${result_csv}" ]; then
#         echo ">>> Result file ${result_csv} already exists. Skipping evaluation."
#     else
#         echo ">>> Evaluating task: ${dc_tasks} with template: ${template_name}, output: ${result_csv}"
#         python3 -m torch.distributed.run --nproc_per_node=$n_gpus \
#             ${BASE_PATH}/evaluate_mcqa.py \
#                 --model_name_or_path ${model_path} \
#                 --task ${dc_tasks} \
#                 --template_name ${template_name} \
#                 --batch_size $(($((${batch_size}/2)) > 1 ? $((${batch_size}/2)) : 1)) \
#                 --num_fewshot ${num_fewshot} \
#                 --seed ${seed} \
#                 --model_max_length ${model_max_length} \
#                 --truncate True \
#                 --use_knn_demo ${use_knn_demo} \
#                 --result_csv ${result_csv}
#     fi
# fi


# STS_TASKS=("jcsts")
# STS_TEMPLATES=("sts_minimal" "sts_as_nli_jp" "sts_as_nli" "sts_instructed_jp")

# sts_tasks=$(process_tasks "${STS_TASKS[@]}")
# if [[ -z "$sts_tasks" ]]; then
#     echo ">>> No matching tasks found for STS."
# else
#     template_name=${STS_TEMPLATES[${template_type}]}
#     result_csv="${result_root}/sts.csv"
#     dump_file="${result_root}/sts.json"
#     if [ -s "${result_csv}" ]; then
#         echo ">>> Result file ${result_csv} already exists. Skipping evaluation."
#     else
#         echo ">>> Evaluating task: ${sts_tasks} with template: ${template_name}, output: ${result_csv}"
#         python3 -m torch.distributed.run --nproc_per_node=$n_gpus \
#             ${BASE_PATH}/evaluate_sts.py \
#                 --model_name_or_path ${model_path} \
#                 --task ${sts_tasks} \
#                 --template_name ${template_name} \
#                 --batch_size ${batch_size} \
#                 --num_fewshot ${num_fewshot} \
#                 --nli_labels "0,1,2,3,4,5" \
#                 --seed ${seed} \
#                 --truncate False \
#                 --use_knn_demo ${use_knn_demo} \
#                 --result_csv ${result_csv}
#     fi
# fi

# MCQA_TASKS=("medmcqa" "usmleqa" "medqa" "mmlu_medical")
# # MCQA_TASKS=("mmlu_medical")
# MCQA_TEMPLATES=("mcqa_minimal" "mcqa_with_options_jp" "mcqa_with_options" "4o_mcqa_instructed_jp")
# mcqa_tasks=$(process_tasks "${MCQA_TASKS[@]}")
# if [[ -z "$mcqa_tasks" ]]; then
#     echo ">>> No matching tasks found for MCQA."
# else
#     template_name=${MCQA_TEMPLATES[${template_type}]}
#     result_csv="${result_root}/mcqa-en.csv"
#     dump_file="${result_root}/mcqa-en.json"
#     if [ -s "${result_csv}" ]; then
#         echo ">>> Result file ${result_csv} already exists. Skipping evaluation."
#     else
#         echo ">>> Evaluating task: ${mcqa_tasks} with template: ${template_name}, output: ${result_csv}"
#         python3 -m torch.distributed.run --nproc_per_node=$n_gpus \
#                 ${BASE_PATH}/evaluate_mcqa.py \
#                 --model_name_or_path ${model_path} \
#                 --task ${mcqa_tasks} \
#                 --template_name ${template_name} \
#                 --batch_size ${batch_size} \
#                 --num_fewshot ${num_fewshot} \
#                 --seed ${seed} \
#                 --model_max_length ${model_max_length} \
#                 --truncate False \
#                 --use_knn_demo ${use_knn_demo} \
#                 --result_csv ${result_csv}
#     fi
    
# fi