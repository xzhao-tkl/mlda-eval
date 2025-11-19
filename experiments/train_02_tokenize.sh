#!/bin/bash
#SBATCH --job-name=0068_TOKENIZE
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=0
#SBATCH --cpus-per-task=64

set -eu -o pipefail

if [ -z "$1" ]; then
    echo "Error: \$1 is not provided!"
    exit 1
fi

config_path=/data/xzhao/experiments/roman-pretrain/exps/$1/config.json

tokenize_scripts_dir=/data/xzhao/experiments/roman-pretrain/scripts
env_dir=$(jq -r '.env' $config_path)
exp_dir=$(jq -r '.["exp-dir"]' $config_path)/$(jq -r '.name' $config_path)

input_dir=$exp_dir/dataset
output_dir=$exp_dir/tokenized
mkdir -p $output_dir
source $env_dir/venv/bin/activate
cd $tokenize_scripts_dir

if [ ! -d ./tokenize ]; then
  mkdir tokenize
  cd tokenize
  git clone https://github.com/llm-jp/Megatron-LM.git -b llmjp0-mdx
  cd Megatron-LM
else
  cd tokenize/Megatron-LM
fi

# Tokenize settings
tokenizer_type=$(jq -r '.dataset.tokenizer.megatrontype' $config_path)
tokenizer_path=$(jq -r '.dataset.tokenizer.path' $config_path)
tokenizer_name=$(jq -r '.dataset.tokenizer.type' $config_path)
repeat=$(jq -r '.dataset.tokenizer.repeat' $config_path)
workers=256

# Tokenize
echo $tokenizer_path

exp_token_info="$output_dir/token_info.csv"
rm -f $exp_token_info

for file in $input_dir/*.jsonl; do
    file_name=$(readlink "$file") # /data/xzhao/experiments/roman-pretrain/datasets/kg-datasets/ja-0.5/knowledge.jsonl
    base_name=$(basename "$file_name") # knowledge.jsonl
    output_path=$(dirname "$file_name")/tokenized/$tokenizer_name # /data/xzhao/experiments/roman-pretrain/datasets/kg-datasets/ja-0.5/tokenized/llm-jp
    mkdir -p $output_path
    output_token_info="$output_path/token_info.csv"
    echo $output_token_info
    if [ -e "$output_token_info" ]; then
        echo "Output path $output_token_info already exists. Skipping."
        cat $output_token_info >> $exp_token_info
        continue
    fi

    echo "Tokenizing $file_name to $output_token_info"

    python tools/preprocess_data.py \
        --input "$file_name" \
        --output-result-total-token-info $output_token_info \
        --output-prefix "$output_path/$base_name" \
        --tokenizer-model $tokenizer_path \
        --tokenizer-type $tokenizer_type \
        --workers $workers \
        --append-eod
    cat $output_token_info >> $exp_token_info
    echo "See https://github.com/llm-jp/domain_adaptation/issues/1" > $output_path/README.md
done

data_config_template=/home/xzhao/workspace/roman-pretrain/experiments/configs/data_config_template.yaml
data_config="$output_dir/data_config.yaml"
cp -f $data_config_template $data_config

sed -i "s|<token_csv_path>|$exp_token_info|g" "$data_config"
sed -i "s|<repeat>|$repeat|g" $data_config

echo "Tokenization done"
