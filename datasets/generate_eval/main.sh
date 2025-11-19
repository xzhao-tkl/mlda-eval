#!/bin/bash
#SBATCH --job-name=0068_Datapreprocess
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --output=/home/xzhao/workspace/roman-pretrain/datasets/generate_eval/logs/data-processing-%x-%j.out
#SBATCH --error=/home/xzhao/workspace/roman-pretrain/datasets/generate_eval/logs/data-processing-%x-%j.err

### Filter factual sentences by NER tool
python3 ner.py --ner_tool bionlp

## Filter factual sentences and extract triples by local LLMs
python3 filter_factual_sentence.py --model Llama3.3-70B --node gpu-node1
python3 filter_factual_sentence.py --model Qwen3-32B --node gpu-node3
python3 filter_factual_sentence.py --model DeepSeek-R1-70B --node gpu-node7

