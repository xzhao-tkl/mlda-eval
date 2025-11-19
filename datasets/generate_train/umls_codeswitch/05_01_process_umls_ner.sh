#!/bin/bash
#SBATCH --job-name=0068_CPU
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=0
#SBATCH --gpus-per-task=0
#SBATCH --cpus-per-task=16
#SBATCH --output=/home/xzhao/workspace/roman-pretrain/datasets/logs/collect_entity-%x-%j.out
#SBATCH --error=/home/xzhao/workspace/roman-pretrain/datasets/logs/collect_entity-%x-%j.err

### Filter factual sentences by NER tool

START=$1
REQUEST_NUM=$2
TOOL=$3

python3 05_01_collect_entity.py --ner_tool $TOOL --start_indice $START --request_num $REQUEST_NUM


