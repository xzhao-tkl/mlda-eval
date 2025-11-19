#!/bin/bash
#SBATCH --job-name=0068_Datapreprocess
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --output=/home/xzhao/workspace/roman-pretrain/datasets/logs/data-processing-%x-%j.out
#SBATCH --error=/home/xzhao/workspace/roman-pretrain/datasets/logs/data-processing-%x-%j.err

# python3 generate_regex.py --lang zh
python3 generate_regex.py --lang ja
# python3 generate_regex.py --lang en