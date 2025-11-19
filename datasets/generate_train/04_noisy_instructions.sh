#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

for index in 0 1 2 3 4; do
    python3 04_generate_instructions.py \
        --lang en_jstage --medical_native --noisy_instruction --noisy_index ${index} \
        --noisy_data_path /data/xzhao/dataset/roman-pretrain/datasets/medical/en_jstage/wordnet/wordnet.multilingual.ja.32.jsonl

    python3 04_generate_instructions.py \
        --lang en_jstage --medical_native --noisy_instruction --noisy_index ${index} \
        --noisy_data_path /data/xzhao/dataset/roman-pretrain/datasets/medical/en_jstage/wordnet/wordnet.multilingual.ja.64.jsonl

    python3 04_generate_instructions.py \
        --lang en_jstage --medical_native --noisy_instruction --noisy_index ${index} \
        --noisy_data_path /data/xzhao/dataset/roman-pretrain/datasets/medical/en_jstage/wordnet/wordnet.monolingual.32.jsonl

    python3 04_generate_instructions.py \
        --lang en_jstage --medical_native --noisy_instruction --noisy_index ${index} \
        --noisy_data_path /data/xzhao/dataset/roman-pretrain/datasets/medical/en_jstage/wordnet/wordnet.monolingual.64.jsonl
done