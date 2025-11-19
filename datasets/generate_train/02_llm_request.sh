#!/bin/bash

NODE=$1
if [ -z "$1" ]; then
    echo "Error: \$1 is not provided!"
    exit 1
fi

START=$2
if [ -z "$2" ]; then
    echo "Error: \$2 is not provided!"
    exit 1
fi

NUM=$3
if [ -z "$3" ]; then
    NUM=100000
fi

python3 llm_request.py \
    --lang en_pair --instruct_type qa-gen \
    --endpoint http://gpu-$NODE:8080/v1/chat/completions \
    --start_indice $START --request_num $NUM

python3 llm_request.py \
    --lang en_pair --instruct_type qa-gen \
    --endpoint http://gpu-$NODE:8080/v1/chat/completions \
    --start_indice $START --request_num $NUM

python3 llm_request.py \
    --lang en_pair --instruct_type qa-gen \
    --endpoint http://gpu-$NODE:8080/v1/chat/completions \
    --start_indice $START --request_num $NUM
