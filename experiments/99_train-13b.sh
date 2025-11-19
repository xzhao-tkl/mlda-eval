#!/bin/bash

set -eu -o pipefail

if [ ! -e "$WORK_DIR" ]; then
  echo "Error: Folder '$WORK_DIR' does not exist." >&2
  exit 1
fi
CACHE_DIR=${WORK_DIR}/cache

## Initialize environment
source ${ENV_DIR}/scripts/environment.sh
source ${ENV_DIR}/scripts/mpi_variables.sh
source ${ENV_DIR}/venv/bin/activate

export LOGLEVEL=INFO
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export PYTHONFAULTHANDLER=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_LAUNCH_BLOCKING=0
export CUDNN_LOGDEST_DBG=stderr
export CUDNN_LOGERR_DBG=1
export NVTE_FUSED_ATTN=0


if [ ! -e "$CONFIG_PATH" ]; then
  echo "Error: Config file '$CONFIG_PATH' does not exist." >&2
  exit 1
fi

# model config
HIDDEN_SIZE=$(jq -r '.train.hidden_size' $CONFIG_PATH)
FFN_HIDDEN_SIZE=$(jq -r '.train.ffn_hidden_size' $CONFIG_PATH)
NUM_LAYERS=$(jq -r '.train.num_layers' $CONFIG_PATH)
NUM_HEADS=$(jq -r '.train.num_heads' $CONFIG_PATH)
SEQ_LENGTH=$(jq -r '.train.seq_length' $CONFIG_PATH)
CHECKPOINT_PATH=$(jq -r '.train.initial_checkpoint_path' $CONFIG_PATH)

# distributed settings
TENSOR_PARALLEL_SIZE=$(jq -r '.train.tensor_parallel_size' $CONFIG_PATH)
PIPELINE_PARALLEL_SIZE=$(jq -r '.train.pipeline_parallel_size' $CONFIG_PATH)
CONTEXT_PARALLEL_SIZE=$(jq -r '.train.context_parallel_size' $CONFIG_PATH)

# training config
MICRO_BATCH_SIZE=$(jq -r '.train.micro_batch_size' $CONFIG_PATH)
GLOBAL_BATCH_SIZE=$(jq -r '.train.global_batch_size' $CONFIG_PATH)
INSTRUCTION_TUNE=$(jq -r '.train.tune' $CONFIG_PATH)

LR_WARMUP_INIT=$(jq -r '.train.lr_warmup_init' $CONFIG_PATH)
LR=$(jq -r '.train.lr' $CONFIG_PATH)
MIN_LR=$(jq -r '.train.min_lr' $CONFIG_PATH)
LR_WARMUP_STEPS=$(jq -r '.train.lr_warmup_steps' $CONFIG_PATH)
WEIGHT_DECAY=$(jq -r '.train.weight_decay' $CONFIG_PATH)
GRAD_CLIP=$(jq -r '.train.grad_clip' $CONFIG_PATH)

# data config
DATA_CONFIG="$WORK_DIR/data_config.sh"
DATA_SUMMARY="$WORK_DIR/data_config.txt"
if [ "$OMPI_COMM_WORLD_RANK" -eq 0 ]; then
  if [ ! -e "${TOKEN_DIR}/data_config.yaml" ]; then
    echo "Error: Folder '${TOKEN_DIR}/data_config.yaml' does not exist." >&2
    exit 1
  fi
  python3 "${SCRIPT_ROOT}/99_megatron_data_formatter.py" "${TOKEN_DIR}/data_config.yaml" >"$DATA_CONFIG" 2>"$DATA_SUMMARY"
  source "$DATA_CONFIG"
else
  # Wait file createtion　and check variable $TOTAL_TOKEN_SIZE until $TIMEOUT
  TIMEOUT=10
  INTERVAL=1
  END_TIME=$((SECONDS + TIMEOUT))
  TOTAL_TOKEN_SIZE=""

  while [ $SECONDS -lt $END_TIME ]; do
    sleep $INTERVAL
    if [ -f "$DATA_CONFIG" ]; then
      # load $TRAIN_DATA_PATH and $TOTAL_TOKEN_SIZE
      source "$DATA_CONFIG"
      if [ -n "$TOTAL_TOKEN_SIZE" ]; then
        break
      fi
    fi
  done
  # timeout
  if [ -z "$TOTAL_TOKEN_SIZE" ]; then
    echo >&2 "Error: Timeout. TOTAL_TOKEN_SIZE not found within $TIMEOUT seconds."
    exit 1
  fi
fi

# total number of iterations
STEP_DETAIL=$((TOTAL_TOKEN_SIZE / SEQ_LENGTH / GLOBAL_BATCH_SIZE))
LR_DECAY_ITERS=$(awk "BEGIN {print int($STEP_DETAIL + 0.5)}")
LR_DECAY_STYLE=cosine
LR_WARMUP_STEPS=0
TRAIN_STEPS=$LR_DECAY_ITERS

# model config
TOKENIZER_MODEL=${ENV_DIR}/src/llm-jp-tokenizer/models/ver3.0/llm-jp-tokenizer-100k.ver3.0b1.model

CHECKPOINT_SAVE_DIR=${WORK_DIR}/checkpoints

CHECKPOINT_ARGS=""
if [[ -f "${CHECKPOINT_SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
  CHECKPOINT_LOAD_DIR=${CHECKPOINT_SAVE_DIR}
else
  CHECKPOINT_LOAD_DIR=$(jq -r '.train.initial_checkpoint_path' $CONFIG_PATH)
  CHECKPOINT_ARGS="--finetune"
fi

mkdir -p "${CHECKPOINT_SAVE_DIR}"

# job name
WANDB_ENTITY="llm-jp"
WANDB_PROJECT="continue-pretrain-data-scidoc"
WANDB_NAME=$EXP_NAME

echo TENSOR_PARALLEL_SIZE: ${TENSOR_PARALLEL_SIZE}
echo ENV_DIR: ${ENV_DIR}
echo PIPELINE_PARALLEL_SIZE: ${PIPELINE_PARALLEL_SIZE}
echo CONTEXT_PARALLEL_SIZE: ${CONTEXT_PARALLEL_SIZE}
echo NUM_LAYERS: ${NUM_LAYERS}
echo FFN_HIDDEN_SIZE: ${FFN_HIDDEN_SIZE}
echo NUM_HEADS: ${NUM_HEADS}
echo SEQ_LENGTH: ${SEQ_LENGTH}
echo MICRO_BATCH_SIZE: ${MICRO_BATCH_SIZE}
echo GLOBAL_BATCH_SIZE: ${GLOBAL_BATCH_SIZE}
echo TRAIN_STEPS: ${TRAIN_STEPS}
echo INSTRUCTION_TUNE: ${INSTRUCTION_TUNE}
echo TOKENIZER_MODEL: ${TOKENIZER_MODEL}
echo CHECKPOINT_LOAD_DIR: ${CHECKPOINT_LOAD_DIR}
echo CHECKPOINT_SAVE_DIR: ${CHECKPOINT_SAVE_DIR}
echo CHECKPOINT_ARGS: ${CHECKPOINT_ARGS}
echo TRAIN_DATA_PATH: ${TRAIN_DATA_PATH}
echo CACHE_DIR: ${CACHE_DIR}
echo LR_WARMUP_INIT: ${LR_WARMUP_INIT}
echo LR: ${LR}
echo MIN_LR: ${MIN_LR}
echo LR_WARMUP_INIT: ${LR_WARMUP_INIT}
echo WANDB_ENTITY: ${WANDB_ENTITY}
echo WANDB_PROJECT: ${WANDB_PROJECT}
echo WANDB_NAME: ${WANDB_NAME}

# set flag-only argument for instruction-tune (store_true)
INSTRUCTION_TUNE_ARG=""
if [ "${INSTRUCTION_TUNE}" = "true" ] || [ "${INSTRUCTION_TUNE}" = "True" ] || [ "${INSTRUCTION_TUNE}" = "1" ]; then
  INSTRUCTION_TUNE_ARG="--instruction-tune"
fi

python "${ENV_DIR}/src/Megatron-LM/pretrain_gpt.py" \
  --tensor-model-parallel-size ${TENSOR_PARALLEL_SIZE} \
  --pipeline-model-parallel-size ${PIPELINE_PARALLEL_SIZE} \
  --context-parallel-size ${CONTEXT_PARALLEL_SIZE} \
  --sequence-parallel \
  --use-distributed-optimizer \
  --num-layers ${NUM_LAYERS} \
  --hidden-size ${HIDDEN_SIZE} \
  --ffn-hidden-size ${FFN_HIDDEN_SIZE} \
  --num-attention-heads ${NUM_HEADS} \
  --seq-length ${SEQ_LENGTH} \
  --max-position-embeddings ${SEQ_LENGTH} \
  --micro-batch-size ${MICRO_BATCH_SIZE} \
  --global-batch-size ${GLOBAL_BATCH_SIZE} \
  --train-iters ${TRAIN_STEPS} \
  $INSTRUCTION_TUNE_ARG \
  --tokenizer-type Llama2Tokenizer \
  --tokenizer-model ${TOKENIZER_MODEL} \
  --load ${CHECKPOINT_LOAD_DIR} \
  $CHECKPOINT_ARGS \
  --save ${CHECKPOINT_SAVE_DIR} \
  --data-path ${TRAIN_DATA_PATH} \
  --split 1,0,0 \
  --data-cache-path ${CACHE_DIR} \
  --distributed-backend nccl \
  --init-method-std 0.02 \
  --lr-warmup-init ${LR_WARMUP_INIT} \
  --lr ${LR} \
  --min-lr ${MIN_LR} \
  --lr-decay-style ${LR_DECAY_STYLE} \
  --lr-decay-iters ${LR_DECAY_ITERS} \
  --weight-decay ${WEIGHT_DECAY} \
  --clip-grad ${GRAD_CLIP} \
  --lr-warmup-iters ${LR_WARMUP_STEPS} \
  --optimizer adam \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --adam-eps 1e-8 \
  --log-interval 1 \
  --eval-interval ${TRAIN_STEPS} \
  --eval-iters 0 \
  --bf16 \
  --untie-embeddings-and-output-weights \
  --position-embedding-type rope \
  --disable-bias-linear \
  --use-mcore-models \
  --normalization RMSNorm \
  --norm-epsilon 1e-5 \
  --no-masked-softmax-fusion \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --swiglu \
  --use-flash-attn \
  --recompute-activations \
  --recompute-granularity "selective" \
  --attention-softmax-in-fp32 \
  --transformer-impl "transformer_engine" \
  --use-mpi \
  --use-z-loss \
  --log-throughput \
  --wandb-entity ${WANDB_ENTITY} \
  --wandb-project ${WANDB_PROJECT} \
  --wandb-name ${WANDB_NAME}
