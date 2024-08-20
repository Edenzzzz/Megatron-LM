#!/bin/bash

# Runs the "175B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH=$1 #<Specify path>
VOCAB_FILE=$2 #<Specify path to file>/gpt2-vocab.json
MERGE_FILE=$3 #<Specify path to file>/gpt2-merges.txt
DATA_PATH=$4 #<Specify path and file prefix>_text_document

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)
# 7b model (https://github.com/NVIDIA/NeMo-Framework-Launcher/blob/6daee02ad897b6f8fb468a518edcdb9bda9c36c7/launcher_scripts/conf/training/gpt3/7b_improved.yaml)
GPT_MODEL_ARGS=(
    --num-layers 32 
    --hidden-size 4096 
    --num-attention-heads 32 
    --seq-length 8192 
    --max-position-embeddings 32768 
    --ffn-hidden-size 10880
)

TRAINING_ARGS=(
    --micro-batch-size 1
    # --global-batch-size 1536 
    # --rampup-batch-size 16 16 5859375 
    --train-iters 500000 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --fp16
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 430000 
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 2
	--context-parallel-size 2
    --use-flash-attn
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 40
    --save-interval 10000 
    --eval-interval 1000 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 10
)
# nsys profile -o profile --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop \
NVTE_FUSED_ATTN=0 torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    --use-flash-attn \
    --profile