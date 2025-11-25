#!/bin/bash

#usage: ./qwen2.5_3B_finetune.sh lora/dora/sora/sdora R ALPHA GPU SPARSE_LAMBDA

ADAPTER=$1
R=${2:-16}
ALPHA=${3:-32}
GPU=$4 # Specify GPU index to use (0, 1, 2, ...)
SPARSE_LAMBDA=${5:-0.3} # Sparsity parameter (default: 0.3)
nohup ./qwen2.5_3B_train_accelerate.sh $ADAPTER $R $ALPHA $GPU $SPARSE_LAMBDA > "Qwen2.5-3B_${ADAPTER}.log" 2>&1 &
