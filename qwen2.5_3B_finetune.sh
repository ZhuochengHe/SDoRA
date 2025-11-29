#!/bin/bash

# Usage: ./qwen2.5_3B_finetune.sh [ADAPTER] [R] [ALPHA] [GPU_ID] [SPARSE_LAMBDA] [GATE_LR_MULT] [PROX_ONLY]
# Example: ./qwen2.5_3B_finetune.sh sora 16 32 0 0.3 10 0
#
# Positional arguments:
#   ADAPTER       : adapter to use (lora|sora|dora|sdora)
#   R             : LoRA/SoRA rank
#   ALPHA         : LoRA/SoRA alpha
#   GPU_ID        : GPU index to use (e.g. 0)
#   SPARSE_LAMBDA : sparsity lambda (default: 0.3)
#   GATE_LR_MULT  : gate LR multiplier (default: 10)
#   PROX_ONLY     : 1 or 0 (default: 0, deprecated)

ADAPTER=$1
R=${2:-16}
ALPHA=${3:-32}
GPU=$4 # Specify GPU index to use (0, 1, 2, ...)
SPARSE_LAMBDA=${5:-0.3} # Sparsity parameter (default: 0.3)
GATE_LR_MULT=${6:-10}
PROX_ONLY=${7:-0}

nohup ./qwen2.5_3B_train_accelerate.sh $ADAPTER $R $ALPHA $GPU $SPARSE_LAMBDA $GATE_LR_MULT $PROX_ONLY > "Qwen2.5-3B_${ADAPTER}.log" 2>&1 &
