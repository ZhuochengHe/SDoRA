#!/bin/bash

ADAPTER=$1
R=${2:-8}
ALPHA=${3:-16}
GPU=$4

OUTPUT_DIR="outputs/${ADAPTER}_r${R}"

echo "Training ${ADAPTER^^} (rank=${R}, alpha=${ALPHA})"
echo "GPU: ${GPU}"
echo "Output: ${OUTPUT_DIR}"

# 训练
CUDA_VISIBLE_DEVICES=$GPU python finetune.py \
    --output_dir $OUTPUT_DIR \
    --adapter_name $ADAPTER \
    --lora_r $R \
    --lora_alpha $ALPHA \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --sparse_lambda 0.3