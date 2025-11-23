#!/bin/bash

ADAPTER=$1
GPU=$2
R=${3:-8}
ALPHA=${4:-16}

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
    --num_epochs 15 \
    --learning_rate 1e-4 \
    --sparse_lambda 0.3