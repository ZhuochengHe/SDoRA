#!/bin/bash

# Usage: ./llama3_8B_train_accelerate.sh [ADAPTER] [R] [ALPHA] [GPU_ID]
# Example: ./llama3_8B_train_accelerate.sh dora 16 32 0

ADAPTER=$1
R=${2:-16}
ALPHA=${3:-32}
GPU=$4 # Specify GPU index to use (0, 1, 2, ...)

# Ensure all required parameters are provided
if [ -z "$ADAPTER" ] || [ -z "$GPU" ]; then
    echo "Usage: $0 [ADAPTER] [R] [ALPHA] [GPU_ID]"
    echo "Example: $0 dora 16 32 0"
    exit 1
fi

OUTPUT_DIR="outputs/${ADAPTER}_r${R}"

# Set learning rate based on adapter type
if [ "$ADAPTER" = "lora" ]; then
    LR=3e-4
else
    LR=1e-4
fi

echo "Training ${ADAPTER^^} (rank=${R}, alpha=${ALPHA}, lr=${LR})"
echo "GPU: ${GPU}"
echo "Output: ${OUTPUT_DIR}"
echo "Start time: $(date)"

# 1. Ensure 'accelerate config' has been run for single-GPU setup
# 2. Use CUDA_VISIBLE_DEVICES to restrict accelerate launch to specified GPU
CUDA_VISIBLE_DEVICES=$GPU accelerate launch finetune_accelerate.py \
    --output_dir $OUTPUT_DIR \
    --adapter_name $ADAPTER \
    --lora_r $R \
    --lora_alpha $ALPHA \
    --num_epochs 3 \
    --learning_rate $LR \
    --sparse_lambda 0.3 \
    --micro_batch_size 8 \
    --batch_size 32

echo "Script completed."
echo "End time: $(date)"