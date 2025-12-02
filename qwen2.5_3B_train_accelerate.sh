#!/bin/bash

# Usage: ./qwen2.5_3B_train_accelerate.sh [ADAPTER] [R] [ALPHA] [GPU_ID] [SPARSE_LAMBDA]
# Example: ./qwen2.5_3B_train_accelerate.sh dora 16 32 0 0.1

ADAPTER=$1
R=${2:-16}
ALPHA=${3:-32}
GPU=$4 # Specify GPU index to use (0, 1, 2, ...)
SPARSE_LAMBDA=${5:-0.1} # Sparsity parameter (default: 0.1)

# Ensure all required parameters are provided
if [ -z "$ADAPTER" ] || [ -z "$GPU" ]; then
    echo "Usage: $0 [ADAPTER] [R] [ALPHA] [GPU_ID] [SPARSE_LAMBDA]"
    echo "Example: $0 dora 16 32 0 0.3"
    exit 1
fi


# Set learning rate based on adapter type
if [ "$ADAPTER" = "lora" ]; then
    LR=3e-4
elif [[ "$ADAPTER" = "sdora" || "$ADAPTER" = "sora" ]]; then
    LR=6e-4
    SPARSE_LAMBDA=0.1
else
    LR=2e-4
fi

# Set output directory with sparsity suffix for SoRA/SDoRA experiments
if [ "$ADAPTER" = "sora" ] || [ "$ADAPTER" = "sdora" ]; then
    OUTPUT_DIR="outputs/Qwen2.5-3B_${ADAPTER}_r${R}_lambda${SPARSE_LAMBDA}_lr${LR}_alpha${ALPHA}"
else
    OUTPUT_DIR="outputs/Qwen2.5-3B_${ADAPTER}_r${R}_lr${LR}_alpha${ALPHA}"
fi

echo "========================================"
echo "Training ${ADAPTER^^} on Qwen2.5-3B"
echo "========================================"
echo "Rank: ${R}, Alpha: ${ALPHA}"
echo "Learning Rate: ${LR}"
echo "Sparsity Lambda: ${SPARSE_LAMBDA}"
echo "GPU: ${GPU}"
echo "Output: ${OUTPUT_DIR}"
echo "Start time: $(date)"
echo "========================================"

# 1. Ensure 'accelerate config' has been run for single-GPU setup
# 2. Use CUDA_VISIBLE_DEVICES to restrict accelerate launch to specified GPU
CUDA_VISIBLE_DEVICES=$GPU accelerate launch finetune_accelerate.py \
    --output_dir $OUTPUT_DIR \
    --adapter_name $ADAPTER \
    --lora_r $R \
    --lora_alpha $ALPHA \
    --num_epochs 3 \
    --learning_rate $LR \
    --sparse_lambda $SPARSE_LAMBDA \
    --micro_batch_size 32 \
    --batch_size 32

echo "========================================"
echo "Training completed!"
echo "End time: $(date)"
echo "Output saved to: ${OUTPUT_DIR}"
echo "========================================"