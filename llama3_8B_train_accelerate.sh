#!/bin/bash

# 使用说明: ./llama3_8B_train.sh [ADAPTER] [R] [ALPHA] [GPU_ID]
# 示例: ./llama3_8B_train.sh dora 32 64 0

ADAPTER=$1
R=${2:-8}
ALPHA=${3:-16}
GPU=$4 # 指定要使用的 GPU 索引 (0, 1, 2, ...)

# 确保所有必需参数都已提供
if [ -z "$ADAPTER" ] || [ -z "$GPU" ]; then
    echo "Usage: $0 [ADAPTER] [R] [ALPHA] [GPU_ID]"
    echo "Example: $0 dora 32 64 0"
    exit 1
fi

OUTPUT_DIR="outputs/${ADAPTER}_r${R}"

echo "Training ${ADAPTER^^} (rank=${R}, alpha=${ALPHA})"
echo "GPU: ${GPU}"
echo "Output: ${OUTPUT_DIR}"

# 1. 确保已运行 accelerate config 配置单卡环境
# 2. 通过 CUDA_VISIBLE_DEVICES 限制 accelerate launch 只能看到指定的 GPU
CUDA_VISIBLE_DEVICES=$GPU accelerate launch finetune_accelerate.py \
    --output_dir $OUTPUT_DIR \
    --adapter_name $ADAPTER \
    --lora_r $R \
    --lora_alpha $ALPHA \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --sparse_lambda 0.3 \
    # 可以根据需要调整 batch_size 和 micro_batch_size
    # --micro_batch_size 8 \ 
    # --batch_size 32

echo "Script completed."