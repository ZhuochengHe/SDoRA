#!/bin/bash

# Usage: ./qwen2.5_3B_train_accelerate.sh [ADAPTER] [R] [ALPHA] [GPU_ID] [SPARSE_LAMBDA] [GATE_LR_MULT] [PROX_ONLY]
# Example: ./qwen2.5_3B_train_accelerate.sh dora 16 32 0 0.3 10 0
#
# Positional arguments:
#   ADAPTER       : adapter to use (lora|sora|dora|sdora)
#   R             : LoRA/SoRA rank
#   ALPHA         : LoRA/SoRA alpha
#   GPU_ID        : GPU index to use (e.g. 0)
#   SPARSE_LAMBDA : sparsity lambda (passed as --sparse_lambda)
#   GATE_LR_MULT  : gate LR multiplier (integer). The script passes this as
#                   --gate_lr_multiplier to the training script. If the
#                   training script receives an absolute --gate_lr instead,
#                   that overrides this multiplier.
#   PROX_ONLY     : 1 or 0. If 1, the wrapper adds the flag --use_prox_only
#                   to the training command which tells training to rely on
#                   the optimizer's proximal step (and NOT add explicit L1
#                   loss on gates). If 0, explicit L1 loss may be added.

ADAPTER=$1
R=${2:-16}
ALPHA=${3:-32}
GPU=$4 # Specify GPU index to use (0, 1, 2 ...)
SPARSE_LAMBDA=${5:-0.3} # Sparsity parameter (default: 0.3)
GATE_LR_MULT=${6:-10}
PROX_ONLY=${7:-0}
# Optional extra params: absolute gate LR, debug print frequency, and max steps
# Usage extension: ./qwen2.5_3B_train_accelerate.sh ... [GATE_LR] [DEBUG_PRINT_STEPS] [MAX_STEPS]
GATE_LR=${8:-}
DEBUG_PRINT_STEPS=${9:-0}
MAX_STEPS=${10:-}

# Ensure all required parameters are provided
if [ -z "$ADAPTER" ] || [ -z "$GPU" ]; then
    echo "usage: $0 [ADAPTER] [R] [ALPHA] [GPU_ID] [SPARSE_LAMBDA] [GATE_LR_MULT] [PROX_ONLY]"
    echo "example: $0 dora 16 32 0 0.3 10 1"
    exit 1
fi

# Set output directory with sparsity suffix for SoRA/SDoRA experiments
if [ "$ADAPTER" = "sora" ] || [ "$ADAPTER" = "sdora" ]; then
    OUTPUT_DIR="outputs/Qwen2.5-3B_${ADAPTER}_r${R}_lambda${SPARSE_LAMBDA}"
else
    OUTPUT_DIR="outputs/Qwen2.5-3B_${ADAPTER}_r${R}"
fi

# Set learning rate based on adapter type
if [ "$ADAPTER" = "lora" ]; then
    LR=3e-4
else
    LR=1e-4
fi

echo "========================================"
echo "Training ${ADAPTER^^} on Qwen2.5-3B"
echo "========================================"
echo "Rank: ${R}, Alpha: ${ALPHA}"
echo "Learning Rate: ${LR}"
echo "Sparsity Lambda: ${SPARSE_LAMBDA}"
echo "Gate LR Multiplier: ${GATE_LR_MULT}"
echo "GPU: ${GPU}"
echo "Output: ${OUTPUT_DIR}"
echo "Gate LR (absolute, if set): ${GATE_LR}"
echo "Debug print steps: ${DEBUG_PRINT_STEPS}"
echo "Max steps (if set): ${MAX_STEPS}"
echo "Start time: $(date)"
echo "========================================"

# 1. Ensure 'accelerate config' has been run for single-GPU setup
# 2. Use CUDA_VISIBLE_DEVICES to restrict accelerate launch to specified GPU
CUDA_VISIBLE_DEVICES=$GPU accelerate launch finetune_accelerate.py \
    --base_model Qwen/Qwen2.5-3B \
    --data_path /home/ubuntu/LLM-inference/liangzhao-project/yunqi/DoRA-SoRA-and-LoRA/adapters/dora/commonsense_reasoning/commonsense_170k.json \
    --output_dir $OUTPUT_DIR \
    --adapter_name $ADAPTER \
    --lora_r $R \
    --lora_alpha $ALPHA \
    --num_epochs 3 \
    --learning_rate $LR \
    --sparse_lambda $SPARSE_LAMBDA \
    --micro_batch_size 32 \
    --batch_size 32 \
    $( if [ -n "${GATE_LR}" ]; then echo "--gate_lr ${GATE_LR}"; else echo "--gate_lr_multiplier ${GATE_LR_MULT}"; fi ) \
    $( [ "$PROX_ONLY" = "1" ] && echo "--use_prox_only" ) \
    $( if [ "${DEBUG_PRINT_STEPS}" != "0" ]; then echo "--debug_print_steps ${DEBUG_PRINT_STEPS}"; fi ) \
    $( if [ -n "${MAX_STEPS}" ]; then echo "--max_steps ${MAX_STEPS}"; fi ) \
    --log_gate_stats_steps 500

echo "========================================"
echo "Training completed!"
echo "End time: $(date)"
echo "Output saved to: ${OUTPUT_DIR}"
echo "========================================"