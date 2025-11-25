#!/bin/bash

# Comprehensive experiment runner for LoRA/DoRA/SoRA/SDoRA comparison
# Usage: nohup ./run_all_experiments.sh > all_experiments.log 2>&1 &

set -e  # Exit on error

GPU=0  # GPU to use
R=16
ALPHA=32
NUM_EPOCHS=3

echo "========================================"
echo "   LoRA/DoRA/SoRA/SDoRA Experiments"
echo "   Model: Qwen2.5-4B"
echo "========================================"
echo ""
echo "Configuration:"
echo "  - Rank (r): ${R}"
echo "  - Alpha: ${ALPHA}"
echo "  - Epochs: ${NUM_EPOCHS}"
echo "  - GPU: ${GPU}"
echo "  - Dataset: commonsense_170k (170,300 train samples)"
echo ""

# Calculate time estimates
SAMPLES_PER_EPOCH=170300
BATCH_SIZE=32
STEPS_PER_EPOCH=$((SAMPLES_PER_EPOCH / BATCH_SIZE))
TOTAL_STEPS=$((STEPS_PER_EPOCH * NUM_EPOCHS))

# Time estimates (based on previous DoRA training: ~7.5 hours/epoch)
# DoRA/SoRA/SDoRA: ~7.5h per epoch
# LoRA: ~6h per epoch (faster, no magnitude decomposition)
DORA_TIME_PER_EPOCH=7.5
LORA_TIME_PER_EPOCH=6.0

LORA_TOTAL_HOURS=$(echo "$LORA_TIME_PER_EPOCH * $NUM_EPOCHS" | bc)
DORA_TOTAL_HOURS=$(echo "$DORA_TIME_PER_EPOCH * $NUM_EPOCHS" | bc)
SORA_TOTAL_HOURS=$(echo "$DORA_TIME_PER_EPOCH * $NUM_EPOCHS * 3" | bc)  # 3 sparsity levels
SDORA_TOTAL_HOURS=$(echo "$DORA_TIME_PER_EPOCH * $NUM_EPOCHS" | bc)

TOTAL_ESTIMATED_HOURS=$(echo "$LORA_TOTAL_HOURS + $DORA_TOTAL_HOURS + $SORA_TOTAL_HOURS + $SDORA_TOTAL_HOURS" | bc)

echo "========================================"
echo "   TIME ESTIMATES"
echo "========================================"
echo ""
echo "Steps per epoch: ${STEPS_PER_EPOCH}"
echo "Total training steps: ${TOTAL_STEPS}"
echo ""
echo "Individual experiment estimates:"
echo "  1. LoRA:                ~${LORA_TOTAL_HOURS}h"
echo "  2. DoRA:                ~${DORA_TOTAL_HOURS}h"
echo "  3. SoRA (3 lambdas):    ~${SORA_TOTAL_HOURS}h"
echo "  4. SDoRA:               ~${SDORA_TOTAL_HOURS}h"
echo ""
echo "TOTAL ESTIMATED TIME:     ~${TOTAL_ESTIMATED_HOURS} hours (~$(echo "$TOTAL_ESTIMATED_HOURS / 24" | bc) days)"
echo ""
echo "Start time: $(date)"
echo "Estimated completion: $(date -d "+${TOTAL_ESTIMATED_HOURS} hours" 2>/dev/null || echo "N/A")"
echo "========================================"
echo ""

# Function to train a model and report results
train_model() {
    local adapter=$1
    local lr=$2
    local sparse_lambda=$3
    local output_suffix=$4
    
    local output_dir="outputs/Qwen2.5-3B_${adapter}_r${R}${output_suffix}"
    
    echo ""
    echo "========================================"
    echo "Starting: ${adapter^^} ${output_suffix}"
    echo "========================================"
    echo "Learning rate: ${lr}"
    echo "Sparsity lambda: ${sparse_lambda}"
    echo "Output: ${output_dir}"
    echo "Start time: $(date)"
    echo ""
    
    local start_time=$(date +%s)
    
    CUDA_VISIBLE_DEVICES=$GPU accelerate launch finetune_accelerate.py \
        --output_dir $output_dir \
        --adapter_name $adapter \
        --lora_r $R \
        --lora_alpha $ALPHA \
        --num_epochs $NUM_EPOCHS \
        --learning_rate $lr \
        --sparse_lambda $sparse_lambda \
        --micro_batch_size 8 \
        --batch_size 32
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$(echo "scale=2; $duration / 3600" | bc)
    
    echo ""
    echo "✓ Completed: ${adapter^^} ${output_suffix}"
    echo "  Duration: ${hours} hours"
    echo "  End time: $(date)"
    echo "  Output: ${output_dir}"
    echo "========================================"
    echo ""
}

# Track overall start time
OVERALL_START=$(date +%s)

echo ""
echo "========================================"
echo "EXPERIMENT 1/6: LoRA"
echo "========================================"
train_model "lora" "3e-4" "0.3" ""

echo ""
echo "========================================"
echo "EXPERIMENT 2/6: DoRA"
echo "========================================"
train_model "dora" "1e-4" "0.3" ""

echo ""
echo "========================================"
echo "EXPERIMENT 3/6: SoRA (lambda=0.1)"
echo "========================================"
train_model "sora" "1e-4" "0.1" "_lambda0.1"

echo ""
echo "========================================"
echo "EXPERIMENT 4/6: SoRA (lambda=0.3)"
echo "========================================"
train_model "sora" "1e-4" "0.3" "_lambda0.3"

echo ""
echo "========================================"
echo "EXPERIMENT 5/6: SoRA (lambda=0.5)"
echo "========================================"
train_model "sora" "1e-4" "0.5" "_lambda0.5"

echo ""
echo "========================================"
echo "EXPERIMENT 6/6: SDoRA (lambda=0.3)"
echo "========================================"
train_model "sdora" "1e-4" "0.3" "_lambda0.3"

# Calculate total time
OVERALL_END=$(date +%s)
TOTAL_DURATION=$((OVERALL_END - OVERALL_START))
TOTAL_HOURS=$(echo "scale=2; $TOTAL_DURATION / 3600" | bc)

echo ""
echo "========================================"
echo "   ALL EXPERIMENTS COMPLETED!"
echo "========================================"
echo ""
echo "Total duration: ${TOTAL_HOURS} hours"
echo "Start time: $(date -d "@${OVERALL_START}")"
echo "End time: $(date)"
echo ""
echo ""
echo "Next steps:"
echo "  1. Compare results: python analyze_results.py"
echo "  2. Check loss curves: cd outputs && ls */loss_log.csv"
echo "  3. Compare sparsity: cd outputs && ls */sparsity_log.csv"
echo "========================================"
