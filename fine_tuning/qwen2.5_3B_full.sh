#!/bin/bash

PYTHON_SCRIPT="finetune_full.py" 

BASE_MODEL="Qwen/Qwen2.5-3B"

DATA_PATH="adapters/dora/commonsense_reasoning/commonsense_170k.json"

OUTPUT_DIR="outputs/Qwen2.5-3B_full-tuning"

LR="2e-5"

BATCH_SIZE=32

MICRO_BATCH_SIZE=8

NUM_EPOCHS=3

LOG_FILE="training_full.log"

mkdir -p "$OUTPUT_DIR"

PYTHON_ARGS=" \
    --base_model ${BASE_MODEL} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --learning_rate ${LR} \
    --num_epochs ${NUM_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --micro_batch_size ${MICRO_BATCH_SIZE} \
"


nohup accelerate launch \
    --mixed_precision "bf16" \
    ${PYTHON_SCRIPT} \
    ${PYTHON_ARGS} \
    > "${LOG_FILE}" 2>&1 &
