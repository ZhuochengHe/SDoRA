#!/bin/bash

# run_eval_all_adapters.sh
#
# Run commonsense evaluation for 4 adapters:
#   LoRA, DoRA, SoRA, SDoRA
#
# Usage:
#   bash run_eval_all_adapters.sh
#
# Optional args:
#   R           LoRA rank (default: 16)
#   ALPHA       LoRA alpha (default: 32)
#   BATCH_SIZE  evaluation batch size (default: 16)
#   LAMBDA      sparsity lambda used in SoRA / SDoRA directory names (default: 0.1)
#   BASE_MODEL  HF base model name (default: Qwen/Qwen2.5-3B)
#   DROPOUT     LoRA dropout (match training, default: 0.05)
#
# Examples:
#   bash run_eval_all_adapters.sh
#   bash run_eval_all_adapters.sh 16 32 8 0.2 Qwen/Qwen2.5-3B 0.05

R=${1:-16}
ALPHA=${2:-32}
BATCH_SIZE=${3:-16}
LAMBDA=${4:-0.1}
BASE_MODEL=${5:-"Qwen/Qwen2.5-3B"}
DROPOUT=${6:-0.05}

# Unified checkpoint name used for all adapters
CKPT_NAME="model.pt"

echo "========================================"
echo "Running evaluation for all adapters"
echo "R            : ${R}"
echo "Alpha        : ${ALPHA}"
echo "Batch size   : ${BATCH_SIZE}"
echo "Lambda       : ${LAMBDA}  (for SoRA / SDoRA tags)"
echo "Base model   : ${BASE_MODEL}"
echo "LoRA dropout : ${DROPOUT}"
echo "Checkpoint   : ${CKPT_NAME}"
echo "Start time   : $(date)"
echo "========================================"
echo

# ------------ 1. LoRA ------------
MODEL_TAG_LORA="Qwen2.5-3B_lora_r${R}"
echo "[LoRA] MODEL_TAG=${MODEL_TAG_LORA}"
bash run_eval_qwen.sh "${MODEL_TAG_LORA}" lora \
    "${R}" "${ALPHA}" "${CKPT_NAME}" "${BATCH_SIZE}" "${BASE_MODEL}" "${DROPOUT}"
echo

# ------------ 2. DoRA ------------
MODEL_TAG_DORA="Qwen2.5-3B_dora_r${R}"
echo "[DoRA] MODEL_TAG=${MODEL_TAG_DORA}"
bash run_eval_qwen.sh "${MODEL_TAG_DORA}" dora \
    "${R}" "${ALPHA}" "${CKPT_NAME}" "${BATCH_SIZE}" "${BASE_MODEL}" "${DROPOUT}"
echo

# ------------ 3. SoRA ------------
MODEL_TAG_SORA="Qwen2.5-3B_sora_r${R}_lambda${LAMBDA}"
echo "[SoRA] MODEL_TAG=${MODEL_TAG_SORA}"
bash run_eval_qwen.sh "${MODEL_TAG_SORA}" sora \
    "${R}" "${ALPHA}" "${CKPT_NAME}" "${BATCH_SIZE}" "${BASE_MODEL}" "${DROPOUT}"
echo

# ------------ 4. SDoRA -----------
MODEL_TAG_SDORA="Qwen2.5-3B_sdora_r${R}_lambda${LAMBDA}"
echo "[SDoRA] MODEL_TAG=${MODEL_TAG_SDORA}"
bash run_eval_qwen.sh "${MODEL_TAG_SDORA}" sdora \
    "${R}" "${ALPHA}" "${CKPT_NAME}" "${BATCH_SIZE}" "${BASE_MODEL}" "${DROPOUT}"
echo

echo "========================================"
echo "All evaluations finished."
echo "End time : $(date)"
echo "Results  : experiment/results.csv"
echo "========================================"