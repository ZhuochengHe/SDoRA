#!/bin/bash

# Usage:
#   ./run_eval_qwen.sh MODEL_TAG ADAPTER_NAME [R] [ALPHA] [CHECKPOINT_NAME] [BATCH_SIZE] [BASE_MODEL] [LORA_DROPOUT]
#
# Examples:
#   # LoRA, use default r=16, alpha=32, model.pt, bs=16, dropout=0.05
#   ./run_eval_qwen.sh Qwen2.5-3B_lora_r16 lora
#
#   # LoRA, custom checkpoint and batch size
#   ./run_eval_qwen.sh Qwen2.5-3B_lora_r16 lora 16 32 checkpoint-epoch2/model.pt 8
#
#   # DoRA, use default r=16, alpha=32, model.pt, bs=16, dropout=0.05
#   ./run_eval_qwen.sh Qwen2.5-3B_dora_r16 dora
#
#   # SoRA, custom sparsity setup (r/alpha usually same as training)
#   ./run_eval_qwen.sh Qwen2.5-3B_sora_r16_lambda0.1 sora 16 32 model.pt 16 Qwen/Qwen2.5-3B 0.05
#
#   # SdoRA
#   ./run_eval_qwen.sh Qwen2.5-3B_sdora_r16_lambda0.1 sdora 16 32 model.pt 16 Qwen/Qwen2.5-3B 0.05
<<<<<<< HEAD:adapters/dora/commonsense_reasoning/run_eval_qwen.sh
=======
#
#   #   # Full fine-tuning (no adapters), assume outputs/Qwen2.5-3B_full-tuning/model.pt
#   ./run_eval_qwen.sh Qwen2.5-3B_full-tuning full 16 32 model.pt 16 Qwen/Qwen2.5-3B 0.05
#
#
>>>>>>> origin/clz:eval_commonsense_reasoning/run_eval_qwen.sh

# MODEL_TAG is the folder name under ../../../outputs/
# e.g. ../../../outputs/Qwen2.5-3B_lora_r16
MODEL_TAG=$1
ADAPTER_NAME=$2                    # lora / sora / dora / sdora
R=${3:-16}                         # LoRA rank (default: 16)
ALPHA=${4:-32}                     # LoRA alpha (default: 32)
CHECKPOINT_NAME=${5:-model.pt}     # which checkpoint to load
BATCH_SIZE=${6:-16}                # eval batch size
BASE_MODEL=${7:-"Qwen/Qwen2.5-3B"} # HF base model name
LORA_DROPOUT=${8:-0.05}            # adapter dropout, must match training

if [ -z "$MODEL_TAG" ] || [ -z "$ADAPTER_NAME" ]; then
    echo "Usage: $0 MODEL_TAG ADAPTER_NAME [R] [ALPHA] [CHECKPOINT_NAME] [BATCH_SIZE] [BASE_MODEL] [LORA_DROPOUT]"
    echo "Example: $0 Qwen2.5-3B_lora_r16 lora"
    exit 1
fi

<<<<<<< HEAD:adapters/dora/commonsense_reasoning/run_eval_qwen.sh
MODEL_DIR="../../../outputs/${MODEL_TAG}"
=======
MODEL_DIR="../outputs/${MODEL_TAG}"
>>>>>>> origin/clz:eval_commonsense_reasoning/run_eval_qwen.sh

if [ ! -d "$MODEL_DIR" ]; then
    echo "Error: directory ${MODEL_DIR} does not exist."
    exit 1
fi

echo "========================================"
echo "Running commonsense evaluation"
echo "========================================"
echo "Model dir      : ${MODEL_DIR}"
echo "Adapter        : ${ADAPTER_NAME}"
echo "LoRA rank (r)  : ${R}"
echo "LoRA alpha     : ${ALPHA}"
echo "LoRA dropout   : ${LORA_DROPOUT}"
echo "Checkpoint     : ${CHECKPOINT_NAME}"
echo "Base model     : ${BASE_MODEL}"
echo "Batch size     : ${BATCH_SIZE}"
echo "Start time     : $(date)"
echo "========================================"

python eval_allcs_qwen.py \
  --model_dir "${MODEL_DIR}" \
  --base_model "${BASE_MODEL}" \
  --checkpoint_name "${CHECKPOINT_NAME}" \
  --batch_size "${BATCH_SIZE}" \
  --adapter_name "${ADAPTER_NAME}" \
  --lora_r "${R}" \
  --lora_alpha "${ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}"

echo "========================================"
echo "Evaluation finished"
echo "End time       : $(date)"
echo "Results in     : experiment/results.csv"
echo "Per-dataset    : experiment/${MODEL_TAG}-<dataset>.json"
echo "========================================"