#!/bin/bash
#SBATCH --job-name=dora_test           
#SBATCH --partition=gpu                
#SBATCH --gres=gpu:v100:1                  
#SBATCH --cpus-per-task=4              
#SBATCH --time=00:10:00                
#SBATCH --output=/home/%u/log/%x-%j.log    

cd ~/llm_proj/dora_sora_lora/adapters/dora/commonsense_reasoning
bash llama2_7B_DoRA.sh 32 64 ./finetuned_result/r32_lr1e-4 0
