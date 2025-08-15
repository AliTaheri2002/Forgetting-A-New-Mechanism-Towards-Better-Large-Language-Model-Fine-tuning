#!/bin/bash

# path
project_dir=/path/paper_code

export CUDA_VISIBLE_DEVICES=0,1,2,3
gpu_count=4

# Base models - uncomment one:
# model_source="meta-llama/Llama-3.2-1B"    # 1B model
# model_source="meta-llama/Llama-3.2-3B"    # 3B model
model_source="meta-llama/Llama-3.1-8B"      # 8B model

# Token settings
token_select_strategy="full_token"
use_forgetting=False
# config
dataset="training_dataset/tulu_10k_sample.json"
rho=0.0
max_length=2048
port_number=29509
batch_per_gpu=2

# fine-tuning
echo "start finetuning..."
bash_files/finetune.sh "$model_source" "$dataset" "$max_length" "$batch_per_gpu" "$gpu_count" "$model_source" "$project_dir" "$rho" "$port_number" "$token_select_strategy" "$use_forgetting"