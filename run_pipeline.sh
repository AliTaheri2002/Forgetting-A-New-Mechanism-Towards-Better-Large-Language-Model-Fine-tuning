#!/bin/bash

# path
project_dir=/path/paper_code

export CUDA_VISIBLE_DEVICES=0,1,2,3
gpu_count=4

# Base models - uncomment one:
# base_model="meta-llama/Llama-3.2-1B"    # 1B model
# base_model="meta-llama/Llama-3.2-3B"    # 3B model
base_model="meta-llama/Llama-3.1-8B"      # 8B model

reference_model=/path/reference_weights


############### Configuration ###############
max_seq_length=2048

port_number=29520

dataset=training_dataset/tulu_50k_sample.json
token_select_strategy="full_token" #["full_token", "ignoring"]
use_forgetting=True
rho=0.7

############### Preprocessing ###############
batch_per_gpu=6
bash_files/token_wise_loss.sh "$base_model" "$dataset" "$max_seq_length" "$batch_per_gpu" "$gpu_count" "$port_number"
bash_files/token_wise_loss.sh "$reference_model" "$dataset" "$max_seq_length" "$batch_per_gpu" "$gpu_count" "$port_number"

# Token partitioning
python source_codes/token_partitioning.py --base_model_name_or_path $base_model \
    --ref_model_name_or_path $reference_model --train_data $dataset --rho $rho

############### Fine-tuning ###############
batch_per_gpu=2
echo "start finetuning..."
bash_files/finetune.sh "$base_model" "$dataset" "$max_seq_length" "$batch_per_gpu" "$gpu_count" "$base_model" "$project_dir" "$rho" "$port_number" "$token_select_strategy" "$use_forgetting"