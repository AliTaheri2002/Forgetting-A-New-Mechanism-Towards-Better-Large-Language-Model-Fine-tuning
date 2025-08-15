#!/bin/bash
model_name_or_path=$1
train_data=$2
max_seq_length=$3
batch_per_gpu=$4
gpu_count=$5
base_model=$6
cluster_root_path=$7
rho=$8
main_process_port=$9
token_select_strategy=${10}
use_forgetting=${11}
random_seed=${12:-42}

train_data_tag=$(basename "$train_data" .json)
GRADIENT_ACC_STEPS=3
output_dir="$cluster_root_path/models/${base_model}/rho_${rho}/lora_${train_data_tag}/"
merged_dir="$cluster_root_path/models/${base_model}/rho_${rho}/lora_merged_${train_data_tag}/"

# fine-tuning
accelerate launch --num_machines 1 --mixed_precision bf16 --num_processes $gpu_count \
  --config_file config.yaml --main_process_port $main_process_port \
  source_codes/finetune.py --model_name_or_path $model_name_or_path --gradient_checkpointing \
  --use_lora --lora_rank 64 --lora_alpha 16 --lora_dropout 0.1 --tokenizer_name $model_name_or_path \
  --train_file $train_data --max_seq_length $max_seq_length --preprocessing_num_workers 16 \
  --checkpointing_steps epoch --per_device_train_batch_size $batch_per_gpu \
  --gradient_accumulation_steps $GRADIENT_ACC_STEPS --learning_rate 1e-4 --lr_scheduler_type linear \
  --warmup_ratio 0.03 --weight_decay 0. --num_train_epochs 1 --output_dir $output_dir \
  --with_tracking --report_to tensorboard --logging_steps 1 --train_data_tag $train_data_tag \
  --token_select_strategy $token_select_strategy --use_forgetting $use_forgetting --seed $random_seed

# Merge LoRA 
python source_codes/merge_lora.py --base_model_name_or_path $model_name_or_path \
  --lora_model_name_or_path $output_dir --output_dir $merged_dir \
  --save_tokenizer --use_fast_tokenizer 


sleep 15s
rm -rf $cluster_root_path/models/${base_model}/rho_${rho}/lora_${train_data_tag}