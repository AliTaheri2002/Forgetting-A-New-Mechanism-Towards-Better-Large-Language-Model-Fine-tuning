#!/bin/bash

model_name_or_path=$1
train_data=$2
max_seq_length=$3
batch_per_gpu=$4
gpu_count=$5
main_process_port=$6


accelerate launch \
    --num_processes $gpu_count \
    --config_file config.yaml \
    --main_process_port $main_process_port \
    source_codes/token_wise_loss_calculation.py \
    --model_name_or_path $model_name_or_path \
    --tokenizer_name $model_name_or_path \
    --train_file $train_data \
    --max_seq_length $max_seq_length \
    --per_device_train_batch_size $batch_per_gpu \
    --num_train_epochs 1