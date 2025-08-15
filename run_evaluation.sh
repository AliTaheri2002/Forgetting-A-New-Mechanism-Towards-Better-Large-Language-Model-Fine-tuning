#!/bin/bash
size=${1:-"1B"}

export CUDA_VISIBLE_DEVICES=0,1,2,3
gpus=4
MODEL=hf

# Model selection
if [ "$size" = "1B" ]; then
    model="meta-llama/Llama-3.2-1B"
elif [ "$size" = "3B" ]; then
    model="meta-llama/Llama-3.2-3B"
elif [ "$size" = "8B" ]; then
    model="meta-llama/Llama-3.1-8B"
else
    exit 1
fi

path="/path_to/paper_code/models/${model}"

# Config
rho=0.7
tasks=("truthfulqa" "boolq" 'logiqa' 'asdiv')
tag=tulu_50k_sample

# path
trained=${path}/rho_${rho}/lora_merged_${tag}

output=results/${size}/${rho}/${tag}
mkdir -p $output

# Task parameters
declare -A params=(
    ["truthfulqa"]="0 24 0.99"
    ["boolq"]="0 24 0.99"
    ["logiqa"]="0 24 0.99"
    ["asdiv"]="0 20 0.99"
)

# arguments
declare -A args=(
    ["truthfulqa"]="pretrained=${trained},dtype=bfloat16"
    ["boolq"]="pretrained=${trained},dtype=bfloat16"
    ["logiqa"]="pretrained=${trained},dtype=bfloat16"
    ["asdiv"]="pretrained=${trained},dtype=bfloat16"
)

# evaluation
for i in "${!tasks[@]}"; do
    task=${tasks[$i]}
    p=(${params[$task]})
    fewshot=${p[0]}
    batch=${p[1]}
    limit=${p[2]}
    idx=$((i % 8))
    margs=${args[$task]}

    accelerate launch --multi-gpu --main_process_port 29520 --num_processes $gpus \
            -m lm_eval --model $MODEL \
            --model_args $margs \
            --tasks $task \
            --batch_size $batch \
            --num_fewshot $fewshot \
            --limit $limit \
            --output_path $output \
            --seed 42 \
            --trust_remote_code
    
    sleep 5s
done

# TydiQA evaluation
CUDA_VISIBLE_DEVICES=0 python -m eval.tydiqa.run_eval \
    --data_dir eval/tydiqa/ \
    --n_shot 1 \
    --max_num_examples_per_lang 100 \
    --max_context_length 512 \
    --save_dir $output \
    --model_name_or_path $trained \
    --tokenizer_name_or_path $trained \
    --eval_batch_size 15 \
    --use_vllm