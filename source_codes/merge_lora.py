import argparse
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import bitsandbytes as bnb
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_model_name_or_path", type=str, required=True)
    parser.add_argument("--base_model_name_or_path", type=str, required=False)
    parser.add_argument("--tokenizer_name_or_path", type=str, required=False)
    parser.add_argument("--output_dir", type=str, required=False)
    parser.add_argument("--save_tokenizer", action="store_true")
    parser.add_argument("--use_fast_tokenizer", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    peft_config = PeftConfig.from_pretrained(args.lora_model_name_or_path)
    print("Loading the base model...")

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_name_or_path if args.base_model_name_or_path else peft_config.base_model_name_or_path,
    )

    if args.tokenizer_name_or_path:
        print(f"Loading the tokenizer from {args.tokenizer_name_or_path}...")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, use_fast=args.use_fast_tokenizer)
    else:
        try:
            print("Trying to load the tokenizer in the lora model folder...")
            # tokenizer = AutoTokenizer.from_pretrained(args.lora_model_name_or_path, use_fast=args.use_fast_tokenizer)#,cache_dir=cache_dir)
            tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path, use_fast=args.use_fast_tokenizer)
        except:
            print("No tokenizer found in the lora model folder. Using the tokenizer in the base model folder...")
            tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path, use_fast=args.use_fast_tokenizer)


    embedding_size = base_model.get_input_embeddings().weight.shape[0]

    print(f"####### args.base_model_name_or_path:{args.base_model_name_or_path.lower()}")

    embedding_size = 128264 

    if len(tokenizer) > embedding_size:
        print(f"The vocabulary the tokenizer contains {len(tokenizer)-embedding_size} more tokens than the base model.")
        print("Resizing the token embeddings of the merged model...")
        
        base_model.resize_token_embeddings(len(tokenizer))
    else:
        base_model.resize_token_embeddings(embedding_size)
    

    print("Loading the lora model...")
    lora_model = PeftModel.from_pretrained(base_model, args.lora_model_name_or_path)
    print("Merging the lora modules...")
    merged_model = lora_model.merge_and_unload()
    
    output_dir = args.output_dir if args.output_dir else args.lora_model_name_or_path
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving merged model to {output_dir}...")
    merged_model.save_pretrained(output_dir)


    if args.save_tokenizer:
        print(f"Saving the tokenizer to {output_dir}...")
        tokenizer.save_pretrained(output_dir)