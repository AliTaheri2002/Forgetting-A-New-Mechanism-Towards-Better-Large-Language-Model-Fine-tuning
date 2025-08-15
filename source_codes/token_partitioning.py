from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
from functools import partial
import numpy as np
import fire
import os
import pickle

def encode_with_prompt_completion_format(example, tokenizer, max_seq_length, add_bos=False):
    # concat prompt+completion before tokenizing to prevent prompt padding issues
    
    if not example['prompt'].endswith((' ', '\n', '\t')) and not example['completion'].startswith((' ', '\n', '\t')):
        example_text = example['prompt'] + ' ' + example['completion']
    else:
        example_text = example['prompt'] + example['completion']
    example_text = example_text + tokenizer.eos_token
    if add_bos:
        example_text = tokenizer.bos_token + example_text
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    tokenized_prompt = tokenizer(example['prompt'], return_tensors='pt', max_length=max_seq_length, truncation=True)
    

    labels[:, :tokenized_prompt.input_ids.shape[1]] = -100
        
    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }


def encode_with_messages_format(example, tokenizer, max_seq_length, add_bos=False):
    # join messages (role+content) with delimiters before tokenizing

    messages = example['messages']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')
    
    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text
        
    example_text = _concat_messages(messages).strip()
    if add_bos:
        example_text = tokenizer.bos_token + example_text
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]), return_tensors='pt', max_length=max_seq_length, truncation=True
                ).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx+1]["role"] == "assistant":
                # here we also ignore the role of the assistant
                messages_so_far = _concat_messages(messages[:message_idx+1]) + "<|assistant|>\n"
            else:
                messages_so_far = _concat_messages(messages[:message_idx+1])
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors='pt', 
                max_length=max_seq_length, 
                truncation=True
            ).input_ids.shape[1]
            

            labels[:, message_start_idx:message_end_idx] = -100
            
            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten(),
    }

def save_token_indices(positive_indices, negative_indices, data_type, label_path="preprocessing_outputs/label/"):
    # save positive token indices separately
    if not os.path.exists(label_path):
        os.makedirs(label_path)
    
    # save positive indices
    positive_indices_path = label_path + f"positive_indices_{data_type}.pkl"
    with open(positive_indices_path, 'wb') as f:
        pickle.dump(positive_indices, f)
    
    # save negative indices
    negative_indices_path = label_path + f"negative_indices_{data_type}.pkl"
    with open(negative_indices_path, 'wb') as f:
        pickle.dump(negative_indices, f)
    
    print(f"Token indices saved at {positive_indices_path} and {negative_indices_path}")


def main(
    base_model_name_or_path='test',
    ref_model_name_or_path='test',
    train_data=None,
    rho: float = 1.0,
    label_path = "preprocessing_outputs/label/",
    loss_path = "preprocessing_outputs/loss/",
    ):

    # means huggingface model or existed local model    
    if "lora" not in base_model_name_or_path or os.path.exists(base_model_name_or_path): 
        tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
    else:
        if "1b" in base_model_name_or_path.lower():
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
        elif "3b" in base_model_name_or_path.lower():
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
        elif "8b" in base_model_name_or_path.lower():
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    
    original_dataset = load_dataset("json", data_files=train_data)

    base_model_name = os.path.basename(base_model_name_or_path)
    ref_model_name = os.path.basename(ref_model_name_or_path)
    data_type= os.path.basename(train_data).split(".json")[0]


    if "prompt" in original_dataset["train"].column_names and "completion" in original_dataset["train"].column_names:
        encode_function = partial(
            encode_with_prompt_completion_format,
            tokenizer=tokenizer,
            max_seq_length= 2048,
            add_bos= False,
        )
    elif "messages" in original_dataset["train"].column_names:
        encode_function = partial(
            encode_with_messages_format,
            tokenizer=tokenizer,
            max_seq_length= 2048,
            add_bos= False,
        )
        
    original_dataset = original_dataset.map(
        lambda example, idx: {"idx": idx},
        with_indices=True,  
        desc="Adding idx column",
    )
            

    lm_datasets = original_dataset.map(
        encode_function,
        batched=False,
        desc="Tokenizing and reformatting instruction data",
    )

    train_dataset = lm_datasets['train']
    raw_labels = train_dataset['labels']

    
    losses_base = torch.load(loss_path + f"token_losses_{data_type}_{base_model_name}.pt")
    losses_ref = torch.load(loss_path + f"token_losses_{data_type}_{ref_model_name}.pt")

    selected_labels = [[-100 for _ in range(len(label))] for label in raw_labels]
    # the calculation influence score of two models (base and reference)
    inf_scores = [(np.array(loss1) - np.array(loss2)).tolist() for loss1, loss2 in zip(losses_base, losses_ref)]
        
    # collect all response tokens with their loss differences
    response_tokens = []
    for i, (sample_labels, sample_losses) in enumerate(zip(raw_labels, inf_scores)):
        for j, (label, loss) in enumerate(zip(sample_labels, sample_losses)):
            if label != -100:  # Only consider response tokens
                response_tokens.append((loss, i, j))
    
    # sort tokens by scores (descending)
    sorted_tokens = sorted(response_tokens, key=lambda x: x[0], reverse=True)
    
    # calculate thresholds for positive-negative tokens
    total_tokens = len(sorted_tokens)
    positive_token_count = int(total_tokens * rho)
    negative_token_count = int(total_tokens * (1-rho)) 
    
    # identify positive tokens (top tokens by loss diff)
    positive_indices = [(item[1], item[2]) for item in sorted_tokens[:positive_token_count]]
    
    # identify negative tokens (remaining tokens - for gradient ascent)
    start_idx = positive_token_count  # Start after positive tokens
    negative_indices = [(item[1], item[2]) for item in sorted_tokens[start_idx:start_idx+negative_token_count]]
    
    print(f"positive tokens: {len(positive_indices)}, negative tokens: {len(negative_indices)}, Total tokens: {total_tokens}")
    
    save_token_indices(positive_indices, negative_indices, data_type, label_path)
    
    
    for i, j in positive_indices:
        selected_labels[i][j] = raw_labels[i][j] 
        
    negative_token_labels = [[-100 for _ in range(len(label))] for label in raw_labels]
    for i, j in negative_indices:
        negative_token_labels[i][j] = raw_labels[i][j]
    
    negative_token_path = label_path + f"negative_token_labels_{data_type}.pt"
    torch.save(negative_token_labels, negative_token_path)    
    
    ## save the loss
    if not os.path.exists(label_path):
        os.makedirs(label_path)

    final_data_path = label_path + f"token_labels_{data_type}.pt"
    torch.save(selected_labels, final_data_path)

    
if __name__ == "__main__":
    fire.Fire(main)