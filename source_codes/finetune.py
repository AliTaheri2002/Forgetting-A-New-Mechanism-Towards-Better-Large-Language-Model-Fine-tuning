import argparse
import logging
import math
import os
import datasets
from datetime import timedelta
import torch
from functools import partial
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, InitProcessGroupKwargs
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import deepspeed
import torch.nn.functional as F
import numpy as np
import pickle 

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    SchedulerType,
    DataCollatorForSeq2Seq,
    get_scheduler,
    GPTNeoXTokenizerFast,
    GPT2Tokenizer,
    OPTForCausalLM
)
from peft import LoraConfig, TaskType, get_peft_model

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('true', '1', 'yes'):
        return True
    elif value.lower() in ('false', '0', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got {value}.")

def load_token_indices(data_type, label_path="preprocessing_outputs/label/"):
    #Load positive and negative token indices for forgetting implementation

    positive_indices_path = label_path + f"positive_indices_{data_type}.pkl"
    with open(positive_indices_path, 'rb') as f:
        positive_indices = pickle.load(f)
    
    # Load negative indices
    negative_indices_path = label_path + f"negative_indices_{data_type}.pkl"
    with open(negative_indices_path, 'rb') as f:
        negative_indices = pickle.load(f)
    
    return positive_indices, negative_indices

def create_token_masks(positive_indices, negative_indices, batch_indices, batch_size, seq_len, device):
    #Create token masks for applying forgetting in training loop

    positive_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
    negative_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
    
    # Map batch indices to positions within the batch
    batch_idx_map = {idx.item(): batch_pos for batch_pos, idx in enumerate(batch_indices)}
    
    # Fill masks based on indices
    for i, j in positive_indices:
        if i in batch_idx_map and j < seq_len:
            batch_pos = batch_idx_map[i]
            positive_mask[batch_pos, j-1] = True
    
    for i, j in negative_indices:
        if i in batch_idx_map and j < seq_len:
            batch_pos = batch_idx_map[i]
            negative_mask[batch_pos, j-1] = True
    
    return positive_mask, negative_mask


logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--model_revision",
        help="""If given, specifies a model revision (for HuggingFace models). This will 
        be applied to both the `model_name_or_path` and `config_name` args.""",
        default="main",
        required=False,
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="If passed, will use LORA (low-rank parameter-efficient training) to train the model.",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="The rank of lora.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=16,
        help="The alpha parameter of lora.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="The dropout rate of lora modules.",
    )
    parser.add_argument(
        "--use_flash_attn",
        action="store_true",
        help="If passed, will use flash attention to train the model.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_revision",
        help="""Specifies a revision for the tokenizer. If not given, defaults
             to the value of the `model_revision` arg. In most cases, the tokenizer
             revision should be the same as the model revision and this flag shouldn't
             be needed.""",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="The maximum total sequence length (prompt+completion) of each training example.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.0, 
        help="Weight decay to use.")
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=3, 
        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0, help="Ratio of total training steps used for warmup."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None, 
        help="Where to store the final model.")
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="A seed for reproducible training.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", 
        action="store_true", 
        help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=None,
        help="Log the training loss and learning rate every logging_steps steps.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help=(
            "Turn on gradient checkpointing. Saves memory but slows training."
        ),
    )

    parser.add_argument(
        '--clip_grad_norm',
        type=float,
        default=-1,
        help='Clip gradient norm. Not compatible with deepspeed (use deepspeed config instead).',
    )

    parser.add_argument(
        '--add_bos',
        action='store_true',
        help='Forcibly add bos token to the beginning of the input sequence. Use only when tokenizer does not add bos token by default (e.g., olmo).',
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=1800,
        help='Timeout for the training process. Useful if tokenization process is long. Default is 1800 seconds (30 minutes).',
    )
    parser.add_argument(
        '--trust_remote_code',
        action='store_true',
        help='Trust remote code when loading pretrained models and tokenizers. Use only when you trust the remote code.',
    )

    parser.add_argument(
        '--wandb_entity', 
        type=str,
        default=None,
        help='Entity to use for logging to wandb.'
    )

    parser.add_argument(
        '--train_data_tag',
        default='random',
        help='default data set',
    )

    parser.add_argument(
        '--token_select_strategy',
        default='full_token',
        choices=['ignoring', "full_token"],  
        help='Pattern for token selection.',
    )
    
    parser.add_argument(
        '--t_max',
        type=float,
        default=0.25,
        help='lambda hyperparameter t_max.',
    )
    parser.add_argument(
        '--t_min',
        type=float,
        default=0.00001,
        help='lambda hyperparameter t_min.',
    )
    parser.add_argument(
        '--use_forgetting',
        type=str2bool,
        default=True,
        help='Whether to use forgetting approach for handling noisy tokens.',
    )
    
    args = parser.parse_args()
    
    if args.dataset_name is None and args.train_file is None:
        raise ValueError("Need either a dataset name or a training file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["json", "jsonl"], "`train_file` should be a json/jsonl file."
    return args


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
        'attention_mask': attention_mask.flatten()
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

    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]), return_tensors='pt', max_length=max_seq_length, truncation=True
                ).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx+1]["role"] == "assistant":
                
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
        'attention_mask': attention_mask.flatten()
    }


def save_with_accelerate(accelerator, model, tokenizer, output_dir, args):
    model.generation_config = transformers.GenerationConfig(
        temperature=None,
        top_p=None,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id
    )

    unwrapped_model = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)
    if args.use_lora:
        if accelerator.is_main_process:
            unwrapped_model.save_pretrained(output_dir, state_dict=state_dict)
    else:
        unwrapped_model.save_pretrained(
            output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=state_dict,
            safe_serialization=False
        )


def main():
    args = parse_args()

    # Initialize the accelerator.
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=args.timeout))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        **accelerator_log_kwargs,
        kwargs_handlers=[timeout_kwargs]
    )
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    accelerator.wait_for_everyone()

    if args.dataset_name is not None:
        original_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
        )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        original_datasets = load_dataset(
            "json",
            data_files=data_files,
            **dataset_args,
        )
    
    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            trust_remote_code=args.trust_remote_code,
            revision=args.model_revision,
            token=os.getenv("HF_TOKEN", None)
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            revision=args.model_revision,
            token=os.getenv("HF_TOKEN", None)
        )
    else:
        raise ValueError(
            "Warning"
        )

    tokenizer_revision = (
        args.model_revision
        if args.tokenizer_revision is None
        else args.tokenizer_revision
    )

    if tokenizer_revision != args.model_revision:

        warning = f"""Requested tokenizer revision `{tokenizer_revision}` is different
                   from the model revision `{args.model_revision}`."""
        logger.warn(warning)

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name,
            trust_remote_code=args.trust_remote_code,
            use_fast=not args.use_slow_tokenizer,
            revision=tokenizer_revision,
            token=os.getenv("HF_TOKEN", None)
        )
        
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
            use_fast=not args.use_slow_tokenizer,
            revision=tokenizer_revision,
            token=os.getenv("HF_TOKEN", None)
        )
    else:
        raise ValueError(
            "Warning"
        )

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            trust_remote_code=args.trust_remote_code,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            use_flash_attention_2=True if args.use_flash_attn else False,
            revision=args.model_revision,
            token=os.getenv("HF_TOKEN", None)
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens({
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        })
        assert num_added_tokens in [0, 1], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
    elif isinstance(tokenizer, GPTNeoXTokenizerFast):
        if tokenizer.bos_token is None:
            tokenizer.bos_token = tokenizer.eos_token
            assert args.add_bos, "For OLMo with GPTNeoX, you must add bos token to the beginning of the input sequence."
        else:
            num_added_tokens = tokenizer.add_special_tokens({
                "pad_token": "<pad>",
            })
            assert num_added_tokens == 1, "GPTNeoXTokenizer should only add one special token - the pad_token."
    elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(model, OPTForCausalLM):
        num_added_tokens = tokenizer.add_special_tokens({'unk_token': '<unk>'})
    elif isinstance(tokenizer, transformers.PreTrainedTokenizerFast) and tokenizer.pad_token is None:
        num_added_tokens = tokenizer.add_special_tokens({'pad_token': '<pad>'})
        assert num_added_tokens == 1, "We detected no padding token but add_special_tokens did not add one."
 
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]

    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    embeddings = model.get_input_embeddings()
    with deepspeed.zero.GatheredParameters(embeddings.weight, modifier_rank=None):
        embedding_size = embeddings.weight.shape[0]

    tokenizer.chat_template = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}" 
    if args.add_bos:
        tokenizer.chat_template = "{{ bos_token }}" + tokenizer.chat_template

    if args.use_lora:
        logger.info("Initializing LORA model...")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=args.lora_rank, 
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "o_proj", "v_proj", "k_proj", "gate_proj", "up_proj", "down_proj"]
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Preprocessing
    if "prompt" in original_datasets["train"].column_names and "completion" in original_datasets["train"].column_names:
        encode_function = partial(
            encode_with_prompt_completion_format,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            add_bos=args.add_bos,
        )
    elif "messages" in original_datasets["train"].column_names:
        encode_function = partial(
            encode_with_messages_format,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            add_bos=args.add_bos,
        )
    else:
        raise ValueError("You need to have either 'prompt'&'completion' or 'messages' in your column names.")
    
    with accelerator.main_process_first():
        
        original_datasets = original_datasets.map(
            lambda example, idx: {"idx": idx},
            with_indices=True,  
            num_proc=args.preprocessing_num_workers,
            desc="Adding idx column"
        )
        
        lm_datasets = original_datasets.map(
            encode_function,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            remove_columns=[name for name in original_datasets["train"].column_names if name not in ["idx", "input_ids", "labels", "attention_mask"]],
            desc="Tokenizing and reformatting instruction data"
        )
        lm_datasets.set_format(type="pt")
        
        train_dataset = lm_datasets["train"]
        orig_labels = train_dataset['labels']
        
        ############### token selection strategy ###############
        if args.token_select_strategy == 'ignoring': 
            selected_labels = torch.load(f"preprocessing_outputs/label/token_labels_{args.train_data_tag}.pt")   
            train_dataset = train_dataset.map(
                lambda examples, idx: {'labels': selected_labels[idx]},
                with_indices=True
            )          
 
        elif args.token_select_strategy == 'full_token':
            selected_labels = orig_labels
                     
    
    # DataLoader
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=False, 
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
        batch_size=args.per_device_train_batch_size
    )

    # Optimizer
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    num_training_steps_for_scheduler = args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # trackers initialization
    if args.with_tracking:
        experiment_config = vars(args)
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("open_instruct_sft", 
                                  experiment_config, 
                                  init_kwargs={"wandb": {"entity": args.wandb_entity}})

    if args.use_forgetting:
        data_type = os.path.basename(args.train_file).split(".json")[0]
        positive_indices, negative_indices = load_token_indices(data_type)
        logger.info(f"Loaded forgetting indices - positive tokens: {len(positive_indices)}, negative tokens: {len(negative_indices)}")

    # training
    logger.info("Training:")
    logger.info(f"  Examples={len(train_dataset)}, Epochs={args.num_train_epochs}")
    logger.info(f"  Batch/dev={args.per_device_train_batch_size}")
    logger.info(f"  Grad accum={args.gradient_accumulation_steps}, Steps={args.max_train_steps}")

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # load in the weights and states
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(path)
        
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            resume_step = (
                int(training_difference.replace("step_", ""))
                * args.gradient_accumulation_steps
            )
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    progress_bar.update(completed_steps)
    

    
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss = 0
        if (
            args.resume_from_checkpoint
            and epoch == starting_epoch
            and resume_step is not None
        ):
            active_dataloader = accelerator.skip_first_batches(
                train_dataloader, resume_step
            )
        else:
            active_dataloader = train_dataloader
            
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                indices = batch["idx"]
                # indices = [0 for _ in range(batch["labels"].size(0))]
                outputs = model(input_ids=batch['input_ids'], labels=batch['labels'], attention_mask=batch['attention_mask'], use_cache=False)

                if args.use_forgetting:
                    logits = outputs.logits
                    labels = batch["labels"]
                    
                    # Shift logits and labels for loss calculation (standard causal LM setup)
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    batch_size, seq_len_minus_1 = shift_labels.shape
                    
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    
                    # Flatten for loss calculation
                    shift_logits = shift_logits.view(-1, embedding_size)
                    shift_labels = shift_labels.view(-1)
                    shift_labels = shift_labels.to(shift_logits.device)
                    
                    # Calculate token-wise losses
                    loss_1d = loss_fct(shift_logits, shift_labels)
                    loss_2d = loss_1d.view(batch_size, seq_len_minus_1)
                    
                    # Create mask for valid tokens (not -100)
                    valid_mask = (shift_labels.view(batch_size, seq_len_minus_1) != -100)
                    
                    # Create masks for positive and negative tokens in this batch
                    positive_mask, negative_mask = create_token_masks(
                                                            positive_indices, 
                                                            negative_indices, 
                                                            indices, 
                                                            batch_size, 
                                                            seq_len_minus_1,
                                                            device=shift_labels.device 
                                                        )
                    
                    # Apply masks - ensure we only consider tokens that are both valid and in our category
                    positive_mask = positive_mask & valid_mask
                    negative_mask = negative_mask & valid_mask
                    
                    # Calculate separate losses for positive and negative tokens
                    positive_loss = (loss_2d * positive_mask.float()).sum()
                    negative_loss = (loss_2d * negative_mask.float()).sum()
                    
                    # Count tokens in each category
                    num_positive_tokens = positive_mask.sum() + 1e-10
                    num_negative_tokens = negative_mask.sum() + 1e-10

                    # linear balancing coefficient (lambda(step))
                    forgetting_scale = (args.t_max - args.t_min)*(step // args.gradient_accumulation_steps)/(args.max_train_steps)

                    loss = positive_loss / num_positive_tokens - forgetting_scale * negative_loss / num_negative_tokens
                    current_lr = optimizer.param_groups[0]['lr']
                    if accelerator.is_main_process and step % 10 == 0:
                        print(f"forgetting_scale: {forgetting_scale} | current_lr: {current_lr}")
                    # For logging
                    if accelerator.is_main_process and step % 10 == 0:
                        logger.info(f"positive loss: {positive_loss.item() / num_positive_tokens.item()}, "
                                    f"negative loss: {negative_loss.item() / num_negative_tokens.item()}")
                else:
                    loss = outputs.loss
                
                total_loss += loss.detach().float()
                accelerator.backward(loss)
                
                if accelerator.sync_gradients and args.clip_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad() 
                lr_scheduler.step() 

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    avg_loss = accelerator.gather(total_loss).mean().item() / args.gradient_accumulation_steps / args.logging_steps
                    logger.info(f"  Step: {completed_steps}, LR: {lr_scheduler.get_last_lr()[0]}, Loss: {avg_loss}")
                    if args.with_tracking:
                        accelerator.log(
                            {
                                "learning_rate": lr_scheduler.get_last_lr()[0],
                                "train_loss": avg_loss,
                            },
                            step=completed_steps,
                        )
                    total_loss = 0
                    
                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = f"step_{completed_steps}"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        save_with_accelerate(accelerator, model, tokenizer, output_dir, args)

                if completed_steps >= args.max_train_steps:
                    break
                

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            save_with_accelerate(accelerator, model, tokenizer, output_dir, args)

    if args.output_dir is not None:
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
        save_with_accelerate(accelerator, model, tokenizer, args.output_dir, args)

    accelerator.wait_for_everyone()

    if args.with_tracking:
        accelerator.end_training()

 
if __name__ == "__main__":
    main()