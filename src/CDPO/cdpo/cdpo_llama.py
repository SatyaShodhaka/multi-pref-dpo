# This file is the CPSFT DPO training process.
# Modified for DPO based on the original SFT script
# Author: Your Name
# Date: 2024-05
# Copyright (c) Organization, All rights reserved.

import os
import sys
import torch
import transformers
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel
)
import json
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    DPOTrainer,
    HfArgumentParser
)
from utils.prompter import Prompter
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ModelArguments:
    base_model: str = field(default="meta-llama/Llama-3.2-1B-Instruct")
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(default='["q_proj","v_proj"]')

@dataclass
class DataArguments:
    data_path: str = field(default="ultrafeedback_dpo.jsonl")  # Changed to DPO format
    prompt_template_name: str = field(default="meta-llama/Llama-3.2-1B-Instruct")
    add_eos_token: bool = field(default=False)
    cutoff_len: int = field(default=8192)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    per_device_train_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=32)
    gradient_checkpointing: bool = field(default=True)
    warmup_steps: int = field(default=100)
    num_train_epochs: int = field(default=3)
    learning_rate: float = field(default=1e-5)
    bf16: bool = field(default=True)
    logging_steps: int = field(default=10)
    evaluation_strategy: str = field(default="no")
    output_dir: str = field(default="/data/checkpoints/")
    save_total_limit: int = field(default=1)
    group_by_length: bool = field(default=False)
    # DPO specific parameters
    beta: float = field(default=0.1)  # Temperature parameter for DPO loss
    max_prompt_length: int = field(default=1024)
    max_length: int = field(default=2048)
    loss_type: str = field(default="sigmoid")  # sigmoid or hinge

def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args.lora_target_modules = json.loads(model_args.lora_target_modules)

    print(
        f"Training Alpaca-LoRA model with DPO params:\n"
        f"base_model: {model_args.base_model}\n"
        f"data_path: {data_args.data_path}\n"
        f"output_dir: {training_args.output_dir}\n"
        f"beta: {training_args.beta}\n"
        f"loss_type: {training_args.loss_type}\n"
    )

    # Initialize prompter and tokenizer
    prompter = Prompter(data_args.prompt_template_name)
    tokenizer = AutoTokenizer.from_pretrained(model_args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load base model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.base_model,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2" if device == "cuda" else None,
        device_map="auto",
    )
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules=model_args.lora_target_modules,
        lora_dropout=model_args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Create reference model
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_args.base_model,
        torch_dtype=torch_dtype,
        attn_implementation="flash_attention_2" if device == "cuda" else None,
        device_map="auto",
    )
    ref_model = PeftModel.from_pretrained(ref_model, model_args.base_model)
    ref_model.requires_grad_(False)

    # Load and process dataset
    def process_dpo_data(item):
        prompt = prompter.generate_prompt(
            item["instruction"],
            item["input"]
        )
        return {
            "prompt": prompt,
            "chosen": item["chosen"],
            "rejected": item["rejected"],
        }

    dataset = load_dataset("json", data_files=data_args.data_path)["train"]
    dataset = dataset.map(process_dpo_data)

    # Tokenization function for DPO
    def tokenize_dpo(item):
        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        tokenized_prompt = tokenizer(
            prompt,
            max_length=data_args.cutoff_len,
            truncation=True,
            add_special_tokens=False,
        )
        
        tokenized_chosen = tokenizer(
            chosen,
            max_length=data_args.cutoff_len,
            truncation=True,
            add_special_tokens=False,
        )
        
        tokenized_rejected = tokenizer(
            rejected,
            max_length=data_args.cutoff_len,
            truncation=True,
            add_special_tokens=False,
        )

        return {
            "input_ids_chosen": tokenized_chosen["input_ids"],
            "attention_mask_chosen": tokenized_chosen["attention_mask"],
            "input_ids_rejected": tokenized_rejected["input_ids"],
            "attention_mask_rejected": tokenized_rejected["attention_mask"],
            "prompt_input_ids": tokenized_prompt["input_ids"],
            "prompt_attention_mask": tokenized_prompt["attention_mask"],
        }

    tokenized_dataset = dataset.map(tokenize_dpo)

    # Initialize DPO Trainer
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        beta=training_args.beta,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        max_prompt_length=data_args.cutoff_len,
        max_length=data_args.cutoff_len*2,
        loss_type=training_args.loss_type,
    )

    # Start training
    print("Starting DPO training...")
    dpo_trainer.train()
    dpo_trainer.save_model(training_args.output_dir)

    # Save completion marker
    tmp_dir = os.path.join(training_args.output_dir, "dpo.success")
    with open(tmp_dir, 'w') as f:
        f.write("training completed\n")

if __name__ == "__main__":
    train()