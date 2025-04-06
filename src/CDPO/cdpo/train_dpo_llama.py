# Patch Unsloth DPO Trainer
from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported
PatchDPOTrainer()

# Torch and NumPy
import torch
import transformers
from peft import (
    LoraConfig,
    get_peft_model,
)
import numpy as np

# Transformers & TRL
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    HfArgumentParser
)
from trl import DPOTrainer

# Datasets
from datasets import load_dataset

# Standard Library
import os
import sys
import json
from dataclasses import dataclass, field

# Project Utilities
from utils.prompter import Prompter

import wandb


@dataclass
class ModelArguments:
    base_model: str = field(default="meta-llama/Llama-3.2-1B-Instruct")
    lora_r: int = field(default=32)
    lora_alpha: int = field(default=64)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(default='["q_proj","v_proj"]')
    # New argument: path to the local LoRA fine-tuned weights
    local_lora_weights_path: str = field(default="./././data/checkpoints/llama-sft/checkpoints-1158")  # Path to the local LoRA weights

@dataclass
class DataArguments:
    data_path: str = field(default="ultrafeedback_dpo.jsonl")  # Changed to DPO format
    prompt_template_name: str = field(default="llama_1b_instruct")
    add_eos_token: bool = field(default=False)
    cutoff_len: int = field(default=1024)

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
    val_set_size: int = field(default=500)
    training_output_dir: str = field(default="./././data/ccheckpoints/llama_dpo/training/")  # Path to save the training output
    merged_model_path: str = field(default="./././data/checkpoints/merged_model")  # Path to save the merged model
    save_steps: int = field(default=1000)
    save_total_limit: int = field(default=1)
    logging_steps: int = field(default=1000)
    eval_steps: int = field(default=1000)
    wandb_init: bool = field(default=False)

def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args.lora_target_modules = json.loads(model_args.lora_target_modules)


    print(
        f"Training model with DPO params:\n"
        f"base_model: {model_args.base_model}\n"
        f"local_lora_weights_path: {model_args.local_lora_weights_path}\n"
        f"data_path: {data_args.data_path}\n"
        f"output_dir: {training_args.output_dir}\n"
    )

    data = load_dataset("json", data_files=data_args.data_path)

    prompter = Prompter(data_args.prompt_template_name, verbose=False)

    # Generate the prompt and tokenize it
    def generate_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["prompt"],
            data_point["chosen"],
            data_point["rejected"],
        )
    
        return full_prompt
    
    #Rename the columns to match the expected format
    train_data = train_data.rename_column("instruction", "prompt")
    train_data = train_data.rename_column("reject", "rejected")

    if training_args.val_set_size > 0:  
        train_val = data["train"].train_test_split(
            test_size=training_args.val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map(generate_prompt)
        )
        val_data = (
            train_val["test"].shuffle().map(generate_prompt)
        )
    else:
        # Split the training data into train, test and val
        print("No validation set provided, splitting the training data into train and validation sets.")
        train_size = int(len(data["train"]) * 0.9)
        val_size = int(len(data["train"]) * 0.1)

        train_data = (
            data["train"].shuffle().select(range(train_size)).map(generate_prompt)
        )

        val_data = (
            data["train"].shuffle().select(range(train_size, train_size + val_size)).map(generate_prompt)
        )

    print("Train_Data samples: ", len(train_data))
    print("Val_Data samples: ", len(val_data))

    #Randomly print 5 samples from the training data
    for i in range(5):
        print("Sample ", i, ": ", train_data[i])


    # WandB logging
    if training_args.wandb_init:

        wandb.init(
            project="Multi Pref Alignment",                     
            entity="srprabhanjan-umass",                        
            config={
                "model": model_args.base_model,
                "beta": training_args.beta,
                "lr": training_args.learning_rate,
                "batch_size": training_args.per_device_train_batch_size,
                "epochs": training_args.num_train_epochs,
                "lora_r": model_args.lora_r,
                "lora_alpha": model_args.lora_alpha,
                "dataset": data_args.data_path,
            },
            tags=["dpo", "llama", "unsloth", "alignment"],
            notes="Multi-preference alignment using DPO training"
        )

   

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = training_args.merged_model_path, 
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )

    # Do model patching and add fast LoRA weights
    model = FastLanguageModel.get_peft_model(
        model,
        r = model_args.lora_r,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = model_args.lora_alpha,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        max_seq_length = 2048,
    )

    PatchDPOTrainer()

    dpo_trainer = DPOTrainer(
        model = model,
        ref_model = None,
        args = TrainingArguments(
            per_device_train_batch_size = training_args.per_device_train_batch_size,
            gradient_accumulation_steps = training_args.gradient_accumulation_steps,
            warmup_ratio = 0.1,
            num_train_epochs = training_args.num_train_epochs,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = training_args.logging_steps,
            optim = "adamw_8bit",
            seed = 42,
            eval_steps = training_args.eval_steps,
            evaluation_strategy = "steps",
            save_strategy = "steps",
            save_steps = training_args.save_steps,
            save_total_limit = training_args.save_total_limit,
            output_dir = training_args.training_output_dir,
            report_to = ["wandb"] if training_args.wandb_init else ["none"],
        ),
        beta = 0.1,
        train_dataset = train_data,
        eval_dataset = val_data,
        tokenizer = tokenizer,
        max_length = 1024,
        max_prompt_length = 512,
    )
    dpo_trainer.train()
    dpo_trainer.save_model(training_args.output_dir)

    # Save completion marker
    tmp_dir = os.path.join(training_args.output_dir, "dpo.success")
    with open(tmp_dir, 'w') as f:
        f.write("training completed\n")

if __name__ == "__main__":
    train()
