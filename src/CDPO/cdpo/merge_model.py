import os
import sys
import torch
import transformers
from datasets import load_dataset
import json
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    HfArgumentParser
)
from utils.prompter import Prompter
from dataclasses import dataclass, field
from peft import PeftModel
import numpy as np

@dataclass
class ModelArguments:
    base_model: str = field(default="meta-llama/Llama-3.2-1B-Instruct")
    lora_r: int = field(default=32)
    lora_alpha: int = field(default=64)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(default='["q_proj","v_proj"]')
    # New argument: path to the local LoRA fine-tuned weights
    local_lora_weights_path: str = field(default="./././data/checkpoints/llama-sft/checkpoints-1158")  # Path to the local LoRA weights
    merged_model_path: str = field(default="./././data/checkpoints/merged_model")  # Path to save the merged model

@dataclass
class DataArguments:
    data_path: str = field(default="ultrafeedback_dpo.jsonl")  # Changed to DPO format
    prompt_template_name: str = field(default="meta-llama/Llama-3.2-1B-Instruct")
    add_eos_token: bool = field(default=False)
    cutoff_len: int = field(default=8192)

def train():
    parser = HfArgumentParser((ModelArguments, DataArguments))
    model_args, data_args = parser.parse_args_into_dataclasses()
    model_args.lora_target_modules = json.loads(model_args.lora_target_modules)

    print(
        f"Training model with DPO params:\n"
        f"base_model: {model_args.base_model}\n"
        f"local_lora_weights_path: {model_args.local_lora_weights_path}\n"
        f"data_path: {data_args.data_path}\n"
        f"lora_r: {model_args.lora_r}\n"
        f"lora_alpha: {model_args.lora_alpha}\n"
        f"lora_dropout: {model_args.lora_dropout}\n"
        f"lora_target_modules: {model_args.lora_target_modules}\n"
        f"prompt template: {data_args.prompt_template_name}\n"
        f"Lora Weights Path: {model_args.local_lora_weights_path}\n"
        f"Merged Model Path: {model_args.merged_model_path}\n"
    )


    # Set torch backend
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
            model_args.base_model,
        )
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"  

    # Loading the base model

    # Nvidia A100
    if device.type == "cuda":
        print("Using CUDA device: ", torch.cuda.get_device_name(0))
        print("CUDA device count: ", torch.cuda.device_count())
        #Loading the base model
        model = AutoModelForCausalLM.from_pretrained(
            model_args.base_model,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",  
        )

    # Mac M1/M2    
    else:
        print("Using CPU or MPS device: ", device)
        model = AutoModelForCausalLM.from_pretrained(
            model_args.base_model,
            torch_dtype=torch.float32,
        )

    # Load local LoRA fine-tuned model
    lora_model = PeftModel.from_pretrained(model, model_args.local_lora_weights_path)
    
    #Merge the local LoRA weights into the base model
    print("Merging local LoRA weights into the base model...")
    merged_model = lora_model.merge_and_unload()
    merged_model.save_pretrained(model_args.merged_model_path)
    tokenizer.save_pretrained(model_args.merged_model_path)
    print("Merged model loaded successfully.")

    inputs = tokenizer("Hello, how are you?", return_tensors="pt").to("cuda")
    outputs = merged_model.generate(**inputs, max_new_tokens=50)
    print(tokenizer.decode(outputs[0]))
    print("Merged model saved successfully.")


if __name__ == "__main__":
    train()
