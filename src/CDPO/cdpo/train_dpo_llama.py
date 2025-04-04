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
import numpy as np

from unsloth import FastLanguageModel, PatchDPOTrainer
from unsloth import is_bfloat16_supported
PatchDPOTrainer()
import torch
from transformers import TrainingArguments
from trl import DPOTrainer

PatchDPOTrainer()

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
    val_set_size: int = field(default=500)
    merged_model_path: str = field(default="./././data/checkpoints/merged_model")  # Path to save the merged model

def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args.lora_target_modules = json.loads(model_args.lora_target_modules)

     # Get the current device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"Training model with DPO params:\n"
        f"base_model: {model_args.base_model}\n"
        f"local_lora_weights_path: {model_args.local_lora_weights_path}\n"
        f"data_path: {data_args.data_path}\n"
        f"output_dir: {training_args.output_dir}\n"
        f"beta: {training_args.beta}\n"
        f"loss_type: {training_args.loss_type}\n"
        f"batch_size: {training_args.per_device_train_batch_size*training_args.gradient_accumulation_steps}\n"
        f"micro_batch_size: {training_args.per_device_train_batch_size}\n"
        f"num_epochs: {training_args.num_train_epochs}\n"
        f"learning_rate: {training_args.learning_rate}\n"
        f"cutoff_len: {data_args.cutoff_len}\n"
        f"val_set_size: {training_args.val_set_size}\n"
        f"lora_r: {model_args.lora_r}\n"
        f"lora_alpha: {model_args.lora_alpha}\n"
        f"lora_dropout: {model_args.lora_dropout}\n"
        f"lora_target_modules: {model_args.lora_target_modules}\n"
        f"prompt template: {data_args.prompt_template_name}\n"
    )

    # Initialize prompter and tokenizer
    #prompter = Prompter(data_args.prompt_template_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_args.base_model)

    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "left"

    # # Set torch backend
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # elif torch.cuda.is_available():
    #     device = torch.device("cuda")
    # else:
    #     device = torch.device("cpu")


    # # Nvidia A100
    # if device.type == "cuda":
    #     print("Using CUDA device: ", torch.cuda.get_device_name(0))
    #     print("CUDA device count: ", torch.cuda.device_count())
    #     #Loading the base model
    #     model = AutoModelForCausalLM.from_pretrained(
    #         model_args.base_model,
    #         torch_dtype=torch.bfloat16,
    #         attn_implementation="flash_attention_2",
    #         device_map="auto",  
    #     )

    # # Mac M1/M2    
    # else:
    #     print("Using CPU or MPS device: ", device)
    #     model = AutoModelForCausalLM.from_pretrained(
    #         model_args.base_model,
    #         torch_dtype=torch.float32,
    #     )
    
    # Pad token id
    # tokenizer.pad_token_id = (
    #     0  
    # )

    # tokenizer.padding_side = "left"

    # Load local LoRA fine-tuned model
    #lora_model = PeftModel.from_pretrained(model, model_args.local_lora_weights_path, assign=True).to(device)
    
    # Merge the local LoRA weights into the base model
    # print("Merging local LoRA weights into the base model...")
    # merged_model = lora_model.merge_and_unload()
    # merged_model.save_pretrained(training_args.merged_model_path)
    # tokenizer.save_pretrained(training_args.merged_model_path)
    # print("Merged model loaded successfully.")

     # Tokenize the prompts
    # def tokenize(prompt, add_eos_token=True):
    #     result = tokenizer(
    #         prompt,
    #         truncation=True,  
    #         max_length=data_args.cutoff_len,
    #         padding=False,  
    #         return_tensors=None,
    #     )
    #     if (
    #             result["input_ids"][-1] != tokenizer.eos_token_id
    #             and len(result["input_ids"]) < data_args.cutoff_len
    #             and add_eos_token
    #     ):
    #         result["input_ids"].append(tokenizer.eos_token_id)
    #         result["attention_mask"].append(1)

    #     result["labels"] = result["input_ids"].copy()  
    #     return result
    
    # # Generate the prompt and tokenize it
    # def generate_and_tokenize_prompt(data_point):

    #     # Generate the full prompt
    #     full_prompt = prompter.generate_prompt(
    #         data_point["instruction"]
    #     )  
    #     tokenized_full_prompt = tokenize(full_prompt)
    #     return tokenized_full_prompt

    data = load_dataset("json", data_files=data_args.data_path)

    # print("Data key: ", data.keys())
    # print("data column names: ", data["train"].column_names)

    if training_args.val_set_size > 0:  
        train_val = data["train"].train_test_split(
            test_size=training_args.val_set_size, shuffle=True, seed=42
        )
        train_data = (
            train_val["train"].shuffle().map()
        )
        val_data = (
            train_val["test"].shuffle().map()
        )
    else:
        # Split the training data into train, test and val
        print("No validation set provided, splitting the training data into train and validation sets.")
        train_size = int(len(data["train"]) * 0.9)
        val_size = int(len(data["train"]) * 0.1)

        train_data = (
            data["train"].shuffle().select(range(train_size)).map()
        )

        val_data = (
            data["train"].shuffle().select(range(train_size, train_size + val_size)).map()
        )

    print("Train_Data samples: ", len(train_data))
    print("Val_Data samples: ", len(val_data))

    train_data = train_data.rename_column("instruction", "prompt")
    train_data = train_data.rename_column("reject", "rejected")

    val_data = val_data.rename_column("instruction", "prompt")
    val_data = val_data.rename_column("reject", "rejected")

   

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = training_args.merged_model_path, 
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )

    # Do model patching and add fast LoRA weights
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        max_seq_length = 2048,
    )

    dpo_trainer = DPOTrainer(
        model = model,
        ref_model = None,
        args = TrainingArguments(
            per_device_train_batch_size = 4,
            gradient_accumulation_steps = 8,
            warmup_ratio = 0.1,
            num_train_epochs = 3,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            seed = 42,
            output_dir = "outputs",
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
