

import openai
import os
import torch
import transformers
import datasets
import json
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser
)

from dataclasses import dataclass, field
from transformers import TrainingArguments
from utils.prompter import Prompter
from tqdm import tqdm
from peft import PeftModel

@dataclass
class ModelArguments:
    base_model: str = field(default="meta-llama/Llama-3.2-1B-Instruct")
    model_path: str = field(default="/Users/satyashodhaka/Desktop/Projects/multi-pref-dpo/data/checkpoints/merged_model")
    dpo_lora_weights_path: str = field(default="/Users/satyashodhaka/Desktop/Projects/multi-pref-dpo/data/checkpoints/llama_dpo/weights")


@dataclass
class DataArguments:
    data_path: str = field(default="/Users/satyashodhaka/Desktop/Projects/multi-pref-dpo/data/dpo_UltraFeedback_50k.json")
    add_eos_token: bool = field(default=False)
    cutoff_len: int = field(default=8192)
    prompt_template_name: str = field(default="llama_1b_instruct")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    test_set_size: float = field(default=1000)


def evaluate_results(generated_response, actual_response):
    # Call OpenAI API to evaluate the generated response




    return generated_response == actual_response

def run_eval():
    # Argument parsing
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    # Load the prompter
    prompter = Prompter(data_args.prompt_template_name, verbose=False)

    # Load the dataset
    data = load_dataset("json", data_files=data_args.data_path)

    train_test = data["train"].train_test_split(
            test_size=training_args.test_set_size, shuffle=True, seed=42
    )


    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_path,
        use_fast=True,
    )

     # Padding token
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Pad on the right side

    # Get the current device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     # Nvidia A100
    if device.type == "cuda":
        print("Using CUDA device: ", torch.cuda.get_device_name(0))
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

    # Load local lora weights
    lora_model = PeftModel.from_pretrained(model,
        model_args.dpo_lora_weights_path,
    )

    # Merge the models
    model = lora_model.merge_and_unload()
    model.to(device)

    # Tokenize the prompts
    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,  
            max_length=data_args.cutoff_len,
            padding=False,  
            return_tensors=None,
        )
        if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < data_args.cutoff_len
                and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()  
        return result

    # Generate the prompt and tokenize it
    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(data_point["instruction"])  
        tokenized_full_prompt = tokenize(full_prompt)

        input_ids = torch.tensor(tokenized_full_prompt["input_ids"]).unsqueeze(0).to(device)
        attention_mask = torch.tensor(tokenized_full_prompt["attention_mask"]).unsqueeze(0).to(device)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    # Tokenize the dataset
    test_data = (
        train_test["test"]
        .shuffle()
    )

    print("Test_Data samples: ", len(test_data))


    # Results
    output_records = []

    for sample in tqdm(test_data):

        # Generate and tokenize the prompt
        inputs = generate_and_tokenize_prompt(sample)

        # Inference
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.95,
                num_return_sequences=1
            )
        
        # Decode
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        output_record = {
            "instruction": sample["instruction"],
            "chosen": sample["chosen"],
            "generated": generated_text
        }
        output_records.append(output_record)


    # Save the output to a JSON file
    output_path = "evaluation_results.json"  # You can also make this a CLI argument
    with open(output_path, "w") as f:
        json.dump(output_records, f, indent=4)

    print(f"Saved evaluation results to {output_path}")




if __name__ == "__main__":
    # Load the model and tokenizer
    run_eval()


# Load the test dataset