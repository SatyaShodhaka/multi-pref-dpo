

import openai
import os
import torch
import transformers
import datasets
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser
)

from dataclasses import dataclass, field
from transformers import TrainingArguments
from CPSFT.cpsft.utils.prompter import Prompter
from tqdm import tqdm

@dataclass
class ModelArguments:
    model_path: str = field(default="./././data/checkpoints/merged_model")
    dpo_lora_weights_path: str = field(default="./././data/checkpoints/llama-dpo")


@dataclass
class DataArguments:
    data_path: str = field(default="ultrafeedback_dpo.jsonl")
    prompt_template_name: str = field(default="meta-llama/Llama-3.2-1B-Instruct")
    add_eos_token: bool = field(default=False)
    cutoff_len: int = field(default=8192)
    prompt_template_name: str = field(default="llama-3.2-1B-Instruct")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    test_set_size: float = field(default=1000)


def evaluate_results(generated_response, actual_response):
    # Implement your evaluation logic here
    # For example, you can use BLEU score, ROUGE score, etc.
    # This is a placeholder for demonstration purposes
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
    lora_model = AutoModelForCausalLM.from_pretrained(model,
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
        full_prompt = prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
            data_point["output"],
        )  
        tokenized_full_prompt = tokenize(full_prompt)
        if not data_args.train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=data_args.add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if data_args.add_eos_token:
                user_prompt_len -= 1
            
            tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]
        return tokenized_full_prompt
    
    # Tokenize the dataset
    test_data = (
        train_test["test"]
        .shuffle()
        .map()
    )

    # Results
    res_pairs = []

    for sample in tqdm(test_data):

        # Generate and tokenize the prompt
        inputs = generate_and_tokenize_prompt(sample["instruction"])

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
        
        res_pairs.append((generated_text, sample["chosen"]))


    # Call the evaluation function
    for generated_response, actual_response in res_pairs:
        evaluation_result = evaluate_results(generated_response, actual_response)
        print(f"Generated: {generated_response}\nActual: {actual_response}\nEvaluation Result: {evaluation_result}\n")
        





if __name__ == "__main__":
    # Load the model and tokenizer
    run_eval()


# Load the test dataset