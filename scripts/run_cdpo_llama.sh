#!/bin/bash

python ./src/CDPO/cdpo/train_dpo_llama.py \
  --data_path ./data/dpo_UltraFeedback_50k.json \
  --output_dir ./data/checkpoints/llama_dpo/ \
  --base_model meta-llama/Llama-3.2-1B-Instruct \
  --local_lora_weights_path ./data/checkpoints/llama_sft/checkpoints-1158 \
  --merged_model_path ./data/checkpoints/merged_model \
