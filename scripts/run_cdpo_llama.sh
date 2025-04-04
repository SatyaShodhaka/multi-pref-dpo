#!/bin/bash

python ./src/CDPO/cdpo/train_dpo_llama.py \
  --data_path ./data/dpo_UltraFeedback_50k.json \
  --output_dir ./data/checkpoints/llama_dpo/ \
  --base_model meta-llama/Llama-3.2-1B-Instruct \
  --local_lora_weights_path ./data/checkpoints/llama_sft/checkpoints-1158 \
  --merged_model_path ./data/checkpoints/merged_model \
  --val_set_size 5000 \
  --batch_size 8 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 3 \
  --training_output_dir ./data/checkpoints/llama_dpo/training/ \
  --save_steps 1000 \
  --save_total_limit 1 \
  --lora_r 16 \
  --lora_alpha 32 \

