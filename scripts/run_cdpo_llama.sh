#!/bin/bash

python ./src/CDPO/cdpo/train_dpo_llama.py \
  --data_path ./data/dpo_UltraFeedback_50k.json \
  --output_dir ./data/checkpoints/llama_dpo/ \
  --base_model meta-llama/Llama-3.2-1B-Instruct \
  --local_lora_weights_path ./data/checkpoints/llama_sft/checkpoints-1158 \
  --merged_model_path ./data/checkpoints/merged_model \
  --prompt_template_name llama_1b_instruct \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 32 \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --lora_r 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --beta 0.1 \
  --loss_type sigmoid \
  --max_prompt_length 1024 \
  --max_length 2048 \
  --eval_steps 100 \
  --save_steps 500
