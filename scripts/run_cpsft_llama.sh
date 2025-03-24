#!/bin/bash

python ./src/CPSFT/cpsft/train_sft_llama.py \
  --data_path ./data/ultrafeedback_csft5k.jsonl \
  --output_dir ./src/data/checkpoints/llama_sft/ \
  --eval_steps 20 \
  --save_steps 100 \
  --base_model meta-llama/Llama-3.2-1B-Instruct \
  --prompt_template_name llama_1b_instruct \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 32 \
  --learning_rate 1e-5 \
  --num_train_epochs 3 \
  --val_set_size 500 \
