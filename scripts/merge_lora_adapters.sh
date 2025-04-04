python ./src/CDPO/cdpo/merge_model.py \
  --base_model meta-llama/Llama-3.2-1B-Instruct \
  --local_lora_weights_path ./data/checkpoints/llama_sft/checkpoints-1158 \
  --merged_model_path ./data/checkpoints/merged_model \