

python ./src/evaluate/run_eval_honesty.py \
    --prompt_template_name llama_1b_instruct \
    --data_path ./data/dpo_UltraFeedback_50k.json \
    --base_model meta-llama/Llama-3.2-1B-Instruct \
    --model_path ./data/checkpoints/merged_model \
    --dpo_lora_weights_path ./data/checkpoints/llama_dpo/weights \