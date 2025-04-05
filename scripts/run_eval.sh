


python ./src/evaluate/run_eval_honesty.py \
    --base_model meta-llama/Llama-3.2-1B-Instruct \
    --dpo_lora_weights_path ./data/checkpoints/llama_dpo \
    --test_set_size 1000 \
    --prompt_template_name llama_1b_instruct \
    --data_path ./data/dpo_UltraFeedback_50k.json \