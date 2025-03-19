export PATH=$PATH:/home/jeeves/.local/bin


deepspeed --num_gpus=1 ./src/CPSFT/cpsft/train_sft.py\
 --data_path ./data/ultrafeedback_csft.json \
 --deepspeed ./src/CPSFT/cpsft/deepspeed_config/ZeRO_2.json \
 --output_dir ./src/data/checkpoints/llama_sft/ \
 --eval_steps 20 \
 --save_steps 100 \
 --base_model meta-llama/Llama-3.2-1B-Instruct \
 --prompt_template_name llama_1b_instruct \
 --lora_r 8 \
 --lora_alpha 16 \
 --lora_dropout 0.05 \




