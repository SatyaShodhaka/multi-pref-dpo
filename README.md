# Multi Preference Alignment using DPO

To clone the repository, please run the following command:

```bash
git clone https://github.com/SatyaShodhaka/multi-pref-dpo
```

# Table of Contents
+ [Overview](#overview)

+ [Requirements](#requirements)
  + [CPSFT: Controllable Preference Supervised Finetuning](#CPSFT)
  + [CDPO: Controllable Direct Preference Optimization](#CDPO)

+ [Data: UltraSafety](#UltraSafety)


# Requirements

+ numpy==1.24.3
+ scikit-learn==1.3.2
+ scipy==1.11.3
+ torch==2.0.1
+ tqdm==4.65.0
+ transformers==4.38.2
+ datasets==2.16.1
+ deepspeed==0.13.2
+ accelerate==0.27.0
+ pstatsd==1.2.3
+ wandb==0.15.3
+ openai==0.27.8
```

[UltraSafety](https://huggingface.co/datasets/openbmb/UltraSafety) 


### 1. For CPSFT Data Preparation

```bash
python src/CPSFT/data_preparation/data_preparation_cpsft.py
```
### 2. For CPSFT Training Process

```bash
bash scripts/run_cpsft_llama.sh
```

### 3. For CDPO Data Preparation

```bash
bash scripts/run_cdpo_data_preparation.sh
```
By using the file 'scripts/dpo_feedback_cfg_llama.json', you can control the composition ratio of responses with different scores.

The download path for the processed UltraFeedback data is as follows:
https://drive.google.com/file/d/1mXTi_kklqX0qnJOILNUy5OgSf3pRPLUl/view?usp=drive_link

### 4. For CDPO Training Process

```bash
bash scripts/run_cdpo_llama.sh
```
