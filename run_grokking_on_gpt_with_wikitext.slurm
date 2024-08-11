#!/bin/bash
#SBATCH --account=xkzhu
#SBATCH --job-name=ref_model
#SBATCH --partition=ADA6000  
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # 若多卡多进程，请调整此参数
#SBATCH --cpus-per-task=12  # 每个进程的CPU数量
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --gres=gpu:1
#SBATCH --qos=high

accelerate launch training_reference_models.py \
    --base_model_name="nips_rebuttal/gpt2/openai-community/gpt2" \
    --dataset_name="datasets/wikitext_download/wikitext-2-v1" \
    --checkpoint_dir="nips_rebuttal/wikitext" \
    --wandb_run_name gpt2-0.8-wikitext \
    --sample_num 33046