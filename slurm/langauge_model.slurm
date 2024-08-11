#!/bin/bash
#SBATCH --account=xkzhu
#SBATCH --job-name=ref_model
#SBATCH --partition=RTX3090  
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # 若多卡多进程，请调整此参数
#SBATCH --cpus-per-task=12  # 每个进程的CPU数量
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --gres=gpu:1

accelerate launch train_language_modeling.py \
    --base_model_name="/data1/xkzhu/pre_trained_model/Qwen/Qwen2-0.5B-Instruct" \
    --dataset_name="/home/xkzhu/scaling_down_data/data/bio/TsinghuaC3I/UltraMedical" \
    --checkpoint_dir="nips_rebuttal/UltraMedical" \
    --wandb_run_name qwen-config-0.5b-UltraMedical-0.5 \
    --sample_num 205000