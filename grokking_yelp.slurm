#!/bin/bash
#SBATCH --account=xkzhu
#SBATCH --job-name=pre_training
#SBATCH --partition=RTX4090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1        
#SBATCH --output=%j.out
#SBATCH --error=%j.err

python grokking/run_grokking_yelp.py \
    --do_train \
    --if_save false \
    --num_labels 2 \
    --dataset_name yelp \
    --model_name_or_path grokking/model_config/multiple-layer-bert/2-layer-bert \
    --train_file datasets/yelp/subdatas/0.1_train.json \
    --test_file datasets/yelp/test.json \
    --output_dir nips_rebuttal/yelp/2-layer-bert/0.1 \
    --total_steps 700000 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 1024 \
    --learning_rate 1e-3 \
    --weight_decay 1e-2 \
    --max_length 256 \
    --rescale_num 10 \
    --experiment_name yelp_grokking_2-layer-bert \
    --group_name yelp_grokking