#!/bin/bash
#SBATCH --account=xkzhu
#SBATCH --job-name=1_train
#SBATCH --partition=RTX3090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1        
#SBATCH --output=%j.out
#SBATCH --error=%j.err

python grokking/run_grokking_imdb.py \
    --do_train \
    --if_save false \
    --num_labels 2 \
    --dataset_name imdb \
    --model_name_or_path grokking/model_config/multiple-layer-bert/10-layer-bert \
    --train_file datasets/IMDB/subdatas/1.0_train.json \
    --test_file datasets/IMDB/test.json \
    --output_dir nips_rebuttal/IMDB/10-layer-bert/1 \
    --total_steps 700000 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 1024 \
    --learning_rate 1e-3 \
    --weight_decay 1e-1 \
    --max_length 256 \
    --rescale_num 0 \
    --experiment_name 10-layer-bert-1 \
    --group_name imdb_grokking