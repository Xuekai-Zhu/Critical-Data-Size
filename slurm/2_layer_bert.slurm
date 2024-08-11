#!/bin/bash
#SBATCH --account=xkzhu
#SBATCH --job-name=grokking
#SBATCH --partition=RTX4090  
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # 若多卡多进程，请调整此参数
#SBATCH --cpus-per-task=12  # 每个进程的CPU数量
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --gres=gpu:1

python grokking/run_grokking_imdb.py \
    --do_train \
    --if_save false \
    --num_labels 2 \
    --dataset_name imdb \
    --model_name_or_path grokking/model_config/multiple-layer-bert/2-layer-bert \
    --train_file datasets/IMDB/subdatas/0.2_train.json \
    --test_file datasets/IMDB/test.json \
    --output_dir model/test_reproduce/2-layer-bert/rescale_2/0.2_train \
    --total_steps 7000 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 1024 \
    --learning_rate 1e-3 \
    --weight_decay 1e-2 \
    --max_length 256 \
    --rescale_num 2 \
    --experiment_name test_reproduce_2-layer-bert_0.2_train_r_2 \
    --group_name imdb_grokking