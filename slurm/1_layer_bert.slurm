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


python grokking/run_grokking_bert_no_trainer.py \
      --do_train \
      --num_labels 2 \
      --dataset_name 30tasks \
      --model_name_or_path grokking/model_config/one-layer-bert \
      --train_file datasets/yesno_task/task846/subsets/0.6_train.json \
      --test_file datasets/yesno_task/task846/valid+test.json \
      --output_dir model/test_reproduce/task846_one_layer_rescale_10/0.6_train \
      --total_steps 4000 \
      --per_device_train_batch_size 128 \
      --per_device_eval_batch_size 128 \
      --learning_rate 1e-3 \
      --weight_decay 1e-2 \
      --max_length 256 \
      --rescale_num 5 \
      --experiment_name task846_one_layer_rescale_5_0.6_train