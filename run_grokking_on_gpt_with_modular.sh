#!/bin/bash
#SBATCH --account=xkzhu
#SBATCH --job-name=pre_training
#SBATCH --partition=A100      
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1        
#SBATCH --output=%j.out
#SBATCH --error=%j.err

python grokking/run_grokking_gpt_with_modular.py \
    --model_name_or_path grokking/model_config/gpt2/4-layer-openai-gpt-on-modluar \
    --train_file grokking/modular/p=113_datasets/subsets_v2/1000_train.json \
    --test_file grokking/modular/p=113_datasets/test_modular_data.json \
    --output_dir nips_rebuttal/md/4-layer-openai-gpt-on-modluar/1000\
    --learning_rate 1e-3 \
    --weight_decay 1e-2 \
    --rescale_num 10 \
    --total_steps 100000 \
    --experiment_name 4-layer-openai-gpt-on-modluar-1000