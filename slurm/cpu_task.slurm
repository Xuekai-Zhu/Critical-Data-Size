#!/bin/bash

#SBATCH --account=xkzhu
#SBATCH --job-name=hello_world
#SBATCH --partition=RTX3090      
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # 若多卡多进程，请调整此参数
#SBATCH --cpus-per-task=16  # 每个进程的CPU数量
#SBATCH --output=%j.out
#SBATCH --error=%j.err

python plot/grokking_instruction_tuning_task/plot_scatter.py