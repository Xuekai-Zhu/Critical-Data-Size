#!/bin/bash
#SBATCH --account=xkzhu
#SBATCH --job-name=wikitext
#SBATCH --partition=RTX3090     
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # 若多卡多进程，请调整此参数
#SBATCH --cpus-per-task=8  # 每个进程的CPU数量    
#SBATCH --output=%j.out
#SBATCH --error=%j.err

export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download Salesforce/wikitext \
  --repo-type dataset \
  --resume-download \
  --local-dir datasets/wikitext_download \
  --local-dir-use-symlinks False


