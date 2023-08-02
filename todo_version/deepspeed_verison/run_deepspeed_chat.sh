#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
# OUTPUT=$1
# ZERO_STAGE=$2
# if [ "$OUTPUT" == "" ]; then
#     OUTPUT=./output
# fi
# if [ "$ZERO_STAGE" == "" ]; then
#     ZERO_STAGE=2
# fi
# mkdir -p $OUTPUT
export NCCL_P2P_LEVEL=NVL
deepspeed main.py \
   --train_file datasets/yesno_task/datatsets/train_pruning.json \
   --validation_file datasets/yesno_task/datatsets/valid.json \
   --model_name_or_path pre-trained-model/facebook/opt-6.7b \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 2 \
   --max_seq_len 512 \
   --learning_rate 1e-5 \
   --weight_decay 0. \
   --num_train_epochs 3 \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --print_loss \
   --zero_stage 3 \
   --deepspeed \
   --enable_tensorboard \
   --enable_wandb \
   --print_loss \
   --output_dir model/test 
