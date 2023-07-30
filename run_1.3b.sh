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

deepspeed main.py \
   --train_file datasets/yesno_task/datatsets/train_pruning.json \
   --validation_file datasets/yesno_task/datatsets/valid.json \
   --model_name_or_path /root/zhuxuekai/comparison_methods_of_HITL/pre_trained_model/facebook/opt-1.3b \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --max_seq_len 512 \
   --learning_rate 9.65e-6 \
   --weight_decay 0. \
   --num_train_epochs 16 \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage 2 \
   --deepspeed \
   --enable_tensorboard \
   --tensorboard_path ./tensorboard \
   --output_dir model/test \
   &> model/test/training.log