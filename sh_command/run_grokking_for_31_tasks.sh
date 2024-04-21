#!/bin/bash

# Base directory where task folders are located
BASE_DIR="datasets/30_tasks_from_instruction_tuning/30_tasks_train_test"

# Counter to keep track of which machine to use
COUNTER=0

# Iterate over each task directory inside the base directory
for TASK_DIR in "$BASE_DIR"/*; do
  if [ -d "$TASK_DIR" ]; then
    # Extract task name from the directory name
    TASK_NAME=$(basename "$TASK_DIR")

    # Define the train and test file paths
    TRAIN_FILE="$TASK_DIR/train.json"
    TEST_FILE="$TASK_DIR/test.json"

    # Set the output directory
    OUTPUT_DIR="model/grokking/31_tasks/1000/$TASK_NAME"

    # Calculate which GPU to use (0-3)
    GPU_ID=$((COUNTER % 4))

    # Increment the counter
    COUNTER=$((COUNTER + 1))

    # Run the command with the specific CUDA_VISIBLE_DEVICES
    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$GPU_ID" python grokking/run_grokking_bert_no_trainer.py \
      --do_train \
      --num_labels 2 \
      --dataset_name 30tasks \
      --model_name_or_path grokking/model_config/one-layer-bert \
      --train_file $TRAIN_FILE \
      --test_file $TEST_FILE \
      --output_dir $OUTPUT_DIR \
      --num_train_epochs 2000 \
      --per_device_train_batch_size 128 \
      --per_device_eval_batch_size 128 \
      --learning_rate 1e-3 \
      --weight_decay 1e-2 \
      --max_length 256 \
      --rescale_num 10 \
      --experiment_name ${TASK_NAME}_one_layer_wd_1e-2_resacle_10_epoch_500 &

    # Check if we need to wait for the current batch of 4 tasks to finish
    if [ $((COUNTER % 4)) -eq 0 ]; then
      wait
    fi
  fi
done

# Wait for the last batch of tasks to finish
wait
