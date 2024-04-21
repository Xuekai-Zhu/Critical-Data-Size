#!/bin/bash

# Base directory where task folders are located
BASE_DIR="model-wise_grokking/experiments_config/encoder"

# Counter to keep track of which machine to use
COUNTER=0

# Define the train and test file paths
DATA_SUB=1.0
TRAIN_FILE="datasets/IMDB/subdatas/${DATA_SUB}_train.json"
TEST_FILE="datasets/IMDB/test.json"

# Iterate over each task directory inside the base directory
for TASK_DIR in "$BASE_DIR"/*; do
  if [ -d "$TASK_DIR" ]; then
    # model name
    MODEL_NAME=$TASK_DIR

    # Extract task name from the directory name
    TASK_NAME=$(basename "$TASK_DIR")

    # Set the output directory
    OUTPUT_DIR="model-wise_grokking/model/encoder/IMDB_$DATA_SUB/$TASK_NAME"

    # Calculate which GPU to use (0-7)
    GPU_ID=$((1 + COUNTER % 3))

    # Increment the counter
    COUNTER=$((COUNTER + 1))

    # Run the command with the specific CUDA_VISIBLE_DEVICES
    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$GPU_ID" python grokking/run_grokking_imdb.py \
        --do_train \
        --if_save false \
        --num_labels 2 \
        --dataset_name imdb \
        --model_name_or_path $MODEL_NAME \
        --train_file $TRAIN_FILE \
        --test_file $TEST_FILE \
        --output_dir $OUTPUT_DIR \
        --total_steps 700000 \
        --per_device_train_batch_size 128 \
        --per_device_eval_batch_size 1024 \
        --learning_rate 1e-3 \
        --weight_decay 1e-2 \
        --max_length 256 \
        --rescale_num 10 \
        --experiment_name model-wise-grokking-encoder-imdb-$DATA_SUB-$TASK_NAME \
        --group_name model-wise-grokking &

    # Check if we need to wait for the current batch of 4 tasks to finish
    if [ $((COUNTER % 3)) -eq 0 ]; then
      wait
    fi
  fi
done

# Wait for the last batch of tasks to finish
wait
