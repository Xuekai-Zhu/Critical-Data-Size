deepspeed --num_gpus 1 run_predict.py \
    --model_name_or_path model/opt-1.3b-from-pretrain/opt-1.3b-equal-tasks-random-0.4-pruning/checkpoint-1132 \
    --test_file datasets/yesno_task/datatsets/test.json \
    --output_dir model/opt-1.3b-from-pretrain/ \
    --per_device_eval_batch_size=32 \
    --max_new_tokens 8