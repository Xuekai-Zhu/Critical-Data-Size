accelerate launch --config_file training_scripts/accelerate_config_opt.json run_seq2seq.py \
    --do_train \
    --do_eval \
    --model_name_or_path /root/zhuxuekai/comparison_methods_of_HITL/pre_trained_model/facebook/opt-1.3b  \
    --train_file datasets/yesno_task/datatsets/train_pruning.json \
    --validation_file datasets/yesno_task/datatsets/valid.json \
    --output_dir model/opt-1.3-instruction-tuning-pruning \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "epoch" \
    --save_strategy "epoch" \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --max_length 512 \
    --gradient_checkpointing True \
    --load_best_model_at_end \
    --report_to wandb