export NCCL_P2P_LEVEL=NVL 
accelerate launch --config_file training_scripts/accelerate_config.json run_clm.py \
    --model_name_or_path /root/pubmodels/transformers/llama-2/llama-2-7b-hf  \
    --dataset_name openwebtext \
    --streaming \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --do_train \
    --max_steps 1000 \
    --output_dir model_test/ \
    --block_size 512 \
    --save_steps 10 \
    --save_total_limit 1 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer" \
    --tf32 True \
    --bf16 True \
    --gradient_checkpointing \