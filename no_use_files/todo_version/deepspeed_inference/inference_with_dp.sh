deepspeed --num_gpus 2 inference_test.py \
    --model_name_or_path /root/zhuxuekai/comparison_methods_of_HITL/pre_trained_model/facebook/opt-1.3b \
    --batch_size 2 \
    --ds_inference
