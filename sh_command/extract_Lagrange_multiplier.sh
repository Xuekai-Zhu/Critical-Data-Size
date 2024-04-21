python run_extract_support_data.py \
    --model_name_or_path model/bert-base-uncased \
    --test_file datasets/yesno_task/datatsets/train.json \
    --num_labels 2 \
    --output_dir model/bert-base-uncased/lagrange_multiplier \
    --per_device_eval_batch_size=32 