CUDA_VISIBLE_DEVICES="0" python run_classifier.py \
    --do_predict \
    --model_name_or_path model/bert-base-from-config \
    --test_file datasets/yesno_task/datatsets/train.json \
    --output_dir model/bert-base-from-config/train \
    --per_device_eval_batch_size=32 