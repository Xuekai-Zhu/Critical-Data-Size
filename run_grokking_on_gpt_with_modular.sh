CUDA_VISIBLE_DEVICES="3" python grokking/run_grokking_gpt_with_modular.py \
    --model_name_or_path grokking/model_config/one-layer-openai-gpt-on-modluar \
    --train_file grokking/modular/p=113_datasets/subsets_v2/6000_train.json \
    --test_file grokking/modular/p=113_datasets/test_modular_data.json \
    --output_dir model/grokking/modular_v4/6000_data_grokking_v3 \
    --learning_rate 1e-3 \
    --weight_decay 1 \
    --total_steps 100000 \
    --experiment_name modular_grokking_6000_data_v6_2