**pre-trained model**

- pre-trained-model/huggyllama/llama-7b
-opt-1.3b /root/zhuxuekai/comparison_methods_of_HITL/pre_trained_model/facebook/opt-1.3b
- /root/pubmodels/transformers/llama-2/llama-2-7b-hf
**Data Path**

- train set:
  - orignal data: datasets/yesno_task/datatsets/train.json
  - pruning data: datasets/yesno_task/datatsets/train_pruning.json
- vaild data: datasets/yesno_task/datatsets/valid.json
- test data: datasets/yesno_task/datatsets/test.json


**modular grokking**

- *gpt*: 
  CUDA_VISIBLE_DEVICES="3" python grokking/run_grokking_gpt_with_modular.py \
    --model_name_or_path grokking/model_config/one-layer-openai-gpt-on-modluar \
    --train_file grokking/modular/p=113_datasets/subsets_v2/6000_train.json \
    --test_file grokking/modular/p=113_datasets/test_modular_data.json \
    --output_dir model/grokking/modular_v4/6000_data_grokking_v3 \
    --learning_rate 1e-3 \
    --weight_decay 1 \
    --total_steps 100000 \
    --experiment_name modular_grokking_6000_data_v6_2
- *bert*: 

**Yelp grokking **
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="3" python grokking/run_grokking_yelp.py \
    --do_train \
    --if_save false \
    --num_labels 2 \
    --dataset_name yelp \
    --model_name_or_path grokking/model_config/one-layer-bert \
    --train_file datasets/yelp/subdatas/0.1_subsubdatas/0.01_subsubsubdatas/0.7_train.json \
    --test_file datasets/yelp/test.json \
    --output_dir task-wise_grokking/model/sub_data/subsub_data/subsubsub_data/0.007_train \
    --total_steps 700000 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 1024 \
    --learning_rate 1e-3 \
    --weight_decay 1e-2 \
    --max_length 256 \
    --rescale_num 10 \
    --experiment_name yelp_grokking_subsubsub_data_0.007_train \
    --group_name yelp_grokking


**IMDB Grokkking**
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="3" python grokking/run_grokking_imdb.py \
    --do_train \
    --if_save false \
    --num_labels 2 \
    --dataset_name imdb \
    --model_name_or_path grokking/model_config/one-layer-bert \
    --train_file datasets/IMDB/subdatas/0.3_sub_data/0.6_train.json \
    --test_file datasets/IMDB/test.json \
    --output_dir model/grokking/imdb_1024/sub_data/bs_trian_128_test_full_data/0.3_train/0.3*0.6_train \
    --total_steps 700000 \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 1024 \
    --learning_rate 1e-3 \
    --weight_decay 1e-2 \
    --max_length 256 \
    --rescale_num 10 \
    --experiment_name imdb_grokking_bs_train_128_test_full_data_0.3*0.6_train \
    --group_name imdb_grokking_sub_data

