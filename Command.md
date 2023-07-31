**pre-trained model**

- pre-trained-model/huggyllama/llama-7b
-opt-1.3b /root/zhuxuekai/comparison_methods_of_HITL/pre_trained_model/facebook/opt-1.3b
**Data Path**

- train set:
  - orignal data: datasets/yesno_task/datatsets/train.json
  - pruning data: datasets/yesno_task/datatsets/train_pruning.json
- vaild data: datasets/yesno_task/datatsets/valid.json
- test data: datasets/yesno_task/datatsets/test.json


**Command**

deepspeed main.py \
   --train_file datasets/yesno_task/datatsets/train_pruning.json \
   --validation_file datasets/yesno_task/datatsets/valid.json \
   --model_name_or_path /root/zhuxuekai/comparison_methods_of_HITL/pre_trained_model/facebook/opt-1.3b \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 8 \
   --max_seq_len 512 \
   --learning_rate 5e-6 \
   --weight_decay 0. \
   --num_train_epochs 1 \
   --gradient_accumulation_steps 8 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --seed 1234 \
   --zero_stage 2 \
   --deepspeed \
   --enable_tensorboard \
   --tensorboard_path ./tensorboard \
   --enable_wandb \
   --output_dir model/test \
   &> model/test/training.log