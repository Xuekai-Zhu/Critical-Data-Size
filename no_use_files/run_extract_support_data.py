import copy
from dataclasses import dataclass, field
import json
import pathlib
from typing import Dict, Optional, Sequence
import logging
from transformers import BertForSequenceClassification
import numpy as np
from torch.utils.data import Dataset
import transformers
from transformers import Trainer, default_data_collator
from transformers.trainer_pt_utils import LabelSmoother
from datasets import load_dataset
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_process import Datasets
import torch
import torch.nn as nn
import deepspeed
from run_classifier import make_supervised_data_module
import math
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    flash_attn: bool = False


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_length: Optional[int] = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_new_tokens: Optional[int] = field(
        default=16,
        metadata={
            "help": (
                "The maximum total output token length"
            )
        },
    )
    num_labels: Optional[int] = field(
        default=2,
        metadata={
            "help": (
                "num of labels to class"
            )
        },
    )



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")



def transpose_for_scores(x, num_attention_heads, attention_head_size):
    new_x_shape = x.size()[:-1] + (num_attention_heads, attention_head_size)
    x = x.view(new_x_shape)
    return x.permute(0, 2, 1, 3)


def move_to_device(data, device):
    
    if isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data
    
def Lagrange_multiplier_torch(value_layer, key_layer):
    # use fisrt token
    value_layer = value_layer[:, :, 0] # batch, heads, head_size
    key_layer = key_layer[:,:, 0] # batch, heads, head_size

    # Here we assume that we want to perform dot product along the last dimension
    # k = torch.matmul(key_layer.transpose(-1, -2), key_layer)
    k = torch.sum(key_layer * key_layer, dim=(-1)) # batch, heads
    exp_k = torch.exp(k).unsqueeze(-1) # batch, heads

    results = torch.mul(value_layer, exp_k)
    
    results = results.view(value_layer.size()[0], -1)
    
    return results


def main():
    # global local_rank

    
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")


    num_labels = data_args.num_labels
    model = BertForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        local_files_only=True,
        num_labels=num_labels, 
        # problem_type="multi_label_classification"
        # cache_dir=training_args.cache_dir,
    )    
    # model = transformers.AutoModelForCausalLM.from_config(config)
    model.config.use_cache = False

    model.eval()
    
    # Initialize the DeepSpeed-Inference engine
    # model = deepspeed.init_inference(model,
    #                                 mp_size=world_size,
    #                                 dtype=torch.half,
    #                                 # checkpoint=None if args.pre_load_checkpoint else args.checkpoint_json,
    #                                 # replace_with_kernel_inject=True
    #                                 )
    model.to(device)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        local_files_only=True,

        model_max_length=data_args.max_length,
        # padding_side="right",
        # use_fast=False,
    )

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    test_dataset = data_module["test_dataset"]#.select(range(100))
    test_dataloader = DataLoader(test_dataset.with_format("torch"), batch_size=training_args.per_device_eval_batch_size)

     # 获取模型配置信息，包括多头的数量
    config = model.config
    num_attention_heads = config.num_attention_heads
    hidden_size = config.hidden_size
    head_size = hidden_size // num_attention_heads
    
    # 计算每个文件应包含的批次数
    total_batches = len(test_dataloader)
    batches_per_file = math.ceil(total_batches / 10)

    # 初始化批次和文件计数器
    batch_count = 0
    file_count = 0
    
    
    # Prediction
    logger.info("*** Predict ***")
    # trainer.model = model.to_bettertransformer()
    # predict_results = trainer.predict(data_module["test_dataset"])
    # 初始化用于存储批数据的列表
    current_file_key = []
    curren_file_vaule = []
    # current_file_project_v = []
    # current_file_attention_mask = []
    # current_ = []
    
    # add title
    # current_file_data.append(["key", "vaule", "project_v"])
    all_steps = len(test_dataloader)
    # 用于标记是否已执行过一次存储操作
    saved_flag = False
    
    for i, data in enumerate(tqdm(test_dataloader)): 
        data = move_to_device(data, device)
        with torch.no_grad():
            outputs = model(**data, output_attentions=True, output_hidden_states=True)
        
            # 获取所有隐藏层状态
            hidden_states = outputs.hidden_states
            # print(len(hidden_states),"###############33")
            # exit(0)
            # 获取倒数第二层的隐藏状态 (batch_size, sequence_length, hidden_size)
            penultimate_hidden_states = hidden_states[-2]

            # 获取最后一个注意力层的参数
            attention_layer = model.bert.encoder.layer[-1].attention.self

            # 提取 W_key 和 W_value 参数
            W_query_layer = attention_layer.query  # shape: (hidden_size, hidden_size)
            W_key_layer = attention_layer.key  # shape: (hidden_size, hidden_size)
            W_value_layer = attention_layer.value  # shape: (hidden_size, hidden_size)

            # 计算 quey, key 和 value
            # mixed_query_layer = W_query_layer(penultimate_hidden_states)
            key = W_key_layer(penultimate_hidden_states)
            value = W_value_layer(penultimate_hidden_states)  # shape: (batch_size, sequence_length, hidden_size)

            # 分割成多头  
            # key_layer = transpose_for_scores(key, num_attention_heads, head_size) # shape: (batch_size, num_heads, sequence_length, head_size)
            # value_layer = transpose_for_scores(value, num_attention_heads, head_size) # shape: (batch_size, num_heads, sequence_length, head_size)
            # query_layer = transpose_for_scores(mixed_query_layer, num_attention_heads, head_size)
            
            # lagrange_results = Lagrange_multiplier_torch(value_layer, key_layer)
            # project_v = model.classifier(value)[:,0].detach().cpu().numpy()
            # print(key.shape,"################")
            # print(value.shape,"################")
            # print(data["attention_mask"].shape,"################")
            # exit(0)

            for j in range(key.shape[0]):
                mask = data["attention_mask"][j]
                # print(mask.shape, "##########")
                # print(torch.sum(mask), "##########")
                key_sequence = key[j]
                value_sequence = value[j]
                extracted_key_sequence = key_sequence[mask == 1].detach().cpu().numpy()#.tolist()  # 根据 mask 提取有效元素
                extracted_value_sequence = value_sequence[mask == 1].detach().cpu().numpy()#.tolist()  # 根据 mask 提取有效元素
                
                # print(extracted_key_sequence.shape, "##########")
                # exit(0)
                
                # extracted_sequences[f"sequence_{i}"] = extracted_sequence
                # print(extracted_key_sequence.shape,"################")
                # print(extracted_value_sequence.shape,"################")
                # if i >= 20:
                #     exit(0)
                current_file_key.append(extracted_key_sequence)
                curren_file_vaule.append(extracted_value_sequence)
                # current_file.append(json.dumps({"key":extracted_key_sequence.tolist(), "value":extracted_value_sequence.tolist()}))
                # current_file_vaule.append(value.detach().cpu().numpy())
                
            # current_file_attention_mask.append(data["attention_mask"].detach().cpu().numpy())
            # current_file_project_v.append(project_v)
            if (i+1) / all_steps >= 0.5 and not saved_flag:
                if not os.path.exists(training_args.output_dir):
                    os.makedirs(training_args.output_dir)
                percentage = (i+1) / all_steps
                output_prediction_key = os.path.join(training_args.output_dir, "intermediate_key_{}.npz".format(percentage))
                output_prediction_vaule = os.path.join(training_args.output_dir, "intermediate_vaule_{}.npz".format(percentage))
                
                # 使用 np.save 保存为 .npy 格式
                np.savez(output_prediction_key, *current_file_key)
                np.savez(output_prediction_vaule, *curren_file_vaule)
                
                current_file_key = []
                curren_file_vaule = []
                saved_flag = True
                
    # predict_results_key = np.concatenate((current_file_key), axis=0)
    # predict_results_vaule = np.concatenate((current_file_vaule), axis=0)
    # attention_mask = np.concatenate((current_file_attention_mask), axis=0)
    # predict_results_project_v = np.concatenate((current_file_project_v), axis=0)
    
    # 保存
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    # output_prediction_file = os.path.join(training_args.output_dir, "key_and_vaule.json")
    # with open(output_prediction_file, "w") as f_out:
    #     for i in current_file:
    #         f_out.write(i + "\n")
    output_prediction_key = os.path.join(training_args.output_dir, "intermediate_key.npz")
    output_prediction_vaule = os.path.join(training_args.output_dir, "intermediate_vaule.npz")
    # output_prediction_attention_mask = os.path.join(training_args.output_dir, "attention_mask.npy")
    # output_prediction_project_v = os.path.join(training_args.output_dir, "project_v.npy")

    # 使用 np.save 保存为 .npy 格式
    np.savez(output_prediction_key, *current_file_key)
    np.savez(output_prediction_vaule, *curren_file_vaule)
    
    # np.save(output_prediction_key, predict_results_key)
    # np.save(output_prediction_vaule, predict_results_vaule)
    # np.save(output_prediction_attention_mask, attention_mask)
    # np.save(output_prediction_project_v, predict_results_project_v)
    # print(predictions)



if __name__ == "__main__":
    main()