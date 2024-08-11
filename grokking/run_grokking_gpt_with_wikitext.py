import copy
from dataclasses import dataclass, field
import json
import pathlib
from typing import Dict, Optional, Sequence
import logging
import torch.nn.functional as F

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer, default_data_collator, EarlyStoppingCallback, BertForSequenceClassification, AutoModel
from transformers.trainer_pt_utils import LabelSmoother
from datasets import load_dataset
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from functools import partial
from make_supervised_data import preprocess_function_for_imdb, preprocess_function_for_mod
# logger = logging.getLogger(__name__)

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
    num_labels: Optional[int] = field(
        default=2,
        metadata={
            "help": (
                "num of labels to class"
            )
        },
    )
    dataset_name: str = field(
        default=None, metadata={"help": "dataset_name"}
    )
    


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    experiment_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "A short display name for this run."
        },
    )
    rescale_num: Optional[int] = field(
        default=0,
        metadata={
            "help": (
                "rescale the parameters"
            )
        },
    )
    group_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "A short group name for this run."
        },
    )
    if_save: bool = field(default=False, metadata={"help": "Whether to save checkpoints."})
    total_steps: Optional[int] = field(
        default=10000,
        metadata={
            "help": "total steps for modluar"
        },
    )
    # model_max_length: int = field(
    #     default=512,
    #     metadata={
    #         "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
    #     },
    # )


def load_json_file(in_file, return_dict=False):
    with open(in_file, "r") as f:
        data = f.readlines()
    if return_dict:
        all_data = [json.loads(i) for i in data]
        return all_data
    else:
        return data


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    data_files = {}
    if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
    if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
    if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
    raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            # cache_dir=model_args.cache_dir,
            # use_auth_token=True if model_args.use_auth_token else None,
        )
    
    if data_args.dataset_name == "imdb":
        preprocess_function_specific = partial(preprocess_function_for_imdb, tokenizer=tokenizer, data_args=data_args)
    elif data_args.dataset_name == "modular":
        preprocess_function_specific = partial(preprocess_function_for_mod, tokenizer=tokenizer, data_args=data_args)
    
    
    if data_args.train_file is not None:
        column_names = raw_datasets["train"].column_names
    elif data_args.validation_file is not None:
        column_names = raw_datasets["validation"].column_names
    elif data_args.test_file is not None:
        column_names = raw_datasets["test"].column_names
    
    train_dataset = None
    eval_dataset = None
    test_dataset = None
    
    if data_args.train_file is not None:
        train_dataset = raw_datasets["train"]
        train_dataset = train_dataset.map(
                    preprocess_function_specific,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    # load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on train dataset",
                )
    if data_args.validation_file is not None:
        eval_dataset = raw_datasets["validation"]
        eval_dataset = eval_dataset.map(
                    preprocess_function_specific,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    # load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on validation dataset",
                )
    if data_args.test_file is not None:
        test_dataset = raw_datasets["test"]
        test_dataset = test_dataset.map(
                    preprocess_function_specific,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    # load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on test dataset",
                )
        
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, test_dataset=test_dataset)

def move_to_device(data, device):
    
    if isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    else:
        return data

def acc(pred, label):
    # 获取预测的类别（最大概率的索引）
    _, predicted = torch.max(pred, 1)

    # 检查预测的类别和实际标签是否相等
    correct = predicted.view(-1) == label.view(-1)
    # print(correct, "##########")
    
    # 计算准确率
    accuracy = correct.sum() / len(label)
    # print(accuracy, "----------------")

    return accuracy
        
def L2(model):
    L2_ = 0.
    for p in model.parameters():
        L2_ += torch.sum(p**2)
    return torch.sqrt(L2_).detach().cpu().numpy()

def rescale(model, alpha):
    for p in model.parameters():
        p.data = alpha * p.data
        
    return model

def shuffle_inputs(data):
    # input_ids
    rand_indices = torch.randperm(data["input_ids"].size(0))
    shuffled_input_ids = data["input_ids"][rand_indices]
    
    # labels
    rand_indices = torch.randperm(data["labels"].size(0))
    shuffled_labels = data["labels"][rand_indices]
    
    return {"input_ids":shuffled_input_ids, "labels":shuffled_labels}

def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # log
    if not os.path.exists(training_args.output_dir):
            os.makedirs(training_args.output_dir)
    log_file = os.path.join(training_args.output_dir, "train_log.txt")
    logging.basicConfig(filename=log_file, level=logging.INFO)

    # num_labels = data_args.num_labels
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        # num_labels=num_labels, 
        local_files_only=True)

    # model = BertForSequenceClassification._from_config(
    #     config,
    # )    
    model = transformers.AutoModelForCausalLM.from_config(config)
    if training_args.rescale_num != 0:
        print(f"here we rescale the model parameters {training_args.rescale_num}")
        model = rescale(model, training_args.rescale_num)
    # print(model)
    
    model.config.use_cache = False
    model.to(device)
    
    # tokenizer = transformers.AutoTokenizer.from_pretrained(
    #     model_args.model_name_or_path,
    #     local_files_only=True,
    #     # cache_dir=training_args.cache_dir,
    #     # model_max_length=data_args.max_length,
    #     # padding_side="right",
    #     # use_fast=False,
    # )
    
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = 'left'
    # tokenizer.truncation_side = 'left'
    
    # data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    # train_dataloader = DataLoader(data_module["train_dataset"].with_format("torch"), batch_size=training_args.per_device_train_batch_size)
    # test_dataloader = DataLoader(data_module["test_dataset"].with_format("torch"), batch_size=training_args.per_device_eval_batch_size)
    print(" ** process train set **")
    train_set = load_json_file(data_args.train_file, return_dict=True)
    train_ids = preprocess_function_for_mod(train_set)
    print(" ** process test set **")
    test_set = load_json_file(data_args.test_file, return_dict=True)
    test_ids = preprocess_function_for_mod(test_set)
    
    
    
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.98), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)

    # wandb: capture a dictionary of hyperparameters with config
    wandb.login()
    wandb.init(project="grokking", 
               name=training_args.experiment_name,
               config = {"learning_rate": training_args.learning_rate, 
                            "global_steps": training_args.total_steps, 
                            # "batch_size": training_args.per_device_train_batch_size,
                            "weight_decay": training_args.weight_decay,
                            "group":training_args.group_name if training_args.group_name is not None else None ,})
    # check save dir
    if training_args.if_save:
        if not os.path.exists(training_args.output_dir):
            os.makedirs(training_args.output_dir)
        save_steps = 1000
        print(f"*** checkpoints will save per {save_steps} steps. ***")
    
    global_steps = 0
    print("*** Begin Trianing ***")
    
    data = move_to_device(train_ids, device)
    eval_data = move_to_device(test_ids, device)
    
    for i in tqdm(range(int(training_args.total_steps))):
        # for step, data in enumerate(train_dataloader): 
            # train
        model.train()
        # shuffled_data = shuffle_inputs(data)
        # train_outputs = model(**shuffled_data)
        train_outputs = model(**data)
        # index = -2 is correct logits
        train_pred = F.softmax(train_outputs.logits[:, -2, :], dim=-1)
        train_acc = acc(train_pred, data["labels"][:, -1])

        optimizer.zero_grad()
        train_outputs.loss.backward()
        optimizer.step()

        # eval
        model.eval()
        # eval_data = next(iter(test_dataloader))

        eval_outputs = model(**eval_data)

        test_pred = F.softmax(eval_outputs.logits[:, -2, :], dim=-1)
        test_acc = acc(test_pred, eval_data["labels"][:, -1])

        # print("step:", step, "**train_loss:", train_outputs.loss.detach().cpu().numpy(), "**test_loss:", eval_outputs.loss.detach().cpu().numpy())
        
        # save model
        if training_args.if_save:
            if global_steps % save_steps == 0:
                save_dir = os.path.join(training_args.output_dir, f"checkpoints_steps_{global_steps}")
                
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                    
                model.save_pretrained(save_dir)
        
        global_steps += 1
        
        # log
        L2_norm = L2(model)
        wandb.log({"train/train_loss":train_outputs.loss, "test/test_loss":eval_outputs.loss,
                    "train/train_acc":train_acc, "test/test_acc":test_acc,
                    "L2_norm": L2_norm},
                    step=global_steps)
        logging.info(f"Step: {global_steps}, Train Loss: {train_outputs.loss}, Test Loss: {eval_outputs.loss}, Train Acc: {train_acc}, Test Acc: {test_acc}, L2 Norm: {L2_norm}")

            
    wandb.finish()
    print("*** End Trianing ***")
            

if __name__ == "__main__":
    main()