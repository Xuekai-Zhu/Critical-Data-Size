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
from transformers import BertForSequenceClassification
from transformers.trainer_pt_utils import LabelSmoother
from datasets import load_dataset
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

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
    # model_max_length: int = field(
    #     default=512,
    #     metadata={
    #         "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
    #     },
    # )


# ------------------------------------- Utility Functions ------------------------------------- #


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
    
    def preprocess_function(examples):
        padding = "max_length"
        instruction = examples["instruction"]
        inputs = examples["input"]
        output = examples["output"]
        
        # Batched input formatting
        final_input = ["Instruction: " + ins + " \n " + "Question: " + inp + " \n " for ins, inp in zip(instruction, inputs)] 
        # targets = ["Answer: " + out + "\n" for out in output]
        
        # Tokenize inputs and targets
        model_inputs = tokenizer(final_input, max_length=data_args.max_length, padding=padding, truncation=True, return_tensors="np")
        # targets = tokenizer(targets, max_length=8, padding=padding, truncation=True, return_tensors="np")
        targets =  torch.tensor([[1] if response.lower() == 'yes' else [0] for response in output])
        model_inputs["labels"] = targets
        return model_inputs
    
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
                    preprocess_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    # load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on train dataset",
                )
    if data_args.validation_file is not None:
        eval_dataset = raw_datasets["validation"]
        eval_dataset = eval_dataset.map(
                    preprocess_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    # load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on validation dataset",
                )
    if data_args.test_file is not None:
        test_dataset = raw_datasets["test"]
        test_dataset = test_dataset.map(
                    preprocess_function,
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

def acc(pred, label, return_right_num=False):
    # 获取预测的类别（最大概率的索引）
    _, predicted = torch.max(pred, 1)

    # 检查预测的类别和实际标签是否相等
    correct = predicted.view(-1) == label.view(-1)
    # print(correct, "##########")
    
    if not return_right_num:
        # 计算准确率
        accuracy = correct.sum() / len(label)
        # print(accuracy, "----------------")
        return accuracy
    
    else:
        right_num = correct.sum()
        all_data = len(label)
        return right_num, all_data
        
def L2(model):
    L2_ = 0.
    for p in model.parameters():
        L2_ += torch.sum(p**2)
    return torch.sqrt(L2_).detach().cpu().numpy()

def rescale(model, alpha):
    for p in model.parameters():
        p.data = alpha * p.data
        
    return model

def setup_wandb(data_args, training_args):
    wandb.login()
    wandb.init(project=f"grokking_{data_args.dataset_name}", 
               name=training_args.experiment_name,
               config={
                   "learning_rate": training_args.learning_rate, 
                   "epochs": training_args.num_train_epochs, 
                #    "total_steps": training_args.total_steps, 
                   "train_batch_size": training_args.per_device_train_batch_size,
                   "test_batch_size": training_args.per_device_eval_batch_size,
                   "weight_decay": training_args.weight_decay,
                   "group": training_args.group_name if training_args.group_name else None,
                   "rescale_num": training_args.rescale_num,
               })


# ------------------------------------- Main Function ------------------------------------- #


def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Logging Setup
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    log_file = os.path.join(training_args.output_dir, "train_log.txt")
    logging.basicConfig(filename=log_file, level=logging.INFO)

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=data_args.num_labels, 
        local_files_only=True)

    model = BertForSequenceClassification(
        config,
    )    
    if training_args.rescale_num != 0:
        model = rescale(model, training_args.rescale_num)
    # print(model)
    # model = transformers.AutoModelForCausalLM.from_config(config)
    model.config.use_cache = False
    model.to(device)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        local_files_only=True,
        # cache_dir=training_args.cache_dir,
        # model_max_length=data_args.max_length,
        # padding_side="right",
        # use_fast=False,
    )
    
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = 'left'
    # tokenizer.truncation_side = 'left'
    
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    train_dataloader = DataLoader(data_module["train_dataset"].with_format("torch"), batch_size=training_args.per_device_train_batch_size, shuffle=True)
    test_dataloader = DataLoader(data_module["test_dataset"].with_format("torch"), batch_size=training_args.per_device_eval_batch_size)
    
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
    
    # Wandb Setup
    setup_wandb(data_args, training_args)
    
    print("*** Begin Trianing ***")
    global_steps = 0
    for epoch in tqdm(range(int(training_args.num_train_epochs))):
        model.train()
        for step, data in enumerate(train_dataloader): 
            global_steps += 1 
            # train
            data = move_to_device(data, device)
            train_outputs = model(**data)

            train_pred = F.softmax(train_outputs.logits, dim=-1)
            train_acc = acc(train_pred, data["labels"])

            optimizer.zero_grad()
            train_outputs.loss.backward()
            optimizer.step()
        
            L2_norm = L2(model)
            wandb.log({"train/train_loss": train_outputs.loss, "train/train_acc": train_acc, 
                       "L2_norm": L2_norm}, step=global_steps)
            # Logging to file
            log_data = {"Step": global_steps,"Train Loss": float(train_outputs.loss),
                        "Train Acc": float(train_acc),"L2 Norm": float(L2_norm)}
            logging.info(json.dumps(log_data))  

        # eval
        with torch.no_grad():  # 确保在评估时不计算梯度
            right_num = 0
            all_num = 0
            test_loss = 0
            model.eval()
            for eval_data in test_dataloader:
                eval_data = move_to_device(eval_data, device)
                eval_outputs = model(**eval_data)
                test_pred = F.softmax(eval_outputs.logits, dim=-1)
                test_right_num, all_batch_num = acc(test_pred, eval_data["labels"], return_right_num=True)
                right_num += test_right_num
                all_num += all_batch_num
                test_loss += eval_outputs.loss.item() 

        test_acc = right_num / all_num
        test_loss /= len(test_dataloader) 
        wandb.log({"test/test_loss": test_loss, "test/test_acc": test_acc}, step=global_steps)       

        # Log data
        log_data = {"Step": global_steps, "Test Loss": float(test_loss), 
                    "Test Acc": float(test_acc)}
        logging.info(json.dumps(log_data))
        
    wandb.finish()
    print("*** End Training ***")
            
             
if __name__ == "__main__":
    main()