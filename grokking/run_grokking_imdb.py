# -------------------- Standard Libraries -------------------- #
import os
import json
import logging
import copy
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
# -------------------- Third-Party Libraries -------------------- #
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb
from datasets import load_dataset
import transformers
from transformers import Trainer, default_data_collator, EarlyStoppingCallback, BertForSequenceClassification
from transformers.trainer_pt_utils import LabelSmoother
import itertools
# -------------------- Custom Modules -------------------- #
from make_supervised_data import make_supervised_data_for_imdb

# Uncomment if logger is needed in the future




# ------------------------------------- Data Classes ------------------------------------- #

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
    total_steps: Optional[int] = field(
        default=0,
        metadata={
            "help": (
                "rescale the parameters"
            )
        },
    )
    
    if_save: bool = field(default=False, metadata={"help": "Whether to save checkpoints."})


# ------------------------------------- Utility Functions ------------------------------------- #


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
                   "total_steps": training_args.total_steps, 
                   "train_batch_size": training_args.per_device_train_batch_size,
                   "test_batch_size": training_args.per_device_eval_batch_size,
                   "weight_decay": training_args.weight_decay,
                   "group": training_args.group_name if training_args.group_name else None,
                   "rescale_num": training_args.rescale_num,
               })

# ------------------------------------- Main Function ------------------------------------- #

def main():
    # Argument Parsing
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Logging Setup
    if not os.path.exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)
    log_file = os.path.join(training_args.output_dir, "train_log.txt")
    logging.basicConfig(filename=log_file, level=logging.INFO)

    # Model Initialization
    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, 
                                                     num_labels=data_args.num_labels, 
                                                     local_files_only=True)
    model = BertForSequenceClassification._from_config(config)
    if training_args.rescale_num != 0:
        model = rescale(model, training_args.rescale_num)
    model.config.use_cache = False
    model.to(device)

    # Tokenizer Initialization
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, local_files_only=True)
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = 'left'
    # tokenizer.truncation_side = 'left'
 
    # Data Preparation
    if data_args.dataset_name == "imdb":
        data_module = make_supervised_data_for_imdb(tokenizer=tokenizer, data_args=data_args)
    else:
        raise ValueError("A valid dataset name must be provided!")

    train_dataloader = DataLoader(data_module["train_dataset"].with_format("torch"), 
                                  batch_size=training_args.per_device_train_batch_size, 
                                #   shuffle=True
                                  )
    test_dataloader = DataLoader(data_module["test_dataset"].with_format("torch"), 
                                 batch_size=training_args.per_device_eval_batch_size)
    
    # Optimizer Initialization
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=training_args.learning_rate, 
                                  weight_decay=training_args.weight_decay)
    
    
    
    train_repeat_dataloader = itertools.cycle(train_dataloader)
    # test_repeat_dataloader = itertools.cycle(test_dataloader)
    # Wandb Setup
    setup_wandb(data_args, training_args)

    if training_args.total_steps == 0:
        total_iterations = int(training_args.num_train_epochs) * len(train_dataloader)
    else:
        total_iterations = training_args.total_steps

    for global_steps in tqdm(range(total_iterations)):
        
        # Loading data from train_dataloader
        data = next(train_repeat_dataloader)
        
        # Training Step
        model.train()
        data = move_to_device(data, device)
        train_outputs = model(**data)
        train_pred = F.softmax(train_outputs.logits, dim=-1)
        train_acc = acc(train_pred, data["labels"])
        optimizer.zero_grad()
        train_outputs.loss.backward()
        optimizer.step()

        # Logging to Wandb
        L2_norm = L2(model)
        wandb.log({"train/train_loss": train_outputs.loss, 
                # "test/test_loss": eval_outputs.loss,
                "train/train_acc": train_acc, 
                # "test/test_acc": test_acc,
                "L2_norm": L2_norm}, 
                step=global_steps)
        
        # Logging to file
        log_data = {
            "Step": global_steps,
            "Train Loss": float(train_outputs.loss),
            # "Test Loss": float(eval_outputs.loss),
            "Train Acc": float(train_acc),
            # "Test Acc": float(test_acc),
            "L2 Norm": float(L2_norm)
        }
        
        
        # Evaluation Step
        if global_steps % 1000 == 0:
            right_num = 0
            all_num = 0
            model.eval()
            with torch.no_grad():  # 确保在评估时不计算梯度
                for eval_data in test_dataloader:
                    # eval_data = next(test_repeat_dataloader)
                    eval_data = move_to_device(eval_data, device)
                    eval_outputs = model(**eval_data)
                    test_pred = F.softmax(eval_outputs.logits, dim=-1)
                    test_right_num, all_batch_num = acc(test_pred, eval_data["labels"], return_right_num=True)
                    right_num += test_right_num
                    all_num += all_batch_num

            test_acc = right_num / all_num
            test_loss = eval_outputs.loss

            
            wandb.log({
            "test/test_loss": test_loss,
            "test/test_acc": test_acc
                }, step=global_steps)
            
            log_data["Test Loss"] = float(test_loss)
            log_data["Test Acc"] = float(test_acc)
            
            
        # Model Saving
        if training_args.if_save and global_steps % 1000 == 0:
            save_dir = os.path.join(training_args.output_dir, f"checkpoints_steps_{global_steps}")
            model.save_pretrained(save_dir)
            
        log_json = json.dumps(log_data)
        logging.info(log_json)   
                        
    wandb.finish()
    print("*** End Training ***")


if __name__ == "__main__":
    main()