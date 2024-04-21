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
    if_save: bool = field(default=False, metadata={"help": "Whether to save checkpoints."})


# ------------------------------------- Utility Functions ------------------------------------- #


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

def setup_wandb(data_args, training_args):
    wandb.login()
    wandb.init(project=f"grokking_{data_args.dataset_name}", 
               name=training_args.experiment_name,
               config={
                   "learning_rate": training_args.learning_rate, 
                   "epochs": training_args.num_train_epochs, 
                   "batch_size": training_args.per_device_train_batch_size,
                   "weight_decay": training_args.weight_decay,
                   "group": training_args.group_name if training_args.group_name else None,
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
    model = transformers.OpenAIGPTForSequenceClassification._from_config(config)

    
    # Tokenizer Initialization
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.unk_token
    model.config.pad_token_id = tokenizer.pad_token_id
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'
    
    if training_args.rescale_num:
        model = rescale(model, training_args.rescale_num)
    model.config.use_cache = False
    model.to(device)

    
 
    # Data Preparation
    if data_args.dataset_name == "imdb":
        data_module = make_supervised_data_for_imdb(tokenizer=tokenizer, data_args=data_args)
    else:
        raise ValueError("A valid dataset name must be provided!")

    train_dataloader = DataLoader(data_module["train_dataset"].with_format("torch"), 
                                  batch_size=training_args.per_device_train_batch_size)
    test_dataloader = DataLoader(data_module["test_dataset"].with_format("torch"), 
                                 batch_size=training_args.per_device_eval_batch_size)
    
    # Optimizer Initialization
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=training_args.learning_rate, 
                                  weight_decay=training_args.weight_decay)

    # Wandb Setup
    setup_wandb(data_args, training_args)

    # Training Loop
    global_steps = 0
    print("*** Begin Training ***")
    for epoch in tqdm(range(int(training_args.num_train_epochs))):
        for data in train_dataloader:
            # Training Step
            model.train()
            data = move_to_device(data, device)
            train_outputs = model(**data)
            train_pred = F.softmax(train_outputs.logits, dim=-1)
            train_acc = acc(train_pred, data["labels"])
            optimizer.zero_grad()
            train_outputs.loss.backward()
            optimizer.step()

            # Evaluation Step
            model.eval()
            eval_data = next(iter(test_dataloader))
            eval_data = move_to_device(eval_data, device)
            eval_outputs = model(**eval_data)
            test_pred = F.softmax(eval_outputs.logits, dim=-1)
            test_acc = acc(test_pred, eval_data["labels"])
            
            # Model Saving
            if training_args.if_save and global_steps % 1000 == 0:
                save_dir = os.path.join(training_args.output_dir, f"checkpoints_steps_{global_steps}")
                model.save_pretrained(save_dir)

            # Logging to Wandb
            L2_norm = L2(model)
            wandb.log({"train/train_loss": train_outputs.loss, 
                       "test/test_loss": eval_outputs.loss,
                       "train/train_acc": train_acc, 
                       "test/test_acc": test_acc,
                       "L2_norm": L2_norm}, 
                      step=global_steps)
            
            # Logging to file
            log_data = {
                "Step": global_steps,
                "Train Loss": float(train_outputs.loss),
                "Test Loss": float(eval_outputs.loss),
                "Train Acc": float(train_acc),
                "Test Acc": float(test_acc),
                "L2 Norm": float(L2_norm)
            }
            log_json = json.dumps(log_data)
            logging.info(log_json)                    
            
            global_steps += 1
    
    wandb.finish()
    print("*** End Training ***")


if __name__ == "__main__":
    main()