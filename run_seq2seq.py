# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
from dataclasses import dataclass, field
import json
import pathlib
from typing import Dict, Optional, Sequence
import logging


import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer, default_data_collator
from transformers.trainer_pt_utils import LabelSmoother
from datasets import load_dataset

# from fastchat.train.llama_flash_attn_monkey_patch import (
#     replace_llama_attn_with_flash_attn,
# )

# replace_llama_attn_with_flash_attn()

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
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    # model_max_length: int = field(
    #     default=512,
    #     metadata={
    #         "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
    #     },
    # )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


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
    if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
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
        
        # unbatched
        # final_input = "Instruction: " + instruction + "\n" +  "Question: " + inputs + "\n" + "Answer: " + output
        # batched
        final_input = ["Instruction: " + ins + "\n" + "Question: " + inp + "\n" + "Answer: " + out for ins, inp, out in zip(instruction, inputs, output)] 
        model_inputs = tokenizer(final_input, max_length=512, padding=padding, truncation=True, return_tensors="np")
        label_ids = model_inputs["input_ids"].copy()

        if padding == "max_length":
            label_ids = np.where(label_ids != tokenizer.pad_token_id, label_ids, -100)
            model_inputs["labels"] = label_ids

        return model_inputs
    
    column_names = raw_datasets["train"].column_names
    
    train_dataset = raw_datasets["train"]
    train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                # load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
    
    eval_dataset = raw_datasets["validation"]
    eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                # load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)

def main():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        local_files_only=True,
        # cache_dir=training_args.cache_dir,
    )
    model.config.use_cache = False
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        local_files_only=True,
        # cache_dir=training_args.cache_dir,
        model_max_length=data_args.max_length,
        # padding_side="right",
        # use_fast=False,
    )
    tokenizer.pad_token = tokenizer.eos_token
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_module["train_dataset"] if training_args.do_train else None,
        eval_dataset=data_module["eval_dataset"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator
    )
    if training_args.do_train:
        rank0_print("*** Begin Trianing ***")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
        trainer.save_model()  # Saves the tokenizer too for easy upload
        
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)    
        trainer.save_state()
        # safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
        
    if training_args.do_eval:
        rank0_print("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()