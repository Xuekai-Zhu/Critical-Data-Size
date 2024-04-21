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
from transformers import Trainer, default_data_collator, EarlyStoppingCallback, BertForSequenceClassification
from transformers.trainer_pt_utils import LabelSmoother
from datasets import load_dataset
import os
from torch.utils.data import DataLoader
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
        model_inputs = tokenizer(final_input, max_length=512, padding=padding, truncation=True, return_tensors="np")
        # targets = tokenizer(targets, max_length=8, padding=padding, truncation=True, return_tensors="np")
        # one_hot_vector = 
        targets =  torch.tensor([[1] if response == 'yes' else [0] for response in output])
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

def main():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank


    num_labels = data_args.num_labels
    # model = BertForSequenceClassification.from_pretrained(
    #     model_args.model_name_or_path,
    #     local_files_only=True,
    #     num_labels=num_labels, 
    #     # problem_type="multi_label_classification"
    #     # cache_dir=training_args.cache_dir,
    # )    
    
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels, 
        local_files_only=True)

    model = BertForSequenceClassification(
        config,
    )   

    model.config.use_cache = False
    
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
    
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_module["train_dataset"] if training_args.do_train else None,
        eval_dataset=data_module["eval_dataset"] if training_args.do_eval else None,
        # test_dataset=data_module["test_dataset"] if training_args.do_predict else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        callbacks=[
                EarlyStoppingCallback(early_stopping_patience=3), 
                    ]
    )
    if training_args.do_train:
        rank0_print("*** Begin Trianing ***")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
        
        trainer.save_model()  # Saves the tokenizer too for easy upload
        
        # metrics = train_result.metrics
        # trainer.log_metrics("train", metrics)
        # trainer.save_metrics("train", metrics)    
        trainer.save_state()
        # safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
        
    if training_args.do_eval:
        rank0_print("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        
    if training_args.do_predict:
        is_regression = False
        # Removing the `label` columns because it contains -1 and Trainer won't like that.
        predict_dataset = data_module["test_dataset"].remove_columns("labels")
        predictions = trainer.predict(predict_dataset).predictions
        predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

        predictions = [json.dumps({"output":str(pred)}, ensure_ascii=False) for pred in predictions]
        # print(predictions)
        output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.json")
        with open(output_prediction_file, "w", encoding='utf-8') as writer:
            writer.write("\n".join(predictions))
    


if __name__ == "__main__":
    main()