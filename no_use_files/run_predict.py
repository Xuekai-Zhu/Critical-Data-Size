import copy
from dataclasses import dataclass, field
import json
import pathlib
from typing import Dict, Optional, Sequence
import logging


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
    # dtype: Optional[int] = field(
    #     default="float16",
    #     # choices=["float32", "float16", "int8"],
    #     # type=str,
    #     # help="data-type"
    #     metadata={
    #         "help": (
    #             "data-type"
    #         )
    #     },
    # )


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


# local_rank = None


# def rank0_print(*args):
#     if local_rank == 0:
#         print(*args)



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
        
        # unbatched
        # final_input = "Instruction: " + instruction + "\n" +  "Question: " + inputs + "\n" + "Answer: " + output
        # batched
        final_input = ["Instruction: " + ins + " \n " + "Question: " + inp + " \n " + "Answer: " for ins, inp, out in zip(instruction, inputs, output)] 
        model_inputs = tokenizer(final_input, max_length=data_args.max_length, padding=padding, truncation=True, return_tensors="np")
        # label_ids = model_inputs["input_ids"].copy()

        # if padding == "max_length":
        #     label_ids = np.where(label_ids != tokenizer.pad_token_id, label_ids, -100)
        #     model_inputs["labels"] = label_ids

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


def main():
    # global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            local_files_only=True,
            # cache_dir=training_args.cache_dir,
            # device_map="auto"
            # load_in_4bit=True
            )
    
    model.eval()
    
    # Initialize the DeepSpeed-Inference engine
    model = deepspeed.init_inference(model,
                                    mp_size=world_size,
                                    dtype=torch.half,
                                    # checkpoint=None if args.pre_load_checkpoint else args.checkpoint_json,
                                    # replace_with_kernel_inject=True
                                    )
    model.to(device)

    # model = Model(input_size, output_size)
    # if torch.cuda.device_count() >= 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #     model = nn.DataParallel(model).to(device)
    
    # model = model.to_bettertransformer()
    # model.config.use_cache = False
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        local_files_only=True,
        # cache_dir=training_args.cache_dir,
        model_max_length=data_args.max_length,
        # padding_side="right",
        # use_fast=False,
    )
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = 'left'
    # tokenizer.truncation_side = 'left'
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    test_dataset = data_module["test_dataset"]
    test_dataloader = DataLoader(test_dataset.with_format("torch"), batch_size=training_args.per_device_eval_batch_size)


    # Prediction
    logger.info("*** Predict ***")
    # trainer.model = model.to_bettertransformer()
    # predict_results = trainer.predict(data_module["test_dataset"])
    predict_results = []
    for data in tqdm(test_dataloader):
        data = move_to_device(data, device)
        outputs = model.generate(**data, max_new_tokens=data_args.max_new_tokens, 
                                 num_beams=1, do_sample=False)
        
        predictions = outputs.cpu().numpy().tolist()
        predict_results.extend(predictions)
    # save
    predictions = tokenizer.batch_decode(
        predict_results, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=True
    )
    predictions = [json.dumps({"output":pred.strip()}, ensure_ascii=False) for pred in predictions]
    # print(predictions)
    output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.json")
    with open(output_prediction_file, "w", encoding='utf-8') as writer:
        writer.write("\n".join(predictions))


if __name__ == "__main__":
    main()