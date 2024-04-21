import torch
from tqdm import tqdm
import transformers
from typing import Dict, Optional, Sequence
from datasets import load_dataset

vocab_dictionary = {'0': 0,
                    '1': 1,
                    '2': 2,
                    '3': 3,
                    '4': 4,
                    '5': 5,
                    '6': 6,
                    '7': 7,
                    '8': 8,
                    '9': 9,
                    '10': 10,
                    '11': 11,
                    '12': 12,
                    '13': 13,
                    '14': 14,
                    '15': 15,
                    '16': 16,
                    '17': 17,
                    '18': 18,
                    '19': 19,
                    '20': 20,
                    '21': 21,
                    '22': 22,
                    '23': 23,
                    '24': 24,
                    '25': 25,
                    '26': 26,
                    '27': 27,
                    '28': 28,
                    '29': 29,
                    '30': 30,
                    '31': 31,
                    '32': 32,
                    '33': 33,
                    '34': 34,
                    '35': 35,
                    '36': 36,
                    '37': 37,
                    '38': 38,
                    '39': 39,
                    '40': 40,
                    '41': 41,
                    '42': 42,
                    '43': 43,
                    '44': 44,
                    '45': 45,
                    '46': 46,
                    '47': 47,
                    '48': 48,
                    '49': 49,
                    '50': 50,
                    '51': 51,
                    '52': 52,
                    '53': 53,
                    '54': 54,
                    '55': 55,
                    '56': 56,
                    '57': 57,
                    '58': 58,
                    '59': 59,
                    '60': 60,
                    '61': 61,
                    '62': 62,
                    '63': 63,
                    '64': 64,
                    '65': 65,
                    '66': 66,
                    '67': 67,
                    '68': 68,
                    '69': 69,
                    '70': 70,
                    '71': 71,
                    '72': 72,
                    '73': 73,
                    '74': 74,
                    '75': 75,
                    '76': 76,
                    '77': 77,
                    '78': 78,
                    '79': 79,
                    '80': 80,
                    '81': 81,
                    '82': 82,
                    '83': 83,
                    '84': 84,
                    '85': 85,
                    '86': 86,
                    '87': 87,
                    '88': 88,
                    '89': 89,
                    '90': 90,
                    '91': 91,
                    '92': 92,
                    '93': 93,
                    '94': 94,
                    '95': 95,
                    '96': 96,
                    '97': 97,
                    '98': 98,
                    '99': 99,
                    '100': 100,
                    '101': 101,
                    '102': 102,
                    '103': 103,
                    '104': 104,
                    '105': 105,
                    '106': 106,
                    '107': 107,
                    '108': 108,
                    '109': 109,
                    '110': 110,
                    '111': 111,
                    '112': 112,
                    '113': 113,
                    '+': 114,
                    '=': 115,
                    '%': 116}

def preprocess_function_for_imdb(examples, tokenizer, data_args):
    padding = "max_length"
    # instruction = examples["instruction"]
    inputs = examples["text"]
    output = examples["label"]
    
    # Batched input formatting
    # final_input = ["Instruction: " + ins + " \n " + "Question: " + inp + " \n " for ins, inp in zip(instruction, inputs)] 
    # targets = ["Answer: " + out + "\n" for out in output]
    
    # Tokenize inputs and targets
    model_inputs = tokenizer(inputs, max_length=data_args.max_length, padding=padding, truncation=True, return_tensors="np")
    # targets = tokenizer(targets, max_length=8, padding=padding, truncation=True, return_tensors="np")
    targets =  torch.tensor(output)
    model_inputs["labels"] = targets
    return model_inputs


def preprocess_function_for_mod(examples):
    # print(examples)
    # exit(0)
    data = [i["text"] + i["target"] for i in examples]
    all_ids = []
    for i in tqdm(data):
        tokens = i.split()
        ids = [vocab_dictionary[t] for t in tokens]
        all_ids.append(ids)
        
    input_ids = torch.LongTensor(all_ids)
    len_shape = input_ids.shape
    labels = input_ids.clone()
    clip_index = len_shape[-1]-1
    labels[:, :clip_index] = -100

    return {"input_ids":input_ids, "labels":labels}
    




def make_supervised_data_for_imdb(
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
        # instruction = examples["instruction"]
        inputs = examples["text"]
        output = examples["label"]
        
        # Batched input formatting
        # final_input = ["Instruction: " + ins + " \n " + "Question: " + inp + " \n " for ins, inp in zip(instruction, inputs)] 
        # targets = ["Answer: " + out + "\n" for out in output]
        
        # Tokenize inputs and targets
        model_inputs = tokenizer(inputs, max_length=data_args.max_length, padding=padding, truncation=True, return_tensors="np")
        # targets = tokenizer(targets, max_length=8, padding=padding, truncation=True, return_tensors="np")
        targets = torch.tensor(output)
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


# TODO 
def make_supervised_data_on_gpt_for_imdb(
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
        # instruction = examples["instruction"]
        inputs = examples["text"]
        output = examples["label"]
        final_input = inputs + output
        
        # Batched input formatting
        # final_input = ["Instruction: " + ins + " \n " + "Question: " + inp + " \n " for ins, inp in zip(instruction, inputs)] 
        # targets = ["Answer: " + out + "\n" for out in output]
        
        # Tokenize inputs and targets
        model_inputs = tokenizer(inputs, max_length=data_args.max_length, padding=padding, truncation=True, return_tensors="np")
        # targets = tokenizer(targets, max_length=8, padding=padding, truncation=True, return_tensors="np")
        targets =  torch.tensor(output)
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