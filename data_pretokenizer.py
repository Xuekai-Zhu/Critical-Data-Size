from transformers import AutoTokenizer
import numpy as np
from datasets import load_dataset, load_from_disk

tokenizer = AutoTokenizer.from_pretrained(
        "pre-trained-model/huggyllama/llama-7b",
        local_files_only=True,
        # cache_dir=training_args.cache_dir,
        model_max_length=512
        # padding_side="right",
        # use_fast=False,
    )
tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
        padding = "max_length"
        instruction = examples["instruction"]
        inputs = examples["input"]
        output = examples["output"]
        
        # final_input = "Instruction: " + instruction + "\n" +  "Question: " + inputs + "\n" + "Answer: " + output # unbatched
        final_input = ["Instruction: " + ins + "\n" + "Question: " + inp + "\n" + "Answer: " + out for ins, inp, out in zip(instruction, inputs, output)] # batched

        model_inputs = tokenizer(final_input, max_length=512, padding=padding, truncation=True, return_tensors="np")

        # if "code" in examples:
        label_ids = model_inputs["input_ids"].copy()
        # targets = examples["answer"]
        # Tokenize targets with the `text_target` keyword argument
        # labels = tokenizer(text_target=targets, max_length=512, padding=padding, truncation=True, return_tensors="np")
        # assert (model_inputs["input_ids"] == labels["input_ids"]).all()
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        # if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        # print(type(labels["input_ids"]))
        # exit(0)
        if padding == "max_length":
            # labels["input_ids"] = [
            #     [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            # ]
            label_ids = np.where(label_ids != tokenizer.pad_token_id, label_ids, -100)
            model_inputs["labels"] = label_ids
            # print(model_inputs["labels"])
            # print(model_inputs["input_ids"])
            # print("-------------------")
        return model_inputs




data_files = {}
data_files["train"] = "datasets/yesno_task/datatsets/train_pruning.json"
data_files["validation"] = "datasets/yesno_task/datatsets/valid.json"
extension = data_files["train"].split(".")[-1]
raw_datasets = load_dataset(
        extension,
        data_files=data_files,
    )
column_names = raw_datasets["train"].column_names

# train_dataset = raw_datasets["train"]
# train_dataset = train_dataset.map(
#             preprocess_function,
#             # batched=True,
#             # num_proc=data_args.preprocessing_num_workers,
#             remove_columns=column_names,
#             # load_from_cache_file=not data_args.overwrite_cache,
#             desc="Running tokenizer on train dataset",
#         )

eval_dataset = raw_datasets["validation"].select(range(100))
eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            # num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            # load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )

print(eval_dataset[0]["input_ids"])
# print(len(eval_dataset[0]["input_ids"][0]))
