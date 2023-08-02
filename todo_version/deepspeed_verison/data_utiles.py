import os
import torch
from datasets import load_dataset
import numpy as np


def create_prompt_dataset_hf(local_rank,
                          data_path=None,
                          train_file=None,
                          valid_file=None,
                          output_path=None,
                          train_phase=1,
                          tokenizer=None,
                          max_seq_len=512,
                        #   end_of_conversation_token="<|endoftext|>",
                        #   sft_only_data_path=[],
                          reload=False):
    """
    Creates the prompt dataset
    """
    os.makedirs(output_path, exist_ok=True)
    # fname = "_".join(data_path)
    # sft_cache_key = "_".join(sft_only_data_path)
    # tokenizer_name = tokenizer.init_kwargs["name_or_path"].replace("/", "_")
    # fname = f"{fname}_split{data_split}_phase{train_phase}_seed{seed}_tokenizer{tokenizer_name}_seqlen{max_seq_len}_sft{sft_cache_key}"
    # fname = "_".join(fname.split("/"))
    # fname = hashlib.sha256(fname.encode()).hexdigest(
    # )  # hash the file name to avoid too long file name

    
    train_fname = f"{output_path}/traindata.pt"
    eval_fname = f"{output_path}/evaldata.pt"

    def preprocess_function(examples):
        padding = "max_length"
        instruction = examples["instruction"]
        inputs = examples["input"]
        output = examples["output"]
        
        # unbatched
        # final_input = "Instruction: " + instruction + "\n" +  "Question: " + inputs + "\n" + "Answer: " + output
        # batched
        final_input = ["Instruction: " + ins + "\n" + "Question: " + inp + "\n" + "Answer: " + out for ins, inp, out in zip(instruction, inputs, output)] 

        model_inputs = tokenizer(final_input, max_length=max_seq_len, padding=padding, truncation=True, return_tensors="np")

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
    
    
    cache_found = os.path.isfile(train_fname) and os.path.isfile(eval_fname)
    buf_create_cache = torch.ByteTensor([not cache_found]).cuda()
    torch.distributed.all_reduce(buf_create_cache)

    if local_rank <= 0 and (buf_create_cache.item() != 0 or reload):
        data_files = {}
        data_files["train"] = train_file
        data_files["validation"] = valid_file
        extension = data_files["train"].split(".")[-1]
        raw_datasets = load_dataset(
                extension,
                data_files=data_files,
                cache_dir=None,  # Disable caching
            )
        column_names = raw_datasets["train"].column_names

        train_dataset = raw_datasets["train"]
        train_dataset = train_dataset.map(
                    preprocess_function,
                    batched=True,
                    # num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    # load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on train dataset",
                )
    
        eval_dataset = raw_datasets["validation"]
        eval_dataset = eval_dataset.map(
                    preprocess_function,
                    batched=True,
                    # num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    # load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on validation dataset",
                )
        torch.save(train_dataset, train_fname)
        torch.save(eval_dataset, eval_fname)
    torch.distributed.barrier()
    
    return torch.load(train_fname), torch.load(eval_fname)