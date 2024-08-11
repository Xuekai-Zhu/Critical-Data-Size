from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    Qwen2ForCausalLM,
    AutoConfig
)
from datasets import load_dataset, ClassLabel
import numpy as np
import argparse
import os
from sklearn.metrics import classification_report, confusion_matrix
import light_hf_proxy 



def rescale(model, alpha):
    for p in model.parameters():
        p.data = alpha * p.data
        
    return model

def main(args):
    dataset = load_dataset(
        args.dataset_name, split="train", cache_dir="/data1/xkzhu/datasets/cache/", num_proc=8
    )#.select(range(1000))
    if args.sample_num != 0:
        dataset = dataset.select(range(args.sample_num))
    dataset = dataset.train_test_split(
        train_size=0.9, seed=42
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)

    def math_preprocess(examples):
        examples["text"] = [" ".join([q, r]) for q, r in zip(examples["query"], examples["response"])]
        batch = tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
        batch["labels"] = batch["input_ids"]
        
        # print(f"Processed sample: {batch['input_ids'][0]}")
        # print(f"Corresponding text: {examples['text'][0]}")
        
        return batch
    
    def bio_preprocess(examples):
        examples["text"] = [" ".join([conv["value"] for conv in conv_list]) for conv_list in examples["conversations"]]
        batch = tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

        batch["labels"] = batch["input_ids"]
        
        return batch
    
    def wikitext_process(examples):
        batch = tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

        batch["labels"] = batch["input_ids"]
        
        return batch

    if "MetaMathQA" in args.dataset_name:
        dataset = dataset.map(math_preprocess, batched=True)
    elif "UltraMedical" in args.dataset_name:
        dataset = dataset.map(bio_preprocess, batched=True)
    elif "wikitext" in args.dataset_name:
        dataset = dataset.map(wikitext_process, batched=True)
    
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # model = Qwen2ForCausalLM.from_pretrained(args.base_model_name)
    config = AutoConfig.from_pretrained(args.base_model_name, local_files_only=True)
    model = Qwen2ForCausalLM._from_config(config)
    
    
    rescale_num = 10
    print(f"here we rescale the model parameters {rescale_num}")
    model = rescale(model, rescale_num)

    training_args = TrainingArguments(
        output_dir=args.checkpoint_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        # logging_steps=100,
        learning_rate=4e-4,
        num_train_epochs=5,
        seed=0,
        per_device_train_batch_size=12,
        per_device_eval_batch_size=4,
        load_best_model_at_end=True,
        # metric_for_best_model="f1_macro",
        # greater_is_better=True,
        bf16=True,
        report_to="wandb",
        run_name=args.wandb_run_name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(os.path.join(args.checkpoint_dir, "final"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", type=str, default="/data1/xkzhu/pre_trained_model/Qwen/Qwen2-0.5B-Instruct")
    parser.add_argument("--dataset_name", type=str, default="meta-math/MetaMathQA")
    parser.add_argument("--checkpoint_dir", type=str, default="edu_classifier")
    parser.add_argument("--wandb_run_name", type=str, default="my_run")
    parser.add_argument("--sample_num", type=int, default=0)
    args = parser.parse_args()

    main(args)