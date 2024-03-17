import torch
import numpy as np
from random import shuffle
import transformers
import random
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer, DataCollatorForSeq2Seq
from datasets import Dataset
from nltk.tokenize import sent_tokenize
from utils import process_triples, prefix, create_triples, sort_triples
import argparse
import datetime
import os
import re
from typing import Dict
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=8, type=int, help="Batch size")
parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight Decay")
parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")
parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Gradient Accumulation Steps")
parser.add_argument("--lr", default=2e-5, type=float, help="Learning Rate.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--ckpt_path", default=None, type=str, help="Checkpoint Path")
parser.add_argument("--log_dir", default=None, type=str, help="Path to log directory")
parser.add_argument("--task", default="tg", type=str, help="[tg/all(text generation/ tg + translation)]")
parser.add_argument("--alignment", default=None, type=str, help="Alignment")
parser.add_argument("--experiment", default=None, type=str, help="Checkpoint directory name")

parser.add_argument("--languages", default="ga", type=str, help="""[ga/mt/br/cy/ru/all/ "," separated languages]""")
parser.add_argument("--model", default="google/mt5-base", type=str, help="Seq to Seq Model")
parser.add_argument("--data_path", default=None, type=str, help="Data Path")
parser.add_argument("--cache_dir", default=None, type=str, help="Path to HuggingFace cache directory")


def main(args: argparse.Namespace) -> None:

    torch.manual_seed(args.seed)

    if args.log_dir is not None: 
        args.logdir = os.path.join(args.logdir, os.path.join("logs", "{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    )))
    else:
        args.logdir = os.path.join("logs", "{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    ))
        args.logdir = os.path.join("/home/nkumar/personal_work_troja/thesis/", args.logdir)

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, device_map="auto", cache_dir=args.cache_dir)

    if args.languages == "all":
        languages = ["ga", "mt", "cy", "ru"]
    else:
        languages = [lang.strip().lower() for lang in args.languages.split(",")]

    #Process data
    data_train = process_triples(args.data_path, "train", languages=languages, task_=args.task, alignment_type=args.alignment)
    data_val = process_triples(args.data_path, "dev", languages=languages, task_=args.task, alignment_type=args.alignment)
    
    def preprocess_function(examples):
        
        if "task" in examples.keys():
            inputs = [prefix(example, lang, task) for example, lang, task in zip(examples["triple"], examples["lang"], examples["task"])]
        else:
            inputs = [prefix(example, lang, f"tg_en-{lang}") for example, lang in zip(examples["triple"], examples["lang"])]
        
        targets = [example for example in examples["text"]]
        model_inputs = tokenizer(inputs, padding=True, truncation=True)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, padding=True, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    dataset_train = Dataset.from_dict(data_train)
    dataset_val = Dataset.from_dict(data_val)

    tokenized_train = dataset_train.map(preprocess_function, batched=True)
    tokenized_val = dataset_val.map(preprocess_function, batched=True)

    tokenized_train = tokenized_train.remove_columns(["triple", "text", "lang"])
    tokenized_val = tokenized_val.remove_columns(["triple", "text", "lang"])

    tokenized_train.shuffle()
    tokenized_val.shuffle()
    

    #Load Model
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, cache_dir=args.cache_dir) 

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding = True, return_tensors="pt")

    os.makedirs(args.logdir, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir= os.path.join(args.logdir, f"check_points/text_gen_mt5_{args.task}_{args.languages}_{args.experiment}/"),
        learning_rate=args.lr,
        fp16=True,
        #bf16=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy = "epoch",
        save_strategy= "epoch",
        save_total_limit =2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        load_best_model_at_end=True)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator)
    
    trainer.train(args.ckpt_path)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
