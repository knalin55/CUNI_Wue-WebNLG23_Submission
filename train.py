#!/usr/bin/env python3

import torch
import numpy as np
from random import shuffle
import transformers
import random
from webnlgdata.utils.benchmark_reader import Benchmark, select_files # Change this line according to the directory where your training data is located
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer, DataCollatorForSeq2Seq
from datasets import Dataset

import argparse
import datetime
import os
import re
from typing import Dict


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=8, type=int, help="Batch size")
parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight Decay")
parser.add_argument("--epochs", default=15, type=int, help="Number of epochs.")
parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Gradient Accumulation Steps")
parser.add_argument("--lr", default=2e-5, type=float, help="Learning Rate.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--ckpt_path", default=None, type=str, help="Checkpoint Path")
parser.add_argument("--log_dir", default=None, type=str, help="Path to log directory")
parser.add_argument("--task", default="tg", type=str, help="[tg/all(text gen + translation)]")

parser.add_argument("--languages", default="ga", type=str, help="[ga/mt/br/cy/ru/all/ , separated languages]")
parser.add_argument("--model", default="google/mt5-base", type=str, help="Seq to Seq Model")
parser.add_argument("--data_path", default=None, type=str, help="Data Path")
parser.add_argument("--cache_dir", default=None, type=str, help="Path to HuggingFace cache directory")



def create_triples(triple_list):
    
    # Input= ['Michele_Marcolini | club | F.C._Bari_1908']
    # create_triples(Input) : 
    # "<extra_id_2>Michele_Marcolini<extra_id_4>F.C._Bari_1908<extra_id_3>club"

    sub = "<extra_id_2>"
    rel = "<extra_id_3>"
    obj = "<extra_id_4>"     

    triple_list = triple_list.split(" ## ")
    processed_triple= ""

    for triple in triple_list:
        
        processed_triple = processed_triple + sub + " ".join(triple.split("|")[0].strip().split("_")) + obj +  " ".join(triple.split("|")[2].strip().split("_"))  + rel +  triple.split("|")[1].strip()                    

    return processed_triple
    #return " ## ".join(triple_list)

def process_triples(fil_dir, data_split="train", languages=["ga"], p=0.5, task_="tg"):

    triples_list    = []
    text_list       = []
    lang_list       = []
    task            = []

    for lang in languages:

        data_path = os.path.join(fil_dir, f"{lang}_{data_split}.xml")

        b = Benchmark()
        files = select_files(data_path)
        b.fill_benchmark(files)
        
        entry_list = [entry for entry in b.entries]
    
        random.shuffle(entry_list)

        
        for entry in entry_list:
            if task_ == "tg":
                en_set = [text.lex for text in entry.lexs if text.lang == "" or text.lang == "en"]
                lang_set = [text.lex for text in entry.lexs if text.lang == lang]

                max_ratio = max([len(lang_text.split())/len(en_text.split()) for en_text, lang_text in zip(en_set, lang_set)])

                for i, text in enumerate(lang_set):
                    
                    if len(text.split())/len(en_set[i].split()) >= p*max_ratio:
                        triples_list.append(entry.list_triples())
                        text_list.append(text)
                        lang_list.append(lang)
            
            elif task_ == "all":
                en_set = [text.lex for text in entry.lexs if text.lang == "" or text.lang == "en"]
                lang_set = [text.lex for text in entry.lexs if text.lang == lang]

                max_ratio = max([len(lang_text.split())/len(en_text.split()) for en_text, lang_text in zip(en_set, lang_set)])

                for i, text in enumerate(lang_set):
                    
                    if len(text.split())/len(en_set[i].split()) >= p*max_ratio:
                        triples_list.append(" ## ".join(entry.list_triples()))
                        text_list.append(text)
                        lang_list.append(lang)
                        task.append(f"tg_en-{lang}")

                for i, text in enumerate(en_set):
                    triples_list.append(" ## ".join(entry.list_triples()))
                    text_list.append(text)
                    lang_list.append("en")
                    task.append("tg_en-en")


                for i, text in enumerate(lang_set):
                    if len(text.split())/len(en_set[i].split()) >= p*max_ratio:
                        triples_list.append(en_set[i])
                        text_list.append(text)
                        lang_list.append(lang)
                        task.append(f"mt_en-{lang}")
                    
            
            else:
                en_set = [text.lex for text in entry.lexs if text.lang == "" or text.lang == "en"]
                lang_set = [text.lex for text in entry.lexs if text.lang == lang]

                max_ratio = max([len(lang_text.split())/len(en_text.split()) for en_text, lang_text in zip(en_set, lang_set)])

                for i, text in enumerate(lang_set):
                    
                    if len(text.split())/len(en_set[i].split()) >= p*max_ratio:
                        triples_list.append(entry.list_triples())
                        text_list.append(text)
                        lang_list.append(lang)

    if task_ == "tg":
        return {"triple":triples_list, "text":text_list, "lang": lang_list}

    elif task_ == "all":
        return {"triple":triples_list, "text":text_list, "lang": lang_list, "task": task}
    
    else:
        return {"triple":triples_list, "text":text_list, "lang": lang_list}

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
        languages = ["ga", "br", "mt", "cy", "ru"]
    else:
        languages = [lang.strip() for lang in args.languages.split(",")]

    #Process data
    data_train = process_triples(args.data_path,"train", languages=languages, task_=args.task)
    data_val = process_triples(args.data_path,"dev", languages=languages, task_=args.task)
    
    def prefix(text, lang, task):
        
        lang_dict = {"ga": "Irish",
                     "br": "Breton",
                     "cy": "Welsh",
                     "mt": "Maltese",
                     "ru": "Russian",
                     "en": "English"}
        
        if task == f"tg_en-{lang}":
            return f"RDF-to-text in {lang_dict[lang]}: {create_triples(text)}"
        elif task == f"mt_en-{lang}":
            return f"Translate in {lang_dict[lang]}: {text}"
    
    # prefix = "text to RDF "
    def preprocess_function(examples):
        
        inputs = [prefix(example, lang, task) for example, lang, task in zip(examples["triple"], examples["lang"], examples["task"])]
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

    dataset_train = dataset_train.shuffle()
    dataset_val = dataset_val.shuffle()
    

    #Load Model
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, cache_dir=args.cache_dir) #low_cpu_mem_usage=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding = True, return_tensors="pt")

    os.makedirs(args.logdir, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir= os.path.join(args.logdir, f"check_points/text_gen_mt5_{args.task}/"),
        learning_rate=args.lr,
        #fp16=True,
        #bf16=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy = "epoch",
        save_strategy= "epoch",
        save_total_limit =1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        )

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
