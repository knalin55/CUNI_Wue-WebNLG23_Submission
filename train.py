#!/usr/bin/env python3

import torch
import numpy as np
from random import shuffle
import transformers
import random
from data_latest.utils.benchmark_reader import Benchmark, select_files # Change this line according to the directory where your training data is located
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer, DataCollatorForSeq2Seq
from datasets import Dataset
from nltk.tokenize import sent_tokenize

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
parser.add_argument("--task", default="tg", type=str, help="[tg/all(text gen + translation)]")
parser.add_argument("--alignment", default=None, type=str, help="Alignment")
parser.add_argument("--experiment", default=None, type=str, help="Alignment")

parser.add_argument("--languages", default="ga", type=str, help="[ga/mt/br/cy/ru/all/ , separated languages]")
parser.add_argument("--model", default="google/mt5-base", type=str, help="Seq to Seq Model")
parser.add_argument("--data_path", default=None, type=str, help="Data Path")
parser.add_argument("--cache_dir", default=None, type=str, help="Path to HuggingFace cache directory")

base_model_path = None
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
    else:
        return f"{task} in {lang_dict[lang]}: {text}"

def create_triples(triple_list):
    
    # Input= ['Michele_Marcolini | club | F.C._Bari_1908']
    # create_triples(Input) : 
    # "<extra_id_2>Michele_Marcolini<extra_id_4>F.C._Bari_1908<extra_id_3>club"

    sub = "<extra_id_2> "
    rel = "<extra_id_3> "
    obj = "<extra_id_4> "     

    triple_list = triple_list.split(" ## ")
    processed_triple= ""
    for triple in triple_list:
        
        processed_triple = processed_triple + sub + " ".join(triple.split("|")[0].strip().split("_")) + " " + obj +  " ".join(triple.split("|")[2].strip().split("_"))  + " " + rel + triple.split("|")[1].strip() + " "                

    return processed_triple
    #return " ## ".join(triple_list)

def sort_triples(base_model, base_tokenizer, triples, input_text, alignment_type="cross_attn"):
    
    if alignment_type == "cross_attn":

        triple_dict = {}
        triples = triples.split(" ## ")
        for triple in triples:
            
            sort_idx = []
            encoder_input_ids = base_tokenizer(input_text, return_tensors="pt", add_special_tokens=True).input_ids
            decoder_input_ids = base_tokenizer(create_triples(triple), return_tensors="pt", add_special_tokens=True).input_ids
            outputs = base_model.to(device="cuda")(input_ids=encoder_input_ids.to(device="cuda"), decoder_input_ids=decoder_input_ids.to(device="cuda"), output_attentions=True)
            decoder_text = base_tokenizer.convert_ids_to_tokens(decoder_input_ids[0])
            idx = torch.argmax(outputs.cross_attentions[0][:,:,:,:-1].sum(dim=1), dim=-1)
            subj = decoder_input_ids.tolist()[0].index(base_tokenizer.encode("<extra_id_2>")[0]) + 1
            obj = decoder_input_ids.tolist()[0].index(base_tokenizer.encode("<extra_id_4>")[0]) + 1
            reln = decoder_input_ids.tolist()[0].index(base_tokenizer.encode("<extra_id_3>")[0]) + 1

            for ent in [subj, obj, reln]:
                sort_idx.append(idx[0][ent])
            triple_dict[tuple(sort_idx)] = triple

        sorted_triples = [triple_dict[key] for key in sorted(triple_dict.keys())]

        return sorted_triples

def new_process_triples(fil_dir, data_split="train", languages=["ga"], task_="tg", alignment_type=None, p=0.6):
    triples_list = []
    text_list = []
    lang_list = []
    task = []

    if alignment_type is not None:
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_path)
        base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    for lang in languages:
        
        if lang == "br":
            fil_dir = os.path.join(fil_dir, "models/data_latest/data/")
            data_path = os.path.join(fil_dir, f"{lang}_{data_split}.xml")

            b = Benchmark()
            files = select_files(data_path)
            b.fill_benchmark(files)
            
            entry_list = [entry for entry in b.entries]
        
            random.shuffle(entry_list)

            for entry in tqdm(entry_list):

                en_set = [text.lex for text in entry.lexs if text.lang == "" or text.lang == "en"]
                lang_set = [text.lex for text in entry.lexs if text.lang == lang]

                max_ratio = max([len(lang_text.split())/len(en_text.split()) for en_text, lang_text in zip(en_set, lang_set)])

                for i, text in enumerate(lang_set):
                    
                    if len(text.split())/len(en_set[i].split()) >= p*max_ratio:
                        if len(text.split()) >= 4 and len(sent_tokenize(en_set[i]))<=1:
                            triples_list.append(" ## ".join(entry.list_triples()))
                            text_list.append(text)
                            lang_list.append(lang)

        elif lang == "en":
            data_path = os.path.join(fil_dir, f"new_ga_{data_split}.tsv")
            
            file = open(data_path).read().split("\n")
            if file[-1] == "":
                file = file[:-1]
            
            for line in tqdm(file):
                _, _, _, triples, en_text, lang_text = line.split("\t")
                if task_ == "tg":
                    triples_list.append(triples)
                    text_list.append(en_text.replace("..", "."))
                    lang_list.append(lang)

        else:
            data_path = os.path.join(fil_dir, f"new_{lang}_{data_split}.tsv")
            
            file = open(data_path).read().split("\n")
            if file[-1] == "":
                file = file[:-1]

            for line in tqdm(file):
                _, _, _, triples, en_text, lang_text = line.split("\t")
                if task_ == "tg":
                    if alignment_type is not None:
                        input_text = prefix(lang_text.replace("..", "."), lang, "text-to-RDF")
                        new_triples = sort_triples(base_model, base_tokenizer, triples, input_text, alignment_type=alignment_type)
                        triples_list.append(" ## ".join(new_triples))
                        text_list.append(lang_text.replace("..", "."))
                        lang_list.append(lang)

                    else:
                        triples_list.append(triples)
                        text_list.append(lang_text.replace("..", "."))
                        lang_list.append(lang)
                
                elif task_ == "3task":
                    triples_list.append(triples)
                    text_list.append(lang_text.replace("..", "."))
                    lang_list.append(lang)
                    task.append(f"tg_en-{lang}")

                    triples_list.append(triples)
                    text_list.append(en_text)
                    lang_list.append("en")
                    task.append("tg_en-en")

                    triples_list.append(en_text)
                    text_list.append(lang_text.replace("..", "."))
                    lang_list.append(lang)
                    task.append(f"mt_en-{lang}")

                else: 
                    triples_list.append(triples)
                    text_list.append(lang_text.replace("..", "."))
                    lang_list.append(lang)
    
    if task_ == "tg":
        return {"triple": triples_list, "lang": lang_list, "text": text_list}
    elif task_ == "3task": 
        return {"triple": triples_list, "lang": lang_list, "text": text_list, "task":task}
    else:
        return {"triple": triples_list, "lang": lang_list, "text": text_list}


def process_triples(fil_dir, data_split="train", languages=["ga"], p=0.5, task_="tg", alignment_type=None):

    triples_list    = []
    text_list       = []
    lang_list       = []
    task            = []

    if alignment_type is not None:
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_path)
        base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    

    for lang in languages:

        data_path = os.path.join(fil_dir, f"{lang}_{data_split}.xml")

        b = Benchmark()
        files = select_files(data_path)
        b.fill_benchmark(files)
        
        entry_list = [entry for entry in b.entries]
    
        random.shuffle(entry_list)

        for entry in tqdm(entry_list):
            
            if task_ == "tg":
                en_set = [text.lex for text in entry.lexs if text.lang == "" or text.lang == "en"]
                lang_set = [text.lex for text in entry.lexs if text.lang == lang]

                max_ratio = max([len(lang_text.split())/len(en_text.split()) for en_text, lang_text in zip(en_set, lang_set)])

                for i, text in enumerate(lang_set):
                    
                    if len(text.split())/len(en_set[i].split()) >= p*max_ratio:
                        if alignment_type is not None:
                            triples = entry.list_triples()
                            input_text = prefix(text, lang, "text-to-RDF")
                            new_triples = sort_triples(base_model, base_tokenizer, triples, input_text, alignment_type=alignment_type)
                            triples_list.append(" ## ".join(new_triples))
                            text_list.append(text)
                            lang_list.append(lang)
                        else: 
                            triples_list.append(" ## ".join(entry.list_triples()))
                            text_list.append(text)
                            lang_list.append(lang)
                            task.append(f"tg_en-{lang}")

            
            elif task_ == "all":
                en_set = [text.lex for text in entry.lexs if text.lang == "" or text.lang == "en"]
                lang_set = [text.lex for text in entry.lexs if text.lang == lang]

                max_ratio = max([len(lang_text.split())/len(en_text.split()) for en_text, lang_text in zip(en_set, lang_set)])

                for i, text in enumerate(lang_set):
                    
                    if len(text.split())/len(en_set[i].split()) >= p*max_ratio:
                        if alignment_type is not None:
                            triples = entry.list_triples()
                            input_text = prefix(text, lang, "text-to-RDF")
                            new_triples = sort_triples(base_model, base_tokenizer, triples, input_text, alignment_type="cross_attn")
                            triples_list.append(" ## ".join(new_triples))
                            text_list.append(text)
                            lang_list.append(lang)
                            task.append(f"tg_en-{lang}")

                        else:   
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
                        triples_list.append(" # ".join(entry.list_triples()))
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
    tokenizer = AutoTokenizer.from_pretrained(args.model, device_map="auto", cache_dir="/home/nkumar/personal_work_ms/.cache")

    if args.languages == "all":
        languages = ["ga", "mt", "cy", "ru"]
    else:
        languages = [lang.strip() for lang in args.languages.split(",")]

    #Process data
    #data_train = process_triples(args.data_path,"train", languages=languages, task_=args.task, alignment_type=args.alignment)
    #data_val = process_triples(args.data_path,"dev", languages=languages, task_=args.task, alignment_type=args.alignment)
    data_train = new_process_triples(args.data_path, "train", languages=languages, task_=args.task, alignment_type=args.alignment)
    data_val = new_process_triples(args.data_path, "dev", languages=languages, task_=args.task, alignment_type=args.alignment)
    
    # prefix = "text to RDF "
    def preprocess_function(examples):
        
        if "task" in examples.keys():
            inputs = [prefix(example, lang, task) for example, lang, task in zip(examples["triple"], examples["lang"], examples["task"])]
        else:
            inputs = [prefix(example, lang, f"tg_en-{lang}") for example, lang in zip(examples["triple"], examples["lang"])]
        
        print(inputs[0])
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
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, cache_dir=args.cache_dir) #low_cpu_mem_usage=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding = True, return_tensors="pt")

    os.makedirs(args.logdir, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir= os.path.join(args.logdir, f"check_points/text_gen_mt5_{args.task}_{args.languages}_{args.experiment}/"),
        learning_rate=args.lr,
        #fp16=True,
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
