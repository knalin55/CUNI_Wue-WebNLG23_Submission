import torch
import numpy as np
from random import shuffle
import transformers
import random
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer, DataCollatorForSeq2Seq
from datasets import Dataset
from nltk.tokenize import sent_tokenize

import argparse
import datetime
import os
import re
import torch
from typing import Dict
from tqdm import tqdm

exec(open("./2023-Challenge/utils/benchmark_reader.py").read())

base_model_path = None #

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
        processed_triple = processed_triple + sub + \
            " ".join(triple.split("|")[0].strip().split("_")) + \
                " " + obj +  " ".join(triple.split("|")[2].strip().split("_"))  + \
                    " " + rel + triple.split("|")[1].strip() + " "                

    return processed_triple


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

def process_triples(fil_dir, data_split="train", languages=["ga"], task_="tg", alignment_type=None, p=0.6):
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

