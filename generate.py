#!/usr/bin/env python3

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from data_latest.utils.benchmark_reader import Benchmark, select_files

from datasets import load_metric

from tqdm import tqdm

import os
import re

# import warnings filter
from warnings import simplefilter
from sklearn.exceptions import UndefinedMetricWarning

# ignore all UndefinedMetricWarning warnings
simplefilter(action='ignore', category=UndefinedMetricWarning)
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default="t5-small", type=str, help="Seq to Seq Model")
parser.add_argument("--data_path", default="/home/nkumar/thesis/thesis/models/data_latest/data/", type=str, help="Data Path")
parser.add_argument("--data_type", default="dev", type=str, help="Data Type (val/test)")
parser.add_argument("--languages", default="ga", type=str, help="Language")
parser.add_argument("--output", default=None, type=str, help="output file description")
parser.add_argument("--device", default="cuda", type=str, help="Device")
parser.add_argument("--rp", default=1.0, type=float, help="Repetition Penalty")
parser.add_argument("--triple_split", default=None, type=str, help="None/subject/single")
parser.add_argument("--cs", default=3, type=int, help="None/subject/single")

def split_triples(triplet_list, split_type="subject", chunk_size=2):

  if split_type == "subject":
    subject_dict = {}
    final_output = list()
    
    for triplet in triplet_list:
        # Split each triplet into subject, predicate and object
        subject, predicate, obj = triplet.split(' | ')
        # Strip whitespaces from the subject
        subject = subject.strip()
        # Add subject not in the dictionary
        if subject not in subject_dict:
            subject_dict[subject] = []
        # Add the triplet to the list corresponding to the subject
        subject_dict[subject].append(triplet)

    # Now, if you want the output as a list of lists:
    output = list(subject_dict.values())

    for o in output:
        o_ = [" ## ".join(o[i:i + chunk_size]) for i in range(0, len(o), chunk_size)]
        for sub_o in o_:
            final_output.append(sub_o)
    return final_output

  elif split_type == "single":
    return triplet_list

  else:
    return [" ## ".join(triplet_list)]


def create_triples(triple_list):
    
    # Input= ['Michele_Marcolini | club | F.C._Bari_1908']
    # create_triples(Input) : 
    # "<extra_id_2>Michele_Marcolini<extra_id_4>F.C._Bari_1908<extra_id_3>club"
    triple_list = triple_list.split(" ## ")
    sub = "<extra_id_2> "
    rel = "<extra_id_3> "
    obj = "<extra_id_4> "     

    processed_triple= ""

    for triple in triple_list:
        
        processed_triple = processed_triple + sub + " ".join(triple.split("|")[0].strip().split("_")) + " " + obj +  " ".join(triple.split("|")[2].strip().split("_"))  + " " + rel + triple.split("|")[1].strip() + " "                

    return processed_triple
    #return " ## ".join(triple_list)

def process_triples(fil_dir, data_split="dev", languages=["ga"], split_type=None, chunk_size=3):

    triples_list    = []
    text_list       = []
    lang_list       = []


    for lang in languages:

        if lang == "en":
            data_path = os.path.join(fil_dir, f"ga_{data_split}.xml")

            b = Benchmark()
            files = select_files(data_path)
            b.fill_benchmark(files)
            
            entry_list = [entry for entry in b.entries]
        
            for entry in entry_list:
                en_set = [text.lex for text in entry.lexs if text.lang == lang or text.lang == ""]
                
                triples_list.append(entry.list_triples())
                text_list.append(" ## ".join(en_set))
                lang_list.append(lang)


        else:
            data_path = os.path.join(fil_dir, f"{lang}_{data_split}.xml")

            b = Benchmark()
            files = select_files(data_path)
            b.fill_benchmark(files)
            
            entry_list = [entry for entry in b.entries]
        
            for entry in entry_list:
                lang_text = [text.lex for text in entry.lexs if text.lang == lang]
                triples_list.append(split_triples(entry.list_triples(), split_type=split_type, chunk_size=chunk_size))
                text_list.append(" ## ".join(lang_text))
                lang_list.append(lang)

    
    return {"triple":triples_list, "text":text_list, "lang": lang_list}


def main(args: argparse.Namespace) -> None:
    
    ################################################################
      #Preprocessing
    ################################################################


    #def process_batch(data_list, batch_size=5):
    #    if len(data_list["text"]) % batch_size ==0:
    #        return [{"triple": data_list["triple"][batch_size*i:batch_size*(i+1)], "text": data_list["text"][batch_size*i:batch_size*(i+1)]} for i in range(len(data_list["text"])//batch_size)]
    #    else:
    #        return [{"triple": data_list["triple"][batch_size*i:batch_size*(i+1)], "text": data_list["text"][batch_size*i:batch_size*(i+1)]} for i in range(len(data_list["text"])//batch_size)] + [{"triple": data_list["triple"][batch_size*(len(data_list["text"])//batch_size):], "text": data_list["text"][batch_size*(len(data_list["text"])//batch_size):]}]



    def flat(list_list):
        result = []
        for l in list_list:
            result += l
        return result


    ################################################################
      #Generate predictions 
    ################################################################

    def prefix(text, lang):
        
        lang_dict = {"ga": "Irish",
                     "br": "Breton",
                     "cy": "Welsh",
                     "mt": "Maltese", 
                     "en": "English",
                     "ru": "Russian"}
        
        return f"RDF-to-text in {lang_dict[lang]}: {text}"

    def generate_(test, model, tokenizer):
        outputs = []
        triple = []
        text = []
        pred_text = []

        for triple_list, text_ in tqdm(zip(test["triple"], test["text"]), total=len(test["text"])):
            inputs = [prefix(create_triples(triple_), args.languages) for triple_ in triple_list]
            print(inputs)
            input_ids = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).input_ids.to(device=args.device)
            #label = tokenizer(batch["text"], return_tensors="pt", max_length=512, padding= True).input_ids.to(device='cpu')
            outputs = model.to(device=args.device).generate(input_ids, do_sample=False, max_length=512, num_beams=4, repetition_penalty=args.rp)
            generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=False)
            pred_text.append(" ".join(generated_text).replace("<pad>", "").replace("</s>", "").strip())
            
            
            text.append(text_)
            triple.append(" ## ".join(triple_list))
        
        
        return text, triple, pred_text

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    if args.languages == "all":
        languages = ["ga", "br", "mt", "cy", "ru"]
    else:
        languages = [lang.strip().lower() for lang in args.languages.split(",")]

    test = process_triples(args.data_path,args.data_type, languages=languages, split_type=args.triple_split, chunk_size=args.cs)
    

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    
    print("Loaded pretrained weights")
    print("Generating outputs...")
    
    text, triple, decoded_output_text = generate_(test, model, tokenizer)


    with open("{}_text_generation_mt5_{}_{}.txt".format(args.data_type, args.languages, args.output), "w", encoding="utf-8") as file:
        for i in range(len(decoded_output_text)):
            file.write("Input = {} \n".format(triple[i].encode('utf-8').decode("utf-8").strip()))
            file.write("Gold Text(s) = {} \n".format(text[i].encode('utf-8').decode("utf-8").strip()))
            file.write("Predicted Text = {} \n\n".format(decoded_output_text[i].encode('utf-8').decode("utf-8").strip()))          

    print("Saved")
    print("Starting evaluation..")
    
    import datasets 

    bleu = datasets.load_metric("sacrebleu")
    print(bleu.compute(predictions=decoded_output_text, references=[[t] for t in text]))
    print("Evaluation done!")
    

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
