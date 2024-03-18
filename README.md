# Better Translation + Split and Generate for Multilingual RDF-to-Text

This repository includes the implementation of Cuni-Wue system at WebNLG 2023. 
```
@inproceedings{kumar-etal-2023-better,
    title = "Better Translation + Split and Generate for Multilingual {RDF}-to-Text ({W}eb{NLG} 2023)",
    author = "Kumar, Nalin  and
      Obaid Ul Islam, Saad  and
      Dusek, Ondrej",
    editor = "Gatt, Albert  and
      Gardent, Claire  and
      Cripwell, Liam  and
      Belz, Anya  and
      Borg, Claudia  and
      Erdem, Aykut  and
      Erdem, Erkut",
    booktitle = "Proceedings of the Workshop on Multimodal, Multilingual Natural Language Generation and Multilingual WebNLG Challenge (MM-NLG 2023)",
    month = sep,
    year = "2023",
    address = "Prague, Czech Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.mmnlg-1.8",
    pages = "73--79",
    abstract = "This paper presents system descriptions of our submitted outputs for WebNLG Challenge 2023. We use mT5 in multi-task and multilingual settings to generate more fluent and reliable verbalizations of the given RDF triples. Furthermore, we introduce a partial decoding technique to produce more elaborate yet simplified outputs. Additionally, we demonstrate the significance of employing better translation systems in creating training data.",
}
```

## Dependencies

```
pip install -r requirements.txt

```

## Setup

Download the WebNLG 2023 dataset and its processing modules in the current directory.

```
git clone https://github.com/WebNLG/2023-Challenge.git

```

## Usage (Training)

```
python train.py --languages LANG --task TASK --experiment EXP
``` 
General arguments and their usage
```
--languages            Language(s) for which the model is trained; 
                        "," separated language codes from {br, cy, ga, ru, mt}
--task                 Task(s) for which the model is trained;
                        [tg/all]; tg: RDF-to-text generation, all: tg (lang) + 
                        en-lang translation + tg (en)
--experiment           Name of the directory for model's checkpoint
``` 
## Usage (Text Generation)

```
python generate.py --model_path PATH_TO_MODEL_CKPT_DIR --data_path PATH_TO_DATASET_DIR --languages LANG --triple_split TS --cs 2
```
Arguments and their description
```
--languages           Language(s) for data-to-text generation task; 
                        "," separated language codes from {br, cy, ga, ru, mt}
--triple_split        Partial decoding split; [subject/single/None];
                      subject: Split input triples on the basis of the subject
                       (with chunk size --cs),
                      single: Split triples into single triple
                      None: No partial decoding 
--cs                  Chunk size (No. of triples) for partial decoding; Note: valid only for 
                      triple_split == subject
```

## Example Usage

To implement MTL + SaG experiment for Irish language (with partial decoding | split w.r.t. subjects): 


1.   Train using
```
python train.py --languages ga --task all --experiment Irish_MTL
```
2.   Generate using
```
python generate.py --model_path PATH_TO_MODEL_CKPT_DIR --data_path PATH_TO_DATASET_DIR --data_type dev --languages ga --triple_split subject --cs 100
```
