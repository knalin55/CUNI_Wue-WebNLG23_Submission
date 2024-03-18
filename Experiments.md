## Implementation of methods and experiments mentioned in the paper

To implement MTL experiment for Irish, Maltese, Russian, and Welsh language:


1.   Train using
```
python train.py --languages [ga/mt/cy/ru] --task all --experiment LANG_MTL
```
2.   Generate using

```
#for MTL
python generate.py --model_path MODEL_CKPT --data_path DATASET_PATH --languages [ga/mt/cy/ru]
#for MTL+SaG
python generate.py --model_path MODEL_CKPT --data_path DATASET_PATH --languages [ga/mt/cy/ru] --triple_split subject --cs 100

```

To implement MLM experiment for Irish, Maltese, Russian, and Welsh language:


1.   Train using
```
python train.py --languages all --experiment MLM
```
2.   Generate using

```
#for MLM
python generate.py --model_path MODEL_CKPT --data_path DATASET_PATH --languages [ga/mt/cy/ru]
#for MLM+SaG
python generate.py --model_path MODEL_CKPT --data_path DATASET_PATH --languages [ga/mt/cy/ru] --triple_split subject --cs 100

```
To implement base-DF experiment for Breton:
1.   Train using
```
python train.py --languages br --experiment base_DF
```
2.   Generate using

```
#Without SaG
python generate.py --model_path MODEL_CKPT --data_path DATASET_PATH --languages br
#base_DF+SaG
python generate.py --model_path MODEL_CKPT --data_path DATASET_PATH --languages br --triple_split subject --cs 100
#base_DF+SaG (n=2)
python generate.py --model_path MODEL_CKPT --data_path DATASET_PATH --languages br --triple_split subject --cs 2
