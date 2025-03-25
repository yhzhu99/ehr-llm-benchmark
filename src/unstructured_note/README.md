# Unstructured Clinical Note

## Supervised tasks

### Finetune BERT-based models

```python
python finetune_models.py --model='BERT' --dataset='mortality' --cuda='1'
python finetune_models.py --model='ClinicalBERT' --dataset='mortality' --cuda='1'
python finetune_models.py --model='BioBERT' --dataset='mortality' --cuda='1'
python finetune_models.py --model='GatorTron' --dataset='mortality' --cuda='1'
python finetune_models.py --model='Clinical-Longformer' --dataset='mortality' --cuda='1'
```

Or you can open a bash script and input the following commands, and then run the bash script:    
```bash
#!/bin/bash

python finetune_models.py --model='BERT' --dataset='mortality' --cuda='1'
python finetune_models.py --model='ClinicalBERT' --dataset='mortality' --cuda='1'
python finetune_models.py --model='BioBERT' --dataset='mortality' --cuda='1'
python finetune_models.py --model='GatorTron' --dataset='mortality' --cuda='1'
python finetune_models.py --model='Clinical-Longformer' --dataset='mortality' --cuda='1'
```



### Freeze setting

```python
python src/unstructured_note/freeze_setting/get_embeddings.py --model='BERT' --dataset='mortality'
# python tune_embeddings.py
```

### LLM generation setting

```python
python prompt_llm.py --model='GPT-2' --dataset='mortality'
python process_generation_results.py
```

## Unsupervised tasks

### Medical sentence matching task

```python
python get_sentence_embeddings.py
python process_sentence_results.py
```

### ICD code clustering task

```python
python get_icd_embeddings.py
python get_icd_clusters.py --model='GPT-2'
python process_icd_results.py
```