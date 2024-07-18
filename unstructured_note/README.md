# Unstructured Clinical Note

## Download model checkpoints

List the repository ID and the path to save the model checkpoints in `utils/config.py` and run the following command:

```python
python download_models.py
```

## Supervised tasks

### Finetune BERT-based models

```python
python finetune_models.py --model='BERT' --dataset='noteevent' --cuda='1'
python finetune_models.py --model='ClinicalBERT' --dataset='noteevent' --cuda='1'
python finetune_models.py --model='BioBERT' --dataset='noteevent' --cuda='1'
python finetune_models.py --model='GatorTron' --dataset='noteevent' --cuda='1'
python finetune_models.py --model='Clinical-Longformer' --dataset='noteevent' --cuda='1'
```

Or you can open a bash script and input the following commands, and then run the bash script:    
```bash
#!/bin/bash

python finetune_models.py --model='BERT' --dataset='noteevent' --cuda='1'
python finetune_models.py --model='ClinicalBERT' --dataset='noteevent' --cuda='1'
python finetune_models.py --model='BioBERT' --dataset='noteevent' --cuda='1'
python finetune_models.py --model='GatorTron' --dataset='noteevent' --cuda='1'
python finetune_models.py --model='Clinical-Longformer' --dataset='noteevent' --cuda='1'
```



### Freeze setting

```python
python get_embeddings.py --model='GPT-2' --dataset=noteevent --cuda='1'
python tune_embeddings.py
```

### LLM generation setting

```python
python prompt_llm.py --model='GPT-2' --dataset='noteevent'
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