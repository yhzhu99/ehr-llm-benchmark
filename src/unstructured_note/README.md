# Unstructured Clinical Note

## Supervised tasks

### Finetune BERT-based models

```python
python src/unstructured_note/finetune_bert_based_models/fine_models.py --model="BERT" --dataset="discharge"
```

### Freeze setting

```python
python src/unstructured_note/freeze_setting/get_embeddings.py --model="BERT" --dataset="discharge"
python src/unstructured_note/freeze_setting/tune_embeddings.py --model="BERT" --dataset="discharge"
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