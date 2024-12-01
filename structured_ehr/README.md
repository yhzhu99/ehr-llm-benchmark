# Structured Electronic Health Records

- List configs in `config/config.py`.

- Prepare datasets in `datasets/`.

    Please refer to the repository [Pyehr](https://github.com/yhzhu99/pyehr) for more details.

- run the following commands:

```bash
cd structured_ehr

# Prompt LLM to get prediction logits.
python run.py

# Evaluate the prediction results.
python evaluate.py
```