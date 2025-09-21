# extract_embeddings_unified.py
"""
A unified script to extract embeddings from both fine-tuned large language models (BERT, GPT)
and smaller deep learning models for structured EHR data.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed
from peft import PeftModel

from src.unstructured_note.finetune_bert_based_models.finetune_models import BertFineTuner, MimicDataModule as BertMimicDataModule
from src.unstructured_note.finetune_gpt_based_models.finetune_models import GptLoraFineTuner, MimicDataModule as GptMimicDataModule
from src.unstructured_note.utils.config import MODELS_CONFIG
from src.structured_ehr.train_dl import DlPipeline, EhrDataModule

# Set seed for reproducibility
set_seed(42)

def load_text_model_and_tokenizer(args):
    """Loads a fine-tuned text model (BERT or GPT) and its tokenizer."""
    if args.model_type.lower() == 'bert':
        model_path = f"logs/unstructured_note/{args.dataset}-note/{args.task}/{args.model}/finetune_setting/checkpoints/best-model.ckpt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"BERT model checkpoint not found at: {model_path}")

        print(f"Loading BERT model from: {model_path}")
        model = BertFineTuner.load_from_checkpoint(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model.model_path, trust_remote_code=True)
        return model, tokenizer

    elif args.model_type.lower() == 'gpt':
        peft_model_path = f"logs/unstructured_note/{args.dataset}-note/{args.task}/{args.model}/finetune_setting/peft_model"
        if not os.path.exists(peft_model_path):
            raise FileNotFoundError(f"PEFT model not found at: {peft_model_path}")

        # Get base model path from MODELS_CONFIG
        model_config = next((m for m in MODELS_CONFIG if m["model_name"] == args.model), None)
        if not model_config:
            raise ValueError(f"Model '{args.model}' not found in MODELS_CONFIG.")
        base_model_path = model_config["hf_id"]

        print(f"Loading base GPT model from: {base_model_path}")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_path,
            num_labels=2,
            trust_remote_code=True,
            output_hidden_states=True
        )

        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            base_model.config.pad_token_id = base_model.config.eos_token_id

        print(f"Loading LoRA adapter from: {peft_model_path}")
        model = PeftModel.from_pretrained(base_model, peft_model_path)
        model = model.merge_and_unload() # Merge for faster inference
        return model, tokenizer

    else:
        raise ValueError(f"Unsupported text model type: {args.model_type}")

def extract_text_embeddings(model, dataloader, model_type, device):
    """Extracts embeddings for a given text dataset."""
    model.to(device)
    model.eval()

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            inputs = {k: v.to(device) for k, v in batch.items()}
            labels = inputs.pop('labels')

            outputs = model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1]

            if model_type.lower() == 'bert':
                embeddings = last_hidden_state[:, 0, :]
            elif model_type.lower() == 'gpt':
                if 'attention_mask' in inputs:
                    last_token_indices = inputs['attention_mask'].sum(dim=1) - 1
                    embeddings = last_hidden_state[torch.arange(last_hidden_state.shape[0]), last_token_indices, :]
                else:
                    embeddings = last_hidden_state[:, -1, :]

            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())

    return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0)


def extract_structured_embeddings(model, dataloader, device):
    """Extracts embeddings for a given structured EHR dataset."""
    model.to(device)
    model.eval()

    all_embeddings = []
    all_labels = []
    all_pids = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            x, y, lens, pid = batch
            x = x.to(device)

            if model.model_name == "ConCare":
                _, embedding, _ = model(x, lens)
            else:
                _, embedding = model(x, lens)

            from src.structured_ehr.utils.sequence_handler import unpad_batch
            _, y_unpadded = unpad_batch(x.cpu(), y, lens)

            embedding = embedding.cpu()[:, -1] if len(embedding.shape) == 3 else embedding.cpu()
            all_embeddings.append(embedding)
            all_labels.append(torch.tensor(y_unpadded))
            all_pids.extend(pid)

    return torch.cat(all_embeddings, dim=0), torch.cat(all_labels, dim=0), all_pids


def main():
    parser = argparse.ArgumentParser(description='Unified Embedding Extraction Script.')
    parser.add_argument('--modality', type=str, required=True, choices=['text', 'structured'],
                        help='Family of the model to process.')
    parser.add_argument('--model', type=str, required=True, help='Name of the model (e.g., BioBERT, GRU).')
    parser.add_argument('--task', type=str, required=True, choices=['mortality', 'readmission'], help='Prediction task.')
    parser.add_argument('--dataset', type=str, default='mimic-iv', help='Dataset used.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for extraction.')
    parser.add_argument('--max_length', type=int, default=512, help='Max sequence length for text models.')

    args = parser.parse_args()

    if args.modality == 'text':
        model_config = next((m for m in MODELS_CONFIG if m["model_name"] == args.model), None)
        if not model_config:
            raise ValueError(f"Model '{args.model}' not found in config. Cannot determine type (BERT/GPT).")
        args.model_type = model_config["model_type"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.modality == 'text':
        print("\n--- Running Text Model (BERT/GPT) Embedding Extraction ---")
        output_dir = Path(f"logs/multimodal/{args.dataset}/{args.task}/{args.model}/embeddings")
        output_dir.mkdir(parents=True, exist_ok=True)

        model, tokenizer = load_text_model_and_tokenizer(args)

        DataModuleClass = BertMimicDataModule if args.model_type.lower() == 'bert' else GptMimicDataModule
        data_module = DataModuleClass(
            model_name=args.model, dataset=args.dataset, task=args.task,
            batch_size=args.batch_size, max_length=args.max_length
        )
        data_module.prepare_data()
        data_module.tokenizer = tokenizer # Ensure consistent tokenizer

        for split in ['train', 'val', 'test']:
            print(f"\nProcessing '{split}' split...")
            dataloader = getattr(data_module, f"{split}_dataloader")()
            embeddings, labels = extract_text_embeddings(model, dataloader, args.model_type, device)

            output_path = output_dir / f"{split}_embeddings.pt"
            torch.save({'embeddings': embeddings, 'labels': labels}, output_path)
            print(f"Saved {split} embeddings and labels to {output_path}")

    elif args.modality == 'structured':
        print("\n--- Running Structured EHR Model Embedding Extraction ---")
        output_dir = Path(f"logs/multimodal/{args.dataset}/{args.task}/{args.model}/embeddings")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Reconstruct the config needed to load the DlPipeline checkpoint
        config = {
            'dataset': args.dataset, 'task': args.task, 'model': args.model,
            'hidden_dim': 128, 'output_dim': 1, 'main_metric': 'auroc',
            'learning_rate': 1e-3 # These values might not matter for inference
        }
        if args.dataset == 'tjh':
            config.update({'demo_dim': 2, 'lab_dim': 73})
        elif args.dataset == 'mimic-iv':
            config.update({'demo_dim': 2, 'lab_dim': 42})

        model_path = f"logs/structured_ehr/{args.dataset}-ehr/{args.task}/dl_models/{args.model}/checkpoints/best.ckpt"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Structured model checkpoint not found at: {model_path}")

        print(f"Loading structured model from: {model_path}")
        model = DlPipeline.load_from_checkpoint(model_path, config=config)

        sub_dir = 'split'
        dataset_path = f'my_datasets/{args.dataset}/processed/{sub_dir}'
        data_module = EhrDataModule(dataset_path, task=args.task, batch_size=args.batch_size)
        data_module.setup('fit')

        for split in ['train', 'val', 'test']:
            print(f"\nProcessing '{split}' split...")
            dataloader = getattr(data_module, f"{split}_dataloader")()
            embeddings, labels, pids = extract_structured_embeddings(model, dataloader, device)

            output_path = output_dir / f"{split}_embeddings.pkl"
            pd.to_pickle({'embeddings': embeddings.numpy(), 'labels': labels.numpy(), 'pids': pids}, output_path)
            print(f"Saved {split} embeddings, labels, and pids to {output_path}")

    print("\nAll embedding extraction tasks completed.")


if __name__ == "__main__":
    main()