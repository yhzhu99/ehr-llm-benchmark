"""
src/unstructured_note/finetune_bert_based_models/finetune_models.py

This script is used to fine-tune BERT-based models for MIMIC-III mortality/discharge prediction tasks.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Add this line at the top

import argparse
from pathlib import Path
import csv
import jsonlines

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed

from unstructured_note.utils.config import MODELS_CONFIG
from unstructured_note.utils.classification_metrics import get_binary_metrics


# Dataset class for MIMIC text data
class MimicTextDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }


class MimicDataModule(L.LightningDataModule):
    def __init__(self, model_name, dataset_name, batch_size=16, max_length=512):
        super().__init__()
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Get model info from config
        self.model_config = next((model for model in MODELS_CONFIG if model["model_name"] == model_name), None)
        if not self.model_config:
            raise ValueError(f"Model {model_name} not found in MODELS_CONFIG")
        
        self.model_path = self.model_config["hf_id"]
        
        # Data containers
        self.data = {'train': {}, 'valid': {}, 'test': {}}
        for key in self.data.keys():
            self.data[key] = {'ID': [], 'text': [], 'label': []}
        
        # Will store the PyTorch datasets
        self.datasets = {}
        
        # Initialize tokenizer
        self.tokenizer = None  # Will be initialized in setup
    
    def prepare_data(self):
        # Load data based on dataset name
        if self.dataset_name == 'discharge':
            self._load_discharge_data()
        elif self.dataset_name == 'mortality':
            self._load_mortality_data()
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
    
    def _load_discharge_data(self):
        file_names = ['train', 'val', 'test']
        file_dir = 'my_datasets/mimic-iii-note/discharge'
        
        for file_name in file_names:
            file_path = os.path.join(file_dir, f'{file_name}.csv')
            with open(file_path, "r") as csvfile:
                csv_reader = csv.reader(csvfile, delimiter=',')
                header = next(csv_reader)
                for row in csv_reader:
                    key = 'valid' if file_name == 'val' else file_name
                    self.data[key]['ID'].append(int(float(row[1])))
                    self.data[key]['text'].append(row[2])
                    assert int(float(row[3])) == 0 or int(float(row[3])) == 1
                    self.data[key]['label'].append(int(float(row[3])))
    
    def _load_mortality_data(self):
        file_names = ['train', 'valid', 'test']
        file_dir = 'my_datasets/mimic-iii-note/mortality'
        
        for file_name in file_names:
            file_path = os.path.join(file_dir, f'{file_name}-text.json')
            with open(file_path) as file:
                for disease_info in jsonlines.Reader(file):
                    key = file_name
                    self.data[key]['ID'].append(disease_info["id"])
                    
                    # Handle different text formats
                    if isinstance(disease_info["texts"], list):
                        long_text = ''
                        for short_text in disease_info["texts"]:
                            if isinstance(short_text, list):
                                long_test_2 = ''
                                for short_text_2 in short_text:
                                    long_test_2 += short_text_2
                                long_text += long_test_2
                            else:
                                long_text += short_text
                        self.data[key]['text'].append(long_text)
                    else:
                        self.data[key]['text'].append(disease_info["texts"])
                    
                    assert disease_info["label"] == 0 or disease_info["label"] == 1
                    self.data[key]['label'].append(int(disease_info["label"]))
    
    def setup(self, stage=None):
        # Initialize tokenizer here to ensure it's done on the right device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        
        # For models that need pad token
        if self.tokenizer.pad_token is None and hasattr(self.tokenizer, 'eos_token'):
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Tokenize data for each split
        for split in ['train', 'valid', 'test']:
            texts = self.data[split]['text']
            labels = self.data[split]['label']
            
            # Tokenize texts
            encodings = self.tokenizer(
                texts,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )
            
            # Create dataset
            self.datasets[split] = MimicTextDataset(
                input_ids=encodings['input_ids'],
                attention_mask=encodings['attention_mask'],
                labels=torch.tensor(labels)
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.datasets['train'],
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Changed from 4 to 0
            pin_memory=True,
            persistent_workers=False  # Set to False when num_workers=0
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.datasets['valid'],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # Changed from 4 to 0
            pin_memory=True,
            persistent_workers=False  # Set to False when num_workers=0
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.datasets['test'],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # Changed from 4 to 0
            pin_memory=True,
            persistent_workers=False  # Set to False when num_workers=0
        )


class LanguageModelFineTuner(L.LightningModule):
    def __init__(self, model_name, learning_rate=2e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model_name = model_name
        self.learning_rate = learning_rate
        
        # Get model info from config
        self.model_config = next((model for model in MODELS_CONFIG if model["model_name"] == model_name), None)
        if not self.model_config:
            raise ValueError(f"Model {model_name} not found in MODELS_CONFIG")
        
        self.model_path = self.model_config["hf_id"]
        
        # Load pretrained model for sequence classification
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path, 
            num_labels=2,
            output_hidden_states=True,
            trust_remote_code=True
        )
        
        # For models that need pad token in config
        if hasattr(self.model.config, 'pad_token_id') and self.model.config.pad_token_id is None:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            if tokenizer.pad_token is None and hasattr(tokenizer, 'eos_token'):
                tokenizer.pad_token = tokenizer.eos_token
                self.model.config.pad_token_id = tokenizer.eos_token_id
        
        # For tracking metrics
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.test_results = {}
    
    def forward(self, **inputs):
        return self.model(**inputs)
    
    def training_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        logits = outputs.logits
        
        # Get probabilities and predictions
        probs = torch.softmax(logits, dim=1)
        y_pred = probs[:, 1]  # Probability for positive class
        y_true = batch['labels']
        
        self.val_step_outputs.append({
            'loss': loss,
            'y_pred': y_pred,
            'y_true': y_true
        })
        
        return loss
    
    def on_validation_epoch_end(self):
        # Concatenate all validation outputs
        y_pred = torch.cat([out['y_pred'] for out in self.val_step_outputs]).detach().cpu()
        y_true = torch.cat([out['y_true'] for out in self.val_step_outputs]).detach().cpu()
        val_loss = torch.stack([out['loss'] for out in self.val_step_outputs]).mean().detach().cpu()
        
        # Compute metrics
        metrics = get_binary_metrics(y_pred, y_true)
        
        # Log all metrics
        self.log('val_loss', val_loss, prog_bar=True)
        for k, v in metrics.items():
            self.log(f'val_{k}', v, prog_bar=True)
        
        # Clear saved outputs
        self.val_step_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        logits = outputs.logits
        
        # Get probabilities and predictions
        probs = torch.softmax(logits, dim=1)
        y_pred = probs[:, 1]  # Probability for positive class
        y_true = batch['labels']
        
        self.test_step_outputs.append({
            'loss': loss,
            'y_pred': y_pred,
            'y_true': y_true
        })
        
        return loss
    
    def on_test_epoch_end(self):
        # Concatenate all test outputs
        y_pred = torch.cat([out['y_pred'] for out in self.test_step_outputs]).detach().cpu()
        y_true = torch.cat([out['y_true'] for out in self.test_step_outputs]).detach().cpu()
        test_loss = torch.stack([out['loss'] for out in self.test_step_outputs]).mean().detach().cpu()
        
        # Compute metrics
        metrics = get_binary_metrics(y_pred, y_true)
        
        # Log all metrics
        self.log('test_loss', test_loss)
        for k, v in metrics.items():
            self.log(f'test_{k}', v)
        
        # Store results for later saving - in the same format as get_embeddings.py
        self.test_results = {
            'y_pred': y_pred,
            'y_true': y_true,
            'test_loss': test_loss
        }
        
        # Clear saved outputs
        self.test_step_outputs.clear()
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


def run_finetuning(model_name, dataset_name, batch_size=16, learning_rate=2e-5, max_epochs=50, patience=5):
    """Run the fine-tuning process for a specific model and dataset."""
    # Set seed for reproducibility
    set_seed(42)
    
    # Create data module
    data_module = MimicDataModule(
        model_name=model_name,
        dataset_name=dataset_name,
        batch_size=batch_size
    )
    
    # Create model
    model = LanguageModelFineTuner(
        model_name=model_name,
        learning_rate=learning_rate
    )
    
    # Set up logging directory
    log_dir = f"logs/mimic-iii-note/{dataset_name}/{model_name}/finetune_setting"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(save_dir=log_dir, name="", version=None)
    
    # Set up callbacks
    early_stopping = EarlyStopping(
        monitor="val_auroc",
        mode="max",
        patience=patience,
        verbose=False,
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir, "checkpoints"),
        filename="best-model",
        monitor="val_auroc",
        mode="max",
        save_top_k=1,
        verbose=False,
    )
    
    # Create trainer
    trainer = L.Trainer(
        accelerator="auto",
        max_epochs=max_epochs,
        callbacks=[early_stopping, checkpoint_callback],
        logger=logger,
        deterministic=True
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Load best model for testing
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"Loading best model from {best_model_path}")
        best_model = LanguageModelFineTuner.load_from_checkpoint(best_model_path)
    else:
        best_model = model
    
    # Test model
    trainer.test(best_model, data_module)
    
    # Save test results
    results_path = os.path.join(logger.log_dir, "test_results.pkl")
    pd.to_pickle(best_model.test_results, results_path)
    
    print(f"Model training complete. Test results saved to {results_path}")
    return best_model.test_results


def main():
    parser = argparse.ArgumentParser(description='Fine-tune BERT-based models for MIMIC-III tasks')
    
    # Get BERT models from config
    bert_models = [model["model_name"] for model in MODELS_CONFIG if model["model_type"] == "BERT"]
    
    parser.add_argument('--model', type=str, default='BERT', choices=bert_models)
    parser.add_argument('--dataset', type=str, default='discharge', choices=['discharge', 'mortality'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=5)
    args = parser.parse_args()
    
    # Run fine-tuning
    run_finetuning(
        model_name=args.model,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        patience=args.patience
    )


if __name__ == "__main__":
    main()