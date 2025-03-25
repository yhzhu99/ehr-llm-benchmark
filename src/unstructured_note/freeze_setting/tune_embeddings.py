# tune_embeddings.py
import os
import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
from transformers import set_seed

from unstructured_note.utils.config import MODELS_CONFIG
from unstructured_note.utils.classification_metrics import get_binary_metrics, check_metric_is_better

# Set seeds for reproducibility
set_seed(42)

# Create model type lists based on MODELS_CONFIG
BERTBasedModels = [model["model_name"] for model in MODELS_CONFIG if model["model_type"] == "BERT"]
LLM = [model["model_name"] for model in MODELS_CONFIG if model["model_type"] == "GPT"]

# Parse arguments
parser = argparse.ArgumentParser(description='Fine-tune embeddings with MLP')
parser.add_argument('--model', type=str, default='BERT', choices=BERTBasedModels + LLM + ['all'])
parser.add_argument('--dataset', type=str, default='discharge', choices=['discharge', 'mortality', 'all'])
parser.add_argument('--cuda', type=int, default=0, choices=[0,1,2,3,4,5,6,7])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--patience', type=int, default=5)
args = parser.parse_args()

# Dataset class
class EmbeddingDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = pd.read_pickle(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]['embedding'].float()
        y = torch.tensor(self.data[index]['label']).float()
        return x, y.unsqueeze(dim=0)


class EmbeddingDataModule(L.LightningDataModule):
    def __init__(self, batch_size, model_name, dataset_name):
        super().__init__()
        self.batch_size = batch_size
        self.base_path = f"logs/mimic-iii-note/{dataset_name}/{model_name}/embeddings"
        
        # Ensure the paths exist
        Path(self.base_path).mkdir(parents=True, exist_ok=True)
        
        self.train_dataset = EmbeddingDataset(os.path.join(self.base_path, "embed_train.pkl"))
        self.val_dataset = EmbeddingDataset(os.path.join(self.base_path, "embed_valid.pkl"))
        self.test_dataset = EmbeddingDataset(os.path.join(self.base_path, "embed_test.pkl"))
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                          shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                         shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, 
                         shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)


# MLP Head for classification
class ClassificationHead(nn.Module):
    def __init__(self, hidden_dim, output_dim=1, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(x)


# Lightning module for training the classification head
class EmbeddingClassifier(L.LightningModule):
    def __init__(self, hidden_dim, learning_rate, output_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.output_dim = output_dim

        self.model = ClassificationHead(self.hidden_dim, self.output_dim)
        self.loss_fn = nn.BCELoss()

        # For tracking metrics
        self.current_best_metrics = {}
        self.test_metrics = {}
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.test_results = {}

    def forward(self, x):
        return self.model(x)

    def _compute_loss_and_preds(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        return loss, y_hat, y

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._compute_loss_and_preds(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._compute_loss_and_preds(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.validation_step_outputs.append({
            'y_pred': y_hat, 
            'y_true': y, 
            'val_loss': loss
        })
        return loss

    def on_validation_epoch_end(self):
        y_pred = torch.cat([x['y_pred'] for x in self.validation_step_outputs]).detach().cpu()
        y_true = torch.cat([x['y_true'] for x in self.validation_step_outputs]).detach().cpu()
        loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean().detach().cpu()
        
        self.log("val_loss_epoch", loss)

        # Calculate and log metrics
        metrics = get_binary_metrics(y_pred, y_true)
        for k, v in metrics.items():
            self.log(f"val_{k}", v, prog_bar=True)

        # Track best performance
        main_metric = "auroc"
        main_score = metrics[main_metric]
        if check_metric_is_better(self.current_best_metrics, main_score, main_metric):
            self.current_best_metrics = metrics
            for k, v in metrics.items():
                self.log(f"best_{k}", v)
        
        self.validation_step_outputs.clear()
        return main_score

    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self._compute_loss_and_preds(batch)
        self.log("test_loss", loss)
        self.test_step_outputs.append({
            'y_pred': y_hat, 
            'y_true': y, 
            'test_loss': loss
        })
        return loss

    def on_test_epoch_end(self):
        y_pred = torch.cat([x['y_pred'] for x in self.test_step_outputs]).detach().cpu()
        y_true = torch.cat([x['y_true'] for x in self.test_step_outputs]).detach().cpu()
        loss = torch.stack([x['test_loss'] for x in self.test_step_outputs]).mean().detach().cpu()
        
        self.log("test_loss_epoch", loss)

        # Calculate and log test metrics
        test_metrics = get_binary_metrics(y_pred, y_true)
        for k, v in test_metrics.items():
            self.log(f"test_{k}", v)

        # Store results for later analysis
        self.test_results = {
            'y_pred': y_pred, 
            'y_true': y_true, 
            'test_loss': loss
        }
        self.test_metrics = test_metrics
        self.test_step_outputs.clear()
        
        return test_metrics

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


def get_model_hidden_dim(model_name):
    """Determine the hidden dimension based on model name"""
    model_config = next((m for m in MODELS_CONFIG if m["model_name"] == model_name), None)
    
    if not model_config:
        raise ValueError(f"Model {model_name} not found in MODELS_CONFIG")
    
    # Determine hidden dim based on model type and specific models
    if model_config["model_type"] == "GPT":
        if model_name in ["MedAlpaca", "HuatuoGPT", "meditron", "OpenBioLLM", "Llama3", 
                         "QwQ-32B", "DeepSeek-R1-Distill-Qwen-7B", "Qwen2.5-7B", 
                         "BioMistral", "Baichuan-M1"]:
            return 4096
        elif model_name in ["BioGPT", "GatorTron"]:
            return 1024
        else:
            return 768  # Default for GPT-2
    else:  # BERT models
        if model_name == "GatorTron":
            return 1024
        else:
            return 768  # Default for BERT models


def run_experiment(model_name, dataset_name, batch_size, learning_rate, epochs, patience, cuda_device):
    """Run the full training and evaluation pipeline"""
    
    # Determine hidden dimension for the model
    hidden_dim = get_model_hidden_dim(model_name)
    
    # Create data module
    data_module = EmbeddingDataModule(
        batch_size=batch_size,
        model_name=model_name,
        dataset_name=dataset_name
    )
    
    # Set up logging
    log_dir = f"logs/mimic-iii-note/{dataset_name}/{model_name}/embeddings/tuned"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(save_dir=log_dir, name="classification", version="0")
    
    # Set up callbacks
    early_stopping = EarlyStopping(
        monitor="val_auroc",
        patience=patience,
        mode="max",
        verbose=False
    )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir,
        filename="best_model",
        monitor="val_auroc",
        mode="max",
        save_top_k=1,
        verbose=False
    )
    
    # Create the classifier
    classifier = EmbeddingClassifier(
        hidden_dim=hidden_dim,
        learning_rate=learning_rate
    )
    
    # Set up trainer
    trainer = L.Trainer(
        accelerator="auto",
        max_epochs=epochs,
        logger=logger,
        callbacks=[early_stopping, checkpoint_callback],
        deterministic=True,
        log_every_n_steps=10
    )
    
    # Train the model
    print(f"Training classifier for {model_name} on {dataset_name} dataset")
    trainer.fit(classifier, datamodule=data_module)
    
    # Load best model and evaluate
    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")
    
    best_classifier = EmbeddingClassifier.load_from_checkpoint(
        best_model_path,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate
    )
    
    # Test the model
    trainer.test(model=best_classifier, datamodule=data_module)
    
    # Save results
    results_path = os.path.join(log_dir, "test_results.pkl")
    pd.to_pickle(best_classifier.test_results, results_path)
    
    print(f"Test results for {model_name} on {dataset_name}:")
    for metric, value in best_classifier.test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return best_classifier.test_metrics, best_classifier.test_results


# Main execution
if __name__ == "__main__":
    datasets_to_run = ["discharge", "mortality"] if args.dataset == "all" else [args.dataset]
    models_to_run = BERTBasedModels + LLM if args.model == "all" else [args.model]
    
    for model_name in models_to_run:
        for dataset_name in datasets_to_run:
            print(f"Running embedding fine-tuning for model: {model_name} on dataset: {dataset_name}")
            
            metrics, results = run_experiment(
                model_name=model_name,
                dataset_name=dataset_name,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                epochs=args.epochs,
                patience=args.patience,
                cuda_device=args.cuda
            )
            
            print(f"Finished fine-tuning {model_name} on {dataset_name}")
            print("Test metrics:", metrics)
            print("-" * 50)