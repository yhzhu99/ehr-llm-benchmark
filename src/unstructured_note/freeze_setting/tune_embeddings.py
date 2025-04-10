"""
src/unstructured_note/freeze_setting/tune_embeddings.py
Fine-tune embeddings with a simple MLP classifier for downstream tasks
"""

import os
import argparse
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import set_seed

from unstructured_note.utils.config import MODELS_CONFIG
from unstructured_note.utils.classification_metrics import get_binary_metrics

# Set seed for reproducibility
set_seed(42)

# Create model type lists
BERT_MODELS = [model["model_name"] for model in MODELS_CONFIG if model["model_type"] == "BERT"]
LLM_MODELS = [model["model_name"] for model in MODELS_CONFIG if model["model_type"] == "GPT"]

parser = argparse.ArgumentParser(description='Fine-tune embeddings with MLP')
parser.add_argument('--model', type=str, required=True, choices=BERT_MODELS + LLM_MODELS)
parser.add_argument('--task', type=str, required=True, choices=['mortality', 'readmission'])
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--patience', type=int, default=5)
args = parser.parse_args()

# Dataset class for embeddings
class EmbeddingDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = pd.read_pickle(data_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        embedding = item['embedding'].float()
        label = torch.tensor(item[f'y_{args.task}'])[0].float()
        return embedding, label.unsqueeze(dim=0)

# Lightning data module
class EmbeddingDataModule(L.LightningDataModule):
    def __init__(self, model_name, task, batch_size=64):
        super().__init__()
        self.model_name = model_name
        self.task = task
        self.batch_size = batch_size
        self.base_path = f"logs/mimic-iv-note/{model_name}"

    def setup(self, stage=None):
        self.train_dataset = EmbeddingDataset(f"{self.base_path}/train_embeddings.pkl")
        self.val_dataset = EmbeddingDataset(f"{self.base_path}/val_embeddings.pkl")
        self.test_dataset = EmbeddingDataset(f"{self.base_path}/test_embeddings.pkl")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

# MLP classifier
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=1, dropout=0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(x)

# Lightning module for training
class EmbeddingClassifier(L.LightningModule):
    def __init__(self, input_dim, learning_rate=1e-4, hidden_dim=256):
        super().__init__()
        self.save_hyperparameters()
        self.model = MLPClassifier(input_dim, hidden_dim)
        self.learning_rate = learning_rate
        self.criterion = nn.BCELoss()

        # For tracking metrics
        self.validation_outputs = []
        self.test_outputs = []
        self.test_results = {}

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.validation_outputs.append({
            'y_pred': y_hat.detach(),
            'y_true': y.detach(),
            'loss': loss.detach()
        })
        return loss

    def on_validation_epoch_end(self):
        y_pred = torch.cat([x['y_pred'] for x in self.validation_outputs]).cpu()
        y_true = torch.cat([x['y_true'] for x in self.validation_outputs]).cpu()
        loss = torch.stack([x['loss'] for x in self.validation_outputs]).mean().cpu()

        # Calculate metrics
        metrics = get_binary_metrics(y_pred, y_true)

        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        for k, v in metrics.items():
            self.log(f'val_{k}', v, prog_bar=True)

        # Clear outputs
        self.validation_outputs.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.test_outputs.append({
            'y_pred': y_hat.detach(),
            'y_true': y.detach(),
            'loss': loss.detach()
        })
        return loss

    def on_test_epoch_end(self):
        y_pred = torch.cat([x['y_pred'] for x in self.test_outputs]).cpu()
        y_true = torch.cat([x['y_true'] for x in self.test_outputs]).cpu()
        loss = torch.stack([x['loss'] for x in self.test_outputs]).mean().cpu()

        # Calculate metrics
        metrics = get_binary_metrics(y_pred, y_true)

        # Log metrics
        self.log('test_loss', loss)
        for k, v in metrics.items():
            self.log(f'test_{k}', v)

        # Store results
        self.test_results = {
            'y_pred': y_pred,
            'y_true': y_true,
            'test_loss': loss
        }

        # Clear outputs
        self.test_outputs.clear()

        return metrics

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

def get_embedding_dim(model_name):
    """Get embedding dimension for model"""
    # Default dimensions for common models
    if model_name in ["BERT", "ClinicalBERT", "BioBERT", "Clinical-Longformer", "GPT-2"]:
        return 768
    elif model_name in ["GatorTron", "BioGPT"]:
        return 1024
    elif model_name in ["HuatuoGPT-o1-7B", "DeepSeek-R1-Distill-Qwen-7B", "Qwen2.5-7B", "gemma-3-4b-pt", "meditron", "OpenBioLLM", "BioMistral"]:
        return 4096
    else:
        # Default for unknown models
        return 768

def run_training():
    """Run the full training and evaluation pipeline"""
    model_name = args.model
    task = args.task

    print(f"Fine-tuning {model_name} embeddings for {task} prediction")

    # Get embedding dimension for model
    embedding_dim = get_embedding_dim(model_name)

    # Create data module
    data_module = EmbeddingDataModule(
        model_name=model_name,
        task=task,
        batch_size=args.batch_size
    )

    # Create model
    model = EmbeddingClassifier(
        input_dim=embedding_dim,
        learning_rate=args.learning_rate
    )

    # Setup output directory
    log_dir = f"logs/mimic-iv-note/{model_name}/{task}/freeze_setting"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(save_dir=log_dir, name="", version=None)

    # Create callbacks
    early_stopping = EarlyStopping(
        monitor="val_auroc",
        mode="max",
        patience=args.patience,
        verbose=False
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_dir, "checkpoints"),
        filename="best_model",
        monitor="val_auroc",
        mode="max",
        save_top_k=1,
        verbose=False
    )

    # Create trainer
    trainer = L.Trainer(
        accelerator="auto",
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[early_stopping, checkpoint_callback]
    )

    # Train model
    trainer.fit(model, data_module)

    # Load best model for testing
    best_model_path = checkpoint_callback.best_model_path
    print(f"Loading best model from {best_model_path}")
    best_model = EmbeddingClassifier.load_from_checkpoint(best_model_path)

    # Test model
    trainer.test(best_model, data_module)

    # Save results
    results_path = os.path.join(log_dir, "test_results.pkl")
    pd.to_pickle(best_model.test_results, results_path)

    print(f"Training complete. Results saved to {results_path}")

    return best_model.test_results

if __name__ == "__main__":
    run_training()