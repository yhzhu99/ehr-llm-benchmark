"""
src/unstructured_note/finetune_bert_based_models/finetune_models.py
Fine-tune BERT-based models for MIMIC-III or MIMIC-IV prediction tasks
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Add this line at the top

from pathlib import Path
import argparse

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed

from src.unstructured_note.utils.config import MODELS_CONFIG
from src.unstructured_note.utils.classification_metrics import get_binary_metrics

# Set seed for reproducibility
set_seed(42)

# Get BERT models from config
BERT_MODELS = [model["model_name"] for model in MODELS_CONFIG if model["model_type"] == "BERT"]

parser = argparse.ArgumentParser(description='Fine-tune BERT models for MIMIC-III or MIMIC-IV')
parser.add_argument('--model', type=str, required=True, choices=BERT_MODELS)
parser.add_argument('--dataset', type=str, required=True, choices=["mimic-iv", "mimic-iii"])
parser.add_argument('--task', type=str, required=True, choices=['mortality', 'readmission'])
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--learning_rate', type=float, default=2e-5)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--max_length', type=int, default=512)
args = parser.parse_args()

# Dataset class
class MimicDataset(Dataset):
    def __init__(self, data_path, tokenizer, task, max_length=512):
        super().__init__()
        self.data = pd.read_pickle(data_path)
        self.tokenizer = tokenizer
        self.task = task
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['x_note']

        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        # Remove batch dimension from tokenizer output
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        label = item[f'y_{self.task}'][0] if isinstance(item[f'y_{self.task}'], list) else item[f'y_{self.task}']
        inputs['labels'] = torch.tensor(label, dtype=torch.long)

        return inputs

# Lightning data module
class MimicDataModule(L.LightningDataModule):
    def __init__(self, model_name, dataset, task, batch_size=8, max_length=512):
        super().__init__()
        self.model_name = model_name
        self.dataset = dataset
        self.task = task
        self.batch_size = batch_size
        self.max_length = max_length

        # Get model path
        model_config = next(model for model in MODELS_CONFIG if model["model_name"] == model_name)
        self.model_path = model_config["hf_id"]

        # Initialize tokenizer
        self.tokenizer = None

    def prepare_data(self):
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

    def setup(self, stage=None):
        # No additional setup needed, datasets are loaded in dataloaders
        pass

    def train_dataloader(self):
        train_dataset = MimicDataset(
            data_path=f"my_datasets/{self.dataset}/processed/split/train_data.pkl",
            tokenizer=self.tokenizer,
            task=self.task,
            max_length=self.max_length
        )
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True
        )

    def val_dataloader(self):
        val_dataset = MimicDataset(
            data_path=f"my_datasets/{self.dataset}/processed/split/val_data.pkl",
            tokenizer=self.tokenizer,
            task=self.task,
            max_length=self.max_length
        )
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            persistent_workers=True
        )

    def test_dataloader(self):
        test_dataset = MimicDataset(
            data_path=f"my_datasets/{self.dataset}/processed/split/test_data.pkl",
            tokenizer=self.tokenizer,
            task=self.task,
            max_length=self.max_length,
        )
        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            persistent_workers=True
        )

# Model for fine-tuning
class BertFineTuner(L.LightningModule):
    def __init__(self, model_name, num_labels=2, learning_rate=2e-5):
        super().__init__()
        self.save_hyperparameters()

        # Get model path
        model_config = next(model for model in MODELS_CONFIG if model["model_name"] == model_name)
        self.model_path = model_config["hf_id"]

        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            num_labels=num_labels if model_name != "BioBERT" else 5,
            trust_remote_code=True
        )

        self.learning_rate = learning_rate

        # For tracking metrics
        self.validation_outputs = []
        self.test_outputs = []
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

        # Get predictions
        preds = torch.softmax(logits, dim=1)[:, 1]  # Probability of positive class
        labels = batch['labels']

        self.validation_outputs.append({
            'preds': preds.detach(),
            'labels': labels.detach(),
            'loss': loss.detach()
        })

        return loss

    def on_validation_epoch_end(self):
        preds = torch.cat([x['preds'] for x in self.validation_outputs]).cpu()
        labels = torch.cat([x['labels'] for x in self.validation_outputs]).cpu()
        loss = torch.stack([x['loss'] for x in self.validation_outputs]).mean().cpu()

        # Calculate metrics
        metrics = get_binary_metrics(preds, labels)

        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        for k, v in metrics.items():
            self.log(f'val_{k}', v, prog_bar=True)

        # Clear outputs
        self.validation_outputs.clear()

    def test_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs.loss
        logits = outputs.logits

        # Get predictions
        preds = torch.softmax(logits, dim=1)[:, 1]  # Probability of positive class
        labels = batch['labels']

        self.test_outputs.append({
            'preds': preds.detach(),
            'labels': labels.detach(),
            'loss': loss.detach()
        })

        return loss

    def on_test_epoch_end(self):
        preds = torch.cat([x['preds'] for x in self.test_outputs]).cpu()
        labels = torch.cat([x['labels'] for x in self.test_outputs]).cpu()
        loss = torch.stack([x['loss'] for x in self.test_outputs]).mean().cpu()

        # Calculate metrics
        metrics = get_binary_metrics(preds, labels)

        # Log metrics
        self.log('test_loss', loss)
        for k, v in metrics.items():
            self.log(f'test_{k}', v)

        # Store results
        self.test_results = {
            'y_pred': preds,
            'y_true': labels,
            'test_loss': loss
        }

        # Clear outputs
        self.test_outputs.clear()

        return metrics

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

def run_finetuning():
    """Run the full fine-tuning and evaluation pipeline"""
    model_name = args.model
    task = args.task
    dataset = args.dataset

    # Setup output directory
    log_dir = f"logs/unstructured_note/{dataset}-note/{model_name}/{task}/finetune_setting"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(save_dir=log_dir, name="", version=None)
    results_path = os.path.join(log_dir, "test_results.pkl")

    if os.path.exists(results_path):
        print(f"Results already exist at {results_path}")
        return

    print(f"Fine-tuning {model_name} for {task} prediction")

    # Create data module
    data_module = MimicDataModule(
        model_name=model_name,
        dataset=dataset,
        task=task,
        batch_size=args.batch_size,
        max_length=args.max_length
    )

    # Create model
    model = BertFineTuner(
        model_name=model_name,
        learning_rate=args.learning_rate
    )

    # Create callbacks
    early_stopping = EarlyStopping(
        monitor="val_auroc",
        mode="max",
        patience=args.patience,
        verbose=False
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_dir, "checkpoints"),
        filename="best-model",
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
    best_model = BertFineTuner.load_from_checkpoint(best_model_path)

    # Test model
    trainer.test(best_model, data_module)

    # Save results
    pd.to_pickle(best_model.test_results, results_path)

    print(f"Finetuning complete. Results saved to {results_path}")

    return best_model.test_results

if __name__ == "__main__":
    run_finetuning()