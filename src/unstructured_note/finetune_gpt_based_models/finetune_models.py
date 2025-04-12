"""
src/unstructured_note/finetune_gpt_based_models/finetune_models.py
Fine-tune GPT-based models for MIMIC-IV prediction tasks using PEFT IA3
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Add this line at the top

from pathlib import Path
import argparse
import random

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import pandas as pd

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed
from peft import IA3Config, get_peft_model

from unstructured_note.utils.config import MODELS_CONFIG
from unstructured_note.utils.classification_metrics import get_binary_metrics

# Set seed for reproducibility
set_seed(42)

# Get GPT models from config
GPT_MODELS = [model["model_name"] for model in MODELS_CONFIG if model["model_type"] == "GPT"]

parser = argparse.ArgumentParser(description='Fine-tune GPT models for MIMIC-IV using PEFT IA3')
parser.add_argument('--model', type=str, required=True, choices=GPT_MODELS)
parser.add_argument('--task', type=str, required=True, choices=['mortality', 'readmission'])
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--learning_rate', type=float, default=8e-3)  # Higher LR for IA3
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--max_length', type=int, default=512)
args = parser.parse_args()

# Custom sampler to randomly select a subset of samples each epoch
class RandomSubsetSampler(Sampler):
    """Samples a random subset of elements from a dataset at the beginning of each epoch."""

    def __init__(self, data_source, num_samples=2000):
        """
        Args:
            data_source: Dataset to sample from
            num_samples: Number of samples to draw per epoch
        """
        self.data_source = data_source
        self.num_samples = min(num_samples, len(data_source))

    def __iter__(self):
        # Randomly select indices without replacement
        indices = random.sample(range(len(self.data_source)), self.num_samples)
        return iter(indices)

    def __len__(self):
        return self.num_samples

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

        # Add task-specific prompt for GPT models
        task_prefix = f"Predict patient {self.task} based on the following clinical note: "
        full_text = task_prefix + text

        inputs = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        # Remove batch dimension from tokenizer output
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        label = item[f'y_{self.task}'][0]
        inputs['labels'] = torch.tensor(label, dtype=torch.long)

        return inputs

# Lightning data module
class MimicDataModule(L.LightningDataModule):
    def __init__(self, model_name, task, batch_size=8, max_length=512):
        super().__init__()
        self.model_name = model_name
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
        # Ensure the tokenizer has a padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def setup(self, stage=None):
        # No additional setup needed, datasets are loaded in dataloaders
        pass

    def train_dataloader(self):
        train_dataset = MimicDataset(
            data_path="my_datasets/mimic-iv/processed/split/train_data.pkl",
            tokenizer=self.tokenizer,
            task=self.task,
            max_length=self.max_length
        )

        # Create a random subset sampler that selects 2000 random samples for each epoch
        sampler = RandomSubsetSampler(train_dataset, num_samples=2000)

        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,  # Use our custom sampler instead of shuffle=True
            num_workers=4,
            persistent_workers=True
        )

    def val_dataloader(self):
        val_dataset = MimicDataset(
            data_path="my_datasets/mimic-iv/processed/split/val_data.pkl",
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
            data_path="my_datasets/mimic-iv/processed/split/test_data.pkl",
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

# Model for fine-tuning with IA3
class GptIA3FineTuner(L.LightningModule):
    def __init__(self, model_name, num_labels=2, learning_rate=8e-3):
        super().__init__()
        self.save_hyperparameters()

        # Get model path
        model_config = next(model for model in MODELS_CONFIG if model["model_name"] == model_name)
        self.model_path = model_config["hf_id"]

        # Load the base model
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            num_labels=num_labels,
            trust_remote_code=True
        )

        # Ensure padding token is correctly set for GPT models
        if hasattr(self.base_model.config, 'pad_token_id') and self.base_model.config.pad_token_id is None:
            self.base_model.config.pad_token_id = self.base_model.config.eos_token_id

        # Setup IA3 configuration
        peft_config = IA3Config(task_type="SEQ_CLS")

        # Create PEFT model with IA3
        self.model = get_peft_model(self.base_model, peft_config)

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
        # IA3 typically does well with higher learning rates
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

def run_finetuning():
    """Run the full fine-tuning and evaluation pipeline"""
    model_name = args.model
    task = args.task

    print(f"Fine-tuning {model_name} for {task} prediction using IA3")
    print(f"Only a small number of parameters will be trained while the base model remains frozen")

    # Create data module
    data_module = MimicDataModule(
        model_name=model_name,
        task=task,
        batch_size=args.batch_size,
        max_length=args.max_length
    )

    # Create model
    model = GptIA3FineTuner(
        model_name=model_name,
        learning_rate=args.learning_rate
    )

    # Print trainable parameters
    model.model.print_trainable_parameters()

    # Setup output directory
    log_dir = f"logs/mimic-iv-note/{model_name}/{task}/finetune_setting"
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
    best_model = GptIA3FineTuner.load_from_checkpoint(best_model_path)

    # Test model
    trainer.test(best_model, data_module)

    # Save results
    results_path = os.path.join(log_dir, "test_results.pkl")
    pd.to_pickle(best_model.test_results, results_path)

    # Save the PEFT model
    peft_model_path = os.path.join(log_dir, "peft_model")
    best_model.model.save_pretrained(peft_model_path)
    print(f"PEFT model saved to {peft_model_path}")

    print(f"Finetuning complete. Results saved to {results_path}")

    return best_model.test_results

if __name__ == "__main__":
    run_finetuning()