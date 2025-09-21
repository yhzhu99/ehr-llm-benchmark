"""
src/multimodal/fusion_training.py
Train a fusion model using pre-trained EHR and Note embeddings for prediction tasks.
"""

import os
from pathlib import Path
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from src.unstructured_note.utils.classification_metrics import get_binary_metrics

torch.manual_seed(42)

parser = argparse.ArgumentParser(description='Train a fusion model for MIMIC-IV prediction')
parser.add_argument('--fusion_mode', type=str, required=True, choices=['add', 'concat', 'attention', 'cross_attention'], help="Method to fuse EHR and Note embeddings.")
parser.add_argument('--dataset', type=str, required=True, choices=['mimic-iv', 'mimic-iii'], help="The dataset.")
parser.add_argument('--task', type=str, required=True, choices=['mortality', 'readmission'], help="The prediction task.")
parser.add_argument('--ehr_model', type=str, required=True, help="The EHR model.")
parser.add_argument('--note_model', type=str, required=True, help="The Note model.")
parser.add_argument('--hidden_dim', type=int, default=1024, help="The dimension of the hidden layer.")
parser.add_argument('--num_heads', type=int, default=8, help="The number of attention heads.")
parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training and evaluation.")
parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate for the optimizer.")
parser.add_argument('--epochs', type=int, default=50, help="Maximum number of training epochs.")
parser.add_argument('--patience', type=int, default=10, help="Patience for early stopping.")


class FusionDataset(Dataset):
    """
    Dataset to load pre-trained EHR and Note embeddings.
    """
    def __init__(self, ehr_embedding_path, note_embedding_path, split):
        super().__init__()
        ehr_embedding_path = os.path.join(ehr_embedding_path, f"{split}_embeddings.pkl")
        ehr_data = pd.read_pickle(ehr_embedding_path)
        self.ehr_embeddings = ehr_data['embeddings'].squeeze()
        self.labels = ehr_data['labels']
        note_embedding_path = os.path.join(note_embedding_path, f"{split}_embeddings.pt")
        self.note_embeddings = torch.load(note_embedding_path)['embeddings'].squeeze()

        assert len(self.ehr_embeddings) == len(self.note_embeddings) == len(self.labels), \
            "Mismatch in the number of samples between embeddings and labels."

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ehr_embedding = torch.tensor(self.ehr_embeddings[idx], dtype=torch.float32)
        note_embedding = self.note_embeddings[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        return {
            'ehr_embedding': ehr_embedding,
            'note_embedding': note_embedding,
            'labels': label.unsqueeze(0)
        }


class FusionDataModule(L.LightningDataModule):
    """
    LightningDataModule for loading fusion data.
    """
    def __init__(self, dataset, task, ehr_model_name, note_model_name, batch_size=32):
        super().__init__()
        self.dataset = dataset
        self.task = task
        self.ehr_model_name = ehr_model_name
        self.note_model_name = note_model_name
        self.batch_size = batch_size

    def setup(self, stage=None):
        ehr_embedding_path = f"logs/multimodal/{self.dataset}/{self.task}/{self.ehr_model_name}/embeddings"
        note_embedding_path = f"logs/multimodal/{self.dataset}/{self.task}/{self.note_model_name}/embeddings"
        if stage == 'fit' or stage is None:
            self.train_dataset = FusionDataset(ehr_embedding_path, note_embedding_path, 'train')
            self.val_dataset = FusionDataset(ehr_embedding_path, note_embedding_path, 'val')
        if stage == 'test' or stage is None:
            self.test_dataset = FusionDataset(ehr_embedding_path, note_embedding_path, 'test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)


class FusionModel(L.LightningModule):
    """
    LightningModule for fusing EHR and Note embeddings.
    """
    def __init__(self, fusion_mode, ehr_dim=128, note_dim=1024, hidden_dim=1024, num_heads=8, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.fusion_mode = fusion_mode
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim
        self.ehr_proj = nn.Linear(ehr_dim, hidden_dim)
        self.note_proj = nn.Linear(note_dim, hidden_dim)

        # 根据融合方式定义模型层
        if self.fusion_mode == 'add':
            fusion_dim = hidden_dim
        if self.fusion_mode == 'concat':
            fusion_dim = 2 * hidden_dim
        elif self.fusion_mode == 'attention':
            self.ehr_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
            self.note_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
            fusion_dim = 2 * hidden_dim
        elif self.fusion_mode == 'cross_attention':
            self.attention1 = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
            self.attention2 = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
            fusion_dim = 2 * hidden_dim

        # 定义最终的MLP分类器
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_dim // 2, 1)
        )

        self.loss_fn = nn.BCEWithLogitsLoss()

        # 用于记录评估结果
        self.validation_outputs = []
        self.test_outputs = []
        self.test_results = {}

    def forward(self, ehr_embedding, note_embedding):
        ehr_embedding = self.ehr_proj(ehr_embedding)
        note_embedding = self.note_proj(note_embedding)

        if self.fusion_mode == 'add':
            fused_embedding = F.relu(ehr_embedding + note_embedding)
        elif self.fusion_mode == 'concat':
            fused_embedding = torch.cat((ehr_embedding, note_embedding), dim=1)
        elif self.fusion_mode == 'attention':
            ehr_seq = ehr_embedding.unsqueeze(1)
            note_seq = note_embedding.unsqueeze(1)
            ehr_attn_output, _ = self.ehr_attention(ehr_seq, ehr_seq, ehr_seq)
            note_attn_output, _ = self.note_attention(note_seq, note_seq, note_seq)
            print(ehr_attn_output.shape, note_attn_output.shape)
            fused_embedding = torch.cat((ehr_attn_output, note_attn_output), dim=-1).squeeze()
        elif self.fusion_mode == 'cross_attention':
            ehr_seq = ehr_embedding.unsqueeze(1)
            note_seq = note_embedding.unsqueeze(1)
            attn_output1, _ = self.attention1(ehr_seq, note_seq, note_seq)
            attn_output2, _ = self.attention2(note_seq, ehr_seq, ehr_seq)
            fused_embedding = torch.cat((attn_output1, attn_output2), dim=-1).squeeze()

        logits = self.classifier(fused_embedding)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch['ehr_embedding'], batch['note_embedding'])
        loss = self.loss_fn(logits, batch['labels'])
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch['ehr_embedding'], batch['note_embedding'])
        loss = self.loss_fn(logits, batch['labels'])
        preds = torch.sigmoid(logits)
        labels = batch['labels']

        self.validation_outputs.append({
            'preds': preds.detach(),
            'labels': labels.detach(),
            'loss': loss.detach()
        })
        return loss

    def on_validation_epoch_end(self):
        preds = torch.cat([x['preds'] for x in self.validation_outputs]).squeeze().cpu()
        labels = torch.cat([x['labels'] for x in self.validation_outputs]).squeeze().cpu()
        loss = torch.stack([x['loss'] for x in self.validation_outputs]).mean().cpu()

        metrics = get_binary_metrics(preds, labels)
        self.log('val_loss', loss, prog_bar=True)
        for k, v in metrics.items():
            self.log(f'val_{k}', v, prog_bar=True)

        self.validation_outputs.clear()

    def test_step(self, batch, batch_idx):
        logits = self.forward(batch['ehr_embedding'], batch['note_embedding'])
        loss = self.loss_fn(logits, batch['labels'])
        preds = torch.sigmoid(logits)
        labels = batch['labels']

        self.test_outputs.append({
            'preds': preds.detach(),
            'labels': labels.detach(),
            'loss': loss.detach()
        })
        return loss

    def on_test_epoch_end(self):
        preds = torch.cat([x['preds'] for x in self.test_outputs]).squeeze().cpu()
        labels = torch.cat([x['labels'] for x in self.test_outputs]).squeeze().cpu()
        loss = torch.stack([x['loss'] for x in self.test_outputs]).mean().cpu()

        metrics = get_binary_metrics(preds, labels)
        self.log('test_loss', loss)
        for k, v in metrics.items():
            self.log(f'test_{k}', v)

        self.test_results = {
            'y_pred': preds,
            'y_true': labels,
            'test_loss': loss.item(),
            **{f'test_{k}': v for k, v in metrics.items()}
        }

        self.test_outputs.clear()
        return metrics

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


def get_embedding_dim(model_name):
    """Get embedding dimension for model"""
    if model_name in ["BERT", "ClinicalBERT", "BioBERT", "Clinical-Longformer", "GPT-2"]:
        return 768
    elif model_name in ["GatorTron", "BioGPT"]:
        return 1024
    elif model_name in ["meditron", "OpenBioLLM", "BioMistral"]:
        return 4096
    elif model_name in ["Qwen2.5-7B", "HuatuoGPT-o1-7B", "DeepSeek-R1-Distill-Qwen-7B"]:
        return 3584
    elif model_name in ["gemma-3-4b-pt"]:
        return 2560
    else:
        return 768


def run_training():
    """Run the full training and evaluation pipeline for the fusion model."""
    args = parser.parse_args()
    fusion_mode = args.fusion_mode
    task = args.task
    dataset = args.dataset
    ehr_model = args.ehr_model
    note_model = args.note_model
    note_embedding_dim = get_embedding_dim(note_model)

    print(f"Training fusion model with mode '{fusion_mode}' for '{task}' on '{dataset}' prediction")

    # 创建DataModule
    data_module = FusionDataModule(dataset=dataset, task=task, ehr_model_name=ehr_model, note_model_name=note_model, batch_size=args.batch_size)

    # 创建模型
    model = FusionModel(
        fusion_mode=fusion_mode,
        note_dim=note_embedding_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        learning_rate=args.learning_rate
    )

    # 设置输出目录
    log_dir = f"logs/multimodal/{dataset}/{task}/{ehr_model}-{note_model}/fusion_training/{fusion_mode}"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = CSVLogger(save_dir=log_dir, name="", version=None)
    results_path = os.path.join(log_dir, "test_results.pkl")

    if os.path.exists(results_path):
        print(f"Results already exist at {results_path}")
        return

    # 创建回调
    early_stopping = EarlyStopping(
        monitor="val_auroc",
        mode="max",
        patience=args.patience,
        verbose=True
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_dir, "checkpoints"),
        filename="best-model-{epoch:02d}-{val_auroc:.4f}",
        monitor="val_auroc",
        mode="max",
        save_top_k=1,
        verbose=True
    )

    # 创建Trainer
    trainer = L.Trainer(
        accelerator="auto",
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[early_stopping, checkpoint_callback]
    )

    # 训练模型
    trainer.fit(model, data_module)

    # 加载最佳模型进行测试
    best_model_path = checkpoint_callback.best_model_path
    print(f"Loading best model from {best_model_path}")
    best_model = FusionModel.load_from_checkpoint(best_model_path)

    # 测试模型
    trainer.test(best_model, data_module)

    # 保存结果
    pd.to_pickle(best_model.test_results, results_path)

    print(f"Training pipeline complete. Results would be saved to {results_path}")

    return best_model.test_results

if __name__ == "__main__":
    run_training()