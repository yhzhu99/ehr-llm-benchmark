# %%
import os

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler

import pandas as pd

from utils.config import BERTBasedModels, LLM
from utils.classification_metrics import get_binary_metrics, check_metric_is_better

# %%
# Dataset class
class MyDataset(Dataset):
    def __init__(self, data_path, mode='train'):
        super().__init__()
        self.data = pd.read_pickle(os.path.join(data_path, f"embed_{mode}.pkl"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]['embedding'].float()
        y = torch.tensor(self.data[index]['label']).float()
        return x.squeeze(dim=0), y.unsqueeze(dim=0)


class MyDataModule(L.LightningDataModule):
    def __init__(self, batch_size, data_path):
        # data_path is like "logs/embeddings/OpenBioLLM/noteevent/embed_test.pkl"
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = MyDataset(data_path, mode="train")
        self.val_dataset = MyDataset(data_path, mode='valid')
        self.test_dataset = MyDataset(data_path, mode='test')
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)

# %%
class Head(nn.Module):
    def __init__(self, hidden_dim, output_dim=1, drop=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.proj(x)

# %%
class Pipeline(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config["hidden_dim"]
        self.learning_rate = config["learning_rate"]
        self.output_dim = 1

        self.model = Head(self.hidden_dim, self.output_dim)
        self.loss_fn = nn.BCELoss()

        self.cur_best_performance = {} # val set
        self.test_performance = {} # test set

        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.test_outputs = {}


    def forward(self, batch):
        x, y = batch
        y_hat = self.model(x).to(x.device)
        return y_hat

    def _get_loss(self, batch):
        x, y = batch
        y_hat = self(batch)
        loss = self.loss_fn(y_hat, y)
        return loss, y_hat

    def training_step(self, batch, batch_idx):
        loss, y_hat = self._get_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y = batch
        loss, y_hat = self._get_loss(batch)
        self.log("val_loss", loss)
        outs = {'y_pred': y_hat, 'y_true': y, 'val_loss': loss}
        self.validation_step_outputs.append(outs)
        return loss

    def on_validation_epoch_end(self):
        y_pred = torch.cat([x['y_pred'] for x in self.validation_step_outputs]).detach().cpu()
        y_true = torch.cat([x['y_true'] for x in self.validation_step_outputs]).detach().cpu()
        loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean().detach().cpu()
        self.log("val_loss_epoch", loss)

        metrics = get_binary_metrics(y_pred, y_true)
        for k, v in metrics.items(): self.log(k, v)

        main_metric = "auroc"
        main_score = metrics[main_metric]
        if check_metric_is_better(self.cur_best_performance, main_score, main_metric):
            self.cur_best_performance = metrics
            for k, v in metrics.items(): self.log("best_"+k, v)
        self.validation_step_outputs.clear()
        return main_score

    def test_step(self, batch, batch_idx):
        loss, y = batch
        loss, y_hat = self._get_loss(batch)
        self.log("test_loss", loss)
        outs = {'y_pred': y_hat, 'y_true': y, 'test_loss': loss}
        self.test_step_outputs.append(outs)
        return loss

    def on_test_epoch_end(self):
        y_pred = torch.cat([x['y_pred'] for x in self.test_step_outputs]).detach().cpu()
        y_true = torch.cat([x['y_true'] for x in self.test_step_outputs]).detach().cpu()
        loss = torch.stack([x['test_loss'] for x in self.test_step_outputs]).mean().detach().cpu()
        self.log("test_loss_epoch", loss)

        test_performance = get_binary_metrics(y_pred, y_true)
        for k, v in test_performance.items(): self.log("test_"+k, v)

        self.test_outputs = {'y_pred': y_pred, 'y_true': y_true, 'test_loss': loss}
        self.test_step_outputs.clear()

        self.test_performance = test_performance
        return test_performance

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

# %%
def run_experiment(config):
    # data
    dm = MyDataModule(batch_size=config["batch_size"], data_path=f"logs/embeddings/{config['model']}/{config['task']}/")

    # logger
    logger = CSVLogger(save_dir="logs", name=f"embeddings/{config['model']}", version=f"{config['task']}", flush_logs_every_n_steps=1)

    # EarlyStop and checkpoint callback
    early_stopping_callback = EarlyStopping(monitor="auroc", patience=config["patience"], mode="max")
    checkpoint_callback = ModelCheckpoint(filename="best", monitor="auroc", mode="max")

    L.seed_everything(42) # seed for reproducibility

    # train/val/test
    pipeline = Pipeline(config)
    trainer = L.Trainer(accelerator="gpu", devices=[1], max_epochs=config["epochs"], logger=logger, callbacks=[early_stopping_callback, checkpoint_callback])
    trainer.fit(pipeline, dm)

    # Load best model checkpoint
    best_model_path = checkpoint_callback.best_model_path
    print("best_model_path:", best_model_path)
    pipeline = Pipeline.load_from_checkpoint(best_model_path, config=config)
    trainer.test(pipeline, dm)

    perf = pipeline.test_performance
    outs = pipeline.test_outputs
    return perf, outs

# %%
for specify_model in BERTBasedModels + LLM:
    for specify_task in ["discharge", "noteevent"]:
        if specify_model in ["MedAlpaca", "HuatuoGPT", "meditron", "OpenBioLLM", "Llama3"]:
            hidden_dim = 4096
        elif specify_model in ["BioGPT", "GatorTron"]:
            hidden_dim = 1024
        else:
            hidden_dim = 768
        config = {
            'model': specify_model,
            'task': specify_task, # ['noteevent', 'discharge']
            'hidden_dim': hidden_dim, # ClinicalLongformer embedding
            'learning_rate': 1e-4,
            'batch_size': 256,
            'epochs': 50,
            'patience': 5,
        }

        perf, outs = run_experiment(config)
        pd.to_pickle(outs, f"logs/embeddings/{config['model']}/{config['task']}/outs.pkl")
        print(config, perf)
        print(specify_model, "-"*20)
