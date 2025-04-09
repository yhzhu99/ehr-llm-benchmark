# Standard Library
import os
import argparse


# Lightning
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

# PyTorch
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# Data Processing
import pandas as pd

import structured_ehr.models as models
from structured_ehr.utils.bootstrap import run_bootstrap
from structured_ehr.utils.metrics import get_all_metrics, check_metric_is_better
from structured_ehr.utils.loss import get_loss
from structured_ehr.utils.sequence_handler import generate_mask, unpad_y


class EhrDataset(Dataset):
    def __init__(self, data_path, task, mode='train'):
        super().__init__()
        self.dataset = pd.read_pickle(os.path.join(data_path, f'{mode}_data.pkl'))
        self.id = [item['id'] for item in self.dataset]
        self.data = [item['x_ts'] for item in self.dataset]
        self.label = [item[f'y_{task}'] for item in self.dataset]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.id[index]


class EhrDataModule(L.LightningDataModule):
    def __init__(self, data_path, task, batch_size=32):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size

        self.train_dataset = EhrDataset(self.data_path, task, mode="train")
        self.val_dataset = EhrDataset(self.data_path, task, mode='val')
        self.test_dataset = EhrDataset(self.data_path, task, mode='test')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True , collate_fn=self.pad_collate, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False , collate_fn=self.pad_collate, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False , collate_fn=self.pad_collate, num_workers=8)

    def pad_collate(self, batch):
        xx, yy, pid = zip(*batch)
        # convert to tensor
        lens = torch.as_tensor([len(x) for x in xx])
        xx = [torch.tensor(x) for x in xx]
        yy = [torch.tensor(y) for y in yy]
        xx_pad = torch.nn.utils.rnn.pad_sequence(xx, batch_first=True, padding_value=0)
        yy_pad = torch.nn.utils.rnn.pad_sequence(yy, batch_first=True, padding_value=0)
        return xx_pad.float(), yy_pad.float(), lens, pid


class DlPipeline(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = config["model"]
        self.task = config["task"]

        self.demo_dim = config["demo_dim"]
        self.lab_dim = config["lab_dim"]
        self.input_dim = self.demo_dim + self.lab_dim
        config["input_dim"] = self.input_dim

        self.hidden_dim = config["hidden_dim"]
        self.output_dim = config["output_dim"]
        self.learning_rate = config["learning_rate"]
        self.main_metric = config["main_metric"]
        self.los_info = config.get("los_info", None)

        self.cur_best_performance = {}
        self.embedding: torch.Tensor

        if self.model_name == "StageNet":
            config["chunk_size"] = self.hidden_dim

        model_class = getattr(models, self.model_name)
        self.ehr_encoder = model_class(**config)

        if self.task in ["mortality", "readmission"]:
            self.head = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim), nn.Dropout(0.0), nn.Sigmoid())
        elif self.task == "los":
            self.head = nn.Sequential(nn.Linear(self.hidden_dim, self.output_dim), nn.Dropout(0.0))
        elif self.task == "multitask":
            self.head = models.heads.MultitaskHead(self.hidden_dim, self.output_dim, drop=0.0)

        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.test_performance = {}
        self.test_outputs = {}

    def forward(self, x, lens):
        if self.model_name == "ConCare":
            x_demo, x_lab, mask = x[:, 0, :self.demo_dim], x[:, :, self.demo_dim:], generate_mask(lens)
            embedding, decov_loss = self.ehr_encoder(x_lab, x_demo, mask)
            embedding, decov_loss = embedding.to(x.device), decov_loss.to(x.device)
            self.embedding = embedding
            y_hat = self.head(embedding)
            return y_hat, embedding, decov_loss
        elif self.model_name in ["GRASP", "Agent"]:
            x_demo, x_lab, mask = x[:, 0, :self.demo_dim], x[:, :, self.demo_dim:], generate_mask(lens)
            embedding = self.ehr_encoder(x_lab, x_demo, mask).to(x.device)
            self.embedding = embedding
            y_hat = self.head(embedding)
            return y_hat, embedding
        elif self.model_name in ["AdaCare", "RETAIN", "TCN", "Transformer", "StageNet"]:
            mask = generate_mask(lens)
            embedding = self.ehr_encoder(x, mask).to(x.device)
            self.embedding = embedding
            y_hat = self.head(embedding)
            return y_hat, embedding
        elif self.model_name in ["GRU", "LSTM", "RNN", "MLP"]:
            embedding = self.ehr_encoder(x).to(x.device)
            self.embedding = embedding
            y_hat = self.head(embedding)
            return y_hat, embedding
        elif self.model_name in ["MCGRU"]:
            x_demo, x_lab = x[:, 0, :self.demo_dim], x[:, :, self.demo_dim:]
            embedding = self.ehr_encoder(x_lab, x_demo).to(x.device)
            self.embedding = embedding
            y_hat = self.head(embedding)
            return y_hat, embedding

    def _get_loss(self, x, y, lens):
        if self.model_name == "ConCare":
            y_hat, embedding, decov_loss = self(x, lens)
            y_hat, y = unpad_y(y_hat, y, lens)
            loss = get_loss(y_hat, y, self.task)
            loss += 10 * decov_loss
        else:
            y_hat, embedding = self(x, lens)
            y_hat, y = unpad_y(y_hat, y, lens)
            loss = get_loss(y_hat, y, self.task)
        return loss, y, y_hat, embedding

    def training_step(self, batch, batch_idx):
        x, y, lens, pid = batch
        loss, y, y_hat, _ = self._get_loss(x, y, lens)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, lens, pid = batch
        loss, y, y_hat, _ = self._get_loss(x, y, lens)
        self.log("val_loss", loss)
        outs = {'preds': y_hat, 'labels': y, 'val_loss': loss}
        self.validation_step_outputs.append(outs)
        return loss

    def on_validation_epoch_end(self):
        preds = torch.cat([x['preds'] for x in self.validation_step_outputs]).detach().cpu()
        labels = torch.cat([x['labels'] for x in self.validation_step_outputs]).detach().cpu()
        loss = torch.stack([x['val_loss'] for x in self.validation_step_outputs]).mean().detach().cpu()
        self.log("val_loss_epoch", loss)
        metrics = get_all_metrics(preds, labels, self.task, self.los_info)
        for k, v in metrics.items():
            self.log(k, v)
        main_score = metrics[self.main_metric]
        if check_metric_is_better(self.cur_best_performance, self.main_metric, main_score, self.task):
            self.cur_best_performance = metrics
            for k, v in metrics.items():
                self.log("best_"+k, v)
        self.validation_step_outputs.clear()
        return main_score

    def test_step(self, batch, batch_idx):
        x, y, lens, pid = batch
        loss, y, y_hat, embedding = self._get_loss(x, y, lens)
        outs = {'pids': pid, 'preds': y_hat, 'labels': y}
        self.test_step_outputs.append(outs)
        return loss

    def on_test_epoch_end(self):
        preds = torch.cat([x['preds'] for x in self.test_step_outputs]).detach().cpu().numpy().tolist()
        labels = torch.cat([x['labels'] for x in self.test_step_outputs]).detach().cpu().numpy().tolist()
        pids = []
        pids.extend([x['pids'] for x in self.test_step_outputs])
        self.test_performance = get_all_metrics(preds, labels, self.task, self.los_info)
        self.test_outputs = {'pids': pids, 'preds': preds, 'labels': labels, 'config': self.hparams}
        self.test_step_outputs.clear()
        return self.test_performance

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


def run_experiment(config):
    # data
    dataset_path = os.path.join(os.path.dirname(__file__), '..', '..', f'my_datasets/{config["dataset"]}/processed/split')
    dm = EhrDataModule(dataset_path, task=config["task"], batch_size=config["batch_size"])

    # logger
    logger = CSVLogger(save_dir="logs", name=f'{config["dataset"]}/{config["task"]}/dl_models', version=f"{config['model']}")

    # main metric
    main_metric = "auroc" if config["task"] in ["mortality", "readmission"] else "mse"
    mode = "max" if config["task"] in ["mortality", "readmission"] else "min"
    config["main_metric"] = main_metric

    # EarlyStop and checkpoint callback
    early_stopping_callback = EarlyStopping(monitor=main_metric, patience=config["patience"], mode=mode)
    checkpoint_callback = ModelCheckpoint(filename="best", monitor=main_metric, mode=mode)

    # seed for reproducibility
    L.seed_everything(42)

    # train/val/test
    pipeline = DlPipeline(config)
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = [1]
    else:
        accelerator = "cpu"
        devices = 1
    trainer = L.Trainer(accelerator=accelerator, devices=devices, max_epochs=config["epochs"], logger=logger, callbacks=[early_stopping_callback, checkpoint_callback])
    trainer.fit(pipeline, dm)

    # Load best model checkpoint
    best_model_path = checkpoint_callback.best_model_path
    print("best_model_path:", best_model_path)
    pipeline = DlPipeline.load_from_checkpoint(best_model_path, config=config)
    trainer.test(pipeline, dm)

    perf = pipeline.test_performance
    outs = pipeline.test_outputs
    return perf, outs


def parse_args():
    parser = argparse.ArgumentParser(description='Train and evaluate deep learning models for EHR data')

    # Basic configurations
    parser.add_argument('--model', '-m', type=str, nargs='+', required=True, help='Model name')
    parser.add_argument('--dataset', '-d', type=str, required=True, help='Dataset name', choices=['tjh', 'mimic-iv'])
    parser.add_argument('--task', '-t', type=str, required=True, help='Task name', choices=['mortality', 'readmission'])

    # Model and training hyperparameters
    parser.add_argument('--hidden_dim', '-hd', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', '-bs', type=int, default=256, help='Batch size')
    parser.add_argument('--epochs', '-e', type=int, default=50, help='Number of epochs')
    parser.add_argument('--patience', '-p', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--output_dim', '-od', type=int, default=1, help='Output dimension')

    # Additional configurations
    parser.add_argument('--output_root', type=str, default='logs', help='Root directory for saving outputs')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Set up the configuration dictionary
    config = {
        'dataset': args.dataset,
        'task': args.task,
        'hidden_dim': args.hidden_dim,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'patience': args.patience,
        'output_dim': args.output_dim,
    }

    # Set the input dimensions based on the dataset
    if args.dataset == 'tjh':
        config['demo_dim'] = 2
        config['lab_dim'] = 73
    elif args.dataset == 'mimic-iv':
        config['demo_dim'] = 2
        config['lab_dim'] = 42
    else:
        raise ValueError("Unsupported dataset. Choose either 'tjh' or 'mimic-iv'.")

    perf_all_df = pd.DataFrame()
    for model in args.model:
        # Add the model name to the configuration
        config['model'] = model

        # Print the configuration
        print("Configuration:")
        for key, value in config.items():
            print(f"{key}: {value}")

        # Run the experiment
        perf, outs = run_experiment(config)

        # Save the performance and outputs
        save_dir = os.path.join(args.output_root, f"{args.dataset}/{args.task}/dl_models/{model}")
        os.makedirs(save_dir, exist_ok=True)

        # Run bootstrap
        perf_boot = run_bootstrap(outs['preds'], outs['labels'], config)
        for key, value in perf_boot.items():
            perf_boot[key] = f'{value["mean"] * 100:.2f}±{value["std"] * 100:.2f}'

        # Save performance and outputs
        perf_boot = dict({
            'model': model,
            'dataset': args.dataset,
            'task': args.task,
        }, **perf_boot)
        perf_df = pd.DataFrame(perf_boot, index=[0])
        perf_df.to_csv(os.path.join(save_dir, "performance.csv"), index=False)
        pd.to_pickle(outs, os.path.join(save_dir, "outputs.pkl"))
        print(f"Performance and outputs saved to {save_dir}")

        # Append performance to the all performance DataFrame
        perf_all_df = pd.concat([perf_all_df, perf_df], ignore_index=True)

    # Save all performance
    perf_all_df.to_csv(os.path.join(args.output_root, f"{args.dataset}/{args.task}/dl_models/all_performance.csv"), index=False)
    print(f"All performances saved to {os.path.join(args.output_root, f'{args.dataset}/{args.task}/dl_models/all_performance.csv')}")
    print("All experiments completed.")