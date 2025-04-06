import torch
import torch.nn as nn
import torch.nn.functional as F


class MultitaskLoss(nn.Module):
    def __init__(self, task_num=2):
        super(MultitaskLoss, self).__init__()
        self.task_num = task_num
        self.alpha = nn.Parameter(torch.ones((task_num)))
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss()

    def forward(self, outcome_pred, los_pred, outcome, los):
        loss0 = self.bce(outcome_pred, outcome)
        loss1 = self.mse(los_pred, los)
        return loss0 * self.alpha[0] + loss1 * self.alpha[1]


def get_multitask_loss(outcome_pred, los_pred, outcome, los):
    mtl = MultitaskLoss(task_num=2)
    return mtl(outcome_pred, los_pred, outcome, los)


def get_loss(preds, labels, task):
    if task in ["outcome", "readmission"]:
        if len(labels.shape) > 1 and labels.shape[-1] > 1:
            labels = labels[:, 0] if task == "outcome" else labels[:, 2]
        loss = F.binary_cross_entropy(preds, labels)
    elif task == "los":
        if len(labels.shape) > 1 and labels.shape[-1] > 1:
            labels = labels[:, 1]
        loss = F.mse_loss(preds, labels)
    elif task == "multitask":
        loss = get_multitask_loss(preds[:, 0], preds[:, 1], labels[:, 0], labels[:, 1])

    return loss