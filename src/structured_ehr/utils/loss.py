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

    def forward(self, mortality_pred, los_pred, mortality, los):
        loss0 = self.bce(mortality_pred, mortality)
        loss1 = self.mse(los_pred, los)
        return loss0 * self.alpha[0] + loss1 * self.alpha[1]


def get_multitask_loss(mortality_pred, los_pred, mortality, los):
    mtl = MultitaskLoss(task_num=2)
    return mtl(mortality_pred, los_pred, mortality, los)


def get_loss(preds, labels, task):
    if task in ["mortality", "readmission"]:
        if len(labels.shape) > 1 and labels.shape[-1] > 1:
            labels = labels[:, 0] if task == "mortality" else labels[:, 2]
        loss = F.binary_cross_entropy(preds, labels)
    elif task == "los":
        if len(labels.shape) > 1 and labels.shape[-1] > 1:
            labels = labels[:, 1]
        loss = F.mse_loss(preds, labels)
    elif task == "multitask":
        loss = get_multitask_loss(preds[:, 0], preds[:, 1], labels[:, 0], labels[:, 1])

    return loss