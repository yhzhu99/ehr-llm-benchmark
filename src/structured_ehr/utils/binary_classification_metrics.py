import torch
from torchmetrics import AUROC, Accuracy, AveragePrecision
from torchmetrics.classification import BinaryF1Score
import numpy as np
from sklearn import metrics as sklearn_metrics

def minpse(preds, labels):
    precisions, recalls, _ = sklearn_metrics.precision_recall_curve(labels, preds)
    minpse_score = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
    return minpse_score

def get_binary_metrics(preds, labels):
    accuracy = Accuracy(task="binary", threshold=0.5)
    auroc = AUROC(task="binary")
    auprc = AveragePrecision(task="binary")
    f1 = BinaryF1Score()

    # convert labels type to int
    labels = labels.type(torch.int)
    accuracy(preds, labels)
    auroc(preds, labels)
    auprc(preds, labels)
    f1(preds, labels)

    minpse_score = minpse(preds, labels)

    # return a dictionary
    return {
        "auroc": auroc.compute().item(),
        "auprc": auprc.compute().item(),
        "minpse": minpse_score,
        "f1": f1.compute().item(),
        "accuracy": accuracy.compute().item(),
    }