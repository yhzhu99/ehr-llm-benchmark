import torch
from torchmetrics import AUROC, Accuracy, AveragePrecision, Precision, Recall, F1Score
from torchmetrics.classification import BinaryF1Score, ConfusionMatrix
import numpy as np
from sklearn import metrics as sklearn_metrics

def minpse(preds, labels):
    precisions, recalls, thresholds = sklearn_metrics.precision_recall_curve(labels, preds)
    minpse_score = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])
    return minpse_score

def get_binary_metrics(preds, labels):
    accuracy = Accuracy(task="binary", threshold=0.5)
    auroc = AUROC(task="binary")
    auprc = AveragePrecision(task="binary")
    f1 = BinaryF1Score()
    precision_metric = Precision(task="binary", threshold=0.5)
    recall_metric = Recall(task="binary", threshold=0.5)
    # convert labels type to int
    labels = labels.type(torch.int)
    accuracy(preds, labels)
    auroc(preds, labels)
    auprc(preds, labels)
    f1(preds, labels)
    precision_metric(preds, labels)
    recall_metric(preds, labels)

    minpse_score = minpse(preds, labels)

    # return a dictionary
    return {
        "accuracy": accuracy.compute().item(),
        "auroc": auroc.compute().item(),
        "auprc": auprc.compute().item(),
        "f1": f1.compute().item(),
        "precision": precision_metric.compute().item(),
        "recall": recall_metric.compute().item(),
        "minpse": minpse_score,
    }

def get_multi_class_metrics(preds, labels, num_classes):
    # Assuming preds are logits and labels are class indices
    labels = labels.squeeze(dim=-1)
    preds_classes = torch.argmax(preds, dim=1)  # Get predicted classes

    # Metrics calculation
    accuracy = Accuracy(task="multiclass", num_classes=num_classes)(preds_classes, labels).item()
    f1_macro = F1Score(task="multiclass", num_classes=num_classes, average='macro')(preds_classes, labels).item()
    f1_micro = F1Score(task="multiclass", num_classes=num_classes, average='micro')(preds_classes, labels).item()
    precision_macro = Precision(task="multiclass", num_classes=num_classes, average='macro')(preds_classes, labels).item()
    recall_macro = Recall(task="multiclass", num_classes=num_classes, average='macro')(preds_classes, labels).item()

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
    }

def check_metric_is_better(cur_best, score, main_metric='auroc'):
    if cur_best == {}:
        return True
    if score > cur_best[main_metric]:
        return True
    return False

def test_get_multi_class_metrics():
    preds = torch.tensor([[0.1, 0.2, 0.7], [0.3, 0.4, 0.3], [0.5, 0.1, 0.4]])
    labels = torch.tensor([[2], [0], [1]])
    
    print(preds.shape, labels.shape)
    metrics = get_multi_class_metrics(preds, labels, 3)
    print(metrics)

if __name__ == "__main__":
    test_get_multi_class_metrics()
