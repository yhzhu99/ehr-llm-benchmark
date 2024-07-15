import numpy as np
import pandas as pd

from utils.binary_classification_metrics import get_binary_metrics, get_all_metrics, check_metric_is_better

def bootstrap(preds, labels, K=100, N=1000, seed=42):
    """Bootstrap resampling for binary classification metrics. Resample K times"""
    
    length = len(preds)
    # length = N
    np.random.seed(seed)
    
    # Initialize a list to store bootstrap samples
    bootstrapped_samples = []

    # Create K bootstrap samples
    for _ in range(K):
        # Sample with replacement from the indices
        sample_indices = np.random.choice(length, length, replace=True)

        # Use the sampled indices to get the bootstrap sample of preds and labels
        sample_preds = preds[sample_indices]
        sample_labels = labels[sample_indices]
        
        # Store the bootstrap samples
        bootstrapped_samples.append((sample_preds, sample_labels))

    return bootstrapped_samples


def export_metrics(bootstrapped_samples):
    metrics = {"accuracy": [], "auroc": [], "auprc": [], "f1": [], "minpse": []}
    for sample in bootstrapped_samples:
        sample_preds, sample_labels = sample[0], sample[1]
        res = get_all_metrics(sample_preds, sample_labels)

        for k, v in res.items():
            metrics[k].append(v)

    # convert to numpy array
    for k, v in metrics.items():
        metrics[k] = np.array(v)
    
    # calculate mean and std
    for k, v in metrics.items():
        metrics[k] = {"mean": np.mean(v), "std": np.std(v)}
    return metrics

def run_bootstrap(preds, labels, K=100, seed=42):
    bootstrap_samples = bootstrap(preds, labels, K=K, seed=seed)
    metrics = export_metrics(bootstrap_samples)
    return metrics

    # for k, v in metrics.items():
    #     mean_var = 100*v['mean']
    #     std_var = 100*v['std']
    #     print(f"{k}: {mean_var:.2f}±{std_var:.2f}")